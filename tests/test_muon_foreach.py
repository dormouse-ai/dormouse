"""CPU call-count and parity proof for the fused Newton–Schulz elementwise chain.

The "before" path is a local copy of the scalar chain that shipped before the change
(`_scalar_polar`): five fp32 elementwise ops per iteration, one launch per op per group.
The "after" path is `dormouse.train.muon._polar_many`: the same ops issued as
`_foreach_mul`/`_foreach_add` over every stacked group in one lockstep chain. Python-level
tensor dispatches are counted, because those are the launches the port is bound by; the
GEMMs are counted separately and must not move.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.train.muon import (
    NS_COEFFS,
    NS_EPS,
    NS_STEPS,
    Muon,
    _matmul,
    _polar_many,
    spectral_bound,
)

EPS = torch.finfo(torch.float32).eps


def _scalar_polar(batch: list[torch.Tensor], steps: int = NS_STEPS) -> list[torch.Tensor]:
    """The pre-change chain: scalar elementwise ops, one full iteration per group."""
    a, b, c = NS_COEFFS
    out: list[torch.Tensor] = []
    for x in batch:
        work = x.to(torch.float32)
        work = work / spectral_bound(work).clamp_min(NS_EPS)
        transposed = work.shape[-2] > work.shape[-1]
        if transposed:
            work = work.transpose(-2, -1)
        for _ in range(steps):
            gram = _matmul(work, work.transpose(-2, -1))
            work = a * work + _matmul(b * gram + c * _matmul(gram, gram), work)
        if transposed:
            work = work.transpose(-2, -1)
        out.append(work.contiguous())
    return out


def _scalar_many(batch: list[torch.Tensor], steps: int = NS_STEPS) -> list[torch.Tensor]:
    return _scalar_polar(batch, steps)


def _conditioned(shape: tuple[int, ...], count: int) -> torch.Tensor:
    """Matrices whose spectrum sits inside the quintic's basin (the orthogonality gate)."""
    out = []
    for _ in range(count):
        left, singular, right = torch.linalg.svd(torch.randn(*shape), full_matrices=False)
        spectrum = torch.linspace(0.8, 1.2, singular.numel())
        out.append(left @ torch.diag(spectrum) @ right)
    return torch.stack(out)


def _groups() -> list[torch.Tensor]:
    """Five shape groups, including a tall one that takes the transpose arm."""
    torch.manual_seed(31)
    return [
        _conditioned((64, 128), 4),
        _conditioned((2, 4), 8),
        _conditioned((4, 4), 4),
        _conditioned((16, 832), 2),
        _conditioned((128, 64), 2),  # tall: takes the transpose arm
    ]


def _count_calls(fn: Callable[[], Any]) -> dict[str, int]:
    """Count Python-level tensor dispatches issued by `fn`."""
    counts: dict[str, int] = defaultdict(int)
    mm, add = torch.Tensor.__mul__, torch.Tensor.__add__
    rmul, radd = torch.Tensor.__rmul__, torch.Tensor.__radd__
    bmm, tmm = torch.bmm, torch.mm
    fe_mul, fe_add = torch._foreach_mul, torch._foreach_add

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return inner

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "__mul__", wrap("mul", mm)))
        stack.enter_context(patch.object(torch.Tensor, "__rmul__", wrap("mul", rmul)))
        stack.enter_context(patch.object(torch.Tensor, "__add__", wrap("add", add)))
        stack.enter_context(patch.object(torch.Tensor, "__radd__", wrap("add", radd)))
        stack.enter_context(patch("torch.bmm", wrap("bmm", bmm)))
        stack.enter_context(patch("torch.mm", wrap("mm", tmm)))
        stack.enter_context(patch("torch._foreach_mul", wrap("foreach_mul", fe_mul)))
        stack.enter_context(patch("torch._foreach_add", wrap("foreach_add", fe_add)))
        fn()
    return dict(counts)


def test_foreach_chain_is_orthogonal_and_matches_scalar_chain() -> None:
    torch.manual_seed(32)
    batch = _groups()
    fused, scalar = _polar_many(batch), _scalar_polar(batch)
    for index, (got, ref) in enumerate(zip(fused, scalar, strict=True)):
        if got.shape[-2] <= got.shape[-1]:
            gram = got @ got.transpose(-2, -1)
        else:
            gram = got.transpose(-2, -1) @ got
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype)
        assert (gram - eye).abs().max().item() <= 1e-5, index
        relative = (got - ref).norm(dim=(-2, -1)) / ref.norm(dim=(-2, -1))
        assert relative.max().item() <= 1e-6, (index, relative.max().item())


def test_elementwise_launches_collapse_and_gems_do_not_move() -> None:
    batch = _groups()
    before = _count_calls(lambda: _scalar_polar(batch))
    after = _count_calls(lambda: _polar_many(batch))
    groups, steps = len(batch), NS_STEPS
    assert before["mul"] == 3 * groups * steps
    assert before["add"] == 2 * groups * steps
    assert after.get("mul", 0) == 0 and after.get("add", 0) == 0
    assert after["foreach_mul"] == 3 * steps
    assert after["foreach_add"] == 2 * steps
    gem_before = before.get("bmm", 0) + before.get("mm", 0)
    gem_after = after.get("bmm", 0) + after.get("mm", 0)
    assert gem_before == gem_after and gem_before > 0
    print(
        f"\naten::mul {before['mul']} -> 0, aten::add {before['add']} -> 0; "
        f"_foreach_mul 0 -> {after['foreach_mul']}, "
        f"_foreach_add 0 -> {after['foreach_add']}; "
        f"bmm+mm {gem_before} -> {gem_after}"
    )


def test_a_full_step_equals_the_pre_change_path() -> None:
    torch.manual_seed(41)
    whole = [torch.nn.Parameter(torch.randn(64, 128)) for _ in range(8)]
    qkv = [torch.nn.Parameter(torch.randn(12, 4)) for _ in range(4)]
    left = whole + qkv
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    for param in left:
        param.grad = torch.randn_like(param)
    for a, b in zip(left, right, strict=True):
        assert a.grad is not None
        b.grad = a.grad.clone()

    def optimizer(params: list[torch.nn.Parameter]) -> Muon:
        return Muon(
            [
                {"params": params[:3], "head_split": None},
                {"params": params[3:], "head_split": (4, 8)},
            ],
            lr=0.02,
        )

    with patch("dormouse.train.muon._polar_many", _scalar_many):
        optimizer(left).step()
    optimizer(right).step()
    for a, b in zip(left, right, strict=True):
        assert a.grad is not None and b.grad is not None
        assert torch.allclose(a, b, atol=EPS, rtol=EPS), tuple(a.shape)
        assert torch.allclose(a.grad, b.grad, atol=EPS, rtol=EPS), tuple(a.shape)


def test_missing_foreach_refuses_loudly() -> None:
    with (
        patch("dormouse.train.muon.torch._foreach_mul", None),
        pytest.raises(LoudError, match="_foreach"),
    ):
        _polar_many([torch.randn(4, 4)])
    with pytest.raises(LoudError, match="no matrices"):
        _polar_many([])
    with pytest.raises(LoudError, match="fp32 work tensors"):
        _polar_many([torch.randn(4, 4, dtype=torch.float64)])
    with pytest.raises(LoudError, match="cpu or cuda"):
        _polar_many([torch.randn(4, 4, device="meta")])
