"""The shape-grouped polar against the per-matrix chain it replaces.

The numbers here are computed by running both schedules on the same fp32 tensors; the
anchors (`‖QQᵀ-I‖∞ ≤ 1e-5`, fp32 epsilon on a whole step) come from the module docstring
and the task, not from either schedule's output.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import MissingReport
from stacks import reference_stack

from dormouse.train.muon import NS_GROUPED_COUNTER, Muon, _polar_many, polar
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

EPS = torch.finfo(torch.float32).eps


def _fixture() -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Two copies of 10 whole matrices and 5 packed qkv weights (35 regions)."""
    torch.manual_seed(21)
    whole = torch.randn(10, 64, 128)
    qkv = torch.randn(5, 12, 4)
    left = [torch.nn.Parameter(t.clone()) for t in whole] + [
        torch.nn.Parameter(t.clone()) for t in qkv
    ]
    right = [torch.nn.Parameter(t.clone()) for t in whole] + [
        torch.nn.Parameter(t.clone()) for t in qkv
    ]
    return left, right


def _optimizer(params: list[torch.nn.Parameter], *, grouped: bool) -> Muon:
    """The router's two roles: whole matrices, and qkv's Q/K heads plus whole V."""
    return Muon(
        [
            {"params": params[:10], "head_split": None},
            {"params": params[10:], "head_split": (4, 8)},
        ],
        lr=0.02,
        grouped=grouped,
    )


def _grads(params: list[torch.nn.Parameter], seed: int = 22) -> None:
    torch.manual_seed(seed)
    for param in params:
        param.grad = torch.randn_like(param)


def _conditioned(shape: tuple[int, ...], count: int) -> torch.Tensor:
    """Matrices with σ in [0.8, 1.2] — the quintic's precision, not §3's ceiling."""
    out = []
    for _ in range(count):
        left, singular, right = torch.linalg.svd(torch.randn(*shape), full_matrices=False)
        spectrum = torch.linspace(0.8, 1.2, singular.numel())
        out.append(left @ torch.diag(spectrum) @ right)
    return torch.stack(out)


def test_grouped_polar_is_orthogonal_and_matches_each_matrix() -> None:
    torch.manual_seed(23)
    groups = {
        (64, 128): _conditioned((64, 128), 10),
        (2, 4): _conditioned((2, 4), 20),
        (4, 4): _conditioned((4, 4), 5),
    }
    for shape, matrices in groups.items():
        batched = polar(matrices)
        one_by_one = torch.stack([polar(matrices[i]) for i in range(len(matrices))])
        gram = batched @ batched.transpose(-2, -1)
        eye = torch.eye(shape[0], dtype=gram.dtype)
        assert (gram - eye).abs().max().item() <= 1e-5, shape
        relative = (batched - one_by_one).norm(dim=(-2, -1)) / one_by_one.norm(dim=(-2, -1))
        assert relative.max().item() <= 1e-6, (shape, relative.max().item())


def test_grouped_step_moves_every_parameter_and_moment_identically() -> None:
    left, right = _fixture()
    _grads(left)
    _grads(right)
    grouped, control = _optimizer(left, grouped=True), _optimizer(right, grouped=False)
    grouped.step()
    control.step()
    # bmm and mm are the same fp32 products with a different accumulation order in
    # the GEMM: the measured difference is at most two ULP, i.e. an fp32 epsilon.
    for a, b in zip(left, right, strict=True):
        assert torch.allclose(a, b, atol=EPS, rtol=EPS), tuple(a.shape)
        assert torch.allclose(a.grad, b.grad, atol=EPS, rtol=EPS), tuple(a.shape)
        assert torch.allclose(
            grouped.state[a]["momentum"], control.state[b]["momentum"], atol=EPS, rtol=EPS
        ), tuple(a.shape)


def test_a_whole_step_is_equal_end_to_end() -> None:
    torch.manual_seed(24)
    model, twin = reference_stack(), reference_stack()
    twin.load_state_dict(model.state_dict())
    control = MixOptimizer(model, muon_grouped=False)
    grouped = MixOptimizer(twin, muon_grouped=True)
    ids = torch.randint(0, 16, (2, 9))
    expected = train_step(model, ids, optimizer=control)
    got = train_step(twin, ids, optimizer=grouped)
    assert got.loss == pytest.approx(expected.loss, abs=EPS, rel=0.0)
    assert got.grad_norm == pytest.approx(expected.grad_norm, abs=EPS, rel=0.0)
    states = dict(twin.named_parameters())
    for name, param in model.named_parameters():
        assert torch.allclose(param, states[name], atol=EPS, rtol=EPS), name
        assert torch.allclose(param.grad, states[name].grad, atol=EPS, rtol=EPS), name


def test_a_per_head_matrix_never_joins_a_whole_bucket_of_the_same_shape() -> None:
    torch.manual_seed(25)
    params = [
        torch.nn.Parameter(torch.randn(12, 4)),  # head_split: 4 heads of [2, 4] + V [4, 4]
        torch.nn.Parameter(torch.randn(2, 4)),  # whole [2, 4]
        torch.nn.Parameter(torch.randn(4, 4)),  # whole [4, 4]
    ]
    for param in params:
        param.grad = torch.randn_like(param)
    calls: list[tuple[torch.Size, ...]] = []
    real_many = _polar_many

    def recording_many(batch: list[torch.Tensor], steps: int = 5) -> list[torch.Tensor]:
        calls.append(tuple(x.shape for x in batch))
        return real_many(batch, steps=steps)

    with patch("dormouse.train.muon._polar_many", recording_many):
        Muon(
            [
                {"params": params[:1], "head_split": (4, 8)},
                {"params": params[1:], "head_split": None},
            ]
        ).step()
    sizes = {(tuple(shape[-2:]), shape[0]) for group in calls for shape in group}
    assert ((2, 4), 4) in sizes, "the four Q/K heads were not batched together"
    assert ((2, 4), 1) in sizes and ((4, 4), 1) in sizes, "a whole matrix joined a head bucket"


def test_the_counter_proves_the_batched_path_ran() -> None:
    left, right = _fixture()
    _grads(left)
    _grads(right)
    counters = Counters()
    _optimizer(left, grouped=True).step(counters=counters)
    assert counters.counts[NS_GROUPED_COUNTER] == 35
    counters.check()

    silent = Counters()
    silent.enable("ns_grouped", counter=NS_GROUPED_COUNTER, why="the flag is on")
    _optimizer(right, grouped=False).step(counters=silent)
    assert silent.counts[NS_GROUPED_COUNTER] == 0
    with pytest.raises(MissingReport, match=NS_GROUPED_COUNTER):
        silent.check()

    via_router = Counters()
    routed = reference_stack()
    ids = torch.randint(0, 16, (2, 9))
    logits = routed(ids[:, :-1])
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    ).backward()
    MixOptimizer(routed, muon_grouped=True).step(counters=via_router)
    assert via_router.counts[NS_GROUPED_COUNTER] > 0
    via_router.check()


def test_grouping_reduces_matmul_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"mm": 0, "bmm": 0}
    real_mm, real_bmm = torch.mm, torch.bmm

    def count_mm(*args: object, **kwargs: object) -> torch.Tensor:
        calls["mm"] += 1
        return real_mm(*args, **kwargs)  # type: ignore[arg-type]

    def count_bmm(*args: object, **kwargs: object) -> torch.Tensor:
        calls["bmm"] += 1
        return real_bmm(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "mm", count_mm)
    monkeypatch.setattr(torch, "bmm", count_bmm)

    left, right = _fixture()
    _grads(left)
    _grads(right)
    _optimizer(left, grouped=False).step()
    before = dict(calls)
    calls.update(mm=0, bmm=0)
    _optimizer(right, grouped=True).step()
    after = dict(calls)
    assert before["mm"] > 0 and after["bmm"] > 0 and after["mm"] == 0
    assert sum(after.values()) < sum(before.values()) // 3
    print(
        f"\nmatmul launches: {before['mm']} mm + {before['bmm']} bmm -> "
        f"{after['mm']} mm + {after['bmm']} bmm"
    )
