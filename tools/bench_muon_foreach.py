"""GPU probe for the fused Newton-Schulz elementwise chain (parity + ms, before/after).

The card is shared, so run this ONLY under the lock:

    flock /tmp/dormouse-gpu.lock uv run python tools/bench_muon_foreach.py

It is deliberately not part of the CPU test suite: on CPU `_foreach_*` decomposes into
one aten op per tensor, so the launch collapse only shows on the card. The script builds
the live shape groups, times the scalar chain ("before") and the `_polar_many` chain
("after") with CUDA events, counts aten op launches with `torch.profiler`, and prints the
three parity gates before any timing number is trusted. Refuses to run without CUDA.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

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
WARMUP = 10
ITERS = 50


def scalar_polar(batch: list[torch.Tensor], steps: int = NS_STEPS) -> list[torch.Tensor]:
    """Run the pre-change scalar chain, fp32, one launch per op per group."""
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


def conditioned(shape: tuple[int, ...], count: int) -> torch.Tensor:
    """Build matrices whose spectrum sits inside the quintic's basin."""
    left, singular, right = torch.linalg.svd(
        torch.randn(*shape, device="cuda"), full_matrices=False
    )
    spectrum = torch.linspace(0.8, 1.2, singular.numel(), device="cuda")
    return torch.stack([left @ torch.diag(spectrum) @ right for _ in range(count)])


def groups() -> list[torch.Tensor]:
    """Build the live rectangles plus the packed qkv heads and one tall matrix."""
    torch.manual_seed(31)
    return [
        conditioned((64, 832), 8),
        conditioned((832, 2048), 4),
        conditioned((2048, 832), 2),
        conditioned((2, 4), 16),
        conditioned((4, 4), 8),
        conditioned((832, 64), 4),  # tall: transpose arm
    ]


def parity(batch: list[torch.Tensor]) -> tuple[float, float, float]:
    """Return orthogonality, direction and elementwise parity against the scalar chain."""
    fused, scalar = _polar_many(batch), scalar_polar(batch)
    ortho = direction = maximum = 0.0
    for got, ref in zip(fused, scalar, strict=True):
        gram = (
            got @ got.transpose(-2, -1)
            if got.shape[-2] <= got.shape[-1]
            else got.transpose(-2, -1) @ got
        )
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        ortho = max(ortho, (gram - eye).abs().max().item())
        direction = max(
            direction,
            ((got - ref).norm(dim=(-2, -1)) / ref.norm(dim=(-2, -1))).max().item(),
        )
        maximum = max(maximum, (got - ref).abs().max().item())
    return ortho, direction, maximum


def step_parity() -> float:
    """Return the worst full-step difference from the pre-change path."""
    torch.manual_seed(41)
    whole = [torch.nn.Parameter(torch.randn(64, 832, device="cuda")) for _ in range(8)]
    qkv = [torch.nn.Parameter(torch.randn(12, 4, device="cuda")) for _ in range(4)]
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
                {"params": params[:8], "head_split": None},
                {"params": params[8:], "head_split": (4, 8)},
            ],
            lr=0.02,
        )

    from dormouse.train import muon as muon_module

    def scalar_chain(batch: list[torch.Tensor], steps: int = NS_STEPS) -> list[torch.Tensor]:
        return scalar_polar(batch, steps)

    with patch.object(muon_module, "_polar_many", scalar_chain):
        optimizer(left).step()
    optimizer(right).step()
    return max(
        max(
            (a - b).abs().max().item(),
            (a.grad - b.grad).abs().max().item()
            if a.grad is not None and b.grad is not None
            else 0.0,
        )
        for a, b in zip(left, right, strict=True)
    )


def count(fn: Callable[[], Any]) -> dict[str, int]:
    """Count the aten launches of one call with the CPU and CUDA profilers."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    wanted = ("aten::mul", "aten::add", "aten::bmm", "aten::_foreach_mul", "aten::_foreach_add")
    return {event.key: event.count for event in prof.key_averages() if event.key in wanted}


def timing(fn: Callable[[], Any]) -> float:
    """Return the mean CUDA milliseconds per call, after warmup."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
        torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
    )
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / ITERS)


def main() -> None:
    """Refuse without CUDA; otherwise print parity gates, launch counts and timings."""
    if not torch.cuda.is_available():
        print(
            "bench_muon_foreach: CUDA unavailable - this is a card probe; refusing.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    batch = groups()
    ortho, direction, maximum = parity(batch)
    whole_step = step_parity()
    print(f"parity  ‖QQᵀ-I‖∞={ortho:.3e} (≤1e-5)  direction={direction:.3e} (≤1e-6)")
    print(f"        max|Δ|={maximum:.3e}  full-step max|Δ|={whole_step:.3e} (≤{EPS:.3e})")
    assert ortho <= 1e-5 and direction <= 1e-6 and maximum <= EPS and whole_step <= EPS

    def before_fn() -> list[torch.Tensor]:
        return scalar_polar(batch)

    def after_fn() -> list[torch.Tensor]:
        return _polar_many(batch)

    before, after = count(before_fn), count(after_fn)
    for key in ("aten::mul", "aten::add", "aten::bmm", "aten::_foreach_mul", "aten::_foreach_add"):
        print(f"launches  {key:24s} {before.get(key, 0):6d} -> {after.get(key, 0):6d}")

    before_ms, after_ms = timing(before_fn), timing(after_fn)
    print(f"ms/step   {before_ms:.1f} -> {after_ms:.1f}  saved {before_ms - after_ms:.1f}")
    print("old engine's measured saving: 738 ms (992 -> 254)")


if __name__ == "__main__":
    main()
