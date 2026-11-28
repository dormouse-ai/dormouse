"""The index-mapped attention against the gather it replaces: parity, and which arm ran.

`dsa.py` keeps its eager gather path (`_Pick` + `_gathered` + `_index_put_`), and it is
the anchor here for the same reason the bank's anchor is kept: a kernel whose forward
matches and whose backward does not is the version that trains a different model. Every
gate below compares this kernel's output and its three input gradients against that arm
on the same seed.

THE TOLERANCE IS DERIVED FROM THE ANCHOR, NOT CHOSEN. In fp32 the two arms compute the
same expression and differ only in summation order: measured 2.4e-7 relative, and the
gate is 1e-5 — two orders above the measurement, below any real defect. On a bf16 step
the anchor moves under itself: `torch.autocast` casts the softmax weights to bf16 before
the `Σ p·v` einsum, so the eager arm's own fp32-to-bf16 departure on `qkv.weight` is
4.7e-2, while this kernel's is 5.6e-3 (it keeps `p` in fp32). The test states both
numbers and asserts the ordering — a kernel that is not closer to its own fp32 result
than the anchor is would fail, and a gate that hid the difference behind one loose
tolerance would pass it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_kernels.engine import sparse

from dormouse.model.attn_variants import dsa as dsa_module
from dormouse.model.attn_variants.dsa import DSA, _Pick

#: The card's own skip, in the wording the bank's gate uses.
REQUIRES_CARD = torch.cuda.is_available()
NO_CARD = "the index-mapped kernel needs a CUDA device; the anchor is what CI runs"
ON_CARD = pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)

#: fp32 parity, as a fraction of the gradient's largest entry: same expression, different
#: summation order. Measured 2.4e-7; 1e-5 is that with two decades of margin.
FP32_PARITY = 1e-5


def _layer(**kwargs: int) -> DSA:
    """A DSA layer on the card, small enough that every gate runs in a second."""
    return DSA(
        int(kwargs.get("d_model", 64 * 4)),
        int(kwargs.get("n_heads", 4)),
        int(kwargs.get("head_dim", 64)),
        k=int(kwargs.get("k", 32)),
        window=int(kwargs.get("window", 16)),
        d_idx=16,
        chunk=4,
        device="cuda",
    )


def _grads(
    layer: DSA, x: torch.Tensor, *, amp: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the layer on `x` under one arm and return its output and every gradient."""
    for parameter in layer.parameters():
        parameter.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = layer(x)
        y.float().square().mean().backward()
    return y.detach().float(), {
        name: p.grad.float() for name, p in layer.named_parameters() if p.grad is not None
    }


def _worst(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference as a fraction of `b`'s own largest entry."""
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


@ON_CARD
def test_the_kernel_is_the_gather_it_replaces_in_fp32() -> None:
    """Forward AND every parameter gradient, fp32, against the eager arm."""
    torch.manual_seed(21)
    layer = _layer()
    x = torch.randn(2, 64, layer.d_model, device="cuda")
    worst = 0.0
    for arm in (False, True):
        sparse.set_sparse(arm)
        y, grads = _grads(layer, x, amp=False)
        if not arm:
            eager, eager_grads = y, grads
        else:
            assert sorted(grads) == sorted(eager_grads)
            worst = max(worst, _worst(y, eager), *(_worst(grads[n], eager_grads[n]) for n in grads))
            assert all(bool(torch.isfinite(g).all()) for g in grads.values())
    sparse.set_sparse(True)
    print(f"\nlargest relative difference against the eager gather: {worst:.1e}")
    assert worst < FP32_PARITY


@ON_CARD
def test_the_bf16_band_belongs_to_the_anchor() -> None:
    """The kernel keeps `p` in fp32; `autocast` rounds the anchor's to bf16.

    Both arms are compared against their OWN fp32 result, so the number that is measured
    is each arm's bf16 departure and not the distance between two different roundings.
    """
    torch.manual_seed(22)
    layer = _layer()
    x = torch.randn(2, 64, layer.d_model, device="cuda")
    departure = {}
    for arm in (False, True):
        sparse.set_sparse(arm)
        _, coarse = _grads(layer, x, amp=True)
        _, exact = _grads(layer, x, amp=False)
        departure[arm] = max(_worst(coarse[n], exact[n]) for n in exact)
    sparse.set_sparse(True)
    eager_band, kernel_band = departure[False], departure[True]
    print(
        f"\nbf16 departure of `qkv.weight` etc.: anchor {eager_band:.1e}, kernel {kernel_band:.1e}"
    )
    assert 4.0 * 2.0**-8 < eager_band < 1.0, "the anchor's own bf16 band is 4.7e-2 here"
    assert kernel_band < eager_band


@ON_CARD
def test_the_scatter_is_the_eager_scatter() -> None:
    """`dq`, `dK` and `dV` from the kernel against `_Pick`'s gather-and-scatter, fp32.

    The layer-level test above cannot say WHICH gradient moved, so this one names them:
    the kernel's `tl.atomic_add` into `dK`/`dV` and its `dq` are compared directly to the
    autograd of the eager expression, on the layer's own index map (window plus top-k,
    duplicates and invisible slots included).
    """
    torch.manual_seed(23)
    layer = _layer()
    x = torch.randn(2, 64, layer.d_model, device="cuda")
    q, key, value = (t.detach().clone().requires_grad_(True) for t in layer._project(x))
    at, visible = layer.attended(x.detach())
    batch, seq, heads, width = q.shape
    keep = at.shape[-1]
    place = at.unsqueeze(1).unsqueeze(-1).expand(batch, heads, seq, keep, width)

    def eager() -> torch.Tensor:
        keys = _Pick.apply(key.transpose(1, 2), place)
        values = _Pick.apply(value.transpose(1, 2), place)
        logits = torch.einsum("bhsd,bhskd->bhsk", q.transpose(1, 2), keys) * layer.scale
        weights = torch.softmax(logits.masked_fill(~visible.unsqueeze(1), -torch.inf), dim=-1)
        return torch.einsum("bhsk,bhskd->bhsd", weights, values).transpose(1, 2)

    torch.manual_seed(24)
    cotangent = torch.randn(batch, seq, heads, width, device="cuda")
    mine = sparse.attention(q, key, value, at, visible, scale=layer.scale)
    assert bool(torch.isfinite(mine).all())
    reference = eager()
    worst = _worst(mine, reference)
    for name, a, b in zip(
        ("q", "k", "v"),
        torch.autograd.grad(mine, (q, key, value), cotangent),
        torch.autograd.grad(reference, (q, key, value), cotangent),
        strict=True,
    ):
        worst = max(worst, _worst(a, b))
        print(f"\n  d{name}: {_worst(a, b):.1e}")
    print(f"forward: {_worst(mine, reference):.1e}")
    assert worst < FP32_PARITY


@ON_CARD
def test_the_kernel_is_the_thing_that_ran() -> None:
    """Two launches per call, counted by the module — a flag alone proves nothing."""
    torch.manual_seed(25)
    layer = _layer()
    x = torch.randn(1, 64, layer.d_model, device="cuda")
    sparse.set_sparse(True)
    sparse.reset_sparse_launches()
    layer(x).float().square().mean().backward()
    assert sparse.sparse_launches() == 2, "one forward launch and one backward launch"
    sparse.reset_sparse_launches()
    sparse.set_sparse(False)
    layer(x).float().square().mean().backward()
    assert sparse.sparse_launches() == 0, "the anchor must not run under the kernel's name"
    sparse.set_sparse(True)


@ON_CARD
def test_the_sparse_path_is_deterministic_and_equals_its_dense_control() -> None:
    """Same index map, same bits — and `dense()` runs through the same kernel.

    `window >= seq` makes the selection empty of arithmetic, so the sparse path and the
    module's dense anchor read the same positions; the kernel is deterministic per launch,
    so the equality is `torch.equal`, not a tolerance.
    """
    torch.manual_seed(26)
    layer = DSA(16, 2, 8, k=32, window=32, d_idx=16, chunk=4, device="cuda")
    x = torch.randn(1, 32, 16, device="cuda")
    sparse.set_sparse(True)
    assert torch.equal(layer(x), layer.dense(x))
    assert torch.equal(layer(x), layer(x))


@ON_CARD
def test_index_reuse_on_the_kernel_matches_an_external_fp64_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimized path's second pass: no indexer pass, kernel launched, fp64 anchor.

    The anchor is numpy's softmax over the SAME `[b, s, k+window]` map, written from
    the formula in fp64 — not the eager gather arm, which is internal to this repo.
    """
    monkeypatch.setattr(dsa_module, "_REUSE_INDEX", True)
    torch.manual_seed(43)
    layer = _layer()
    x = torch.randn(2, 64, layer.d_model, device="cuda")
    sparse.set_sparse(True)
    sparse.reset_sparse_launches()
    first = layer(x)
    reused_counters = Counters()
    second = layer(x, reused_counters)
    assert torch.equal(first, second), "the reused map must give the identical write"
    assert sparse.sparse_launches() >= 2, "the optimized path must actually have run"
    assert reused_counters.counts["dsa_index_macs"] == 0
    assert reused_counters.counts["reuse_tokens"] == 2 * 64
    reused_counters.check()

    q, key, value = layer._project(x)
    tail = layer.indexer.select(x, layer.k)
    at, visible = layer._layout(tail, x.shape[1])
    q64 = q.detach().double().cpu().numpy()
    k64 = key.detach().double().cpu().numpy()
    v64 = value.detach().double().cpu().numpy()
    at64 = at.detach().cpu().numpy()
    vis = visible.detach().cpu().numpy().astype(bool)
    width = layer.head_dim
    out = np.empty_like(q64)
    for b in range(q64.shape[0]):
        for h in range(q64.shape[2]):
            for t in range(q64.shape[1]):
                idx, live = at64[b, t], vis[b, t]
                logits = k64[b, idx, h, :] @ q64[b, t, h, :] / np.sqrt(width)
                logits = np.where(live, logits, -np.inf)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                out[b, t, h, :] = p @ v64[b, idx, h, :]
    proj = layer.proj.weight.detach().double().cpu().numpy()
    want = out.reshape(q64.shape[0], q64.shape[1], layer.d_kv) @ proj.T
    got = second.detach().double().cpu().numpy()
    worst = float(np.abs(got - want).max() / np.abs(want).max())
    print(f"\nreuse + kernel against external fp64 anchor: {worst:.1e}")
    assert worst < 2e-5


def test_the_kernel_refuses_rather_than_pretending() -> None:
    """The entry point is not a fallback: off, or on the CPU, it names its escape."""
    cpu = torch.zeros(1, 4, 2, 8)
    at = torch.zeros(1, 4, 4, dtype=torch.long)
    visible = torch.ones(1, 4, 4, dtype=torch.bool)
    sparse.set_sparse(True)
    try:
        with pytest.raises(LoudError, match="cpu tensor or with the kernel off"):
            sparse.attention(cpu, cpu, cpu, at, visible, scale=0.1)
    finally:
        sparse.set_sparse(True)


def test_the_refusal_names_its_escape_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_require` is the kernel entry point: no kernels is a named refusal, not a fallback."""
    monkeypatch.setattr(sparse, "_build", lambda: None)
    with pytest.raises(LoudError, match="ran without Triton"):
        sparse._require()
    monkeypatch.setattr(sparse, "_build", lambda: {"fwd": None, "bwd": None})
    assert sparse._require() == {"fwd": None, "bwd": None}


def test_the_launch_counter_reads_and_resets() -> None:
    """The counter the card gate reads; on a CPU job it is simply zero."""
    sparse.reset_sparse_launches()
    assert sparse.sparse_launches() == 0
    sparse._count(2)
    assert sparse.sparse_launches() == 2
    sparse.reset_sparse_launches()
    assert sparse.sparse_launches() == 0


def test_an_empty_index_map_is_refused() -> None:
    """`keep = 0` is a caller's bug, and the check does not need a card to fire."""
    empty = torch.zeros(1, 4, 2, 8)
    with pytest.raises(LoudError, match="names no key"):
        sparse.attention(
            empty,
            empty,
            empty,
            torch.zeros(1, 4, 0, dtype=torch.long),
            torch.ones(1, 4, 0, dtype=torch.bool),
            scale=0.1,
        )


def test_the_tiles_are_powers_of_two() -> None:
    """`tl.arange` needs a power of two, and the block is never larger than `keep`."""
    assert [_pow2(n) for n in (1, 12, 32, 640)] == [16, 16, 32, 1024]
    assert min(sparse.BN, _pow2(12)) == 16
    assert min(sparse.BN, _pow2(640)) == sparse.BN


def _pow2(n: int) -> int:
    """The module's own rounding, imported by name so the assertion is on the code."""
    return sparse._pow2(n)
