"""E4: the batched KDA prefix-scan core is the same operator as the shipped loop.

The shipped KO1 path solves the per-chunk systems in two batched calls but still walks
the 16 chunks in Python. `_chunks_scan` composes the affine state transition with a
log2 parallel prefix. These gates pin the arithmetic: an fp64 anchor against the token
recurrence, equality with the shipped loop including gradients, the dispatch receipt,
and a 20-step paired parameter trace under `<= 1e-5`.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.model.attn_variants import kda as kda_module
from dormouse.model.attn_variants.kda import KDA, scan_calls

D_MODEL = 32
N_HEADS = 2
HEAD_DIM = 8


def _kernel_args(kda: KDA, seq: int, *, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Project a batch through the module's own weights, in kernel layout."""
    torch.manual_seed(seed)
    x = torch.randn(2, seq, kda.d_model, dtype=kda.qkv.weight.dtype)
    q, k, v, alpha, beta = kda._project(x)
    return (*(z.transpose(1, 2) for z in (q, k, v, alpha)), beta.transpose(1, 2))


def test_scan_is_the_token_recurrence_in_fp64(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix association must not move the output or the final state."""
    monkeypatch.setattr(kda_module, "SCAN", True)
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=7).double()
    args = _kernel_args(kda, 35)
    fast, state = kda._chunkwise(*args)
    slow, slow_state = kda.recurrent(*args)
    assert torch.allclose(fast, slow, atol=1e-9, rtol=0)
    assert torch.allclose(state, slow_state, atol=1e-9, rtol=0)


def _chunked_args(kda: KDA, seq: int) -> tuple[torch.Tensor, ...]:
    """The five core operands, reshaped to the chunked layout `_chunks_*` takes."""
    q, k, v, alpha, beta = _kernel_args(kda, seq)
    b, h, s, _ = q.shape
    nc = s // kda.chunk
    return (
        q.reshape(b, h, nc, kda.chunk, q.shape[-1]),
        k.reshape(b, h, nc, kda.chunk, k.shape[-1]),
        v.reshape(b, h, nc, kda.chunk, v.shape[-1]),
        alpha.reshape(b, h, nc, kda.chunk, alpha.shape[-1]),
        beta.reshape(b, h, nc, kda.chunk),
    )


def test_scan_equals_the_shipped_loop_including_gradients() -> None:
    """Same operands, same per-chunk arithmetic: fwd/state/d-inputs agree."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8).float()
    args = _chunked_args(kda, 32)
    shipped = kda._chunks_batched_solves(*args)
    scanned = kda._chunks_scan(*args)
    assert torch.allclose(shipped[0], scanned[0], atol=1e-5, rtol=0)
    assert torch.allclose(shipped[1], scanned[1], atol=1e-5, rtol=0)

    q = args[0].detach().clone().requires_grad_(True)
    v = args[2].detach().clone().requires_grad_(True)
    rest = (q, args[1].detach(), v, args[3].detach(), args[4].detach())
    a = kda._chunks_batched_solves(*rest)
    (a[0].sum() + a[1].sum()).backward()
    dq_a, dv_a = q.grad.clone(), v.grad.clone()
    q.grad = None
    v.grad = None
    b = kda._chunks_scan(*rest)
    (b[0].sum() + b[1].sum()).backward()
    assert torch.allclose(dq_a, q.grad, atol=1e-5, rtol=0)
    assert torch.allclose(dv_a, v.grad, atol=1e-5, rtol=0)


def test_flag_dispatches_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF is the shipped loop and issues no scan receipt; ON issues exactly one."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    args = _kernel_args(kda, 16)
    before = scan_calls()
    monkeypatch.setattr(kda_module, "SCAN", False)
    kda._chunkwise(*args)
    assert scan_calls() == before
    monkeypatch.setattr(kda_module, "SCAN", True)
    kda._chunkwise(*args)
    assert scan_calls() == before + 1


def test_twenty_steps_stay_within_onee_minus_five(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paired 20-step traces on one KDA block; scan vs shipped loop."""
    torch.set_num_threads(1)
    traces: list[list[torch.Tensor]] = []
    for use_scan in (False, True):
        monkeypatch.setattr(kda_module, "SCAN", use_scan)
        torch.manual_seed(11)
        kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8).float()
        opt = torch.optim.AdamW(kda.parameters(), lr=1e-2)
        trace = []
        for step in range(20):
            torch.manual_seed(100 + step)
            x = torch.randn(2, 16, D_MODEL)
            loss = kda(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(torch.cat([p.detach().flatten() for p in kda.parameters()]))
        traces.append(trace)
    worst = max((a - b).abs().max().item() for a, b in zip(traces[0], traces[1], strict=True))
    assert worst <= 1e-5
