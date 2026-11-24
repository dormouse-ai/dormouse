"""Sparse attention's read runs THROUGH the index map: no gathered slab, no `index_put_`.

WHAT THE LAYER DID BEFORE THIS FILE, AND WHAT IT COST. `dsa.py` built
`keys = source[place]` for every query, ran the scores and the softmax on that
`[batch, heads, queries, k+window, head_dim]` tile, and scattered the gradient back with
`index_put_(accumulate=True)`. At b8 x s1024 that tile is 5.4 GiB per tensor per layer:
32 GiB written and re-read per step across the three DSA layers, in 86 query slabs each,
plus a `zeros_like` of the key/value rows per slab in the backward. `tools/profile_step.py`
measured the bill as `_index_put_impl_` 780 ms + `indexing_backward_kernel` 761 ms —
1.54 s of a 3.6 s step, and it was all traffic: the arithmetic over the same index list is
`2·b·h·s·keep·d = 5.4 GFLOP` per layer, 0.1 ms at this card's peak.

WHAT THIS FILE DOES INSTEAD. One program owns ONE (batch, head, query). It reads its
`k+window` key rows straight out of the `[b, h, s, d]` projection through the index map,
runs the softmax online over the same rows, weights the values, and writes one output row.
Nothing of shape `[..., keep, d]` is ever allocated. The backward re-reads the same rows
(recompute, not store: the weights would be 168 MiB per layer at b8 and the keys are
8.4 MiB, so they are in L2) and scatters `dK`/`dV` with `tl.atomic_add` into the
`[b, h, s, d]` gradient — 16.8 MiB, L2-resident, instead of a generic `index_put_` over a
materialised buffer.

WHY THE ARITHMETIC IS UNCHANGED. `o_t = Σ_j p_j v_j` with `p = softmax(q_t·k[at_t]·scale)`
over the SAME `keep` positions, masked by the same visibility list: the selection is an
index list, and the kernel reads it. Only the summation order differs from `einsum`'s —
the reduction is over the same `head_dim`-wide dot and the same `keep` positions — which
is why the parity gate is bf16 epsilon and not bit-identity. `dsa.dense()` runs through
this kernel too, so `sparse == dense` under `window >= T` stays `torch.equal`.

THE REQUIRED PARITY IS THE ANCHOR'S. `tests/test_sparse_attn.py` compares this kernel
against `dsa.py`'s eager gather path (the arm that shipped) on the same seed: forward,
and every one of `dq`, `dK`, `dV`. The eager arm is kept, not deleted: it is the CPU path
and it is what the card gate measures against.

NO SILENT FALLBACK. `available()` is false only when Triton or CUDA is missing, and then
`dsa.py` runs the eager path — visibly. `sparse_launches()` is what the tests read to
prove a kernel ran rather than a flag being set.
"""
# ruff: noqa: N803

from __future__ import annotations

import os
from typing import Any, cast

import torch
from dormouse_core.errors import LoudError

from . import mfu40 as mfu40_kernels

#: One program owns one query, and reads its `BN` keys at a time. 64 is two sectors per
#: row at `d = 64` and keeps a `[BN, d]` fp32 tile at 16 KiB, which is what lets the
#: forward run at 4 warps without spilling (`tools/bench_sparse.py` prints `n_regs`).
BN = 128

#: The backward holds `k`, `v`, `p` and the two atomic tiles at once, so it reads 64 keys
#: at a time; measured, 128 there is 167 registers (two thirds of the file) and 41 ms
#: against this one's 23.
BN_BWD = 32 if mfu40_kernels.enabled("KO3") else 64

#: Warps per program. One query is a 64-wide row, so 4 warps already cover it and the
#: extra warps only buy latency hiding on the scattered loads.
WARPS = 4

#: Software pipelining of the key-row loop. 2 covers one load in flight; 3 was measured
#: equal and doubles the shared-memory bill.
STAGES = 2

#: Copies of the `dK`/`dV` buffer the backward scatters into, one per `pid % SHARDS`.
#: The layer's index map is not uniform — the prefix top-k makes early positions sticky,
#: so many queries vote for the same key row, and one fp32 buffer serialises those
#: atomics. Measured on the layer's own map: 50.6 ms per backward at one copy, 25 ms at
#: eight (16.8 MiB each, summed once: 134 MiB of read per tensor per layer).
SHARDS = 4 if mfu40_kernels.enabled("KO3") else 8

_KERNELS: dict[str, Any] = {}
_LAUNCHES = 0
_ON = os.environ.get("DORMOUSE_SPARSE", "1") != "0"


def set_sparse(on: bool) -> None:
    """Turn the fused gather-attention on or off for this process (the anchor's arm)."""
    global _ON
    _ON = bool(on)


def sparse_launches() -> int:
    """Return how many Triton launches this module has issued since the last reset."""
    return _LAUNCHES


def reset_sparse_launches() -> None:
    """Zero the launch counter; the flag keeps its value."""
    global _LAUNCHES
    _LAUNCHES = 0


def available() -> bool:
    """Return whether the kernel can run here: Triton compiled, CUDA present."""
    return torch.cuda.is_available() and _build() is not None


def enabled() -> bool:
    """Return whether the forward should take the kernel: asked for AND possible."""
    return _ON and available()


def _build() -> dict[str, Any] | None:  # pragma: no cover - requires triton + a device
    """Compile-once Triton kernels; `None` when Triton is not installed."""
    if _KERNELS:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit  # type: ignore[untyped-decorator]
    def fwd_kernel(  # type: ignore[no-untyped-def]
        Q,
        K,
        V,
        AT,
        VIS,
        OUT,
        ML,
        S,
        H,
        D: tl.constexpr,
        KEEP: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """One query: online softmax over its index map, then `Σ p·v`.

        Two passes over the same rows. The first finds `(m, l)` — the softmax's max and
        normaliser - and the second weights the values. One pass would have to hold the
        whole `keep` axis of logits in registers (640 fp32 = 2.5 KiB per program, which
        Triton cannot index dynamically), and storing them to global is 168 MiB per layer
        for a 8.4 MiB read. The second pass costs 7 ms per layer; the store cost 20.
        """
        pid = tl.program_id(0)
        qpos = pid % S
        bh = pid // S
        batch = bh // H
        head = bh % H
        off = bh * S * D
        cols = tl.arange(0, BD)
        dcol = cols < D
        q = tl.load(Q + off + qpos * D + cols, mask=dcol, other=0.0).to(tl.float32)
        at_row = AT + (batch * S + qpos) * KEEP
        vis_row = VIS + (batch * S + qpos) * KEEP

        mx = float("-inf")
        norm = 0.0
        for n0 in range(0, KEEP, BN):
            n = n0 + tl.arange(0, BN)
            nmask = n < KEEP
            idx = tl.load(at_row + n, mask=nmask, other=0)
            vis = tl.load(vis_row + n, mask=nmask, other=0)
            k = tl.load(
                K + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * SCALE
            s = tl.where((vis != 0) & nmask, s, float("-inf"))
            new_m = tl.maximum(mx, tl.max(s, 0))
            #: A block with nothing visible leaves `new_m` at the old max; only the very
            #: first block can be `-inf`, and it never is (a query always reads itself).
            safe = tl.where(new_m == float("-inf"), 0.0, new_m)
            norm = norm * tl.exp(mx - safe) + tl.sum(tl.exp(s - safe), 0)
            mx = new_m

        acc = tl.zeros([BD], tl.float32)
        for n0 in range(0, KEEP, BN):
            n = n0 + tl.arange(0, BN)
            nmask = n < KEEP
            idx = tl.load(at_row + n, mask=nmask, other=0)
            vis = tl.load(vis_row + n, mask=nmask, other=0)
            k = tl.load(
                K + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * SCALE
            p = tl.where((vis != 0) & nmask, tl.exp(s - mx), 0.0)
            v = tl.load(
                V + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(p[:, None] * v, 0)

        o = acc / tl.maximum(norm, 1e-30)
        tl.store(OUT + (batch * S + qpos) * H * D + head * D + cols, o, mask=dcol)
        tl.store(ML + 2 * pid, mx)
        tl.store(ML + 2 * pid + 1, norm)

    @triton.jit  # type: ignore[untyped-decorator]
    def fwd_single_kernel(  # type: ignore[no-untyped-def]
        Q,
        K,
        V,
        AT,
        VIS,
        OUT,
        ML,
        S,
        H,
        D: tl.constexpr,
        KEEP: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """KO3: one online-softmax pass; each key row and index slot is fetched once."""
        pid = tl.program_id(0)
        qpos = pid % S
        bh = pid // S
        batch = bh // H
        head = bh % H
        off = bh * S * D
        cols = tl.arange(0, BD)
        dcol = cols < D
        q = tl.load(Q + off + qpos * D + cols, mask=dcol, other=0.0).to(tl.float32)
        at_row = AT + (batch * S + qpos) * KEEP
        vis_row = VIS + (batch * S + qpos) * KEEP
        mx = float("-inf")
        norm = 0.0
        acc = tl.zeros([BD], tl.float32)
        for n0 in range(0, KEEP, BN):
            n = n0 + tl.arange(0, BN)
            nmask = n < KEEP
            idx = tl.load(at_row + n, mask=nmask, other=0)
            vis = tl.load(vis_row + n, mask=nmask, other=0)
            k = tl.load(
                K + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * SCALE
            s = tl.where((vis != 0) & nmask, s, float("-inf"))
            new_m = tl.maximum(mx, tl.max(s, 0))
            safe = tl.where(new_m == float("-inf"), 0.0, new_m)
            alpha = tl.exp(mx - safe)
            p = tl.where(s == float("-inf"), 0.0, tl.exp(s - safe))
            v = tl.load(
                V + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
            norm = norm * alpha + tl.sum(p, 0)
            mx = new_m
        o = acc / tl.maximum(norm, 1e-30)
        tl.store(OUT + (batch * S + qpos) * H * D + head * D + cols, o, mask=dcol)
        tl.store(ML + 2 * pid, mx)
        tl.store(ML + 2 * pid + 1, norm)

    @triton.jit  # type: ignore[untyped-decorator]
    def bwd_kernel(  # type: ignore[no-untyped-def]
        Q,
        K,
        V,
        AT,
        VIS,
        DO,
        OUT,
        ML,
        DQ,
        DK,
        DV,
        S,
        H,
        SHARDS,
        D: tl.constexpr,
        KEEP: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """One query's `dq`, and its scatter-add into `dK`/`dV` by key row.

        `dlogits_j = p_j (do·v_j - do·o)` and `dq = scale·Σ_j dlogits_j k_j` is the whole
        derivation; `dk_j += scale·dlogits_j q` and `dv_j += p_j·do` are the contributions
        of the same row to the keys it came from. Invisible positions have `p_j = 0`, so
        their atomics add exactly zero and need no mask.
        """
        pid = tl.program_id(0)
        qpos = pid % S
        bh = pid // S
        batch = bh // H
        off = bh * S * D
        #: The scatter's target copy: `SHARDS` interleaved by program, so the queries that
        #: vote for the same key row land on different copies and the atomics do not
        #: serialise. `tl.num_programs(0)` is `batch·heads·seq`, one copy's element count.
        dko = (pid % SHARDS) * tl.num_programs(0) * D + off
        cols = tl.arange(0, BD)
        dcol = cols < D
        q = tl.load(Q + off + qpos * D + cols, mask=dcol, other=0.0).to(tl.float32)
        dout = tl.load(DO + off + qpos * D + cols, mask=dcol, other=0.0).to(tl.float32)
        out = tl.load(OUT + off + qpos * D + cols, mask=dcol, other=0.0).to(tl.float32)
        mx = tl.load(ML + 2 * pid)
        norm = tl.load(ML + 2 * pid + 1)
        delta = tl.sum(dout * out, 0)
        at_row = AT + (batch * S + qpos) * KEEP
        vis_row = VIS + (batch * S + qpos) * KEEP

        dq = tl.zeros([BD], tl.float32)
        for n0 in range(0, KEEP, BN):
            n = n0 + tl.arange(0, BN)
            nmask = n < KEEP
            idx = tl.load(at_row + n, mask=nmask, other=0)
            vis = tl.load(vis_row + n, mask=nmask, other=0)
            k = tl.load(
                K + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + off + idx[:, None] * D + cols[None, :],
                mask=nmask[:, None] & dcol[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * SCALE
            p = tl.where((vis != 0) & nmask, tl.exp(s - mx) / tl.maximum(norm, 1e-30), 0.0)
            dl = p * (tl.sum(v * dout[None, :], 1) - delta)
            dq += SCALE * tl.sum(dl[:, None] * k, 0)
            ptr = dko + idx[:, None] * D + cols[None, :]
            tl.atomic_add(DV + ptr, p[:, None] * dout[None, :], mask=nmask[:, None] & dcol[None, :])
            tl.atomic_add(
                DK + ptr,
                SCALE * dl[:, None] * q[None, :],
                mask=nmask[:, None] & dcol[None, :],
            )
        tl.store(DQ + off + qpos * D + cols, dq, mask=dcol)

    _KERNELS.update(
        {
            "fwd": fwd_single_kernel if mfu40_kernels.enabled("KO3") else fwd_kernel,
            "bwd": bwd_kernel,
        }
    )
    return _KERNELS


def _sparse_compute(  # pragma: no cover - the launches are the card's
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    at: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Launch the sparse forward, returning the write and the backward's saved tensors.

    The eager autograd Function and the opaque custom op share this one copy of the launch.
    """
    built = _require()
    batch, seq, heads, width = q.shape
    keep = at.shape[-1]
    bd = _pow2(width)
    #: `[b·h, s, d]` planes: the transposes are 8.4 MiB each at the live shape, and the
    #: kernel reads rows of `d` contiguous — the layout a gather wants.
    # Clone: contiguous() can be a no-op and custom_op forbids an output aliasing an input.
    qh = q.transpose(1, 2).contiguous().clone()
    kh = key.transpose(1, 2).contiguous().clone()
    vh = value.transpose(1, 2).contiguous().clone()
    flat = (qh, kh, vh)
    # `.clone()` is not redundant: dynamo's higher-order-op capture rejects an
    # intermediate that aliases the Function's input (`RuntimeError: idx ... aliases
    # an input`), so the index buffer must own its storage.
    idx = at.to(torch.int32).contiguous().clone()
    vis = visible.to(torch.uint8).contiguous().clone()
    out = torch.empty((batch, seq, heads, width), dtype=torch.float32, device=q.device)
    ml = torch.empty((batch * heads * seq, 2), dtype=torch.float32, device=q.device)
    _count(1)
    if mfu40_kernels.enabled("KO3"):
        mfu40_kernels.count("ko3_sparse_single_pass")
    built["fwd"][(batch * heads * seq,)](
        *flat,
        idx,
        vis,
        out,
        ml,
        seq,
        heads,
        width,
        keep,
        min(BN, _pow2(keep)),
        bd,
        float(scale),
        num_warps=WARPS,
        num_stages=STAGES,
    )
    # `.to()` is identity when dtypes match; clone so output 0 never aliases output 6.
    return out.to(q.dtype).clone(), qh, kh, vh, idx, vis, out, ml


class _Sparse(torch.autograd.Function):
    """The layer's read, as one program per query, in and out of the index map."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]  # pragma: no cover - device only
        ctx, q, key, value, at, visible, scale
    ):
        """Return `o [b, s, h, d]` and keep what the backward recomputes from."""
        write, qh, kh, vh, idx, vis, out, ml = _sparse_compute(q, key, value, at, visible, scale)
        ctx.save_for_backward(qh, kh, vh, idx, vis, out, ml)
        ctx.meta = (q.shape[0], q.shape[1], q.shape[2], q.shape[3], at.shape[-1], float(scale))
        return write

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # pragma: no cover
        """`dq` from the same rows, `dK`/`dV` scattered by key row with atomics."""
        built = _require()
        qh, kh, vh, idx, vis, out, ml = ctx.saved_tensors
        batch, seq, heads, width, keep, scale = ctx.meta
        bd = _pow2(width)
        dout = grad.transpose(1, 2).contiguous()
        #: `out` is saved in the layer's `[b, s, h, d]` write layout and the kernel reads
        #: `[b·h, s, d]` like the projections: 8.4 MiB at the live shape, once per layer.
        out_h = out.transpose(1, 2).contiguous()
        dq = torch.empty_like(qh)
        #: fp32 accumulates the scatter at full width and is cast once at the end; bf16
        #: atomics do not exist, and a bf16 buffer would round every one of the 640 adds
        #: a key receives per step.
        dk = torch.zeros(
            (SHARDS, batch * heads * seq, width), dtype=torch.float32, device=qh.device
        )
        dv = torch.zeros_like(dk)
        _count(1)
        if mfu40_kernels.enabled("KO3"):
            mfu40_kernels.count("ko3_sparse_single_pass")
        built["bwd"][(batch * heads * seq,)](
            qh,
            kh,
            vh,
            idx,
            vis,
            dout,
            out_h,
            ml,
            dq,
            dk,
            dv,
            seq,
            heads,
            SHARDS,
            width,
            keep,
            min(BN_BWD, _pow2(keep)),
            bd,
            scale,
            num_warps=WARPS,
            num_stages=STAGES,
        )
        dq_out = dq.transpose(1, 2).contiguous().to(grad.dtype)
        dk_out = dk.sum(0).reshape(batch, heads, seq, width).transpose(1, 2).to(kh.dtype)
        dv_out = dv.sum(0).reshape(batch, heads, seq, width).transpose(1, 2).to(vh.dtype)
        return dq_out, dk_out, dv_out, None, None, None


#: The opaque boundary dynamo sees: one node per sparse layer instead of a trace through
#: the Triton launch. The seven saved tensors are extra outputs because a custom op's
#: `setup_context` cannot read attributes its implementation set under FakeTensor.
@torch.library.custom_op(
    "dormouse::sparse_attention",
    mutates_args=(),
    schema=(
        "(Tensor q, Tensor key, Tensor value, Tensor at, Tensor visible, float scale) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _sparse_op(  # pragma: no cover - the kernels need a card
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    at: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Run sparse attention, returning the write and the backward's saved tensors."""
    return _sparse_compute(q, key, value, at, visible, scale)


@_sparse_op.register_fake
def _sparse_op_fake(  # pragma: no cover - only dynamo's fake path reaches it
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    at: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Declare the write and the seven saved buffers without touching data."""
    del scale
    batch, seq, heads, width = q.shape
    keep = at.shape[-1]
    write = q.new_empty((batch, seq, heads, width))
    # N3 P0: fake must allocate SEPARATE tensors for qh/kh/vh. Aliasing one
    # `plane` made autograd's compiled path see a shared destination and
    # silently produce wrong dq/dk/dv (maxdiff 0.6-0.9) with correct forward.
    qh = q.new_empty((batch, heads, seq, width))
    kh = q.new_empty((batch, heads, seq, width))
    vh = q.new_empty((batch, heads, seq, width))
    idx = q.new_empty((batch, seq, keep), dtype=torch.int32)
    vis = q.new_empty((batch, seq, keep), dtype=torch.uint8)
    raw = torch.empty((batch, seq, heads, width), dtype=torch.float32, device=q.device)
    # N3 P1 was about the LIBRARY fake; this app op really returns (b*h*s, 2).
    ml = torch.empty((batch * heads * seq, 2), dtype=torch.float32, device=q.device)
    return write, qh, kh, vh, idx, vis, raw, ml


def _sparse_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
    """Save the operands and the seven intermediates in `_Sparse.backward`'s order."""
    q, key, value, at, visible, scale = inputs
    ctx.save_for_backward(
        output[1], output[2], output[3], output[4], output[5], output[6], output[7]
    )
    ctx.meta = (q.shape[0], q.shape[1], q.shape[2], q.shape[3], at.shape[-1], float(scale))
    del key, value, visible


def _sparse_backward(ctx: Any, *grads: torch.Tensor | None) -> tuple[Any, ...]:
    """Reuse the eager backward; the scale argument has no gradient."""
    grad = cast("torch.Tensor", grads[0])
    # _Sparse.backward already returns one gradient per schema input (dq, dk, dv,
    # and None for at/visible/scale); appending another makes 7 grads for 6 inputs.
    return _Sparse.backward(ctx, grad)


_sparse_op.register_autograd(_sparse_backward, setup_context=_sparse_setup)


def _pow2(n: int) -> int:
    """Return `n` rounded up to a power of two, at least 16 (one warp's tile)."""
    return max(16, 1 << max(0, n - 1).bit_length())


def _count(n: int) -> None:  # pragma: no cover - only a kernel launch reaches it
    """Count `n` launches."""
    global _LAUNCHES
    _LAUNCHES += n


def _require() -> dict[str, Any]:
    """Return the kernels or refuse: a flag that is on and cannot run is not a fallback."""
    built = _build()
    if built is None:
        raise LoudError(
            "sparse attention ran without Triton",
            "call it only when kernels.sparse.enabled() is true; the eager path in "
            "dsa.py is the fallback and it is chosen there",
        )
    return built


def attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    at: torch.Tensor,
    visible: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """`o [b, s, h, d] = softmax(q·k[at]·scale)·v[at]`, reading `at` and nothing gathered."""
    if at.shape[-1] < 1:
        raise LoudError(
            "the index map names no key",
            "k + window is at least 2; an empty read set is a caller's bug",
        )
    if not enabled() or not q.is_cuda:
        raise LoudError(
            f"kernels.sparse.attention called on a {q.device.type} tensor or with the kernel off",
            "this is the kernel entry point; dsa.py picks the anchor instead",
        )
    #: The launch itself: no card, no line, and `dsa.py` chose the anchor above.
    out = _sparse_op(q, key, value, at, visible, scale)[0]
    return cast("torch.Tensor", out)  # pragma: no cover
