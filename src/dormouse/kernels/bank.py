"""The bank as one fused expert chain per group: no `[rows, d_ffn]` slab in global memory.

THE ARITHMETIC IS THE ANCHOR'S, TILED DIFFERENTLY. An expert is three rank-`r` matrices
(`tsct.py`), so one routed row costs

    xr_g = x·u_g,  g = xr_g·v_g          [rows, r] then [rows, f]
    xr_u = x·u_u,  u = xr_u·v_u
    A = Σ_k w_k·f_k(g),  B = Σ_k w_k·f_k(u),  h = A⊙B
    q = h·u_d,  y = q·v_d                [rows, f] is read once here and never written

`kernels/fused.mixture` collapsed the per-entry chain into one launch and `ffn.py`
collapsed `K` copies of `z` into one buffer — but the PyTorch chain still writes `g`,
`u`, `A`, `B` and `h` as `[rows, f]` slabs and reads each back. At the live bank
(b8×s1024, 64 experts, capacity 512, f=6144) one group's slab is 126 MiB in fp32 and the
group is read and written eleven times per layer. That is the traffic this module
deletes: the rank-space tile `[rows, r]` is 24× narrower, so the two things that must
survive between loop iterations (`xr_g`, `xr_u` and `q`) are the cheap ones.

THE TILE SCHEDULE, AND WHY IT FITS. A program owns `BM=16` rows of one expert and loops
over `f` in `BN=128` chunks. Live across the f-loop:

    xr_g, xr_u   [16, 256] each        (the step's dtype; 8 KiB each at bf16)
    q            [16, 256] fp32        (the down-projection's rank-space accumulator)
    g, u, A, B, h    [16, 128] fp32    (five tiles, consumed within the iteration)

At `num_warps=8` that is ~110 32-bit registers per thread (`tools/bench_bank.py` prints
`n_regs`/`n_spills` for every kernel it runs), so the intermediates this module exists to
keep are kept. The refused alternative is worth stating: the whole `[rows, f]` hidden in
one program is `16·6144·4 = 384 KiB`, i.e. 1536 registers per thread at 256 threads —
6× the 255-register file — so the f-loop is not an optimisation of that idea, it is the
only tile that exists at `r=256, f=6144`.

WHY THE BACKWARD IS TWO KERNELS. `dV_g = xr_gᵀ·dg` reduces over rows and
`dxr_g = dg·v_gᵀ` reduces over `f`; no program owns both without either materialising
`dg` (`[rows, f]`, the slab this module deletes) or atomically adding a `[r, f]` gradient
once per row tile (8 tiles per expert at `BM=64` ⇒ 11 GiB of atomics per layer — the
wrong answer, measured, not guessed). So `bwd_rows` owns rows and reduces over `f`
(producing `dq`, `dxr_g`, `dxr_u`, `dw`) and `bwd_cols` owns an `f`-strip, walks every
row of the expert and holds `dV_g`, `dV_u`, `dU_d` in registers until the last row: one
store each, no atomics, bitwise reproducible.

WHAT IS NOT FUSED, AND WHAT IT COSTS. The rank-side GEMMs (`dU_g`, `dU_u`, `dV_d`,
`dx`, and the mixture gate's `dW`) run as cuBLAS `bmm` on a `[group, capacity, d]`
layout. They are `d=832` wide — 7.4× narrower than the slab deleted here — they are the
backward only, and a fifth and sixth Triton kernel for `[d]`-wide reductions would buy
6% of the traffic for twice the kernel surface. `tools/bench_bank.py` counts both paths'
bytes; the report states the split.

THE MIXTURE IS EVALUATED FROM CODES, IN THE EPILOGUE. `kinds` is the constexpr tuple the
dictionary resolves to (`fused.MIX_CODES`), so each entry is one static branch inside the
tile that already holds `g`, and the per-row coefficients are six `[BM]` vectors loaded
once per program. No per-entry pass, no Python loop.

NO SILENT FALLBACK. `bank_enabled()` is false only when Triton or CUDA is absent, and
then the anchor is what runs — visibly, as the CPU path. On the card a launch that
cannot happen raises `LoudError`, and `bank_launches()` is what the tests read to prove
the kernel was the thing that ran.
"""
# ruff: noqa: N803

from __future__ import annotations

import os
from typing import Any, cast

import torch
from dormouse_core.errors import LoudError

from . import fused

#: Rows a program owns. 16 is the smallest `tl.dot` M and doubling it doubles the live
#: `xr_*` tiles; the docstring's register table is this number's arithmetic.
BM = 16

#: Rows the prep kernel owns. It holds two `[BM, r]` accumulators AND a `[BM, d]` input
#: tile across its `d` loop, so it spills at 32 (measured 255 registers, 140 B spilled)
#: and is kept at 16.
BM_P = 16

#: `f`-columns a forward program reads at a time. 64 keeps `g`, `u`, `A`, `B` and `h` in
#: the register file together with the `[BM, r]` tiles; at `r=256` its three `[r, BN]`
#: operands are 32 KiB each at bf16, which is what a block's 101 KiB of shared memory
#: holds. 128 asks for 147 KiB and the launch refuses (`OutOfResources`), measured.
BN = 64

#: The same strip when the step is fp32: a `[r, BN]` fp32 operand is twice the bf16 one,
#: and 128 asks for 172 KiB against the 101 KiB a block may hold — the launch refuses
#: (`OutOfResources`), measured. 32 is the widest fp32 strip that fits; fp32 is the
#: parity and CI path, so its tile is chosen for fitting, not for throughput.
BN_FP32 = 32

#: The backward's row-major strip. Four `[r, BN]` operands are live in its f-loop
#: (`v_g`, `v_u` and their transposes), so bf16 takes 64 here where the forward takes 128.
BN_BWD = 32

#: `d`-columns the input projection and the store walk at a time.
BK = 64

#: `f`-columns one backward program owns: three `[r, BN]`-shaped accumulators held to the
#: last row, so this is the number the register file sizes.
BNC = 32

#: Rows the column-major backward walks per iteration.
BMC = 16

#: The mixture's constexpr width. `tl.dot` needs N ≥ 16 and the live dictionary is six
#: entries, so the gate is computed at width 16 and the extra columns carry zeros.
MM = 16

#: Warps per program; see the docstring's register arithmetic.
WARPS = 8

#: Pipeline stages per program. The f-loop's operands are `[r, BN]` fp32 (128 KiB at
#: `r=256`), so a default of three stages asks for 278 KiB of shared memory against the
#: card's 101 KiB per block and the launch refuses (`OutOfResources`). One stage is the
#: tile that fits; the loop is long enough (48 iterations) that the miss costs
#: prefetch, not throughput, and `tools/bench_bank.py` prints ms/fwd either way.
STAGES = 1

#: The flag: on where a kernel can run, and then the anchor is not a fallback.
_BANK = os.environ.get("DORMOUSE_BANK", "") not in ("0", "off", "false")

#: Triton launches since the last reset. Zero with the flag on is the historical defect
#: this counter exists to catch.
_launches = 0

#: The kernels, built once. `None` from `_build` means Triton is absent.
_KERNELS: dict[str, Any] = {}

#: `triton.language`, bound by `_build` and left as a module global on purpose: Triton
#: resolves a nested kernel's `tl.constexpr` annotations against `__globals__`, so a
#: helper defined inside `_build` cannot see a *local* import of it. `None` here means
#: Triton is absent, which is the CPU path.
tl: Any = None


def set_bank(on: bool) -> None:
    """Turn the fused bank on or off for this process."""
    global _BANK
    _BANK = bool(on)


def bank_enabled() -> bool:
    """Return whether the fused bank is requested AND can run on this device."""
    return _BANK and torch.cuda.is_available() and _build() is not None


@torch._dynamo.disable  # type: ignore[untyped-decorator]
def bank_launches() -> int:
    """Return how many Triton launches the bank has issued since the last reset.

    Disabled for dynamo: `_launches` changes every step, and tracing this accessor would
    install a guard on the mutable global (`G['…bank']._launches == N`) and recompile the
    graph on the next step. The read happens on the host at the counter call site instead.
    """
    return _launches


def reset_bank_launches() -> None:
    """Zero the launch counter; the flag keeps its value."""
    global _launches
    _launches = 0


def _tally(n: int) -> None:
    """Count `n` bank launches."""
    global _launches
    _launches += n


def bf16_step(device: torch.device) -> bool:
    """Return whether this call is a bf16 step: autocast on for the device, or bf16 data."""
    try:
        return bool(torch.is_autocast_enabled(device.type))
    except TypeError:  # pragma: no cover - older torch signature
        return bool(torch.is_autocast_enabled())


def _build() -> dict[str, Any] | None:  # pragma: no cover - requires triton + a device
    """Compile-once Triton kernels; `None` when Triton is not installed."""
    global tl
    if _KERNELS:
        return _KERNELS
    base = fused._build()
    if base is None:
        return None
    import triton
    import triton.language

    tl = triton.language

    activate = base["activate"]
    grad = base["grad"]

    @triton.jit  # type: ignore[untyped-decorator]
    def _dot(a, b, BF16: tl.constexpr):  # type: ignore[no-untyped-def]
        """One matmul: bf16 tensor cores on a bf16 step, IEEE fp32 otherwise.

        The dtype belongs to the STEP, not to a preference here: a bf16 step converts its
        matmul operands before the MMA and an fp32 step does not, so a kernel that always
        used bf16 would move an fp32 run's numbers while claiming to be a drop-in.
        """
        if BF16:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)
        return tl.dot(
            a.to(tl.float32), b.to(tl.float32), input_precision="ieee", out_dtype=tl.float32
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _mix(  # type: ignore[no-untyped-def]
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        z,
        K: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
    ):
        """`Σₖ wₖ·fₖ(z)` for one tile, from the dictionary's integer codes."""
        acc = w0[:, None] * activate(z, K0)
        if K > 1:
            acc += w1[:, None] * activate(z, K1)
        if K > 2:
            acc += w2[:, None] * activate(z, K2)
        if K > 3:
            acc += w3[:, None] * activate(z, K3)
        if K > 4:
            acc += w4[:, None] * activate(z, K4)
        if K > 5:
            acc += w5[:, None] * activate(z, K5)
        return acc

    @triton.jit  # type: ignore[untyped-decorator]
    def _dmix(  # type: ignore[no-untyped-def]
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        z,
        K: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
    ):
        """`Σₖ wₖ·fₖ'(z)`: the mixture's own derivative, one static branch per entry."""
        acc = w0[:, None] * grad(z, K0)
        if K > 1:
            acc += w1[:, None] * grad(z, K1)
        if K > 2:
            acc += w2[:, None] * grad(z, K2)
        if K > 3:
            acc += w3[:, None] * grad(z, K3)
        if K > 4:
            acc += w4[:, None] * grad(z, K4)
        if K > 5:
            acc += w5[:, None] * grad(z, K5)
        return acc

    @triton.jit  # type: ignore[untyped-decorator]
    def _weights(WGT, slot, live, K: tl.constexpr, P: tl.constexpr):  # type: ignore[no-untyped-def]
        """Load the per-row coefficients, once per program, as `[BM]` vectors."""
        w0 = tl.load(WGT + slot * P, mask=live, other=0.0)
        w1 = tl.load(WGT + slot * P + 1, mask=live, other=0.0) if K > 1 else w0
        w2 = tl.load(WGT + slot * P + 2, mask=live, other=0.0) if K > 2 else w0
        w3 = tl.load(WGT + slot * P + 3, mask=live, other=0.0) if K > 3 else w0
        w4 = tl.load(WGT + slot * P + 4, mask=live, other=0.0) if K > 4 else w0
        w5 = tl.load(WGT + slot * P + 5, mask=live, other=0.0) if K > 5 else w0
        return w0, w1, w2, w3, w4, w5

    @triton.jit  # type: ignore[untyped-decorator]
    def prep_kernel(  # type: ignore[no-untyped-def]
        X,
        TOK,
        CNT,
        UG,
        UU,
        MW,
        MB,
        CF,
        XRG,
        XRU,
        WGT,
        SG,
        CAP,
        D: tl.constexpr,
        RK: tl.constexpr,
        BKc: tl.constexpr,
        MMv: tl.constexpr,
        K: tl.constexpr,
        BMv: tl.constexpr,
        BF16: tl.constexpr,
    ):
        """One tile's rank-space projection, its mixture gate and its weights.

        The only pass that reads `x`: it emits `xr_g`, `xr_u` (`[rows, r]`), the gate's
        pre-activation `s` and the normalised weights `w`, so neither the f-loop nor the
        backward touches the `d`-wide input again.
        """
        e = tl.program_id(0)
        m0 = tl.program_id(1) * BMv
        cnt = tl.load(CNT + e)
        if m0 >= cnt:
            #: The capacity is `CAPACITY_RATIO` (2x) the mean, so half the grid's tiles are
            #: pure padding; returning here costs the schedule nothing and issues no work
            #: for them. The masked rows inside a live tile still guard every access.
            return
        rows = m0 + tl.arange(0, BMv)
        live = rows < cnt
        slot = e * CAP + rows
        tok = tl.load(TOK + slot, mask=live, other=0)
        rk = tl.arange(0, RK)
        mm = tl.arange(0, MMv)
        acc_g = tl.zeros((BMv, RK), tl.float32)
        acc_u = tl.zeros((BMv, RK), tl.float32)
        acc_m = tl.zeros((BMv, MMv), tl.float32)
        for k0 in range(0, D, BKc):
            cols = k0 + tl.arange(0, BKc)
            dmask = cols < D
            xv = tl.load(
                X + tok[:, None] * D + cols[None, :],
                mask=live[:, None] & dmask[None, :],
                other=0.0,
            )
            ug = tl.load(
                UG + e * D * RK + cols[:, None] * RK + rk[None, :], mask=dmask[:, None], other=0.0
            )
            acc_g += _dot(xv, ug, BF16)
            uu = tl.load(
                UU + e * D * RK + cols[:, None] * RK + rk[None, :], mask=dmask[:, None], other=0.0
            )
            acc_u += _dot(xv, uu, BF16)
            mw = tl.load(
                MW + e * D * K + cols[:, None] * K + mm[None, :],
                mask=dmask[:, None] & (mm[None, :] < K),
                other=0.0,
            )
            acc_m += _dot(xv, mw, BF16)
        bias = tl.load(MB + e * K + mm, mask=mm < K, other=0.0)
        coef = tl.load(CF + e * K + mm, mask=mm < K, other=0.0)
        s = acc_m + bias[None, :]
        r = tl.sigmoid(s) * coef[None, :]
        w = r / tl.sum(r, axis=1)[:, None]
        dt = XRG.dtype.element_ty
        tl.store(XRG + slot[:, None] * RK + rk[None, :], acc_g.to(dt), mask=live[:, None])
        tl.store(XRU + slot[:, None] * RK + rk[None, :], acc_u.to(dt), mask=live[:, None])
        tl.store(WGT + slot[:, None] * MMv + mm[None, :], w, mask=live[:, None])
        tl.store(SG + slot[:, None] * MMv + mm[None, :], s, mask=live[:, None])

    @triton.jit  # type: ignore[untyped-decorator]
    def fwd_kernel(  # type: ignore[no-untyped-def]
        VG,
        VU,
        UD,
        VD,
        XRG,
        XRU,
        WGT,
        CNT,
        OUT,
        QQ,
        CAP,
        D: tl.constexpr,
        F: tl.constexpr,
        RK: tl.constexpr,
        BNc: tl.constexpr,
        BDc: tl.constexpr,
        MMv: tl.constexpr,
        K: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
        BI: tl.constexpr,
        BMv: tl.constexpr,
        BF16: tl.constexpr,
    ):
        """One tile through gate/up, the mixture, the rank read and the write.

        Nothing `[rows, f]` is ever addressed: `g`, `u`, the mixture and `h` exist only as
        a `[BMv, BN]` tile inside one iteration, and the only `[rows, f]`-shaped object in
        the chain — the expert's `v` factors — is an operand, read once per tile.
        """
        #: Tile-major: consecutive programs walk the row tiles of ONE expert, so that
        #: expert's factor slab stays resident in L2 across the whole wave instead of being
        #: evicted between tiles by the factors of the experts in the next column.
        e = tl.program_id(1)
        m0 = tl.program_id(0) * BMv
        cnt = tl.load(CNT + e)
        if m0 >= cnt:
            #: The capacity is `CAPACITY_RATIO` (2x) the mean, so half the grid's tiles are
            #: pure padding; returning here costs the schedule nothing and issues no work
            #: for them. The masked rows inside a live tile still guard every access.
            return
        rows = m0 + tl.arange(0, BMv)
        live = rows < cnt
        slot = e * CAP + rows
        rk = tl.arange(0, RK)
        w0, w1, w2, w3, w4, w5 = _weights(WGT, slot, live, K, MMv)
        xrg = tl.load(XRG + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
        xru = tl.load(XRU + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
        q = tl.zeros((BMv, RK), tl.float32)
        for n0 in range(0, F, BNc):
            nc = n0 + tl.arange(0, BNc)
            fmask = nc < F
            vg = tl.load(
                VG + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
            )
            g = _dot(xrg, vg, BF16)
            vu = tl.load(
                VU + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
            )
            u = _dot(xru, vu, BF16)
            a = _mix(w0, w1, w2, w3, w4, w5, g, K, K0, K1, K2, K3, K4, K5)
            b = _mix(w0, w1, w2, w3, w4, w5, u, K, K0, K1, K2, K3, K4, K5) if BI else u
            ud = tl.load(
                UD + e * F * RK + nc[:, None] * RK + rk[None, :], mask=fmask[:, None], other=0.0
            )
            q += _dot(a * b, ud, BF16)
        tl.store(
            QQ + slot[:, None] * RK + rk[None, :], q.to(QQ.dtype.element_ty), mask=live[:, None]
        )
        for c0 in range(0, D, BDc):
            cc = c0 + tl.arange(0, BDc)
            dmask = cc < D
            vd = tl.load(
                VD + e * RK * D + rk[:, None] * D + cc[None, :], mask=dmask[None, :], other=0.0
            )
            tl.store(
                OUT + slot[:, None] * D + cc[None, :],
                _dot(q, vd, BF16).to(OUT.dtype.element_ty),
                mask=live[:, None] & dmask[None, :],
            )

    @triton.jit  # type: ignore[untyped-decorator]
    def bwd_rows_kernel(  # type: ignore[no-untyped-def]
        D_OUT,
        VD,
        VG,
        VU,
        UD,
        XRG,
        XRU,
        WGT,
        CNT,
        DQ,
        DXRG,
        DXRU,
        DW,
        CAP,
        D: tl.constexpr,
        F: tl.constexpr,
        RK: tl.constexpr,
        BKc: tl.constexpr,
        BNc: tl.constexpr,
        MMv: tl.constexpr,
        K: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
        BI: tl.constexpr,
        BMv: tl.constexpr,
        BF16: tl.constexpr,
    ):
        """Row-major backward: `dq`, the two rank-space gradients, the mixture's `dw`.

        One program owns `BM` rows, so both `[rows, r]` results and `dw [rows, K]` are
        plain stores — no atomics, and no reduction order to argue about.
        """
        e = tl.program_id(0)
        m0 = tl.program_id(1) * BMv
        cnt = tl.load(CNT + e)
        if m0 >= cnt:
            #: The capacity is `CAPACITY_RATIO` (2x) the mean, so half the grid's tiles are
            #: pure padding; returning here costs the schedule nothing and issues no work
            #: for them. The masked rows inside a live tile still guard every access.
            return
        rows = m0 + tl.arange(0, BMv)
        live = rows < cnt
        slot = e * CAP + rows
        rk = tl.arange(0, RK)
        w0, w1, w2, w3, w4, w5 = _weights(WGT, slot, live, K, MMv)
        xrg = tl.load(XRG + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
        xru = tl.load(XRU + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
        dq = tl.zeros((BMv, RK), tl.float32)
        for k0 in range(0, D, BKc):
            cols = k0 + tl.arange(0, BKc)
            dmask = cols < D
            dy = tl.load(
                D_OUT + slot[:, None] * D + cols[None, :],
                mask=live[:, None] & dmask[None, :],
                other=0.0,
            )
            vd = tl.load(
                VD + e * RK * D + rk[None, :] * D + cols[:, None], mask=dmask[:, None], other=0.0
            )
            dq += _dot(dy, vd, BF16)
        tl.store(
            DQ + slot[:, None] * RK + rk[None, :], dq.to(DQ.dtype.element_ty), mask=live[:, None]
        )
        dxrg = tl.zeros((BMv, RK), tl.float32)
        dxru = tl.zeros((BMv, RK), tl.float32)
        dw0 = tl.zeros((BMv,), tl.float32)
        dw1 = tl.zeros((BMv,), tl.float32)
        dw2 = tl.zeros((BMv,), tl.float32)
        dw3 = tl.zeros((BMv,), tl.float32)
        dw4 = tl.zeros((BMv,), tl.float32)
        dw5 = tl.zeros((BMv,), tl.float32)
        for n0 in range(0, F, BNc):
            nc = n0 + tl.arange(0, BNc)
            fmask = nc < F
            vg = tl.load(
                VG + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
            )
            g = _dot(xrg, vg, BF16)
            vu = tl.load(
                VU + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
            )
            u = _dot(xru, vu, BF16)
            a = _mix(w0, w1, w2, w3, w4, w5, g, K, K0, K1, K2, K3, K4, K5)
            b = _mix(w0, w1, w2, w3, w4, w5, u, K, K0, K1, K2, K3, K4, K5) if BI else u
            ud = tl.load(
                UD + e * F * RK + nc[None, :] * RK + rk[:, None], mask=fmask[None, :], other=0.0
            )
            dh = _dot(dq, ud, BF16)
            da = dh * b
            db = dh * a
            dg = da * _dmix(w0, w1, w2, w3, w4, w5, g, K, K0, K1, K2, K3, K4, K5)
            du = db * _dmix(w0, w1, w2, w3, w4, w5, u, K, K0, K1, K2, K3, K4, K5) if BI else db
            vgt = tl.load(
                VG + e * RK * F + rk[None, :] * F + nc[:, None], mask=fmask[:, None], other=0.0
            )
            dxrg += _dot(dg, vgt, BF16)
            vut = tl.load(
                VU + e * RK * F + rk[None, :] * F + nc[:, None], mask=fmask[:, None], other=0.0
            )
            dxru += _dot(du, vut, BF16)
            dw0 += tl.sum(activate(g, K0) * da + (activate(u, K0) * db if BI else 0.0), axis=1)
            if K > 1:
                dw1 += tl.sum(activate(g, K1) * da + (activate(u, K1) * db if BI else 0.0), axis=1)
            if K > 2:
                dw2 += tl.sum(activate(g, K2) * da + (activate(u, K2) * db if BI else 0.0), axis=1)
            if K > 3:
                dw3 += tl.sum(activate(g, K3) * da + (activate(u, K3) * db if BI else 0.0), axis=1)
            if K > 4:
                dw4 += tl.sum(activate(g, K4) * da + (activate(u, K4) * db if BI else 0.0), axis=1)
            if K > 5:
                dw5 += tl.sum(activate(g, K5) * da + (activate(u, K5) * db if BI else 0.0), axis=1)
        dt = DXRG.dtype.element_ty
        tl.store(DXRG + slot[:, None] * RK + rk[None, :], dxrg.to(dt), mask=live[:, None])
        tl.store(DXRU + slot[:, None] * RK + rk[None, :], dxru.to(dt), mask=live[:, None])
        tl.store(DW + slot * MMv, dw0, mask=live)
        if K > 1:
            tl.store(DW + slot * MMv + 1, dw1, mask=live)
        if K > 2:
            tl.store(DW + slot * MMv + 2, dw2, mask=live)
        if K > 3:
            tl.store(DW + slot * MMv + 3, dw3, mask=live)
        if K > 4:
            tl.store(DW + slot * MMv + 4, dw4, mask=live)
        if K > 5:
            tl.store(DW + slot * MMv + 5, dw5, mask=live)

    @triton.jit  # type: ignore[untyped-decorator]
    def bwd_cols_kernel(  # type: ignore[no-untyped-def]
        VG,
        VU,
        UD,
        XRG,
        XRU,
        WGT,
        CNT,
        DQ,
        DVG,
        DVU,
        DUD,
        CAP,
        F: tl.constexpr,
        RK: tl.constexpr,
        BNc: tl.constexpr,
        MMv: tl.constexpr,
        K: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
        BI: tl.constexpr,
        BMCv: tl.constexpr,
        BF16: tl.constexpr,
    ):
        """Column-major backward: `dV_g`, `dV_u`, `dU_d` held in registers to the last row.

        One program owns one `[r, BN]` strip of the expert's `f` axis and every row, so
        each of the three widest gradients is written exactly once.
        """
        e = tl.program_id(0)
        n0 = tl.program_id(1) * BNc
        cnt = tl.load(CNT + e)
        rk = tl.arange(0, RK)
        nc = n0 + tl.arange(0, BNc)
        fmask = nc < F
        vg = tl.load(
            VG + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
        )
        vu = tl.load(
            VU + e * RK * F + rk[:, None] * F + nc[None, :], mask=fmask[None, :], other=0.0
        )
        ud = tl.load(
            UD + e * F * RK + nc[None, :] * RK + rk[:, None], mask=fmask[None, :], other=0.0
        )
        dvg = tl.zeros((RK, BNc), tl.float32)
        dvu = tl.zeros((RK, BNc), tl.float32)
        dud = tl.zeros((BNc, RK), tl.float32)
        for m0 in range(0, cnt, BMCv):
            rows = m0 + tl.arange(0, BMCv)
            live = rows < cnt
            slot = e * CAP + rows
            w0, w1, w2, w3, w4, w5 = _weights(WGT, slot, live, K, MMv)
            xrg = tl.load(XRG + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
            xru = tl.load(XRU + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
            dq = tl.load(DQ + slot[:, None] * RK + rk[None, :], mask=live[:, None], other=0.0)
            g = _dot(xrg, vg, BF16)
            u = _dot(xru, vu, BF16)
            a = _mix(w0, w1, w2, w3, w4, w5, g, K, K0, K1, K2, K3, K4, K5)
            b = _mix(w0, w1, w2, w3, w4, w5, u, K, K0, K1, K2, K3, K4, K5) if BI else u
            dh = _dot(dq, ud, BF16)
            da = dh * b
            db = dh * a
            dg = da * _dmix(w0, w1, w2, w3, w4, w5, g, K, K0, K1, K2, K3, K4, K5)
            du = db * _dmix(w0, w1, w2, w3, w4, w5, u, K, K0, K1, K2, K3, K4, K5) if BI else db
            dvg += _dot(tl.trans(xrg), dg, BF16)
            dvu += _dot(tl.trans(xru), du, BF16)
            dud += _dot(tl.trans(a * b), dq, BF16)
        dt = DVG.dtype.element_ty
        tl.store(DVG + e * RK * F + rk[:, None] * F + nc[None, :], dvg.to(dt), mask=fmask[None, :])
        tl.store(DVU + e * RK * F + rk[:, None] * F + nc[None, :], dvu.to(dt), mask=fmask[None, :])
        tl.store(DUD + e * F * RK + nc[:, None] * RK + rk[None, :], dud.to(dt), mask=fmask[:, None])

    _KERNELS.update(
        {
            "prep": prep_kernel,
            "fwd": fwd_kernel,
            "bwd_rows": bwd_rows_kernel,
            "bwd_cols": bwd_cols_kernel,
        }
    )
    return _KERNELS


def _require(what: str) -> dict[str, Any]:
    """Return the built kernels, or refuse: a flag that is on must not become the anchor."""
    built = _build()
    if built is None or not torch.cuda.is_available():
        raise LoudError(
            f"bank={what} is on and cannot run here (triton="
            f"{'absent' if built is None else 'present'}, cuda="
            f"{'absent' if not torch.cuda.is_available() else 'present'})",
            "run with DORMOUSE_BANK=0, or use the CUDA torch build; a flag that silently "
            "runs the anchor reports a kernel that never executed",
        )
    return built  # pragma: no cover - the card is required to get here


def _validate(rank: int, d: int, f: int, kinds: tuple[int, ...], sides: str) -> None:
    """Refuse a shape the tile schedule does not cover, naming the escape."""
    if rank < 16 or rank & (rank - 1):
        raise LoudError(
            f"the fused bank's rank is {rank}, not a power of two >= 16",
            "a rank-space tile is an aligned `tl.dot` operand; use 16, 32, 64, 128 or 256 "
            "(the live expert is 256), or run with DORMOUSE_BANK=0",
        )
    if min(d, f) < 16:
        raise LoudError(
            f"the fused bank's widths are d={d}, f={f}",
            "both must be at least 16 (a `tl.dot` operand); the live widths are 832/6144",
        )
    if not 1 <= len(kinds) <= fused.MAX_MIX_ENTRIES:
        raise LoudError(
            f"a fused bank over {len(kinds)} dictionary entries",
            f"the kernel sums 1 to {fused.MAX_MIX_ENTRIES}; run with DORMOUSE_BANK=0 for a "
            "longer dictionary",
        )
    unknown = [code for code in kinds if code not in fused.MIX_CODES.values()]
    if unknown:
        raise LoudError(
            f"the fused mixture has no entry for code {unknown[0]}",
            f"the epilogue computes {', '.join(fused.MIX_CODES)}; run with DORMOUSE_BANK=0 "
            "for a dictionary outside its table",
        )
    if sides not in ("one", "bi"):
        raise LoudError(f"sides={sides!r} is not an FFN topology", "use 'one' or 'bi'")


def gather(
    x: torch.Tensor, tok: torch.Tensor, cnt: torch.Tensor, cap: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Return the group's rows in the slot layout, `[group, cap, d]`, zeros in the padding.

    The one materialised gather in this module, and it is the BACKWARD's: `d=832` is 7.4×
    narrower than the `[rows, f]` slab the kernel deletes, and the two reductions that
    need `x` by row (`dU = xᵀ·dxr` and the mixture gate's `dW = dsᵀ·x`) are `d`-wide. The
    forward reads `x` through the slot index map and gathers nothing.
    """
    e = cnt.shape[0]
    live = torch.arange(cap, device=x.device).unsqueeze(0) < cnt.unsqueeze(1)
    idx = torch.arange(e * cap, device=x.device)[live.reshape(-1)]
    out = torch.zeros((e * cap, x.shape[1]), dtype=out_dtype, device=x.device)
    out.index_copy_(0, idx, x.index_select(0, tok[idx]).to(out_dtype))
    return out.view(e, cap, x.shape[1])


def _bank_compute(  # pragma: no cover - the launches are the card's
    x: torch.Tensor,
    tok: torch.Tensor,
    cnt: torch.Tensor,
    gate_u: torch.Tensor,
    gate_v: torch.Tensor,
    up_u: torch.Tensor,
    up_v: torch.Tensor,
    down_u: torch.Tensor,
    down_v: torch.Tensor,
    mix_w: torch.Tensor,
    mix_b: torch.Tensor,
    coef: torch.Tensor,
    meta: tuple[int, tuple[int, ...], str, bool],
) -> tuple[torch.Tensor, ...]:
    """Issue the prep and forward kernels, returning every tensor the backward needs.

    Both the eager autograd Function and the opaque custom op call this, so the kernel
    code has one copy and the two paths cannot drift apart.
    """
    cap, kinds, sides, bf16 = meta
    built = _require("forward")
    e, d, rk = gate_u.shape
    f = gate_v.shape[2]
    _validate(rk, d, f, kinds, sides)
    slots = e * cap
    dev = x.device
    dt = torch.bfloat16 if bf16 else x.dtype
    out = torch.zeros((slots, d), dtype=dt, device=dev)
    xrg = torch.zeros((slots, rk), dtype=dt, device=dev)
    xru = torch.zeros_like(xrg)
    qq = torch.zeros_like(xrg)
    wgt = torch.zeros((slots, MM), dtype=torch.float32, device=dev)
    sg = torch.zeros_like(wgt)
    tile = (e, -(-cap // BM_P))
    pads = (0,) * (fused.MAX_MIX_ENTRIES - len(kinds))
    built["prep"][tile](
        x,
        tok,
        cnt,
        gate_u,
        up_u,
        mix_w,
        mix_b,
        coef,
        xrg,
        xru,
        wgt,
        sg,
        cap,
        d,
        rk,
        BK,
        MM,
        len(kinds),
        BM,
        bf16,
        num_warps=WARPS,
        num_stages=STAGES,
    )
    built["fwd"][(-(-cap // BM), e)](
        gate_v,
        up_v,
        down_u,
        down_v,
        xrg,
        xru,
        wgt,
        cnt,
        out,
        qq,
        cap,
        d,
        f,
        rk,
        BN if bf16 else BN_FP32,
        BK,
        MM,
        len(kinds),
        *kinds,
        *pads,
        sides == "bi",
        BM,
        bf16,
        num_warps=WARPS,
        num_stages=STAGES,
    )
    _tally(2)
    return out, xrg, xru, qq, wgt, sg


class _Bank(torch.autograd.Function):
    """The routed bank's forward and backward, per expert group, over the slot layout.

    The group's parameters arrive as the `[group, ...]` stacks `MoEFFN._stacked` builds,
    so autograd's own `torch.stack` carries every per-expert gradient back to the arm that
    owns it: this Function never names an expert module.
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]  # pragma: no cover - device only
        ctx,
        x,
        tok,
        cnt,
        gate_u,
        gate_v,
        up_u,
        up_v,
        down_u,
        down_v,
        mix_w,
        mix_b,
        coef,
        meta,
    ):
        """Return the group's `[group·cap, d]` per-slot write."""
        out, xrg, xru, qq, wgt, sg = _bank_compute(
            x, tok, cnt, gate_u, gate_v, up_u, up_v, down_u, down_v, mix_w, mix_b, coef, meta
        )
        ctx.save_for_backward(
            x,
            tok,
            cnt,
            gate_u,
            gate_v,
            up_u,
            up_v,
            down_u,
            down_v,
            mix_w,
            mix_b,
            coef,
            xrg,
            xru,
            qq,
            wgt,
            sg,
        )
        ctx.meta = meta
        return out

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]  # pragma: no cover - device only
        ctx, d_out
    ):
        """Return the six weight gradients, `dx`, and the gate's `dw`/`db`/`dcoef`."""
        built = _require("backward")
        (
            x,
            tok,
            cnt,
            gate_u,
            gate_v,
            up_u,
            up_v,
            down_u,
            down_v,
            mix_w,
            mix_b,
            coef,
            xrg,
            xru,
            qq,
            wgt,
            sg,
        ) = ctx.saved_tensors
        cap, kinds, sides, bf16 = ctx.meta
        e, d, rk = gate_u.shape
        f = gate_v.shape[2]
        slots = e * cap
        k = len(kinds)
        pads = (0,) * (fused.MAX_MIX_ENTRIES - k)
        d_out = d_out.reshape(slots, d).contiguous()
        dq = torch.zeros_like(xrg)
        dxrg = torch.zeros_like(xrg)
        dxru = torch.zeros_like(xru)
        dw = torch.zeros((slots, MM), dtype=torch.float32, device=x.device)
        built["bwd_rows"][(e, -(-cap // BM))](
            d_out,
            down_v,
            gate_v,
            up_v,
            down_u,
            xrg,
            xru,
            wgt,
            cnt,
            dq,
            dxrg,
            dxru,
            dw,
            cap,
            d,
            f,
            rk,
            BK,
            (BN_BWD if bf16 else BN_FP32),
            MM,
            k,
            *kinds,
            *pads,
            sides == "bi",
            BM,
            bf16,
            num_warps=WARPS,
            num_stages=STAGES,
        )
        dvg = torch.zeros_like(gate_v, dtype=torch.float32)
        dvu = torch.zeros_like(up_v, dtype=torch.float32)
        dud = torch.zeros_like(down_u, dtype=torch.float32)
        built["bwd_cols"][(e, -(-f // BNC))](
            gate_v,
            up_v,
            down_u,
            xrg,
            xru,
            wgt,
            cnt,
            dq,
            dvg,
            dvu,
            dud,
            cap,
            f,
            rk,
            BNC,
            MM,
            k,
            *kinds,
            *pads,
            sides == "bi",
            BMC,
            bf16,
            num_warps=WARPS,
            num_stages=STAGES,
        )
        _tally(2)
        # The rank-side GEMMs: `d`-wide, backward only, cuBLAS rather than two more
        # kernels — the module docstring has the byte count of that choice.
        xp = gather(x, tok, cnt, cap, gate_u.dtype)
        xr_g = dxrg.view(e, cap, rk)
        xr_u = dxru.view(e, cap, rk)
        dug = torch.bmm(xp.transpose(1, 2), xr_g)
        duu = torch.bmm(xp.transpose(1, 2), xr_u)
        dvd = torch.bmm(qq.view(e, cap, rk).transpose(1, 2), d_out.view(e, cap, d))
        dxp = torch.bmm(xr_g, gate_u.transpose(1, 2)) + torch.bmm(xr_u, up_u.transpose(1, 2))
        # The gate: `w = α⊙σ(s)/Σ`, and `ds_k = (dw_k − Σ_j dw_j w_j)·w_k·(1−σ(s_k))` is
        # `dw/ds` written out — the normaliser cancels, so only `s` is needed here.
        w2, s2, dw2 = wgt[:, :k], sg[:, :k], dw[:, :k]
        live = (torch.arange(cap, device=x.device).unsqueeze(0) < cnt.unsqueeze(1)).reshape(-1)
        idx = torch.arange(slots, device=x.device)[live]
        cent = (dw2 * w2).sum(1, keepdim=True)
        sig = torch.sigmoid(s2)
        ds = (dw2 - cent) * w2 * (1.0 - sig)
        d_mix_w = torch.bmm(ds.view(e, cap, k).transpose(1, 2), xp).transpose(1, 2)
        d_mix_b = ds.view(e, cap, k).sum(1)
        # `dα_j = (σ_j/Σr)·(dw_j − Σ_k dw_k w_k)`: the priors enter `r = α⊙σ(s)` as the
        # numerator, so unlike `w`'s other two gradients this one keeps the normaliser.
        total = (coef.unsqueeze(1) * sig.view(e, cap, k)).sum(-1, keepdim=True)
        d_coef = ((dw2.view(e, cap, k) - cent.view(e, cap, 1)) * sig.view(e, cap, k) / total).sum(1)
        #: `dxp` is the step's own dtype (bf16 under autocast); `x` may be fp32 master
        #: weights, so the scatter runs in the gradient's dtype and casts once at the end.
        dx = torch.zeros((x.shape[0], d), dtype=dxp.dtype, device=x.device)
        dx.index_add_(0, tok[idx], dxp.reshape(slots, d)[idx])
        return (
            dx.to(x.dtype),
            None,
            None,
            dug.to(gate_u.dtype),
            dvg.to(gate_v.dtype),
            duu.to(up_u.dtype),
            dvu.to(up_v.dtype),
            dud.to(down_u.dtype),
            dvd.to(down_v.dtype),
            d_mix_w.contiguous().to(mix_w.dtype),
            d_mix_b.to(mix_b.dtype),
            d_coef.to(coef.dtype),
            None,
        )


#: The opaque boundary dynamo sees. Hand-written Triton launches stay behind one node, so
#: a compiled step keeps a single graph instead of breaking at every Python launch site.
#: The five saved intermediates are extra outputs (unused by the caller) because a custom
#: op's `setup_context` cannot see attributes set by its implementation under FakeTensor.
@torch.library.custom_op(
    "dormouse::bank",
    mutates_args=(),
    schema=(
        "(Tensor x, Tensor tok, Tensor cnt, Tensor gate_u, Tensor gate_v, Tensor up_u, "
        "Tensor up_v, Tensor down_u, Tensor down_v, Tensor mix_w, Tensor mix_b, "
        "Tensor coef, int cap, int[] kinds, str sides, bool bf16) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _bank_op(  # pragma: no cover - the kernels need a card
    x: torch.Tensor,
    tok: torch.Tensor,
    cnt: torch.Tensor,
    gate_u: torch.Tensor,
    gate_v: torch.Tensor,
    up_u: torch.Tensor,
    up_v: torch.Tensor,
    down_u: torch.Tensor,
    down_v: torch.Tensor,
    mix_w: torch.Tensor,
    mix_b: torch.Tensor,
    coef: torch.Tensor,
    cap: int,
    kinds: tuple[int, ...],
    sides: str,
    bf16: bool,
) -> tuple[torch.Tensor, ...]:
    """Run the fused expert chain, returning the write and the backward's saved tensors."""
    meta = (cap, tuple(kinds), sides, bf16)
    return _bank_compute(
        x, tok, cnt, gate_u, gate_v, up_u, up_v, down_u, down_v, mix_w, mix_b, coef, meta
    )


@_bank_op.register_fake
def _bank_op_fake(  # pragma: no cover - only dynamo's fake path reaches it
    x: torch.Tensor,
    tok: torch.Tensor,
    cnt: torch.Tensor,
    gate_u: torch.Tensor,
    gate_v: torch.Tensor,
    up_u: torch.Tensor,
    up_v: torch.Tensor,
    down_u: torch.Tensor,
    down_v: torch.Tensor,
    mix_w: torch.Tensor,
    mix_b: torch.Tensor,
    coef: torch.Tensor,
    cap: int,
    kinds: tuple[int, ...],
    sides: str,
    bf16: bool,
) -> tuple[torch.Tensor, ...]:
    """Declare the write and the five rank-space buffers without touching data."""
    del tok, cnt, up_u, up_v, down_u, down_v, mix_w, mix_b, coef, kinds, sides
    e, d, rk = gate_u.shape
    f = gate_v.shape[2]
    del f
    slots = e * cap
    dt = torch.bfloat16 if bf16 else x.dtype
    out = x.new_empty((slots, d), dtype=dt)
    rank = x.new_empty((slots, rk), dtype=dt)
    gate = x.new_empty((slots, MM), dtype=torch.float32)
    return out, rank, rank, rank, gate, gate


def _bank_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
    """Save the operands and the five intermediates in `_Bank.backward`'s exact order."""
    x, tok, cnt, gate_u, gate_v, up_u, up_v, down_u, down_v, mix_w, mix_b, coef = inputs[:12]
    cap, kinds, sides, bf16 = inputs[12:]
    ctx.save_for_backward(
        x,
        tok,
        cnt,
        gate_u,
        gate_v,
        up_u,
        up_v,
        down_u,
        down_v,
        mix_w,
        mix_b,
        coef,
        output[1],
        output[2],
        output[3],
        output[4],
        output[5],
    )
    ctx.meta = (cap, tuple(kinds), sides, bool(bf16))


def _bank_backward(ctx: Any, *grads: torch.Tensor | None) -> tuple[Any, ...]:
    """Reuse the eager backward; the shape/flag arguments have no gradient."""
    d_out = cast("torch.Tensor", grads[0])
    return (*_Bank.backward(ctx, d_out), None, None, None, None)  # type: ignore[no-untyped-call]


_bank_op.register_autograd(_bank_backward, setup_context=_bank_setup)


def bank(
    x: torch.Tensor,
    tok: torch.Tensor,
    cnt: torch.Tensor,
    cap: int,
    stacked: Any,
    *,
    kinds: tuple[int, ...],
    sides: str,
) -> torch.Tensor:
    """Return the group's `[group·cap, d]` write, from the fused expert chain.

    `tok` maps a slot to the row of `x` it holds (the index map the dispatch already
    built), `cnt` is the real row count per expert of the group, and `stacked` is the
    group's `[group, ...]` parameter batch. One `prep` launch and one `fwd` launch for the
    forward; `bwd_rows` and `bwd_cols` for the backward.
    """
    if not bank_enabled() or not x.is_cuda:
        raise LoudError(
            f"bank.bank ran with the fused bank off or on a {x.device.type} tensor",
            "this is the kernel entry point, not a fallback: call it only when "
            "bank.bank_enabled() is true, or the anchor reports a kernel that never ran",
        )
    return _launch(  # pragma: no cover - the launch is the card's, not the CPU's
        cap,
        kinds,
        sides,
        x,
        tok,
        cnt,
        cast(tuple[torch.Tensor, torch.Tensor], stacked.mix),
        stacked,
    )


def _launch(  # pragma: no cover - the arguments are copied, then the kernels run
    cap: int,
    kinds: tuple[int, ...],
    sides: str,
    x: torch.Tensor,
    tok: torch.Tensor,
    cnt: torch.Tensor,
    mix: tuple[torch.Tensor, torch.Tensor],
    stacked: Any,
) -> torch.Tensor:
    """Hand the group to the opaque custom op, with every operand contiguous for the pointer.

    Split out so the CPU job can cover the checks above without pretending it launched a
    kernel: `gather`'s and the layouts' arithmetic is tested on the CPU, the launch is not.
    """
    mix_w, mix_b = mix
    bf16 = bf16_step(x.device)
    #: `.contiguous()` is not cosmetic: a Triton program receives a raw pointer and
    #: indexes with the layouts stated in `_build`, so `Stacked.mix`'s transposed
    #: `[group, d, K]` view (stride `d` on a K-sized last axis) would be read as garbage on
    #: every dictionary wider than one entry. Measured: it was, at `K=2` and above.
    tensors = [
        t.contiguous()
        for t in (
            x,
            tok,
            cnt,
            stacked.gate[0],
            stacked.gate[1],
            stacked.up[0],
            stacked.up[1],
            stacked.down[0],
            stacked.down[1],
            mix_w,
            mix_b,
            stacked.coef,
        )
    ]
    return _bank_op(*tensors, cap, tuple(kinds), sides, bf16)[0]  # type: ignore[no-any-return]
