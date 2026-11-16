"""The two pointwise fusions a step can take, and the eager path they must be a drop-in for.

WHY THESE TWO, AND NOT SOMETHING ELSE. A training step's time is matmuls plus the
pointwise work between them, and the second term is invisible in a matmul-only bill: the
old engine's step was 323 ms of which 73 ms was not matmul. Two of that 73 ms are here,
because they are the two blocks that touch a whole residual row more than once for
nothing:

* `residual_norm(x, delta, norm)` — the block's `x = x + attn(...)` immediately followed
  by `norm2(x)`. Eager reads `x` and `delta` to write the sum, reads the sum again to
  compute its mean square, and reads it a third time to scale it. The kernel keeps the
  sum in registers and writes the residual and the normalised copy once each.
* `glu(gate, up)` — `silu(gate) * up`, which eager computes as three elementwise kernels
  over the same `d_ffn`-wide buffer. One kernel here.

WHAT IS *NOT* CLAIMED. The final projection, the cross-entropy and the optimizer are not
fused here; they are the profiler's own columns (`tools/profile_step.py`) and each needs
its own argument. A fusion that took credit for a kernel it did not replace is the thing
this project's counters exist to catch.

THE PARITY BOUND, DERIVED AND NOT CHOSEN. Eager and the kernel compute the same
expression; the only free parameter is the order of the `d`-term sum inside
`mean(x²)` — a tree over `BLOCK = next_pow2(d)` lanes here, torch's own reduction there.
With `u = 2⁻²⁴` the unit roundoff of fp32, a `d`-term sum has a relative error of at most
`gamma_{d-1} ≈ d·u` (Higham, Thm 3.1) and `√d·u` in the random-accumulation case a padded
tree realises. At `d = 832` that is `28.8·5.96e-8 = 1.7e-6` on `mean(x²)`; `rsqrt`
halves the relative error to `0.86e-6`, and `y = r·s·g` multiplies it by `|r|`, which is
`≤ ~4` for a 4-sigma row of a unit-RMS stream. So the honest worst-case bound on an fp32
output is `4·0.86e-6 ≈ 3.4e-6` and the typical one is `≈ 4e-7`; the gate is `FP32_PARITY
= 2e-6`, between the two with margin on the typical case and inside the worst case. A
kernel that misses it is not a drop-in, and the tolerance is not the thing to move. On a
bf16 stream the output is quantised to `2⁻⁸` relative anyway, so that path's gate is 2
ulp of bf16 (`BF16_PARITY`) and is stated separately instead of folded into one number
that hides which path was measured.

BIT-IDENTITY WHEN THE FLAG IS OFF. `residual_norm` with the flag off is
`residual = x + delta` then `norm(residual, counters)` — the same two ops, in the same
order, on the same tensors as the code it replaces, so a step is bit-identical and the
quality baseline cannot move. `tests/test_kernels.py` asserts that with `torch.equal`,
not `allclose`, for the op, for a whole block, and for a whole step.

NO SILENT FALLBACK. A flag that is on and cannot fuse (no CUDA, no Triton) raises
`LoudError`. The alternative — quietly running eager under a flag that claims a kernel —
is this project's recorded failure class: a mechanism that was enabled and never ran,
with every test green.
"""
# ruff: noqa: N803

from __future__ import annotations

import os
from typing import Any, Protocol, cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

#: Absolute parity gate for the fp32 path; the derivation is in the module docstring.
FP32_PARITY = 2e-6

#: Two units in the last place of bf16, as a *relative* gate: the bound is
#: `BF16_PARITY · max|y|`, because bf16's step is relative and a normalised row's
#: largest entry is not 1 — a 4-sigma row of a unit-RMS stream reaches |y| ≈ 4, where one
#: ulp is 0.03. An absolute gate here would be a gate on the row's scale, not on the
#: kernel.
BF16_PARITY = 2.0 * 2.0**-8

#: The Triton kernels, built once on first use. Empty until then; `None` from `_build`
#: means Triton is absent, which `_require` acts on rather than falls back from.
_KERNELS: dict[str, Any] = {}

#: The dictionary entries the mixture kernel knows, as the codes its constexpr branch
#: reads. A name outside this table is not a mixture the kernel claims, and
#: `mixture_code` refuses it instead of guessing a formula.
GELU, SILU, RELU2, LEAKY, TANH, SITU = range(6)
MIX_CODES: dict[str, int] = {
    "gelu": GELU,
    "silu": SILU,
    "relu2": RELU2,
    "leaky": LEAKY,
    "tanh": TANH,
    "situ": SITU,
}

#: Entries one fused mixture launch can sum. The live dictionary is six; a seventh
#: would need a seventh constexpr branch, so a longer one is refused with the count.
MAX_MIX_ENTRIES = 6

#: Columns one mixture program reads at a time. 1024 keeps a program's working set in
#: registers for `d_ffn=6144` (six tiles) at any row count.
MIX_BLOCK = 1024

#: K3's SiTU scale, as `activations.SITU_BETA`; the kernel sits below `model` in the
#: tree and cannot import it, so the constant is stated against its owner.
SITU_BETA = 4.0

#: The flag. Off by default: a fused path nobody measured is not a default.
_FUSED = os.environ.get("DORMOUSE_FUSED", "") not in ("", "0")

#: Fused launches since the last `reset_fused_ops()`. Zero while the flag is off, and
#: zero while the flag is on is a defect, not a state.
_fused_ops = 0


class Norm(Protocol):
    """What the fused op needs from an `RMSNorm`: its gain, its eps, and its eager path."""

    weight: torch.nn.Parameter
    eps: float

    def __call__(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Normalise `x` on the last axis, returning the input's dtype."""
        ...


def set_fused(on: bool) -> None:
    """Turn the fused path on or off for this process."""
    global _FUSED
    _FUSED = bool(on)


def fused() -> bool:
    """Return whether the fused path is on."""
    return _FUSED


def fused_ops() -> int:
    """Return how many fused launches have run since the last reset."""
    return _fused_ops


def reset_fused_ops() -> None:
    """Zero the launch counter; the flag keeps its value."""
    global _fused_ops
    _fused_ops = 0


def next_pow2(n: int) -> int:
    """Return the smallest power of two `>= n`: the width a Triton block has to be."""
    return 1 << (n - 1).bit_length()


def available() -> bool:
    """Return whether a fused launch can run here: Triton installed and a CUDA device."""
    return torch.cuda.is_available() and _build() is not None


def _build() -> dict[str, Any] | None:  # pragma: no cover - requires triton + a device
    """Compile-once Triton kernels; `None` when Triton is not installed."""
    if _KERNELS:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except ImportError:
        return None

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def residual_norm_kernel(X, D, R, Y, W, eps, n_cols: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """One residual row: sum, write it, normalise from the same registers."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        base = row * n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0)
        d = tl.load(D + base + cols, mask=mask, other=0.0)
        # The add happens in the stream's own dtype, exactly as the eager `x + delta`
        # does; only the statistic is fp32, exactly as `RMSNorm.forward` upcasts it.
        r = x + d
        tl.store(R + base + cols, r, mask=mask)
        rf = r.to(tl.float32)
        var = tl.sum(rf * rf, axis=0) / n_cols
        gain = tl.load(W + cols, mask=mask, other=0.0)
        normed = rf * tl.rsqrt(var + eps) * gain
        tl.store(Y + base + cols, normed.to(Y.dtype.element_ty), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def glu_kernel(G, U, OUT, n, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """`silu(gate) * up` over one block of the flattened buffer."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        g = tl.load(G + off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + off, mask=mask, other=0.0).to(tl.float32)
        out = g * tl.sigmoid(g) * u
        tl.store(OUT + off, out.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def _activate(x, kind: tl.constexpr):  # type: ignore[no-untyped-def]
        """One dictionary entry, fp32 in and out — the `activations.py` formula, verbatim."""
        if kind == 0:  # gelu: the erf form, `nn.functional.gelu`'s default
            return 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))
        if kind == 1:  # silu
            return x * tl.sigmoid(x)
        if kind == 2:  # relu2
            relu = tl.maximum(x, 0.0)
            return relu * relu
        if kind == 3:  # leaky, PyTorch's default slope
            return tl.where(x >= 0.0, x, 0.01 * x)
        if kind == 4:  # tanh
            return libdevice.tanh(x)
        # situ: 4·tanh(x/4)·sigmoid(x)
        return 4.0 * libdevice.tanh(x * 0.25) * tl.sigmoid(x)

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def _grad(x, kind: tl.constexpr):  # type: ignore[no-untyped-def]
        """`d/dx` of `_activate`, derived there rather than by a second forward."""
        if kind == 0:  # gelu: ½(1+erf(x/√2)) + x·φ(x)
            return 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476)) + (
                0.3989422804014327 * x * libdevice.exp(-0.5 * x * x)
            )
        if kind == 1:  # silu: s(1 + x(1-s))
            sig = tl.sigmoid(x)
            return sig * (1.0 + x * (1.0 - sig))
        if kind == 2:  # relu2: 2·relu(x)
            return 2.0 * tl.maximum(x, 0.0)
        if kind == 3:  # leaky
            return tl.where(x >= 0.0, 1.0, 0.01)
        if kind == 4:  # tanh
            t = libdevice.tanh(x)
            return 1.0 - t * t
        # situ: (1-t²)·s + 4·t·s·(1-s), t = tanh(x/4)
        sig = tl.sigmoid(x)
        t = libdevice.tanh(x * 0.25)
        return (1.0 - t * t) * sig + 4.0 * t * sig * (1.0 - sig)

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def _coef(W, which: tl.constexpr, row, K: tl.constexpr, PER_ROW: tl.constexpr):  # type: ignore[no-untyped-def]
        """Return the k-th coefficient: one row's entry, or one shared entry."""
        if PER_ROW:
            return tl.load(W + row * K + which)
        return tl.load(W + which)

    @triton.jit  # type: ignore[untyped-decorator]  # pragma: no cover - compiled on device
    def mix_kernel(  # type: ignore[no-untyped-def]
        Z,
        W,
        OUT,
        DZ,
        DW,
        DOUT,
        n_cols,
        K: tl.constexpr,
        PER_ROW: tl.constexpr,
        BACK: tl.constexpr,
        NEED_DW: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        K3: tl.constexpr,
        K4: tl.constexpr,
        K5: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`Σₖ wₖ·fₖ(z)` for one row, or its two gradients, in one pass over `z`.

        One program per row and one pass over the row's `d_ffn` columns: the forward
        writes the mixture, the backward (`BACK`) writes `dz` and, when the coefficients
        need a gradient (`NEED_DW`), `dw [rows, K]`. Combining the two directions here
        keeps ONE copy of each activation formula, which is the thing a second kernel
        would have had to duplicate and then drift from.
        """
        row = tl.program_id(0)
        dw0 = 0.0
        dw1 = 0.0
        dw2 = 0.0
        dw3 = 0.0
        dw4 = 0.0
        dw5 = 0.0
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            z = tl.load(Z + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            if BACK:
                d = tl.load(DOUT + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
                dz = tl.zeros([BLOCK], tl.float32)
            else:
                acc = tl.zeros([BLOCK], tl.float32)
            if K > 0:
                w0 = _coef(W, 0, row, K, PER_ROW)
                a = _activate(z, K0)
                if BACK:
                    dz += w0 * _grad(z, K0) * d
                    if NEED_DW:
                        dw0 += tl.sum(a * d, axis=0)
                else:
                    acc += w0 * a
            if K > 1:
                w1 = _coef(W, 1, row, K, PER_ROW)
                a = _activate(z, K1)
                if BACK:
                    dz += w1 * _grad(z, K1) * d
                    if NEED_DW:
                        dw1 += tl.sum(a * d, axis=0)
                else:
                    acc += w1 * a
            if K > 2:
                w2 = _coef(W, 2, row, K, PER_ROW)
                a = _activate(z, K2)
                if BACK:
                    dz += w2 * _grad(z, K2) * d
                    if NEED_DW:
                        dw2 += tl.sum(a * d, axis=0)
                else:
                    acc += w2 * a
            if K > 3:
                w3 = _coef(W, 3, row, K, PER_ROW)
                a = _activate(z, K3)
                if BACK:
                    dz += w3 * _grad(z, K3) * d
                    if NEED_DW:
                        dw3 += tl.sum(a * d, axis=0)
                else:
                    acc += w3 * a
            if K > 4:
                w4 = _coef(W, 4, row, K, PER_ROW)
                a = _activate(z, K4)
                if BACK:
                    dz += w4 * _grad(z, K4) * d
                    if NEED_DW:
                        dw4 += tl.sum(a * d, axis=0)
                else:
                    acc += w4 * a
            if K > 5:
                w5 = _coef(W, 5, row, K, PER_ROW)
                a = _activate(z, K5)
                if BACK:
                    dz += w5 * _grad(z, K5) * d
                    if NEED_DW:
                        dw5 += tl.sum(a * d, axis=0)
                else:
                    acc += w5 * a
            if BACK:
                tl.store(DZ + row * n_cols + cols, dz.to(DZ.dtype.element_ty), mask=mask)
            else:
                tl.store(OUT + row * n_cols + cols, acc.to(OUT.dtype.element_ty), mask=mask)
        if BACK and NEED_DW:
            if K > 0:
                tl.store(DW + row * K + 0, dw0)
            if K > 1:
                tl.store(DW + row * K + 1, dw1)
            if K > 2:
                tl.store(DW + row * K + 2, dw2)
            if K > 3:
                tl.store(DW + row * K + 3, dw3)
            if K > 4:
                tl.store(DW + row * K + 4, dw4)
            if K > 5:
                tl.store(DW + row * K + 5, dw5)

    _KERNELS.update(
        {
            "triton": triton,
            "residual_norm": residual_norm_kernel,
            "glu": glu_kernel,
            "mix": mix_kernel,
        }
    )
    return _KERNELS


def _require(what: str) -> dict[str, Any]:
    """Return the built kernels, or refuse: a flag that is on must not become eager."""
    built = _build()
    if built is None or not torch.cuda.is_available():
        raise LoudError(
            f"fused={what} is on and cannot run here (triton="
            f"{'absent' if built is None else 'present'}, cuda="
            f"{'absent' if not torch.cuda.is_available() else 'present'})",
            "run with the fused flag off, or use the CUDA torch build; a flag that "
            "silently runs the eager path reports a kernel that never executed",
        )
    return built


def _flat(x: torch.Tensor, width: int) -> torch.Tensor:
    """Return `x` as contiguous `[rows, width]`, copying only when it is not."""
    rows = x.reshape(-1, width)
    return rows if rows.is_contiguous() else rows.contiguous()


def _tally() -> None:
    """Count one fused launch; the caller's `Counters` is updated at the call site."""
    global _fused_ops
    _fused_ops += 1


class _ResidualNorm(torch.autograd.Function):
    """`r = x + delta; y = rmsnorm(r)`: one forward kernel, a torch-op backward.

    The backward is composed of torch ops rather than a second kernel: it is not the
    measured hot path (the profiler's `backward` column is matmuls whichever way this is
    written), and a hand-derived kernel there would be a second place the norm's
    gradient is written down — the defect class this project has paid for twice.
    """

    @staticmethod
    def forward(  # pragma: no cover - device only
        ctx: Any, x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - device kernels
        """Return `(x + delta, rmsnorm(x + delta))`."""
        built = _require("residual_norm")
        width = x.shape[-1]
        xf, df = _flat(x, width), _flat(delta, width)
        residual = torch.empty_like(xf)
        normed = torch.empty_like(xf)
        built["residual_norm"][(xf.shape[0],)](
            xf, df, residual, normed, weight, eps, width, next_pow2(width)
        )
        _tally()
        ctx.save_for_backward(residual, normed, weight)
        ctx.eps = eps
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        return residual.view(x.shape), normed.view(x.shape)

    @staticmethod
    def backward(  # pragma: no cover - device only
        ctx: Any, d_residual: torch.Tensor | None, d_normed: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        """Gradients of `x`, `delta` and the gain.

        With `y = z·s·g`, `s = rsqrt(mean(z²)+eps)` and `z = x + delta`:

            dz_j = s·g_j·dy_j - (s³/n)·z_j·Σ_k(dy_k·z_k·g_k),   dg_j = Σ_rows dy·z·s

        and `dr = d_residual + dz` is the gradient of both inputs, because `+` is.
        """
        residual, _normalised, weight = ctx.saved_tensors
        width = residual.shape[-1]
        z = residual.detach().float()
        s = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + ctx.eps)
        gain = weight.detach().float()
        if d_normed is None:
            dz = torch.zeros_like(z)
            d_gain = torch.zeros_like(gain)
        else:
            dy = _flat(d_normed, width).detach().float()
            d_gain = (dy * z * s).sum(0)
            a = (dy * z * gain).sum(-1, keepdim=True)
            dz = dy * s * gain - (s.pow(3) / width) * z * a
        if d_residual is not None:
            dz = dz + _flat(d_residual, width).detach().float()
        grad = dz.reshape(ctx.shape).to(ctx.dtype)
        return grad, grad, d_gain.to(weight.dtype), None


class _Glu(torch.autograd.Function):
    """`silu(gate) * up` in one kernel; the backward is the two chain rules."""

    @staticmethod
    def forward(
        ctx: Any, gate: torch.Tensor, up: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - device only
        """Return `silu(gate) * up`, elementwise."""
        built = _require("glu")
        width = gate.shape[-1]
        gf, uf = _flat(gate, width), _flat(up, width)
        out = torch.empty_like(gf)
        n = gf.numel()
        block = 1024
        built["glu"][((n + block - 1) // block,)](gf, uf, out, n, block)
        _tally()
        ctx.save_for_backward(gf, uf)
        ctx.shape = gate.shape
        return out.view(gate.shape)

    @staticmethod
    def backward(  # pragma: no cover - device only
        ctx: Any, d_out: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """`dg = dout·u·sigma(g)(1 + g(1-sigma(g)))`, `du = dout·silu(g)`."""
        if d_out is None:
            return None, None
        gate, up = ctx.saved_tensors
        d = d_out.detach().float().reshape(-1)
        g = gate.detach().float().reshape(-1)
        u = up.detach().float().reshape(-1)
        sig = torch.sigmoid(g)
        d_gate = d * u * sig * (1.0 + g * (1.0 - sig))
        d_up = d * (g * sig)
        shape = ctx.shape
        return d_gate.reshape(shape).to(gate.dtype), d_up.reshape(shape).to(up.dtype)


class _Mixture(torch.autograd.Function):
    """`Σₖ wₖ·fₖ(z)` in one launch, its two gradients in one more.

    `kinds` is a tuple of the codes `MIX_CODES` names — integers, not callables. That is
    the whole point: the per-entry path handed `torch.utils.checkpoint` a Python function
    per dictionary entry, and dynamo cannot see through a callable argument (2.13 stops
    at `lift_tracked_freevar_to_input` on the root subgraph tracer), so the bank's forward
    was opaque to `torch.compile` and a graph break of six checkpoints per slice. The
    codes are constants, so the kernel is one static dispatch and the region is traceable.
    """

    @staticmethod
    def forward(  # pragma: no cover - device only
        ctx: Any, z: torch.Tensor, w: torch.Tensor, kinds: tuple[int, ...], per_row: bool
    ) -> torch.Tensor:  # pragma: no cover - device kernels
        """Return the mixture on `z`'s shape; `w` is `[rows, K]` or the shared `[K]`."""
        built = _require("mixture")
        cols = z.shape[-1]
        zf = _flat(z, cols)
        rows, k = zf.shape[0], len(kinds)
        wf = w.reshape(rows, k).contiguous() if per_row else w.reshape(-1).contiguous()
        out = torch.empty_like(zf)
        slots = kinds + (0,) * (MAX_MIX_ENTRIES - k)
        built["mix"][(rows,)](
            zf, wf, out, out, out, out, cols, k, per_row, False, False, *slots, MIX_BLOCK
        )
        _tally()
        ctx.save_for_backward(zf, wf)
        ctx.kinds = kinds
        ctx.per_row = per_row
        ctx.shape = z.shape
        ctx.w_dtype = w.dtype
        ctx.w_shape = w.shape
        ctx.need_dw = bool(ctx.needs_input_grad[1])
        return out.view(z.shape)

    @staticmethod
    def backward(  # pragma: no cover - device only
        ctx: Any, d_out: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:  # pragma: no cover
        """`dz_j = Σₖ wₖ·fₖ'(z_j)·dy_j`, `dwₖ = Σ_j fₖ(z_j)·dy_j`."""
        if d_out is None:
            return None, None, None, None
        built = _require("mixture")
        zf, wf = ctx.saved_tensors
        cols = zf.shape[-1]
        rows, k = zf.shape[0], len(ctx.kinds)
        df = _flat(d_out, cols)
        dz = torch.empty_like(zf)
        dw = torch.zeros((rows, k), dtype=torch.float32, device=zf.device) if ctx.need_dw else dz
        slots = ctx.kinds + (0,) * (MAX_MIX_ENTRIES - k)
        built["mix"][(rows,)](
            zf, wf, dz, dz, dw, df, cols, k, ctx.per_row, True, ctx.need_dw, *slots, MIX_BLOCK
        )
        _tally()
        grad_w = None
        if ctx.need_dw:
            grad_w = dw.reshape(ctx.w_shape) if ctx.per_row else dw.sum(0)
            grad_w = grad_w.to(ctx.w_dtype)
        return dz.view(ctx.shape), grad_w, None, None


def mixture_code(name: str) -> int:
    """Return the kernel's code for dictionary entry `name`; an unknown one is refused."""
    try:
        return MIX_CODES[name]
    except KeyError:
        raise LoudError(
            f"{name!r} is not a dictionary entry the fused mixture computes",
            f"the kernel knows: {', '.join(MIX_CODES)}; run with the fused flag off "
            "for a dictionary it does not",
        ) from None


def mixture(
    z: torch.Tensor, w: torch.Tensor, kinds: tuple[int, ...], *, per_row: bool
) -> torch.Tensor:
    """Return `Σₖ wₖ·fₖ(z)`, one pass over `z` and one launch — the eager chain's twin.

    `per_row` says whether `w` is the `[rows, K]` per-token mixture (`moa`) or one shared
    `[K]` vector (`la`). The caller is `model.ffn._accumulate`, which keeps the chunked,
    checkpointed chain behind the same flag-off branch so a step with the flag off is the
    step it was before this kernel existed.
    """
    if not _FUSED:
        raise LoudError(
            "fused.mixture ran with the fused flag off",
            "this is the kernel entry point, not a fallback: call it only when "
            "fused.fused() is true, or the eager chain reports a launch that never ran",
        )
    if not z.is_cuda:
        _require("mixture")
    if not 1 <= len(kinds) <= MAX_MIX_ENTRIES:
        raise LoudError(
            f"a fused mixture of {len(kinds)} entries",
            f"the kernel sums 1 to {MAX_MIX_ENTRIES} entries with one constexpr branch "
            "each; shorten the dictionary or run with the fused flag off",
        )
    return cast(torch.Tensor, _Mixture.apply(z, w, tuple(kinds), per_row))


def residual_norm(
    x: torch.Tensor, delta: torch.Tensor, norm: Norm, counters: Counters | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(x + delta, norm(x + delta))`: the block's add and its pre-norm, fused.

    Drop-in for the two statements it replaces (`r = x + delta` then `norm(r)`), with
    `norm_fwd` counted exactly as `RMSNorm.forward` counts it.
    """
    if not _FUSED or not x.is_cuda:
        if _FUSED:
            _require("residual_norm")
        residual = x + delta
        return residual, norm(residual, counters)
    residual, normed = _ResidualNorm.apply(x, delta, norm.weight, norm.eps)
    if counters is not None:
        counters.count("fused_ops")
        counters.count("norm_fwd", x.numel() // x.shape[-1])
    return residual, normed


def glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return `silu(gate) * up`: the SwiGLU activation and its multiply, fused."""
    if not _FUSED or not gate.is_cuda:
        if _FUSED:
            _require("glu")
        return torch.nn.functional.silu(gate) * up
    return cast(torch.Tensor, _Glu.apply(gate, up))
