"""The quantisations the stack can be built with, and the blocks that prove them.

A quantization is a mechanism like any other here: it changes the arithmetic of every
forward, so it must move a counter that the step reads. The protocol is the same as
`model/attn_variants'`: the number in the plan is only a claim, the counter is the
proof.

WEIGHTS: TERNARY, AND WHY NOT `NN.UTILS.PARAMETRIZE`. `ternary` is the BitNet b1.58
straight-through estimator — round the weight to `s·{-1,0,1}` with `s` the per-matrix
mean magnitude, and pass the gradient through unchanged. A parametrization would rename
the parameter to `parametrizations.weight.original`, and Muon's routing reads names, so
the class swap keeps the names stable.

ACTIVATIONS: A4, PER TOKEN. Block-FWHT rotation, `absmax/7` scale at each normalised
stream and a `sigma·2.5/7` scale on the bank's wide slabs; int4 codes packed two per byte,
inverse rotation, straight-through gradient. A single config slot (`act_quant="a4"`)
turns on BOTH the two pre-sublayer norm boundaries and the bank's slab code-saving —
not an env flag. The round can be nearest-even or deterministically seeded stochastic;
the config is the model from step 0, with no warm-up or ramp.

The one fused path lives in `dormouse_kernels.engine.a4_fused`: one Triton launch each
for the norm boundary's forward and backward. There is no kernel copy in this app. The
bank's own packed-slab kernels stay here because they are part of its autograd boundary,
and a flag asking for a fused kernel on a machine without CUDA is a named refusal, never
an eager path under a kernel's name.

`int4` remains the older per-token fake quant of the comparison arms. Its conflict with
`torch.compile` is measured rather than assumed: on the four-layer stack
`torch._dynamo.explain` reports 12 graphs and 11 graph breaks with int4 against 11 and
10 without, so `spec.validate` refuses the pair instead of pricing a kernel that never
ran.
"""

from __future__ import annotations

# Triton tile names are conventionally upper-case; the project's kernel modules
# carry the same per-file exception.
# ruff: noqa: N803
import math
import os
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from .tsct import scale_for_rms, ternarized

try:  # the CPU-only build has no Triton; the torch path below is the anchor
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the environment
    triton, tl = None, None

__concepts__ = ("QuantBlock", "QuantLinear", "NativeQuantLinear", "BankA4Save", "A4Save")


def ternary(weight: torch.Tensor) -> torch.Tensor:
    """`s·round(w/s)` clipped to `{-1, 0, 1}`, with the gradient passed through."""
    scale = weight.detach().abs().mean().clamp_min(1e-8)
    return weight + ((weight / scale).round().clamp(-1, 1) * scale - weight).detach()


def int4(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int4 fake-quantization, straight-through."""
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7
    return x + ((x / scale).round().clamp(-7, 7) * scale - x).detach()


#: The model flag. `DORMOUSE_ACT_A4=1` asserts the native a4+ternary model.
A4_FLAG = "DORMOUSE_ACT_A4"
#: `0` saves the fp32 activation instead of packed int4 codes (VRAM attribution arm).
CODESAVE_FLAG = "DORMOUSE_ACT_A4_CODESAVE"
#: The fused-kernel arm. Off keeps the Python/FWHT path; on requires Triton/CUDA.
FUSED_FLAG = "DORMOUSE_A4_FUSED"
#: `near` (default) or `sr`: nearest-even or deterministically seeded stochastic round.
ROUND_FLAG = "DORMOUSE_A4_ROUND"
#: `absmax` (default) or `pc`: per-token absmax scale, or its per-channel learned multiple.
SCALE_FLAG = "DORMOUSE_A4_SCALE"
_JOURNAL_CAP = 20000


def a4_on() -> bool:
    """Whether the model flag is set; the config still decides what is built."""
    return os.getenv(A4_FLAG, "0").strip().lower() not in {"", "0", "off", "false", "no"}


def codesave_on() -> bool:
    """Whether autograd saves packed codes rather than a fp32 activation."""
    return os.getenv(CODESAVE_FLAG, "1").strip().lower() not in {"0", "off", "false", "no"}


def fused_on() -> bool:
    """Whether the single-kernel FWHT/quant/pack path is requested."""
    return os.getenv(FUSED_FLAG, "0").strip().lower() not in {"", "0", "off", "false", "no"}


def round_mode() -> str:
    """Which round decision the int4 grid uses; anything but `sr` is nearest-even."""
    return os.getenv(ROUND_FLAG, "near").strip().lower()


def scale_mode() -> str:
    """Which scale the int4 grid uses; anything but `pc` is per-token absmax/7."""
    return os.getenv(SCALE_FLAG, "absmax").strip().lower()


# The round stream is a pure function of (run_seed, step, forward-call index). A step
# boundary resets the call index so the same seed reproduces the same decisions even if
# a caller invokes a different number of A4 forwards in an earlier step.
_round_seed_step: tuple[int, int] = (0, 0)
_call_index = 0
_journal: list[dict[str, int | float | str]] = []
#: Set by `StackSpec.validate`: the config, not an env flag, is the source of truth for
#: whether the bank slabs are quantised and saved as codes on this run.
_a4_config = False


def set_a4_config(enabled: bool) -> None:
    """Bind bank quant+codesave to `act_quant="a4"` rather than to an env flag."""
    global _a4_config
    _a4_config = bool(enabled)


def a4_config_on() -> bool:
    """Whether the last validated spec named the native a4 mechanism."""
    return _a4_config


def set_step_seed(run_seed: int, step: int) -> None:
    """Bind the stochastic-round stream to one deterministic `(run_seed, step)`."""
    global _round_seed_step, _call_index
    _round_seed_step = (int(run_seed), int(step))
    _call_index = 0


def take_round_journal() -> list[dict[str, int | float | str]]:
    """Return and clear the round events recorded since the previous read."""
    global _journal
    events, _journal = _journal, []
    return events


def _next_seed() -> int:
    """Mix the three coordinates into a positive 31-bit torch generator seed."""
    run_seed, step = _round_seed_step
    return (run_seed * 1000003 + step * 9176 + _call_index * 69069) & 0x7FFFFFFF


def _record(events: int, total: int, base_mean: float) -> None:
    """Append one round event, bounding the journal so a long run cannot grow without end."""
    global _journal
    if len(_journal) >= _JOURNAL_CAP:
        _journal = _journal[-_JOURNAL_CAP + 1 :]
    _journal.append(
        {
            "run_seed": _round_seed_step[0],
            "step": _round_seed_step[1],
            "call": _call_index,
            "mode": round_mode(),
            "rounded_up": events,
            "total": total,
            "base_mean": base_mean,
        }
    )


def _round_codes(u: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Nearest-even or seeded stochastic round on the int4 grid; return codes and ups."""
    if round_mode() != "sr":
        return u.round(), 0
    lo = u.floor()
    probability = (u - lo).clamp(0.0, 1.0)
    generator = torch.Generator().manual_seed(_next_seed())
    draw = torch.rand(u.shape, generator=generator, dtype=torch.float32).to(u.device)
    up = draw < probability
    return torch.where(up, lo + 1, lo), int(up.sum().item())


#: One FWHT block. `d_ffn=6144` is 96 blocks of 64 exactly.
FWHT_BLOCK = 64
#: Symmetric int4 codes are `[-7, 7]`; adding 8 maps them to 1..15, two per byte.
INT4_MAX = 7
INT4_BIAS = 8
#: The bank's codesaving A/B. Off by default: the fused bank already deletes the
#: `[rows, d_ffn]` slabs, so an enabled counter there would be a MissingReport.
BANK_FLAG = "DORMOUSE_BANK_A4_CODESAVE"
#: The model mechanism: INT4 on the bank's two `[rows, d_ffn]` reads. Both the
#: storage control and the codesaving arm quantize the same way; only autograd's
#: retained form differs, so a BPB A/B isolates the storage change. The model
#: flag has the same name as the architecture's A4 activation mechanism.
BANK_QUANT_FLAG = "DORMOUSE_BANK_A4"

#: Boundaries issued by `BankA4Save`, read and zeroed once per step.
_bank_fwd = 0
_bank_bwd = 0
_bank_codes = 0


def _flag_on(name: str) -> bool:
    """Return whether one of the bank's flags is set."""
    return os.environ.get(name, "0").strip().lower() not in {"", "0", "off", "false", "no"}


def bank_codesave_on() -> bool:
    """Whether the bank retains INT4 codes instead of wide activations for backward."""
    return _a4_config or _flag_on(BANK_FLAG)


def bank_quant_on() -> bool:
    """Whether the bank's two wide reads are INT4 activations from step 0."""
    return _a4_config or _flag_on(BANK_QUANT_FLAG) or bank_codesave_on()


def take_bank_codesave_counters() -> tuple[int, int, int]:
    """Return and reset `(fwd, bwd, codes_saved)` since the previous read."""
    global _bank_fwd, _bank_bwd, _bank_codes
    seen = (_bank_fwd, _bank_bwd, _bank_codes)
    _bank_fwd = _bank_bwd = _bank_codes = 0
    return seen


@cache
def _hadamard(block: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the orthonormal `H` of one block, cached per dtype and device.

    The 64-wide matrix is 96 tiles of the 6144-wide bank slab; one matmul moves it
    where the butterfly chain used ~24 elementwise kernels per slab. `H` is
    symmetric and `H @ H` is the identity, so the inverse is the same read.
    """
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    h = base
    while h.shape[0] < block:
        h = torch.kron(h, base)
    if h.shape[0] != block or block & (block - 1):
        raise ValueError(f"_hadamard: {block} is not a power of two")
    return (h / math.sqrt(block)).to(dtype=dtype, device=device)


def _fit_block(width: int, block: int) -> int:
    """Return the largest power-of-two divisor of `width` not above `block`.

    A narrow anchor stream (`d=32`) is a legal host for the same transform; the
    bank's 64-wide schedule is the live shape, not a requirement of the math.
    """
    chosen = block
    while chosen > 1 and width % chosen:
        chosen //= 2
    if width % chosen or chosen < 8:
        raise ValueError(f"width {width} is not tiled by power-of-two block {block}")
    return chosen


def fwht(x: torch.Tensor, block: int = FWHT_BLOCK) -> torch.Tensor:
    """Orthonormal block Walsh-Hadamard transform: the involutive `H = W/√block`.

    One batched matmul over the block tiles, not a chain of `unbind`/`stack`
    butterflies: same H, one launch. Blockwise because `d_ffn=6144` is not a power
    of two, and a rotation that mixed only part of the stream could not be
    inverted by the same call.
    """
    width = x.shape[-1]
    block = _fit_block(width, block)
    h = _hadamard(block, x.dtype, x.device)
    y = x.reshape(*x.shape[:-1], width // block, block) @ h
    return y.reshape(x.shape)


def fwht_a4(x: torch.Tensor, block: int = FWHT_BLOCK) -> torch.Tensor:
    """Return the norm boundary's butterfly rotation: the fused kernel's schedule.

    `fwht` above is the bank's one-batched-matmul rotation (the same matrix H, a
    different addition order). At the 832-wide norm stream the two differ by ulps
    on CUDA; the kernel's six unrolled butterflies must match the eager path
    bit for bit or a 20-step trace drifts, so the norm path uses this schedule.
    """
    width = x.shape[-1]
    block = _fit_block(width, block)
    y = x.reshape(*x.shape[:-1], width // block, block)
    step = 1
    while step < block:
        y = y.reshape(*y.shape[:-2], width // block, block // (2 * step), 2, step)
        a, b = y.unbind(-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], width // block, block)
        step *= 2
    return y.reshape(x.shape) / math.sqrt(block)


class A4Save(torch.autograd.Function):
    """INT4 activation quantisation at a normalised stream: codes, not the activation.

    Forward: `z = H(x)`, `s = per_token_absmax(z)/7` times an optional per-channel
    multiplier, `q = round(z/s)` clipped to `[-7, 7]`, two codes packed per uint8,
    `y = H(q·s)`. Backward: `H` is orthonormal and involutive and STE makes the round's
    Jacobian the identity, so `dL/dx = H(H(dL/dy)) = dL/dy`; a per-channel multiplier,
    when present, gets the real gradient `Σ_t H(g)[t,c]·q[t,c]·absmax[t]/7`.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        counters: Counters | None = None,
        channel_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:  # pragma: no cover - covered on card and by the eager CPU arm
        """Rotate, quantise, pack, rotate back; save codes or the control activation."""
        global _call_index
        z = fwht_a4(x.to(torch.float32))
        base = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / INT4_MAX
        scale = base if channel_scale is None else base * channel_scale.reshape(1, -1)
        raw, ups = _round_codes(z / scale)
        q = raw.clamp(-INT4_MAX, INT4_MAX).to(torch.int8)
        packed = _pack(q)
        _record(ups, q.numel(), float(base.mean().item()))
        _call_index += 1
        ctx.channel_scale = channel_scale
        ctx.codesave = codesave_on()
        if ctx.codesave:
            ctx.save_for_backward(packed, base, channel_scale)
        else:
            ctx.save_for_backward(x, q, base, channel_scale)
        ctx.counters = counters
        y = fwht_a4(q.to(torch.float32) * scale).to(x.dtype)
        if counters is not None:
            counters.count("a4_fwd", x.numel())
            counters.count("a4_codes_saved", packed.numel() if ctx.codesave else 0)
            if round_mode() == "sr":
                counters.count("a4_sr_rounds", ups)
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, Any]:
        """Straight-through rotation round trip, plus the optional scale gradient."""
        if ctx.counters is not None:
            ctx.counters.count("a4_bwd", grad.numel())
        g = fwht_a4(grad.to(torch.float32))  # dL/d(q·s); one of the two Hadamards
        saved = ctx.saved_tensors
        if ctx.channel_scale is None:
            return fwht_a4(g).to(grad.dtype), None, None  # H(H(g)) = g
        if ctx.codesave:
            packed, base, _ = saved
            q = _unpack(packed, g.shape[-1])
        else:
            _, q, base, _ = saved
            q = q.to(torch.float32)
        grad_scale = (g * q * base).sum(dim=tuple(range(g.dim() - 1))).to(ctx.channel_scale.dtype)
        if ctx.counters is not None:
            ctx.counters.count("a4_scale_grad", grad_scale.numel())
        return fwht_a4(g).to(grad.dtype), None, grad_scale  # H(H(g)) = g


def a4(
    x: torch.Tensor, counters: Counters | None = None, channel_scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply A4: one library kernel per direction when requested, eager otherwise."""
    if fused_on():
        if round_mode() == "sr" or channel_scale is not None:
            raise LoudError(
                "DORMOUSE_A4_FUSED=1 with SR or per-channel scale calibration",
                "the fused kernel implements nearest-even absmax/7 only: set "
                "DORMOUSE_A4_ROUND=near and DORMOUSE_A4_SCALE=absmax, or unset the flag",
            )
        from dormouse_kernels.engine import a4_fused

        return a4_fused.a4_fused(x, counters)
    return cast(
        "torch.Tensor",
        A4Save.apply(x, counters, channel_scale),
    )


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack two signed int4 codes into one uint8 byte, low nibble first."""
    low = codes[..., 0::2] + INT4_BIAS
    high = codes[..., 1::2] + INT4_BIAS
    return (low | (high << 4)).to(torch.uint8)


def _unpack(packed: torch.Tensor, width: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Undo `_pack`; the audited inverse, used by the bank's backward."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], width).to(dtype) - INT4_BIAS


#: The per-token scale is `rms·2.5/7` of the rotated row, clipped at ±7. A
#: Gaussian row's own absmax/7 (the norm-output arm's scale) wastes four of the
#: eight levels on outliers: at d_ffn=6144 that costs 15.9% relative RMSE against
#: 11.3% here, and the backward's derivative is evaluated at the reconstruction.
BANK_SCALE_SIGMA = 2.5


if tl is not None:  # pragma: no cover - device kernels compiled on first use

    @triton.jit  # type: ignore[untyped-decorator]
    def _quant_pack(  # type: ignore[no-untyped-def]
        Z,
        P,
        S,
        L,
        width: tl.constexpr,
        sigma: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One row's scale, INT4 codes and pre-rotation latent in one pass."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        z = tl.load(Z + row * width + cols, mask=cols < width, other=0.0).to(tl.float32)
        std = tl.sqrt(tl.sum(z * z, 0) / width)
        scale = tl.maximum(sigma * std / 7.0, 1e-8)
        ratio = z / scale
        nearest = tl.where(ratio >= 0, tl.floor(ratio + 0.5), tl.ceil(ratio - 0.5))
        q = tl.minimum(tl.maximum(nearest, -7.0), 7.0).to(tl.int32)
        tl.store(
            L + row * width + cols,
            (q.to(tl.float32) * scale).to(L.dtype.element_ty),
            mask=cols < width,
        )
        q2 = tl.reshape(q, (BLOCK // 2, 2))
        pair = tl.arange(0, 2)
        even = tl.sum(tl.where(pair[None, :] == 0, q2, 0), 1)
        odd = tl.sum(tl.where(pair[None, :] == 1, q2, 0), 1)
        byte = ((even + 8) | ((odd + 8) << 4)).to(tl.uint8)
        idx = tl.arange(0, BLOCK // 2)
        tl.store(P + row * (width // 2) + idx, byte, mask=idx < width // 2)
        tl.store(S + row, scale)

    @triton.jit  # type: ignore[untyped-decorator]
    def _quant_unpack(P, S, L, width: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """One row's two-codes-per-byte read back to a pre-rotation latent."""
        row = tl.program_id(0)
        idx = tl.arange(0, BLOCK // 2)
        byte = tl.load(P + row * (width // 2) + idx, mask=idx < width // 2, other=0).to(tl.int32)
        low, high = (byte & 0x0F) - 8, (byte >> 4) - 8
        q = tl.reshape(tl.join(low, high), (BLOCK,))
        scale = tl.load(S + row)
        cols = tl.arange(0, BLOCK)
        tl.store(
            L + row * width + cols,
            (q.to(tl.float32) * scale).to(L.dtype.element_ty),
            mask=cols < width,
        )


def _shape_ok(x: torch.Tensor) -> None:
    """Refuse only a width with no power-of-two tile at all."""
    width = x.shape[-1]
    if width % 8:
        raise ValueError(f"pack_slab: width {width} is not tiled by 64")


def pack_slab_latent(
    x: torch.Tensor, z: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(packed codes, per-token scale, pre-rotation INT4 latent)`.

    The latent is what `fwht` maps to the slab the bank reads, so the forward
    needs one rotation and one kernel — not a separate unpack pass. `z` lets a
    caller that already rotated reuse that read.
    """
    _shape_ok(x)
    #: Quantisation is a non-differentiable projection. Neither the scale nor
    #: the codes carry a gradient; the STE lives in `a4_bank_readout`/`BankA4Save`.
    if x.is_cuda and tl is not None:
        z = (fwht(x) if z is None else z).detach().contiguous()
        width = x.shape[-1]
        half = width // 2
        leading = x.shape[:-1]
        packed = torch.empty((*leading, half), device=x.device, dtype=torch.uint8)
        scale = torch.empty((*leading, 1), device=x.device, dtype=torch.float32)
        latent = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _quant_pack[(x.numel() // width,)](
            z,
            packed,
            scale,
            latent,
            width,
            BANK_SCALE_SIGMA,
            triton.next_power_of_2(width),
            num_warps=8,
        )
        return packed, scale, latent
    z = (fwht(x) if z is None else z).detach()
    std = z.square().mean(dim=-1, keepdim=True).sqrt()
    scale = (BANK_SCALE_SIGMA * std / INT4_MAX).clamp_min(1e-8)
    q = (z / scale).round().clamp(-INT4_MAX, INT4_MAX).to(torch.int8)
    latent = (q.to(z.dtype) * scale).to(x.dtype)
    return _pack(q), scale, latent


def a4_bank_readout(x: torch.Tensor) -> torch.Tensor:
    """Return the INT4 bank read with STE for the wide-retaining arm.

    Same forward as `BankA4Save` and the same straight-through gradient: the
    code path is detached, so only the activation's stored form differs.
    """
    z = fwht(x)
    readout = fwht(pack_slab_latent(x, z)[2])
    return readout + (z - z.detach())


def pack_slab(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a `[rows, width]` bank slab into two INT4 codes per uint8 and a row scale."""
    packed, scale, _ = pack_slab_latent(x)
    return packed, scale


def unpack_latent(
    packed: torch.Tensor, scale: torch.Tensor, width: int, dtype: torch.dtype
) -> torch.Tensor:
    """Undo the pack to the pre-rotation latent; `fwht` of it is the slab."""
    if packed.is_cuda and tl is not None:
        half = width // 2
        leading = packed.shape[:-1]
        latent = torch.empty((*leading, width), device=packed.device, dtype=dtype)
        _quant_unpack[(packed.numel() // half,)](
            packed, scale, latent, width, triton.next_power_of_2(width), num_warps=8
        )
        return latent
    return _unpack(packed, width, dtype) * scale.to(dtype)


def unpack_slab(
    packed: torch.Tensor, scale: torch.Tensor, width: int, dtype: torch.dtype
) -> torch.Tensor:
    """Undo `pack_slab`: unpack -> FWHT -> the reconstructed bank slab.

    The rounding's Jacobian is the identity under STE, so the downstream gradient
    passes through the code unchanged.
    """
    return fwht(unpack_latent(packed, scale, width, dtype))


def _rebuild(
    saved: tuple[torch.Tensor, ...],
    selected: set[int],
    shapes: list[torch.Size],
    dtypes: list[torch.dtype],
    n_inputs: int,
) -> list[torch.Tensor]:
    """Reassemble one `fn` call from packed codes and the cheap saved inputs."""
    wide = [
        unpack_slab(saved[2 * i], saved[2 * i + 1], shape[-1], dtype)
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
    ]
    others = iter(saved[2 * len(shapes) :])
    return [wide.pop(0) if i in selected else next(others) for i in range(n_inputs)]


class BankA4Save(torch.autograd.Function):
    """A checkpoint whose wide bank slabs are INT4 codes, not bf16/fp32.

    Forward packs each selected slab with a block FWHT and per-token scale, and
    feeds the same code readout back to `fn` under autograd off: no wide tensor is
    kept by the ops inside `fn`. Backward unpacks, rotates with the same involutive
    H, recomputes `fn` at the SAME readout the forward used, and passes that
    gradient straight through the rounding (STE). Forward and backward therefore
    agree on the activation; the code is the only wide state autograd retains.

    `indices` names which tensor inputs are the wide slabs. Other inputs (e.g. the
    `[rows, K]` mixture weights) are cheap and carried with real gradients. `fn` is
    a callable, so this boundary is eager-only; `StackSpec.validate` refuses it with
    `torch_compile=true` rather than pricing a graph that would break.
    """

    @staticmethod
    def forward(
        ctx: Any, fn: Callable[..., torch.Tensor], indices: Sequence[int], *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Pack the selected slabs, then run `fn` on their code readout."""
        global _bank_fwd, _bank_codes
        selected = set(indices)
        saved: list[torch.Tensor] = []
        shapes: list[torch.Size] = []
        dtypes: list[torch.dtype] = []
        latents: list[torch.Tensor] = []
        for index in indices:
            packed, scale, latent = pack_slab_latent(inputs[index])
            saved.extend((packed, scale))
            latents.append(fwht(latent))
            shapes.append(inputs[index].shape)
            dtypes.append(inputs[index].dtype)
            _bank_codes += packed.numel()
            _bank_fwd += 1
        others = [inputs[i] for i in range(len(inputs)) if i not in selected]
        ctx.save_for_backward(*saved, *others)
        ctx.fn = fn
        ctx.selected = selected
        ctx.shapes = shapes
        ctx.dtypes = dtypes
        ctx.n_wide = len(indices)
        ctx.n_inputs = len(inputs)
        latents_iter = iter(latents)
        others_iter = iter(others)
        readouts = [
            next(latents_iter) if i in selected else next(others_iter) for i in range(len(inputs))
        ]
        outputs = fn(*readouts)
        return (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor | None) -> tuple[Any, ...]:
        """Unpack -> FWHT -> recompute -> STE; one gradient per tensor input."""
        global _bank_bwd
        _bank_bwd += 1
        inputs = _rebuild(ctx.saved_tensors, ctx.selected, ctx.shapes, ctx.dtypes, ctx.n_inputs)
        need = ctx.needs_input_grad[2:]
        leaves = [t.detach().requires_grad_(True) for t, n in zip(inputs, need, strict=True) if n]
        with torch.enable_grad():
            outputs = ctx.fn(*leaves)
            outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
            parts = torch.autograd.grad(outputs, leaves, grad_outputs=grads, allow_unused=True)
        part_iter = iter(parts)
        slots = {index: slot for slot, index in enumerate(sorted(ctx.selected))}
        result: list[Any] = [None, None]
        for i, required in enumerate(need):
            grad = next(part_iter) if required else None
            if grad is not None and i in ctx.selected:
                #: The control's STE path rotates `z = H(x)` in forward and its
                #: adjoint `H` here; applying the same one rotation keeps both
                #: arms' gradients identical, down to bf16 rounding.
                grad = fwht(grad.to(ctx.dtypes[slots[i]]))
            result.append(grad)
        return tuple(result)


def bank_a4(
    fn: Callable[..., torch.Tensor], indices: Sequence[int], *inputs: torch.Tensor
) -> torch.Tensor:
    """Apply `BankA4Save` to the wide inputs of one bank read; one output."""
    out = cast("tuple[torch.Tensor, ...]", BankA4Save.apply(fn, tuple(indices), *inputs))
    if len(out) != 1:
        raise ValueError(f"bank_a4: expected one output, got {len(out)}")
    return out[0]


class QuantLinear(nn.Linear):
    """An `nn.Linear` whose weight is ternarised for the forward, never in the state."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run `nn.Linear`'s contract, with one `ternary` call on the weight."""
        return functional.linear(x, ternary(self.weight), self.bias)


class NativeQuantLinear(nn.Linear):
    """An `nn.Linear` whose weight is ternary from step 1, with a learnable scale.

    The fp32 weight is the optimizer's shadow state only: `forward` reads the
    `scale·{-1,0,+1}` code, sized at init to keep the layer's fp32 RMS. The older
    `QuantLinear` uses a detached `mean|x|` scale that shrinks a Gaussian weight
    about 6x; training from scratch with that shrink is a different (much worse)
    model, so the native arm does not use it.
    """

    scale: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run `nn.Linear`'s contract on the ternary code, never on the fp32 shadow."""
        return functional.linear(x, ternarized(self.weight, self.scale), self.bias)


def native_ternarize(model: nn.Module) -> int:
    """Swap every plain `nn.Linear` to the native arm; return how many were swapped."""
    swapped = 0
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = NativeQuantLinear
            with torch.no_grad():
                module.scale = nn.Parameter(scale_for_rms(module.weight))
            swapped += 1
    return swapped


def ternarize(model: nn.Module) -> int:
    """Swap every plain `nn.Linear` for `QuantLinear`; return how many were swapped."""
    swapped = 0
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = QuantLinear
            swapped += 1
    return swapped


class QuantBlock(nn.Module):
    """A block with the quantizations applied where they belong, and counted.

    `matrices` is how many ternary matrices this block's forward reads (shared bank
    matrices are read by every block, so each block reports its own read), and
    `act_int4` quantizes the activations entering it. Both counters are the proof the
    plan's bits are real: an enabled quantization that never ran raises
    `MissingReport`, which is the defect this file exists to make impossible.
    """

    def __init__(self, inner: nn.Module, *, matrices: int = 0, act_int4: bool = False) -> None:
        """Wrap a block; `matrices` is a claim about the inner module's weights."""
        super().__init__()
        self.inner = inner
        self.matrices = matrices
        self.act_int4 = act_int4

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Quantize what the block reads, then run it."""
        if self.act_int4:
            x = int4(x)
            if counters is not None:
                counters.count("act_int4", x.shape[0] * x.shape[1])
        if counters is not None and self.matrices:
            counters.count("weight_ternary", self.matrices)
            counters.count("ternary_fwd", self.matrices)
        return cast(torch.Tensor, self.inner(x, counters))
