"""NVFP4: e2m1 codes, one e4m3 scale per 16 elements, one fp32 scale per tensor.

THE FORMAT, in the arithmetic it actually is (NVIDIA's NVFP4 — the MXFP4 family
with a second scale level; OCP MX v1.0 §4, and the block-16 rule is the one
`torch._scaled_mm`'s own size check enforces for `float4_e2m1fn_x2`):

    x  ≈  S_tensor · S_block · q

    q        ∈ E2M1 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}      4 bits
    S_block  ∈ E4M3 = 1 sign, 4 exponent, 3 mantissa            8 bits, max 448
    S_tensor ∈ fp32, one per tensor, a power of two

one `S_block` per `BLOCK = 16` elements along the LAST axis, which for a weight
`[out, in]` is the contraction axis — the axis a matmul sums, so the 16 products
that share a scale are 16 terms of one dot product. Cost per value:
`4 + 8/16 = 4.5` bits.

    S_tensor = 2^ceil(log2(amax / (MAX_CODE · E4M3_MAX)))
    S_block  = e4m3( amax_block / (MAX_CODE · S_tensor) )       clipped to e4m3's range
    s        = S_tensor · S_block
    q        = round_e2m1(x / s)
    x̂        = q · s

THE LEVEL TABLE IS DERIVED, NOT TYPED. E2M1 is sign, 2 exponent, 1 mantissa, bias
1, and no implicit bit when `e = 0`:

    e = 0:  mag = m · 0.5                       subnormal
    e > 0:  mag = 2^(e-1) · (1 + m/2)

    e / m      0        1
    0        0.0      0.5
    1        1.0      1.5
    2        2.0      3.0
    3        4.0      6.0

The top level is **6.0**: e2m1's exponent range stops one doubling short of what
a `2^e` table gives. A magnitude table that returns 2.0 for the sixth level is the
exactness bug this file exists to make impossible — `tests/test_fp4.py` decodes
the table from the formula above, not from `LEVELS`.

WHY `S_tensor` IS A POWER OF TWO, AND WHY THAT IS THE SECOND BUG FIX. The divisor
a code is formed against is `s = S_tensor · S_block`. E4M3 carries 4 significant
bits, so `s` is exactly representable in fp32 as long as `S_tensor` is a power of
two — multiplying by `2^k` moves the exponent and never rounds the mantissa. That
makes `x / s` ONE correctly-rounded IEEE division: a value exactly on a midpoint
between two codes lands exactly ON that midpoint and the tie-break below decides
it, instead of an ulp to one side of it. The measured failure in the earlier
kernel was this: `x · (6/amax)` (a reciprocal, itself rounded) applied where
`x / s` was meant, which is 2 ulp off at the midpoints and flips them.

ROUNDING IS RNE, AND THE TIE-BREAK IS THE LADDER'S ALTERNATION. Ties go to the
code with an even mantissa bit. That bit alternates 0,1,0,1 down the magnitude
ladder (0.0 has m=0, 0.5 m=1, 1.0 m=0, 1.5 m=1, ...), so a midpoint rounds DOWN
for a pair starting at an even index and UP for one starting at an odd index:
`(0, .5)` down, `(.5, 1)` up, `(1, 1.5)` down, `(1.5, 2)` up, `(2, 3)` down,
`(3, 4)` up, `(4, 6)` down. `_LADDER` is that alternation written out with its
boundary, so `round_e2m1` is one comparison per level and no lookup that can
disagree with the arithmetic.

WHAT A FORWARD CAN AND CANNOT USE. `torch._scaled_mm` is the only kernel that
touches the format, and on sm_120 it is forward-only: e2m1 operands, no backward.
A training run is therefore QAT only — quantise in the forward, carry a
straight-through estimator (the identity's gradient) in the backward, keep
bf16/fp32 masters — and `linear` refuses the kernel while autograd is on, because
a matmul that drops its gradient is an STE with a wrong slope, silently. On a
forward-only run the kernel is tried and every outcome is recorded; a run that
claims fp4 and never touches `_scaled_mm` says so in `tally()`.

`tally()` is not a `Counters` entry on purpose: a step's counters answer "did the
rung run" (per step, and silent means `MissingReport`), while the tally answers
"which kernel served this build" — a property of the machine and the wheel, not
of the step.

Ceiling (`ponytail:`), and it is a MEASURED one: neither wheel on this machine can even
pack the dtype — `torch 2.14.1+cu130` on the card and `torch 2.15.0+cpu` both raise
`copy_kernel not implemented for 'Float4_e2m1fn_x2'` at the cast, before `_scaled_mm`
is called. So the kernel path below is written and never reached here, and the plain
`[rows, k/16]` e4m3 scale layout it would hand the kernel is untested against NVFP4's
padded 128x4-swizzled one. Both are wired on the first wheel that can pack the dtype
and a free card to check the element order on: the swizzle (`_scaled_mm_v2` with an
explicit `swizzle_a` also exists), then `tally()` proves it ran.
"""

from __future__ import annotations

from collections import Counter

import torch
from dormouse_core.errors import LoudError
from torch.nn import functional

#: Elements per block scale. 16 is the format's number, not a knob: `_scaled_mm`
#: refuses `float4_e2m1fn_x2` with any other block size.
BLOCK = 16

#: The largest magnitude e2m1 can encode. The top level is 6.0, not 8.
MAX_CODE = 6.0

#: E4M3's range: largest finite value, and its smallest subnormal.
E4M3_MAX = 448.0
E4M3_MIN = 2.0**-9

#: Positive e2m1 magnitudes, low to high — the table of the docstring's formula.
LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: The seven steps between those levels: boundary, and whether the tie goes up.
#: The alternation is the mantissa bit's, spelled out rather than recomputed.
_LADDER: tuple[tuple[float, bool], ...] = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)

#: Which path served the format in this process.
_TALLY: Counter[str] = Counter({"fp4_fakequant": 0, "fp4_scaled_mm": 0})

#: One verdict per device: `""` when `_scaled_mm` ran, else the reason it did not.
_VERDICTS: dict[str, str] = {}

#: Devices whose kernel output has already been checked against the reference.
_CHECKED: set[str] = set()


def tally(device: torch.device | str | None = None) -> dict[str, int | str]:
    """Return the two path counters and, for `device`, the kernel's verdict.

    `device` is the device the model RAN on, and it is a parameter for that reason:
    the tally must describe the machine the matmuls happened on, not whichever one
    this process happens to have. `None` asks the default device.
    """
    return {**dict(_TALLY), "verdict": verdict(device)}


def reset() -> None:
    """Zero the path counters; the verdicts are a property of the build and stay."""
    for key in _TALLY:
        _TALLY[key] = 0


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 magnitude, sign kept, ties to the even mantissa."""
    magnitude = x.detach().abs()
    index = torch.zeros_like(magnitude)
    for edge, tie_up in _LADDER:
        index = index + (magnitude >= edge if tie_up else magnitude > edge).to(magnitude.dtype)
    table = torch.tensor(LEVELS, dtype=x.dtype, device=x.device)
    return table[index.long()] * torch.sign(x).masked_fill(x == 0, 1.0)


def block_scales(
    x: torch.Tensor, block: int = BLOCK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(S_tensor, S_block, s)` in fp32: a power of two, an e4m3 value, its product.

    `S_block` is returned in its stored dtype, `torch.float8_e4m3fn`: the format's
    second level IS an 8-bit float, and returning it as one keeps the scheme from
    drifting away from the file it claims to produce. Both `S_tensor` and `s` are fp32
    whatever `x` is — 4 significant bits of e4m3 times a power of two is exact in any
    float with at least 4 mantissa bits, so the arithmetic here is dtype-independent.
    """
    if x.shape[-1] % block:
        raise LoudError(
            f"the last axis is {x.shape[-1]}, not a multiple of the fp4 block {block}",
            f"a block scale covers {block} contiguous elements of the contraction axis: "
            "pad the width, or drop the rung",
        )
    grid = x.detach().abs().float().reshape(*x.shape[:-1], -1, block)
    block_max = grid.amax(-1)
    amax = block_max.amax().clamp_min(E4M3_MIN * MAX_CODE)
    #: `ceil(log2(...))` through `exp2`: the tensor scale is a power of two exactly.
    tensor_scale = torch.exp2(torch.log2(amax / (MAX_CODE * E4M3_MAX)).ceil())
    raw = block_max / (MAX_CODE * tensor_scale)
    block_scale = raw.clamp(E4M3_MIN, E4M3_MAX).to(torch.float8_e4m3fn)
    return tensor_scale, block_scale, tensor_scale * block_scale.float()


def _spread(scale: torch.Tensor, x: torch.Tensor, block: int) -> torch.Tensor:
    """Broadcast one scale per block back onto `x`'s shape, along the last axis."""
    return scale.unsqueeze(-1).expand(*scale.shape, block).reshape(x.shape)


def quantize(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Quantize to the format's grid without a gradient: `x̂ = q·s`, `q` the nearest code."""
    _, _, s = block_scales(x, block)
    #: `s` is exact in `x`'s dtype too (4 mantissa bits, a power of two), so the
    #: division stays a correctly-rounded operation on the value's own grid.
    wide = _spread(s, x, block).to(x.dtype)
    return round_e2m1(x / wide) * wide


def fake_quant(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """`quantize` in the forward and the identity in the backward — the STE.

    `y = x + (x̂ - x).detach()`, so `dy/dx = 1` exactly: not `x̂/x` (the secant) and
    not 0 (the derivative of a step function). `tests/test_fp4.py` pins the gradient
    to `torch.ones_like` and pins the forward to `quantize`, bit for bit.
    """
    _TALLY["fp4_fakequant"] += 1
    return x + (quantize(x, block) - x).detach()


def verdict(device: torch.device | str | None = None) -> str:
    """Attempt one NVFP4 matmul; return `""` when this build runs it, else the reason.

    The attempt IS the measurement: no list of supported dtypes is trusted, because
    the last one of those disagreed with the wheel. Cached per device, so a build
    that cannot pack the format pays for exactly one exception.
    """
    key = str(device if device is not None else _default_device())
    if key not in _VERDICTS:
        try:
            _scaled_mm(torch.ones(16, 64), torch.ones(16, 64), None)
        except Exception as error:
            _VERDICTS[key] = f"{type(error).__name__}: {error}".split(" (Triggered")[0]
        else:
            _VERDICTS[key] = ""
    return _VERDICTS[key]


def _default_device() -> str:
    """Return the device a caller with no opinion means: the accelerator when there is one."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_note() -> str:
    """One sentence on which path fp4 can take on this build, for the plan to print."""
    reason = verdict()
    if reason:
        return f"fp4 here is the reference fake-quant, not the kernel: {reason}"
    return "fp4 can use `_scaled_mm` forward; the backward is the reference STE"


def _scaled_mm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """`x·Wᵀ + b` through the format's kernel; raises if this build cannot take it."""
    _, scaled_x, _ = block_scales(x)
    _, scaled_w, _ = block_scales(weight)
    return torch._scaled_mm(
        quantize(x).to(torch.float4_e2m1fn_x2),
        quantize(weight).to(torch.float4_e2m1fn_x2).transpose(0, 1),
        scaled_x.detach(),
        scaled_w.detach(),
        bias=bias,
        out_dtype=torch.bfloat16,
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activations: bool = True,
) -> torch.Tensor:
    """`x·Wᵀ + b` with the operands on the fp4 grid.

    `activations=False` is the weight-only rung: `x` is read as it is, so the kernel
    (which needs both operands at ONE format) is not a candidate. Autograd on is the
    same kind of refusal: there is no fp4 backward on this hardware, so the kernel
    would drop the gradient instead of passing it through.
    """
    key = str(x.device)
    grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    if activations and not grad and not verdict(key):
        kernel = _kernel(key, x, weight, bias)
        if kernel is not None:
            return kernel
    return _reference(x, weight, bias, activations)


def _kernel(
    key: str, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    """One kernel attempt; a failure is recorded and never retried on this device."""
    try:
        out = _scaled_mm(x, weight, bias)
    except Exception as error:
        _VERDICTS[key] = f"_scaled_mm refused: {type(error).__name__}: {error}"
        return None
    if key not in _CHECKED:
        #: A layout that is ACCEPTED but read in the wrong order produces numbers
        #: rather than an error, so the first result is checked against the format.
        _CHECKED.add(key)
        reference = _reference(x, weight, bias, activations=True).float()
        if not torch.allclose(out.float(), reference, rtol=2e-2, atol=2e-2):
            gap = (out.float() - reference).abs().max().item()
            _VERDICTS[key] = f"the kernel ran and disagreed with the reference by {gap:.3g}"
            return None
    _TALLY["fp4_scaled_mm"] += 1
    return out


def _reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activations: bool
) -> torch.Tensor:
    """Return the fake-quant matmul: the operands on the grid, the product in their dtype."""
    return functional.linear(fake_quant(x) if activations else x, fake_quant(weight), bias)
