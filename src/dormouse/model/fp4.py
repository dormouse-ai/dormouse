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

PACKING IS OURS BECAUSE THE WHEEL HAS NO COPY KERNEL FOR THE DTYPE. Every cast to
`torch.float4_e2m1fn_x2` on this build (`torch 2.14.1+cu130`) raises
`copy_kernel not implemented for 'Float4_e2m1fn_x2'`, which is why the previous
measurement of this rung measured a fake-quant and called it FP4. Nothing about
the format needs that kernel: two codes are one byte (the FIRST code in the low
nibble, the second in the high nibble — `pack` below, and the kernel's own
arithmetic agrees, `tests/test_fp4.py`), and a `torch.uint8` buffer of the right
last-dim ratio is viewed as the dtype in place (`as_fp4`). `view()` reinterprets
the same storage — it is a stride/shape declaration, not a cast, so the missing
copy kernel never runs. `.view()` was the one that works; `as_strided` on the
uint8 buffer also reaches the dtype but only by rebuilding the layout the view
already gives for free, and `torch.empty(..., dtype=float4_e2m1fn_x2)` then
`copy_` into a uint8 view of the destination works too but copies through a
second buffer for nothing.

THE SCALES ARE SWIZZLED, AND THAT IS NOT OPTIONAL. `_scaled_mm` answers
`ValueError: scale_a must be swizzled to SWIZZLE_32_4_4 format` for the plain
`[rows, k/16]` layout — cuBLAS's block-scaled GEMM reads scale factors out of a
128x4 tile of (128 rows x 4 k-blocks), internally 32x4x4 (cuBLAS §3.1.4.3.2). The
tile is padded to 128 rows and 4 k-blocks, which is why a `[832, 52]` scale is
handed to the kernel as 896 x 52 values. `swizzle` is that permutation, and
`tests/test_fp4.py` checks it against the reference implementation that ships in
torch's own test utilities, element for element.

WHAT A FORWARD CAN AND CANNOT USE. `torch._scaled_mm` is the only kernel that
touches the format, and on sm_120 it is forward-only: e2m1 operands, no backward.
A training run is therefore QAT only — quantise in the forward, carry a
straight-through estimator (the identity's gradient) in the backward, keep
bf16/fp32 masters — because a matmul that drops its gradient is an STE with a
wrong slope, silently. `fp4_mm` is the primitive: it runs the kernel or raises
with the reason, and never substitutes the reference. `linear` is the rung: under
autograd it takes the STE, without autograd it takes the kernel, and when the
device has no kernel it counts `fp4_fallback` so a run cannot claim the kernel it
did not use.

`tally()` answers "which path served this build", not "did the rung run" (the step
counters answer that): `fp4_scaled_mm` is the kernel, `fp4_fakequant` the STE, and
`fp4_fallback` the times an eligible matmul was served by the reference because
the device refused the kernel. `require_kernel` is the gate over those numbers,
used by the test and by `scripts/bench_fp4.py` — a run on a device whose verdict is
"the kernel runs" with `fp4_scaled_mm == 0` fails, which is the failure this file
was rewritten for.

Ceiling (`ponytail:`): the kernel path is forward-only here, and the per-tensor
scale is applied as one elementwise multiply over the `[m, n]` output rather than
folded into the epilogue (`_scaled_mm`'s `scale_result` is fp32-only). Both are
fine at the measured sizes and both are visible in `scripts/bench_fp4.py`.
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

#: Rows of a scale tile, and k-blocks per tile: the 128x4 shape cuBLAS reads.
_TILE_ROWS = 128
_TILE_K = 4

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

#: The sign bit of a code, and the mask of its magnitude.
_SIGN = 8
_MAG = 7

#: Which path served the format in this process.
_TALLY: Counter[str] = Counter({"fp4_fakequant": 0, "fp4_scaled_mm": 0, "fp4_fallback": 0})

#: One verdict per device: `""` when `_scaled_mm` runs, else the reason it did not.
_VERDICTS: dict[str, str] = {}


def tally(device: torch.device | str | None = None) -> dict[str, int | str]:
    """Return the three path counters and, for `device`, the kernel's verdict.

    `device` is the device the model RAN on, and it is a parameter for that reason:
    the tally must describe the machine the matmuls happened on, not whichever one
    this process happens to have. `None` asks the default device.
    """
    return {**dict(_TALLY), "verdict": verdict(device)}


def reset() -> None:
    """Zero the path counters; the verdicts are a property of the build and stay."""
    for key in _TALLY:
        _TALLY[key] = 0


def require_kernel(device: torch.device | str | None = None) -> None:
    """Fail unless the rung's matmuls were served by the kernel on a device that has one.

    This is the gate the rung needs and did not have: a run whose tests passed while
    `fp4_scaled_mm` stayed at zero was a measurement of the quantiser. A device whose
    verdict carries a reason has no kernel to require — the counter is not a claim
    there, and the tally says so in the same breath.
    """
    reason = verdict(device)
    if reason or _TALLY["fp4_scaled_mm"]:
        return
    raise LoudError(
        f"the fp4 rung is on and {device if device is not None else _default_device()} "
        "can run the kernel, but no matmul reached it",
        "fp4.fp4_mm is the only path to `_scaled_mm`: a rung that quantises and "
        "dequantises without calling it pays the format's cost and buys none of it",
    )


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


def codes(x: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(q, S_block, S_tensor)`: the 4-bit codes as bytes, the two scales.

    `q` has `x`'s shape and holds one code per value in the low nibble, `0x8` being
    the sign — the arithmetic `unpack` reverses and `_scaled_mm` consumes. Its dtype
    is uint8 because that is the byte the format is stored in; two codes per byte is
    `pack`'s business, not this function's.
    """
    tensor_scale, block_scale, s = block_scales(x, block)
    grid = x.detach().reshape(*x.shape[:-1], -1, block)
    quotient = grid / s.unsqueeze(-1).to(grid.dtype)
    magnitude = quotient.abs()
    index = torch.zeros_like(magnitude)
    for edge, tie_up in _LADDER:
        index = index + (magnitude >= edge if tie_up else magnitude > edge).to(magnitude.dtype)
    code = index.to(torch.uint8) | ((quotient < 0).to(torch.uint8) * _SIGN)
    return code.reshape(x.shape), block_scale, tensor_scale


def unpack(code: torch.Tensor) -> torch.Tensor:
    """Expand one byte per value into the 2-bit-aligned code it holds in each nibble."""
    code = code.view(torch.uint8)
    wide = torch.empty(
        (*code.shape[:-1], code.shape[-1] * 2), dtype=torch.uint8, device=code.device
    )
    wide[..., 0::2] = code & 0xF
    wide[..., 1::2] = code >> 4
    return wide


def decode(code: torch.Tensor) -> torch.Tensor:
    """Turn codes back into the exact values they stand for, scale-free (fp32)."""
    wide = code.view(torch.uint8).to(torch.int64)
    table = torch.tensor(LEVELS, dtype=torch.float32, device=wide.device)
    magnitude = table[wide & _MAG]
    return torch.where((wide & _SIGN) != 0, -magnitude, magnitude)


def pack(code: torch.Tensor) -> torch.Tensor:
    """Two codes per byte: the even element in the low nibble, the odd in the high.

    The order is the format's, not a choice: the first of the pair is the low nibble
    in every MX and NVFP4 description, and `_scaled_mm` reads it that way — the
    swapped packing is the layout bug that produces plausible numbers, so
    `tests/test_fp4.py` runs the kernel against an fp64 reference to pin it.
    """
    if code.shape[-1] % 2:
        raise LoudError(
            f"a width of {code.shape[-1]} codes cannot be packed: two per byte",
            "the contraction axis is a multiple of the fp4 block (16), so this is a "
            "width that never got that far",
        )
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def as_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Reinterpret the packed bytes as `float4_e2m1fn_x2` — the same storage, in place.

    `view()` is a dtype declaration over the existing bytes: no copy kernel, which is
    the entire reason this function exists (`tensor.to(torch.float4_e2m1fn_x2)` raises
    `copy_kernel not implemented`). On this build the dtype's item size is one byte,
    so the view keeps the buffer's shape and every byte stays exactly where `pack` put
    it. `tests/test_fp4.py` pins the storage pointer, so a cast sneaking in shows up.
    """
    if packed.dtype is not torch.uint8:
        raise LoudError(
            f"a float4_e2m1fn_x2 operand is packed from uint8 bytes, not {packed.dtype}",
            "pack the codes first: two per byte, even element in the low nibble",
        )
    if not packed.is_contiguous():
        raise LoudError(
            "the packed codes are not contiguous, so the dtype view would be a copy",
            "call .contiguous() on the quantiser's output, or transpose after `as_fp4`",
        )
    return packed.view(torch.float4_e2m1fn_x2)


def swizzle(scale: torch.Tensor) -> torch.Tensor:
    """Lay block scales out in cuBLAS's 128x4 SWIZZLE_32_4_4 tile, flattened.

    `scale` is `[rows, k-blocks]` — one e4m3 per 16 elements of one row. The kernel
    reads a 128x4 tile of those as a 32x4x4 cube, so the tile's rows come in groups of
    32 and the four k-blocks of a group are contiguous (cuBLAS §3.1.4.3.2, and torch's
    own `to_blocked` in `torch/testing/_internal/common_quantized.py`). Padding to a
    whole tile is part of the layout: a `[832, 52]` scale is passed as 896 x 52
    values, and the extra rows are never read.
    """
    rows, cols = scale.shape
    row_tiles = -(-rows // _TILE_ROWS)
    col_tiles = -(-cols // _TILE_K)
    padded = scale.new_zeros(row_tiles * _TILE_ROWS, col_tiles * _TILE_K)
    padded[:rows, :cols] = scale
    blocks = padded.view(row_tiles, _TILE_ROWS, col_tiles, _TILE_K).permute(0, 2, 1, 3)
    return blocks.reshape(-1, _TILE_K, 32, 4).transpose(1, 2).reshape(-1)


def packed(x: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(operand, scale, S_tensor)` ready for the kernel: 2-D `x`, packed bytes.

    `operand` is one `float4_e2m1fn_x2` tensor of `x`'s shape, `scale` the swizzled
    e4m3 block scales, and `S_tensor` the fp32 tensor scale the caller multiplies the
    result by. Splitting this out of `fp4_mm` is what lets the benchmark weigh the
    quantiser against the matmul it feeds, instead of reporting their sum as one
    number and calling it the format.
    """
    if x.dim() != 2:
        raise LoudError(
            f"the fp4 kernel takes a 2-D operand, not {tuple(x.shape)}",
            "flatten the leading axes before packing: the block scale runs along the "
            "contraction axis, which is the last one",
        )
    code, block_scale, tensor_scale = codes(x, block)
    return as_fp4(pack(code)), swizzle(block_scale), tensor_scale


def fp4_mm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """`a·bᵀ + bias` through `_scaled_mm`, both operands on the NVFP4 grid.

    `b` is `[out, in]` like every weight in this repository, so the kernel gets
    `bᵀ`; both operands are quantised here, which is the cost this function charges
    for. There is no fallback: a device or a shape the kernel will not take raises
    with the reason, because a matmul that quietly becomes the reference fake-quant
    is how this rung came to be measured as a cost with no benefit.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise LoudError(
            f"fp4_mm takes two matrices, got {tuple(a.shape)} and {tuple(b.shape)}",
            "flatten the leading axes; the kernel is a 2-D GEMM",
        )
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        raise LoudError(
            "the fp4 kernel has no backward on this hardware, so it cannot serve autograd",
            "a training forward is fp4.fake_quant (the STE): fp4.linear picks it, and "
            "calling fp4_mm under autograd would drop the gradient silently",
        )
    reason = verdict(a.device)
    if reason:
        raise LoudError(
            f"the fp4 kernel is not available on {a.device}: {reason}",
            "run the rung on a device whose verdict is empty, or use fp4.linear, which "
            "counts the reference path as fp4_fallback instead of claiming the kernel",
        )
    left, scale_a, tensor_a = packed(a)
    right, scale_b, tensor_b = packed(b)
    out = _scaled_mm(left, right, scale_a, scale_b)
    result = out * (tensor_a * tensor_b)
    if bias is not None:
        result = result + bias
    return result.to(a.dtype)


def _scaled_mm(
    left: torch.Tensor,
    right: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    count: bool = True,
) -> torch.Tensor:
    """One `_scaled_mm` call: fp4 operands, swizzled e4m3 scales, bf16 accumulation.

    `count=False` is for `verdict`'s probe, whose whole job is to find out whether the
    kernel runs. A probe that moved `fp4_scaled_mm` would let `require_kernel` pass on
    a run that never asked the model for a single fp4 matmul.
    """
    out: torch.Tensor = torch._scaled_mm(
        left,
        right.transpose(0, 1),
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
    )
    if count:
        _TALLY["fp4_scaled_mm"] += 1
    return out


def _probe(device: torch.device | str) -> str:
    """Attempt one matmul on `device` and return `""` or the reason; never counts.

    The attempt IS the measurement: no list of supported dtypes is trusted, because
    the last one of those disagreed with the wheel. It must not touch the tally, or
    the gate `require_kernel` enforces could be satisfied by the probe itself rather
    than by a matmul the model asked for.
    """
    try:
        left, scale_a, _ = packed(torch.ones(128, 64, device=device))
        right, scale_b, _ = packed(torch.ones(128, 64, device=device))
        _scaled_mm(left, right, scale_a, scale_b, count=False)
    except Exception as error:
        return f"{type(error).__name__}: {error}".split(" (Triggered")[0]
    return ""


def verdict(device: torch.device | str | None = None) -> str:
    """`""` when this build runs the kernel on `device`, else the reason it does not."""
    key = str(device if device is not None else _default_device())
    if key not in _VERDICTS:
        _VERDICTS[key] = _probe(device if device is not None else _default_device())
    return _VERDICTS[key]


def _default_device() -> str:
    """Return the device a caller with no opinion means: the accelerator when there is one."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_note() -> str:
    """One sentence on which path fp4 can take on this build, for the plan to print."""
    reason = verdict()
    if reason:
        return f"fp4 here is the reference fake-quant, not the kernel: {reason}"
    return "fp4 uses `_scaled_mm` forward; the backward is the reference STE"


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activations: bool = True,
) -> torch.Tensor:
    """`x·Wᵀ + b` with the operands on the fp4 grid, by the path this build allows.

    `activations=False` is the weight-only rung: `x` is read as it is, so the kernel
    (which needs both operands at ONE format) is not a candidate. Autograd on is the
    same kind of refusal: there is no fp4 backward on this hardware, so the kernel
    would drop the gradient instead of passing it through, and the STE is the rung's
    training path. Neither is silent — the STE counts `fp4_fakequant`, and a device
    without the kernel counts `fp4_fallback` on every matmul it serves.
    """
    if not activations:
        return functional.linear(x, fake_quant(weight), bias)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return functional.linear(fake_quant(x), fake_quant(weight), bias)
    flat = x.reshape(-1, x.shape[-1])
    if verdict(x.device):
        _TALLY["fp4_fallback"] += 1
        return functional.linear(fake_quant(x), fake_quant(weight), bias)
    return fp4_mm(flat, weight, bias).reshape(*x.shape[:-1], weight.shape[0])
