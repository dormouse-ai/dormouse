"""NVFP4's gates: the level table, the scales, the STE, the wiring, and the kernel.

Each gate here is a bug this project has already paid for, or the one thing that makes
the rung worth enabling:

* the level table is decoded from E2M1's own formula, so a table that returns 2.0 for
  the 6.0 level cannot pass (`test_the_level_table_is_the_format_formula`);
* every midpoint between two codes resolves by the even mantissa bit AND by exactly one
  ulp on either side, so a rounding path that is 2 ulp off at a midpoint fails both
  (`test_every_midpoint_takes_the_even_mantissa`, `test_a_midpoint_never_loses_to_an_ulp`);
* the two-level scale is exact: `S_tensor` is a power of two, so `s` is exact and every
  value exactly on the grid survives quantise→dequantise as `torch.equal`
  (`test_the_two_level_scale_is_exact_at_the_midpoints`);
* the backward is the identity's, bit for bit, and neither the step function's zero
  derivative nor the quantiser's secant slope could pass (`test_the_ste_...`);
* the rung is wired into BOTH weight readers — `forward` and the rank-factored path —
  because a quantizer on `forward` alone leaves every expert in the bank unquantized
  while its counter still moves (`test_a_quantized_weight_is_read_by_the_factored_path`);
* the packed bytes are the format's — two codes per byte, the first in the low nibble —
  and the dtype is a VIEW of them, so a copy kernel sneaking back in shows up as a new
  storage pointer (`test_two_codes_share_a_byte_...`, `test_the_dtype_view_...`);
* the scale tile is cuBLAS's 128x4 SWIZZLE_32_4_4, checked against the offset formula
  from §3.1.4.3.2 rather than against the implementation it is testing
  (`test_the_scale_tile_is_cublas_3_1_4_3_2`);
* the quantiser's error is bounded by e2m1's own half-ulp, per code, not by a tolerance
  somebody picked (`test_the_quantiser_is_within_half_an_ulp_of_the_format`);
* the kernel's result is checked against an fp64 matmul of the very bytes it was handed,
  so a nibble order or a swizzle read backwards fails (`test_the_kernel_matches_fp64_...`);
* and the counter is the gate: an fp4 rung on a device that HAS the kernel must show
  `fp4_scaled_mm > 0`, which is the measurement this rung never had
  (`test_the_gate_refuses_a_rung_that_never_reached_the_kernel`).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from dormouse.model import fp4, quant, tsct
from dormouse.model.assemble import assemble
from dormouse.model.blocks import Block
from dormouse.model.spec import StackSpec

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "n_layers": 4,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 32,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}


@pytest.fixture(autouse=True)
def _clean_tally() -> object:
    """The path tally is per process; every test starts from zero."""
    fp4.reset()
    yield
    fp4.reset()


def _smoke(**rungs: str) -> StackSpec:
    return replace(StackSpec.from_toml(CONFIGS / "dormouse_1b.toml"), **SMOKE, **rungs)


def _formula_levels() -> tuple[float, ...]:
    """E2M1's magnitudes from its exponent and mantissa, not from `fp4.LEVELS`."""
    table = []
    for exponent in range(4):
        for mantissa in range(2):
            table.append(
                mantissa * 0.5 if exponent == 0 else 2.0 ** (exponent - 1) * (1 + mantissa / 2)
            )
    return tuple(table)


# --------------------------------------------------------------- the level table
def test_the_level_table_is_the_format_formula() -> None:
    """16 codes, and the sixth magnitude is 6.0: e2m1's top level, not 8.0 and not 2.0."""
    formula = _formula_levels()
    assert formula == fp4.LEVELS
    assert len(fp4.LEVELS) == 8
    assert max(fp4.LEVELS) == fp4.MAX_CODE == 6.0
    assert fp4.LEVELS[6:] == (4.0, 6.0)


def test_every_code_is_a_fixed_point() -> None:
    codes = torch.tensor(fp4.LEVELS)
    assert torch.equal(fp4.round_e2m1(codes), codes)
    assert torch.equal(fp4.round_e2m1(-codes), -codes)


def test_every_midpoint_takes_the_even_mantissa() -> None:
    """The tie goes to the even mantissa bit, which alternates down the ladder."""
    for index in range(len(fp4.LEVELS) - 1):
        low, high = fp4.LEVELS[index], fp4.LEVELS[index + 1]
        midpoint = (low + high) / 2
        even = low if index % 2 == 0 else high
        assert float(fp4.round_e2m1(torch.tensor(midpoint))) == even, (low, high, "up")
        assert float(fp4.round_e2m1(torch.tensor(-midpoint))) == -even, (low, high, "down")


def test_a_midpoint_never_loses_to_an_ulp() -> None:
    """One ulp each side of every midpoint must cross it — no 2-ulp rounding anywhere."""
    up, down = torch.tensor(float("inf")), torch.tensor(float("-inf"))
    for index in range(len(fp4.LEVELS) - 1):
        low, high = fp4.LEVELS[index], fp4.LEVELS[index + 1]
        midpoint = torch.tensor((low + high) / 2)
        assert float(fp4.round_e2m1(torch.nextafter(midpoint, up))) == high
        assert float(fp4.round_e2m1(torch.nextafter(midpoint, down))) == low


# ------------------------------------------------------------------- the scales
def test_the_two_level_scale_is_exact_at_the_midpoints() -> None:
    """A block whose amax is 48 gets `s = 2^-5 · 256 = 8.0` exactly, and 15 midpoints."""
    block = torch.tensor(
        [
            48.0,
            40.0,
            28.0,
            20.0,
            14.0,
            10.0,
            6.0,
            2.0,
            -2.0,
            -6.0,
            -10.0,
            -14.0,
            0.0,
            -20.0,
            -28.0,
            -40.0,
        ]
    ).reshape(1, 16)
    tensor_scale, block_scale, s = fp4.block_scales(block)
    assert torch.equal(tensor_scale, torch.tensor(2.0**-5))
    assert torch.equal(block_scale.float(), torch.tensor([[256.0]]))
    assert torch.equal(s, torch.tensor([[8.0]]))
    #: 48 is 6·s and survives; every other value is exactly ON a midpoint.
    expected = torch.tensor(
        [
            48.0,
            32.0,
            32.0,
            16.0,
            16.0,
            8.0,
            8.0,
            0.0,
            0.0,
            -8.0,
            -8.0,
            -16.0,
            0.0,
            -16.0,
            -32.0,
            -32.0,
        ]
    ).reshape(1, 16)
    assert torch.equal(fp4.quantize(block), expected)


def test_the_scale_product_is_exact_and_the_tensor_scale_is_a_power_of_two() -> None:
    """`s` must be `S_tensor · S_block` with no rounding: that is what makes `x / s` exact."""
    generator = torch.Generator().manual_seed(7)
    for _ in range(8):
        block = torch.randn(3, 16, generator=generator) * torch.rand(1, generator=generator) * 100
        tensor_scale, block_scale, s = fp4.block_scales(block)
        assert torch.equal(tensor_scale, torch.exp2(torch.log2(tensor_scale)))
        assert torch.equal(s, tensor_scale * block_scale.float())


def test_a_value_on_the_grid_survives_round_trip_bit_for_bit() -> None:
    """Quantise→dequantise of a representable value is the identity, not `allclose`."""
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(5, 64, generator=generator) * 40
    once = fp4.quantize(x)
    assert torch.equal(fp4.quantize(once), once)
    assert torch.equal(fp4.fake_quant(once), once)
    #: And every dequantized value is a code times its block's scale, exactly.
    _tensor_scale, _block_scale, s = fp4.block_scales(x)
    assert torch.equal(
        once, fp4.round_e2m1(once / fp4._spread(s, once, 16)) * fp4._spread(s, once, 16)
    )


def test_the_format_keeps_the_dtype_it_was_handed() -> None:
    """A bf16 activation must come back bf16: `s` is exact in 8 mantissa bits too."""
    generator = torch.Generator().manual_seed(19)
    for dtype in (torch.float32, torch.bfloat16):
        x = (torch.randn(4, 32, generator=generator) * 10).to(dtype)
        quantized = fp4.quantize(x)
        assert quantized.dtype == dtype
        assert torch.equal(fp4.quantize(quantized), quantized)


def test_a_width_the_block_cannot_tile_is_refused_with_the_remainder() -> None:
    with pytest.raises(LoudError) as caught:
        fp4.quantize(torch.zeros(2, 24))
    assert "24" in str(caught.value) and "16" in str(caught.value)


# -------------------------------------------------------------------- the STE
def test_the_ste_gradient_is_the_identity_bit_for_bit() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(8, 64, generator=generator) * 3
    identity = x.clone().requires_grad_()
    quantized = x.clone().requires_grad_()
    (identity).sum().backward()
    fp4.fake_quant(quantized).sum().backward()
    assert torch.equal(identity.grad, quantized.grad)
    assert torch.equal(quantized.grad, torch.ones_like(quantized))


def test_neither_a_step_derivative_nor_a_secant_slope_would_pass_the_ste_gate() -> None:
    """The three candidate slopes are distinct, so the gate above can only pass one."""
    x = torch.linspace(0.5, 4.0, 64).reshape(4, 16)
    quantized = x.clone().requires_grad_()
    fp4.fake_quant(quantized).sum().backward()
    assert torch.equal(quantized.grad, torch.ones_like(quantized))
    assert not torch.equal(quantized.grad, torch.zeros_like(quantized))
    secant = fp4.quantize(x) / x
    assert not torch.allclose(quantized.grad, secant), "the STE must not be the secant"


def test_a_quantized_linear_passes_the_master_gradient_through() -> None:
    """`sum(out)`'s upstream is ones, so the master's gradient is the identity's."""
    torch.manual_seed(5)
    plain = nn.Linear(64, 32, bias=False)
    rung = nn.Linear(64, 32, bias=False)
    rung.load_state_dict(plain.state_dict())
    quant.quantize_weights(rung, "fp4")
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(4, 64, generator=generator)
    x_plain, x_rung = x.clone().requires_grad_(), x.clone().requires_grad_()
    y_plain, y_rung = plain(x_plain), rung(x_rung)
    assert not torch.equal(y_plain, y_rung), "the rung has to have changed the arithmetic"
    y_plain.sum().backward()
    y_rung.sum().backward()
    assert type(rung) is quant.QuantLinear
    assert torch.equal(plain.weight.grad, rung.weight.grad)


# ------------------------------------------------------------------ the wiring
def test_a_quantized_weight_is_read_by_the_factored_path_too() -> None:
    """`low_rank` never calls `forward`: the scheme has to live on the weight read."""
    torch.manual_seed(13)
    rung = nn.Linear(32, 32, bias=False)
    quant.quantize_weights(rung, "fp4")
    twin = nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        twin.weight.copy_(fp4.fake_quant(rung.weight))
    x = torch.randn(4, 32)
    assert torch.equal(tsct.low_rank(rung, x, 8), tsct.low_rank(twin, x, 8))
    plain = nn.Linear(32, 32, bias=False)
    plain.load_state_dict(rung.state_dict())
    assert not torch.equal(tsct.low_rank(rung, x, 8), tsct.low_rank(plain, x, 8))


def test_the_counters_prove_both_fp4_rungs_ran() -> None:
    spec = _smoke(weight_qat="fp4", act_quant="fp4")
    assembled = assemble(spec, smoke=True)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len), generator=torch.Generator())
    counters = Counters()
    assembled.model(ids, counters)
    counters.check()
    assert counters.counts["weight_fp4"] > 0
    assert counters.counts["act_fp4"] == spec.batch * spec.seq_len * spec.n_layers
    assert counters.counts["weight_ternary"] == 0


def test_the_rung_off_runs_no_fp4_code_at_all() -> None:
    spec = _smoke(weight_qat="none", act_quant="none")
    assembled = assemble(spec, smoke=True)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len))
    fp4.reset()
    before = dict(fp4._TALLY)
    assembled.model(ids)
    assert dict(fp4._TALLY) == before
    assert type(assembled.model.blocks[0]) is Block
    assert not any(isinstance(module, quant.QuantLinear) for module in assembled.model.modules())
    assert assembled.quant_report() == {}


def test_a_weight_on_the_grid_makes_the_rung_a_no_op_with_torch_equal() -> None:
    """The degeneracy gate: a weight the format can represent exactly must not move."""
    torch.manual_seed(17)
    plain = nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        plain.weight.copy_(fp4.quantize(plain.weight))
    rung = nn.Linear(32, 32, bias=False)
    rung.load_state_dict(plain.state_dict())
    quant.quantize_weights(rung, "fp4")
    x = torch.randn(3, 32)
    assert torch.equal(plain(x), rung(x))


# ----------------------------------------------------------------------- the bytes
def test_two_codes_share_a_byte_and_the_first_is_the_low_nibble() -> None:
    """The format's packing, pinned on one byte: 1 then 2 is `0x21`, not `0x12`."""
    code = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.uint8)
    packed = fp4.pack(code)
    assert packed.tolist() == [[0x21, 0x43, 0x65, 0x07]]
    assert torch.equal(fp4.unpack(packed), code)
    generator = torch.Generator().manual_seed(23)
    wide = torch.randint(0, 16, (7, 64), generator=generator, dtype=torch.uint8)
    assert torch.equal(fp4.unpack(fp4.pack(wide)), wide)


def test_the_dtype_view_is_the_same_storage_not_a_copy() -> None:
    """`view` is the trick: a cast would need the copy kernel this wheel lacks."""
    generator = torch.Generator().manual_seed(29)
    code, _, _ = fp4.codes(torch.randn(4, 64, generator=generator) * 5)
    packed = fp4.pack(code)
    operand = fp4.as_fp4(packed)
    assert operand.dtype is torch.float4_e2m1fn_x2
    assert operand.data_ptr() == packed.data_ptr()
    #: On this build the dtype's item size is one byte, so the view keeps the buffer's
    #: shape: the kernel reads `shape[-1]/2` bytes and doubles them back into K.
    assert operand.element_size() == 1
    assert operand.numel() == packed.numel()
    #: And the bytes the kernel will read are the bytes the quantiser wrote.
    assert torch.equal(fp4.unpack(operand.view(torch.uint8)), code)


def test_the_dtype_view_refuses_bytes_it_cannot_read() -> None:
    """A wrong dtype or a non-contiguous buffer would silently become a cast."""
    with pytest.raises(LoudError) as wrong_dtype:
        fp4.as_fp4(torch.zeros(4, 8))
    assert "uint8" in str(wrong_dtype.value)
    with pytest.raises(LoudError) as strided:
        fp4.as_fp4(torch.zeros(4, 8, dtype=torch.uint8).t())
    assert "contiguous" in strided.value.args[0]


def test_the_scale_tile_is_cublas_3_1_4_3_2() -> None:
    """The swizzle, against the offset formula from cuBLAS — an independent derivation."""

    def cublas_offset(rows: int, cols: int) -> list[tuple[int, int, int]]:
        col_tiles = -(-cols // 4)
        placed = []
        for row in range(rows):
            for column in range(cols):
                tile = 512 * ((row // 128) * col_tiles + column // 4)
                outer, inner = row % 128, column % 4
                placed.append((tile + (outer % 32) * 16 + (outer // 32) * 4 + inner, row, column))
        return placed

    for rows, cols in ((128, 4), (129, 5), (300, 52)):
        scale = torch.randn(rows, cols).to(torch.float8_e4m3fn)
        expected = torch.zeros((-(-rows // 128)) * 128 * (-(-cols // 4)) * 4)
        expected = expected.to(torch.float8_e4m3fn)
        for offset, row, column in cublas_offset(rows, cols):
            expected[offset] = scale[row, column]
        assert torch.equal(fp4.swizzle(scale), expected), (rows, cols)


def test_the_scale_tile_is_padded_to_whole_tiles() -> None:
    """A `[832, 52]` scale is 896 x 52 values: 832 is 6.5 tiles of 128 rows."""
    scale = torch.zeros(832, 52).to(torch.float8_e4m3fn)
    assert fp4.swizzle(scale).numel() == 896 * 52
    assert fp4.swizzle(scale).is_contiguous()


# ------------------------------------------------------------------- the grid's error
def _half_ulp(code: torch.Tensor) -> torch.Tensor:
    """Half the distance to the further neighbour of each code, in scale units."""
    levels = torch.tensor(fp4.LEVELS, dtype=torch.float32)
    index = code.long() & 0x7
    here = levels[index]
    below = torch.where(index > 0, levels[(index - 1).clamp_min(0)], torch.zeros_like(here))
    #: The top code has no neighbour above it: the grid clamps at 6, so its step is 6 - 4.
    above = torch.where(index < 7, levels[(index + 1).clamp_max(7)] - here, here - below)
    return torch.maximum(here - below, above) / 2


def test_the_quantiser_is_within_half_an_ulp_of_the_format() -> None:
    """Every value is within half a step of the code it took — e2m1's own grid, per code.

    Two bounds, both derived from the format and neither one a tolerance:

    * absolute: `|x̂ - x| ≤ ½·Δ(code)·s`, where `Δ(code)` is the distance to the code's
      FURTHER neighbour. This is the one that catches a quantiser one code off, or one
      whose divisor `s` was built from a rounded reciprocal;
    * relative to the code's own value: ≤ 25% inside the normal range (`½ ulp` of a
      1-mantissa format, reached at the bottom of every binade) and ≤ 50% at 0.5, whose
      neighbour below is zero. Below ¼·s the format's error is absolute, not relative:
      the code is zero and `x̂` is `s`-sized, so it is bounded by the absolute rule only.
    """
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(4096, 832, generator=generator) * torch.rand(4096, 1, generator=generator)
    code, block_scale, tensor_scale = fp4.codes(x)
    scale = torch.repeat_interleave(tensor_scale * block_scale.float(), 16, dim=-1)
    error = (fp4.quantize(x) - x).abs()
    assert (error <= _half_ulp(code) * scale * (1 + 1e-6)).all()

    value = fp4.decode(code) * scale
    relative = error / value.abs().clamp_min(1e-30)
    nonzero = value != 0
    assert relative[nonzero].max().item() <= 0.5 * (1 + 1e-6)
    normal = value.abs() >= scale
    assert relative[normal].max().item() <= 0.25 * (1 + 1e-6)
    print(
        f"max relative quantisation error {relative[nonzero].max().item():.4f} "
        f"(half-ulp ceilings: 0.5 at the smallest code, 0.25 in the normal range)"
    )


# -------------------------------------------------------------------- the kernel
def test_fp4_mm_refuses_a_device_without_the_kernel() -> None:
    """No silent substitution of the reference: the primitive raises the reason."""
    x = torch.randn(4, 64)
    with pytest.raises(LoudError) as caught:
        fp4.fp4_mm(x, torch.randn(8, 64))
    assert "cpu" in str(caught.value)
    assert fp4.tally(torch.device("cpu"))["fp4_scaled_mm"] == 0


def test_fp4_mm_refuses_to_drop_a_gradient() -> None:
    """A kernel with no backward would silently be an STE with the wrong slope."""
    x = torch.randn(4, 64, requires_grad=True)
    with pytest.raises(LoudError) as caught:
        fp4.fp4_mm(x, torch.randn(8, 64))
    assert "backward" in str(caught.value)


def test_the_probe_does_not_count_as_a_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verdict's attempt must not satisfy the gate it exists to make honest."""
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    assert fp4.verdict(torch.device("cpu")) == ""
    assert fp4.tally(torch.device("cpu"))["fp4_scaled_mm"] == 0


def test_the_gate_refuses_a_rung_that_never_reached_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rung on, the kernel available, zero calls: that run must fail, loudly.

    This is the gate the previous measurement did not have — it passed its own tests
    with `fp4_scaled_mm == 0` on every step and reported the quantiser's cost as FP4's.
    """
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    with pytest.raises(LoudError) as caught:
        fp4.require_kernel(torch.device("cpu"))
    assert "never" in str(caught.value) or "no matmul" in str(caught.value)
    fp4._TALLY["fp4_scaled_mm"] = 1
    fp4.require_kernel(torch.device("cpu"))


def test_the_kernel_is_not_used_while_autograd_is_on() -> None:
    """No fp4 backward exists, so a training forward must never reach `_scaled_mm`."""
    torch.manual_seed(37)
    rung = nn.Linear(64, 8, bias=False)
    quant.quantize_weights(rung, "fp4", activations_fp4=True)
    x = torch.randn(4, 64, requires_grad=True)
    out = rung(x)
    assert out.requires_grad
    assert fp4.tally(torch.device("cpu"))["fp4_scaled_mm"] == 0
    assert fp4.tally(torch.device("cpu"))["fp4_fakequant"] > 0


def test_a_device_without_the_kernel_counts_the_fallback() -> None:
    """The reference path is not silent: every matmul it serves is counted."""
    torch.manual_seed(41)
    rung = nn.Linear(64, 8, bias=False)
    quant.quantize_weights(rung, "fp4", activations_fp4=True)
    with torch.no_grad():
        rung(torch.randn(4, 64))
    report = fp4.tally(torch.device("cpu"))
    assert report["fp4_fallback"] == 1
    assert report["fp4_scaled_mm"] == 0
    assert isinstance(report["verdict"], str) and report["verdict"], "a reason, here"


def test_a_weight_only_rung_never_takes_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kernel needs both operands at one format; activations are not quantized here."""
    calls: list[int] = []
    monkeypatch.setattr(fp4, "_scaled_mm", lambda *a, **k: calls.append(1))
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    with torch.no_grad():
        fp4.linear(torch.randn(4, 64), torch.randn(8, 64), activations=False)
    assert not calls


def test_a_run_that_claims_fp4_and_never_touches_the_kernel_says_so() -> None:
    """Training cannot use the kernel at all, and the report carries both numbers."""
    spec = _smoke(weight_qat="fp4", act_quant="fp4")
    assembled = assemble(spec, smoke=True)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len))
    assembled.model(ids)
    report = assembled.quant_report()
    assert report["fp4_fakequant"] > 0
    assert report["fp4_scaled_mm"] == 0, "no fp4 backward exists: a step cannot use it"
    assert isinstance(report["verdict"], str)  # a reason, or the machine can run it
    assert "fp4" in fp4.build_note()


# ----------------------------------------------------------------- the card, marked
@pytest.mark.gpu
def test_the_kernel_matches_fp64_on_the_bytes_it_was_handed() -> None:
    """The kernel's result against an fp64 matmul of the decoded codes, not of `x`."""
    for rows, out_features, width in ((256, 256, 256), (512, 384, 832), (248, 512, 832)):
        a = torch.randn(rows, width, device="cuda")
        w = torch.randn(out_features, width, device="cuda")
        with torch.no_grad():
            got = fp4.fp4_mm(a, w)
            code_a, scale_a, tensor_a = fp4.codes(a)
            code_w, scale_w, tensor_w = fp4.codes(w)
            exact_a = _exact(code_a, scale_a, tensor_a, rows, width)
            exact_w = _exact(code_w, scale_w, tensor_w, out_features, width)
            reference = exact_a @ exact_w.T
        #: The output is bf16, so one ulp of the largest |out| is the bound: 2^-8 of it.
        bound = 2.0**-8 * got.abs().max().item()
        assert (got.double() - reference).abs().max().item() <= bound, (rows, out_features, width)

    assert fp4.tally("cuda")["fp4_scaled_mm"] == 3


@pytest.mark.gpu
def test_the_fp4_rung_reaches_the_kernel_on_the_card() -> None:
    """The gate, live: the rung on and a forward must move `fp4_scaled_mm`."""
    torch.manual_seed(43)
    rung = nn.Linear(64, 32, bias=True).cuda()
    quant.quantize_weights(rung, "fp4", activations_fp4=True)
    with torch.no_grad():
        out = rung(torch.randn(4, 64, device="cuda"))
    assert out.shape == (4, 32)
    fp4.require_kernel(torch.device("cuda"))
    report = fp4.tally(torch.device("cuda"))
    assert report["fp4_scaled_mm"] == 1
    assert report["fp4_fallback"] == 0


def _exact(
    code: torch.Tensor, block_scale: torch.Tensor, tensor_scale: torch.Tensor, rows: int, width: int
) -> torch.Tensor:
    """The quantised operand in fp64: decoded codes times their own two-level scale."""
    scale = (tensor_scale.double() * block_scale.double()).unsqueeze(-1)
    values = fp4.decode(code).double().reshape(rows, -1, fp4.BLOCK) * scale
    return values.reshape(rows, width)
