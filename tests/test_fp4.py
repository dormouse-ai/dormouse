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
* a run that claims fp4 and never reaches `_scaled_mm` says so (`test_..._says_so`).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from torch import nn
from torch.nn import functional

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
    from dormouse_core.errors import LoudError

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


# -------------------------------------------------------------------- the kernel
def test_the_kernel_is_not_used_while_autograd_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fp4 backward exists, so a training forward must never reach `_scaled_mm`."""
    calls: list[int] = []

    def stub(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        calls.append(1)
        return functional.linear(fp4.quantize(x), fp4.quantize(w), b)

    monkeypatch.setattr(fp4, "_scaled_mm", stub)
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    monkeypatch.setattr(fp4, "_CHECKED", set())
    cpu = torch.device("cpu")
    x = torch.randn(4, 64, requires_grad=True)
    w = torch.randn(8, 64)
    out = fp4.linear(x, w)
    assert not calls, "the kernel has no backward: it must not run under autograd"
    assert fp4.tally(cpu)["fp4_scaled_mm"] == 0
    assert out.requires_grad
    with torch.no_grad():
        fp4.reset()
        fp4.linear(x, w)
    assert len(calls) == 1
    assert fp4.tally(cpu)["fp4_scaled_mm"] == 1


def test_a_kernel_that_disagrees_with_the_reference_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layout accepted but read in the wrong order returns numbers: they are checked."""
    monkeypatch.setattr(
        fp4, "_scaled_mm", lambda x, w, b: torch.full((x.shape[0], w.shape[0]), 9.0)
    )
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    monkeypatch.setattr(fp4, "_CHECKED", set())
    x = torch.randn(4, 64)
    w = torch.randn(8, 64)
    with torch.no_grad():
        out = fp4.linear(x, w)
    assert torch.equal(out, fp4._reference(x, w, None, activations=True))
    assert fp4.tally(torch.device("cpu"))["fp4_scaled_mm"] == 0
    assert "disagreed" in fp4.verdict(torch.device("cpu"))


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


def test_a_weight_only_rung_never_takes_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kernel needs both operands at one format; activations are not quantized here."""
    calls: list[int] = []
    monkeypatch.setattr(fp4, "_scaled_mm", lambda x, w, b: calls.append(1))
    monkeypatch.setitem(fp4._VERDICTS, str(torch.device("cpu")), "")
    with torch.no_grad():
        fp4.linear(torch.randn(4, 64), torch.randn(8, 64), activations=False)
    assert not calls
