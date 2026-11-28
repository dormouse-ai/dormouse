"""A4 quality gates: seeded stochastic rounding, scale calibration, anchor, flow, refusal.

The 3x200-step paired measurement itself is `tools/a4_quality_gate.py` (a card run);
these are the runnable checks that fail if a mechanism stops doing what it claims.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model import quant
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import _skip

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
    "torch_compile": False,
}


@pytest.fixture(autouse=True)
def _native_tern_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native arm is explicit: every spec in this file builds it with its flag on."""
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")


def _spec(*, a4: bool) -> StackSpec:
    """Build the smoke spec from the shipped text, overriding only the quant slots."""
    raw = tomllib.loads((CONFIGS / "dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {f.name for f in fields(StackSpec) if f.init}
    values = {key: value for key, value in raw.items() if key in known}
    values.update(SMOKE)
    values["act_quant"] = "a4" if a4 else "none"
    values["weight_qat"] = "ternary_native" if a4 else "none"
    return StackSpec.from_mapping(values)


def _np_fwht(x: np.ndarray, block: int = quant.FWHT_BLOCK) -> np.ndarray:
    y = x.reshape(*x.shape[:-1], x.shape[-1] // block, block).copy()
    step = 1
    while step < block:
        y = y.reshape(*y.shape[:-2], x.shape[-1] // block, block // (2 * step), 2, step)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y = np.stack((a + b, a - b), axis=-2).reshape(*x.shape[:-1], x.shape[-1] // block, block)
        step *= 2
    return y.reshape(x.shape) / math.sqrt(block)


def test_nearest_round_matches_external_fp64_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unchanged arm is still the bit-exact numpy formula plus an STE gradient."""
    monkeypatch.setenv(quant.ROUND_FLAG, "near")
    monkeypatch.setenv(quant.SCALE_FLAG, "absmax")
    quant.set_step_seed(7, 0)
    x = np.random.default_rng(7).standard_normal((3, 128))
    tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    z = _np_fwht(x)
    base = np.maximum(np.abs(z).max(axis=-1, keepdims=True), 1e-8) / 7
    expected = _np_fwht(np.clip(np.round(z / base), -7, 7) * base)
    actual = quant.a4(tensor)
    assert np.allclose(actual.detach().numpy(), expected, atol=2e-5, rtol=1e-5)
    upstream = torch.tensor(np.random.default_rng(9).standard_normal((3, 128)), dtype=torch.float32)
    actual.backward(upstream)
    assert torch.allclose(tensor.grad, upstream, atol=1e-6)


def test_sr_is_deterministic_per_step_and_unbiased_on_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One `(run_seed, step)` reproduces the same codes; many seeds have the right mean."""
    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    monkeypatch.setenv(quant.SCALE_FLAG, "absmax")
    torch.manual_seed(3)
    x = torch.randn(4, 128)
    quant.take_round_journal()  # a previous test's events are not this stream
    quant.set_step_seed(11, 5)
    first = quant.a4(x)
    journal = quant.take_round_journal()
    quant.set_step_seed(11, 5)
    assert torch.equal(first, quant.a4(x)), "same seed and step must replay the same rounds"
    quant.set_step_seed(11, 6)
    assert not torch.equal(first, quant.a4(x)), "a different step must advance the stream"
    assert all(e["run_seed"] == 11 and e["mode"] == "sr" for e in journal)
    assert sum(e["rounded_up"] for e in journal) > 0

    z = _np_fwht(x.numpy())
    base = np.maximum(np.abs(z).max(axis=-1, keepdims=True), 1e-8) / 7
    target = np.clip(z / base, -7, 7)
    sums = np.zeros_like(target)
    draws = 200
    for step in range(draws):
        quant.set_step_seed(91, step)
        packed_journal = quant.take_round_journal()
        del packed_journal
        out = quant.a4(x).numpy()
        sums += _np_fwht(out) / base
    assert np.abs(sums / draws - target).max() < 0.20, "SR must be unbiased on the int4 grid"


def test_per_channel_scale_gets_a_real_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """The learned multiplier is a parameter, not a detached absmax statistic."""
    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    monkeypatch.setenv(quant.SCALE_FLAG, "pc")
    quant.set_step_seed(5, 0)
    x = torch.randn(3, 128, requires_grad=True)
    scale = torch.ones(128, requires_grad=True)
    y = quant.a4(x, channel_scale=scale)
    y.sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    assert scale.grad is not None and bool(torch.isfinite(scale.grad).all())
    assert float(scale.grad.abs().sum()) > 0.0
    assert quant.take_round_journal()


def test_legacy_paths_and_fp32_saved_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old fake-quant helpers and the codesave-off control still run and count."""
    from torch import nn

    weight = torch.randn(8, 8)
    assert quant.ternary(weight).shape == weight.shape
    assert quant.int4(weight).shape == weight.shape
    linear = nn.Linear(8, 8)
    assert quant.QuantLinear.forward(linear, weight).shape == weight.shape
    model = nn.Sequential(nn.Linear(8, 8))
    assert quant.ternarize(model) == 1
    native_model = nn.Sequential(nn.Linear(8, 8))
    assert quant.native_ternarize(native_model) == 1
    native = native_model[0]
    assert isinstance(native, quant.NativeQuantLinear)
    with torch.no_grad():
        native.scale.fill_(1.0)
    assert quant.NativeQuantLinear.forward(native, weight).shape == weight.shape
    counters = Counters()
    wrapped = quant.QuantBlock(lambda x, _c: x, matrices=2, act_int4=True)
    wrapped(weight, counters)
    assert counters.counts["act_int4"] > 0 and counters.counts["ternary_fwd"] == 2

    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    monkeypatch.setenv(quant.SCALE_FLAG, "pc")
    monkeypatch.setenv(quant.CODESAVE_FLAG, "0")
    quant.set_step_seed(3, 0)
    x = torch.randn(2, 128, requires_grad=True)
    scale = torch.ones(128, requires_grad=True)
    quant.a4(x, channel_scale=scale).sum().backward()
    assert scale.grad is not None and float(scale.grad.abs().sum()) > 0
    assert not quant.codesave_on()


def test_model_flag_refuses_the_bf16_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DORMOUSE_ACT_A4", "1")
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "0")
    with pytest.raises(LoudError) as caught:
        StackSpec(act_quant="none")
    assert "DORMOUSE_ACT_A4=1" in str(caught.value)
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")
    with pytest.raises(LoudError):
        StackSpec(act_quant="a4", weight_qat="none")
    assert _spec(a4=True).act_quant == "a4"


def test_two_step_smoke_moves_every_flow_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A4 fwd/bwd/codes, the scale gradient and the ternary STE all report."""
    from dormouse_kernels.engine import fused, sparse

    monkeypatch.setattr(sparse, "_ON", False)  # CPU arm under a CUDA build
    monkeypatch.setattr(fused, "_FUSED", False)  # read at import; env alone is too late
    monkeypatch.setenv("DORMOUSE_ACT_A4", "1")
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")
    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    monkeypatch.setenv(quant.SCALE_FLAG, "pc")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    spec = _spec(a4=True)
    built = assemble(spec, smoke=True)
    optimizer = MixOptimizer(built.model, steps=2)
    ids = torch.randint(
        0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=torch.Generator().manual_seed(7)
    )
    for step in range(2):
        quant.set_step_seed(7, step)
        result, counters = built.step(ids, optimizer=optimizer)
        assert not result.skipped and math.isfinite(result.loss)
    for name in (
        "a4_fwd",
        "a4_bwd",
        "a4_codes_saved",
        "a4_sr_rounds",
        "a4_scale_grad",
        "tsct_ternary_fwd",
        "tsct_ternary_bwd",
    ):
        assert counters.counts[name] > 0, name
    assert counters.silent() == []


def test_nan_firewall_is_a_named_skip_not_a_number() -> None:
    """A non-finite step is refused and counted; no loss is printed as a result."""
    counters = Counters()
    result = _skip(counters, float("nan"), "injected", bad_loss=True, stage="logits")
    assert result.skipped and math.isnan(result.loss)
    assert result.bad_stage == "logits"
    assert counters.counts["nan_guard"] > 0 and counters.counts["nan_loss"] > 0


def test_twelve_step_cpu_determinism_from_step_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bank's `index_add_` is thread-order dependent: pin one thread for this gate."""
    from dormouse_kernels.engine import fused, sparse

    monkeypatch.setattr(sparse, "_ON", False)  # CPU arm under a CUDA build
    monkeypatch.setattr(fused, "_FUSED", False)  # read at import; env alone is too late
    monkeypatch.setenv("DORMOUSE_ACT_A4", "1")
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")
    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    monkeypatch.setenv(quant.SCALE_FLAG, "pc")

    def run() -> list[float]:
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(13)
            built = assemble(_spec(a4=True), smoke=True)
            optimizer = MixOptimizer(built.model, steps=12)
            ids = torch.randint(0, 32_000, (2, 129), generator=torch.Generator().manual_seed(13))
            losses: list[float] = []
            for step in range(12):
                quant.set_step_seed(13, step)
                result, _ = built.step(ids, optimizer=optimizer)
                losses.append(result.loss)
            return losses
        finally:
            torch.set_num_threads(previous)

    assert run() == run()
