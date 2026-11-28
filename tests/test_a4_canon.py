"""The canonical A4 path: one config, two sites, one kernel home, eight receipts.

`act_quant=\"a4\"` is the whole mechanism: A4 on the block norms, INT4 codes on the
bank's wide slabs, and native ternary weights. A runtime flag may select the fused
library kernel, but it cannot turn half the mechanism on. These checks pin that
contract without a card; the card measurements live in `tools/a4_bank_gate.py`,
`tools/a4_fused_ab.py` and `tools/a4_integrate_gate.py`.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model import quant
from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.registry import QUANT
from dormouse.model.spec import ACT_QUANT, StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.runconfig import RunRecipe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train_200m_a4_2b.toml"
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
RECEIPTS = (
    "a4_fwd",
    "a4_bwd",
    "a4_codes_saved",
    "bank_a4_fwd",
    "bank_a4_bwd",
    "bank_a4_codes_saved",
    "tsct_ternary_fwd",
    "tsct_ternary_bwd",
)


@pytest.fixture(autouse=True)
def _native_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native arm is flag-explicit; every spec here is built under its flag."""
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")
    monkeypatch.delenv("DORMOUSE_ACT_A4", raising=False)
    monkeypatch.delenv("DORMOUSE_A4_FUSED", raising=False)
    monkeypatch.delenv(quant.BANK_FLAG, raising=False)
    monkeypatch.delenv(quant.BANK_QUANT_FLAG, raising=False)


def _spec() -> StackSpec:
    """The shipped A4 recipe at the smallest widths the mechanism accepts."""
    raw = tomllib.loads((ROOT / "configs" / "dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {field.name for field in fields(StackSpec) if field.init}
    values = {key: value for key, value in raw.items() if key in known}
    values.update(SMOKE, act_quant="a4", weight_qat="ternary_native")
    return StackSpec.from_mapping(values)


def test_config_is_runnable_and_its_plan_prices_both_quant_slots() -> None:
    """The shipped config validates, has no warm-up and can be priced without a card."""
    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    known = {field.name for field in fields(StackSpec) if field.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in known})
    assert (spec.act_quant, spec.weight_qat) == ("a4", "ternary_native")
    assert raw["warmup"] == 0
    recipe = RunRecipe.read(CONFIG)
    assert recipe.warmup == 0
    lines = plan_lines(spec)
    assert any("weight_qat=ternary_native" in line and "act_quant=a4" in line for line in lines)
    assert any(line.startswith("step:") for line in lines)


def test_a4_is_a_registry_literal_with_a_named_counter() -> None:
    """A config value that never reached the registry would price no mechanism."""
    assert "a4" in ACT_QUANT
    assert QUANT["a4"].counter == "a4_fwd"


def test_config_turns_on_both_bank_sites_without_an_env_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`act_quant=a4` is the switch; the bank env flags stay off."""
    _spec()
    assert quant.a4_config_on() and quant.bank_quant_on() and quant.bank_codesave_on()
    assert not quant.a4_on()  # the config assertion flag is deliberately unset
    with monkeypatch.context() as m:
        m.delenv("DORMOUSE_TSCT_NATIVE_TERN", raising=False)
        StackSpec.from_mapping({**SMOKE, "act_quant": "none", "weight_qat": "none"})
    assert not quant.a4_config_on() and not quant.bank_quant_on()
    assert not quant.bank_codesave_on()


def test_half_quant_pairings_are_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """4-bit activations against wider weights, or two schemes, are not the model."""
    with monkeypatch.context() as m:
        m.delenv("DORMOUSE_TSCT_NATIVE_TERN", raising=False)
        with pytest.raises(LoudError, match="half-quantised"):
            StackSpec.from_mapping({**SMOKE, "act_quant": "a4", "weight_qat": "ternary"})
        with pytest.raises(LoudError, match="half-quantised"):
            StackSpec.from_mapping({**SMOKE, "act_quant": "a4", "weight_qat": "none"})
    with pytest.raises(LoudError, match="half-quantised"):
        StackSpec.from_mapping({**SMOKE, "act_quant": "int4", "weight_qat": "ternary_native"})
    monkeypatch.setenv("DORMOUSE_ACT_A4", "1")
    with monkeypatch.context() as m:
        m.delenv("DORMOUSE_TSCT_NATIVE_TERN", raising=False)
        with pytest.raises(LoudError, match="DORMOUSE_ACT_A4=1"):
            StackSpec.from_mapping({**SMOKE, "act_quant": "none", "weight_qat": "none"})


def test_kernel_lives_in_the_library_not_the_app() -> None:
    """One home for the fused kernel: app code imports, it does not define."""
    from dormouse_kernels.engine import a4_fused

    assert not (ROOT / "src" / "dormouse" / "kernels" / "a4_fused.py").exists()
    assert Path(a4_fused.__file__).parent.name == "engine"
    assert list(a4_fused.A4FusedSave.forward.__code__.co_varnames[:3]) == [
        "ctx",
        "x",
        "counters",
    ]


def _run(steps: int) -> tuple[list[float], dict[str, int]]:
    """Pinned-thread CPU replay of the whole canonical step."""
    from dormouse_kernels.engine import fused, sparse

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch_no_cuda = (sparse, fused)
        for module in monkeypatch_no_cuda:
            for name, value in (("_ON", False), ("_FUSED", False)):
                if hasattr(module, name):
                    setattr(module, name, value)
        torch.manual_seed(7)
        built = assemble(_spec(), smoke=True)
        optimizer = MixOptimizer(built.model, steps=steps)
        ids = torch.randint(
            0,
            32_000,
            (SMOKE["batch"], SMOKE["seq_len"] + 1),
            generator=torch.Generator().manual_seed(7),
        )
        losses: list[float] = []
        for _ in range(steps):
            result, counters = built.step(ids, optimizer=optimizer)
            assert not result.skipped and math.isfinite(result.loss)
            losses.append(result.loss)
        for module in monkeypatch_no_cuda:
            for name, value in (("_ON", True), ("_FUSED", True)):
                if hasattr(module, name):
                    setattr(module, name, value)
        return losses, dict(counters.snapshot())
    finally:
        torch.set_num_threads(previous)


def test_eight_receipts_move_and_the_step_is_bitwise_replayable() -> None:
    """Both A4 sites and both native ternary directions report; 20 steps replay."""
    losses, counts = _run(steps=20)
    for name in RECEIPTS:
        assert counts.get(name, 0) > 0, name
    replay, replay_counts = _run(steps=20)
    assert losses == replay
    assert {name: counts[name] for name in RECEIPTS} == {
        name: replay_counts[name] for name in RECEIPTS
    }
