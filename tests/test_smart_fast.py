"""The smart-fast config must parse, fit the FLOP budget, and carry only live levers.

No REFUTED lever may be on, every enabled lever names its gate, and the IQ smoke
suite is the sole quality gate: random=0, oracle=1, with the blind-judge counter
proving the gate actually ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dormouse_core.counters import Counters
from dormouse_core.errors import MissingReport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import smart_fast_bench as bench  # noqa: E402

CONFIG = ROOT / "configs" / "smart_fast.toml"


def test_the_config_parses_as_a_stack() -> None:
    """The price is read straight from spec.py, the same parser `dorm plan` uses."""
    spec, _text = bench.load(CONFIG)
    assert (spec.batch, spec.seq_len) == (2, 1024)
    assert spec.attn == "hybrid" and spec.n_layers % 4 == 0
    assert (spec.n_experts, spec.topk, spec.n_shared, spec.expert_tsct_rank) == (64, 2, 1, 256)
    assert spec.torch_compile is True
    assert spec.objective == "ntp"
    assert spec.optimizer == "muon"
    assert spec.act_quant == "none" and spec.weight_qat == "none"
    assert spec.tokens == 108_000_000


def test_flop_bill_fits_the_two_hour_budget() -> None:
    """6*P_stored*D <= 92 TFLOP/s x 39.92% x 2h, priced from the built tree."""
    spec, _text = bench.load(CONFIG)
    bill = bench.price(spec)
    assert 390e6 <= bill["stored"] <= 410e6
    assert 60e6 <= bill["active"] <= 75e6
    assert bill["tokens"] == 108_000_000
    assert 2.5e17 <= bill["headline_flops"] <= bench.BUDGET_FLOPS
    assert bill["headline_hours"] <= bench.BUDGET_HOURS
    assert 0 < bill["engine_hours"] < bill["headline_hours"]


def test_no_refuted_lever_is_enabled() -> None:
    """xmod/fff/loop_l/engram/mtp/jepa/fsz/sc_k5 are hard-off, in manifest and text."""
    spec, text = bench.load(CONFIG)
    refuted = [lever for lever in bench.LEVERS if lever.status == "refuted"]
    assert {lever.name for lever in refuted} == bench.REFUTED
    assert all(not lever.enabled for lever in refuted)
    assert not any(f"{name} = true" in text or f"{name}=true" in text for name in bench.REFUTED)
    assert bench.problems(spec, text) == []


def test_surviving_levers_are_wired_to_a_gate() -> None:
    """Every enabled lever names a counter/gate; the conflicted TST lever stays gated."""
    for lever in bench.LEVERS:
        if lever.enabled:
            assert lever.status == "survived"
            assert lever.counter and lever.gate
        if lever.name == "tst_r_0.3":
            assert lever.status == "gated" and not lever.enabled
    iq = next(lever for lever in bench.LEVERS if lever.name == "iq_smoke")
    assert iq.enabled and iq.counter == "iq_judge_calls"


def test_a_refuted_toggle_is_refused_by_the_gate() -> None:
    """The gate can fail: flipping a REFUTED lever on is named, not ignored."""
    spec, _text = bench.load(CONFIG)
    fake = bench.Lever("xmod", "refuted", True, "xmod_fwd", "wrongly enabled")
    original = bench.LEVERS
    try:
        object.__setattr__(bench, "LEVERS", (*original, fake))
        bad = bench.problems(spec, _text)
        assert any("xmod" in reason for reason in bad)
    finally:
        object.__setattr__(bench, "LEVERS", original)


def test_an_over_budget_bill_is_refused() -> None:
    """Doubling the token budget breaks the envelope and is named by the gate."""
    from dataclasses import replace

    spec, text = bench.load(CONFIG)
    big = replace(spec, tokens=spec.tokens * 2)
    assert any("envelope" in reason for reason in bench.problems(big, text))


def test_iq_smoke_gate_has_a_measured_floor_ceiling_and_counter() -> None:
    """random=0 and oracle=1 on the same tasks, with the judge counter reporting."""
    result = bench.iq_gate()
    assert result["n"] > 0
    assert result["floor"] == 0
    assert result["ceiling"] == result["n"]
    assert result["counter"] == result["n"]


def test_an_enabled_flag_without_its_counter_is_a_missing_report() -> None:
    """A lever whose counter never fired must not pass as a clean gate."""
    counters = Counters()
    counters.enable("iq_smoke", counter="iq_judge_calls")
    with pytest.raises(MissingReport, match="iq_smoke"):
        counters.check()
