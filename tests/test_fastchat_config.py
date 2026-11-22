"""The fastchat config must parse, fit two hours, and carry a falsifiable IQ gate."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fastchat_bench as bench  # noqa: E402

from dormouse.eval.iq import MockJudge, build_suite, evaluate  # noqa: E402
from dormouse.eval.model_adapter import KnownAdapter, RandomAdapter  # noqa: E402
from dormouse.model.spec import StackSpec  # noqa: E402

CONFIG = ROOT / "configs" / "fastchat_400m.toml"


def test_the_config_parses_as_a_stack() -> None:
    """The price is read straight from spec.py, the same parser `dorm plan` uses."""
    spec = StackSpec.from_toml(CONFIG)
    assert (spec.batch, spec.seq_len) == (2, 1024)
    assert spec.attn == "hybrid" and spec.n_layers % 4 == 0
    assert (spec.n_experts, spec.topk, spec.n_shared, spec.expert_tsct_rank) == (64, 2, 1, 256)
    assert spec.torch_compile is True
    assert spec.tokens == 108_000_000


def test_the_headline_flop_arithmetic_fits_two_hours() -> None:
    """6*P*D <= the 92 TFLOP/s x 39.92% x 2h envelope, priced from the built tree."""
    bill = bench.price(StackSpec.from_toml(CONFIG))
    assert 390e6 <= bill["stored"] <= 410e6
    assert bill["tokens"] == 108_000_000
    assert 2.5e17 <= bill["headline_flops"] <= 2.64e17
    assert bill["headline_hours"] <= bench.BUDGET_HOURS
    # The routed engine bill cannot be the larger one: the gate is conservative.
    assert 0 < bill["engine_hours"] < bill["headline_hours"]


def test_the_spec_still_refuses_a_broken_width() -> None:
    """A valid config is not a free pass: n_layers must still tile the 3:1 period."""
    with pytest.raises(LoudError, match="2 layers are left over"):
        replace(StackSpec.from_toml(CONFIG), n_layers=6)


def test_the_iq_smoke_suite_has_a_measured_floor_and_ceiling() -> None:
    """random=0 and oracle=1 on the same tasks: the gate can fail, not just pass."""
    suite = build_suite("smoke", seed=0)
    assert suite.name == "smoke" and len(suite) > 0
    floor = evaluate(suite.tasks, RandomAdapter(0), MockJudge())
    ceiling = evaluate(suite.tasks, KnownAdapter(suite.tasks), MockJudge())
    assert len(floor) == len(ceiling) == len(suite)
    assert all(row.score == 0.0 for row in floor)
    assert all(row.score == 1.0 for row in ceiling)
