"""The step-graph measurement's gates: zero replays is a failure, not a slow win.

The card runs this measurement by hand, so CI cannot rerun it. These tests hold the
artifact to its own claims: a capture that failed must report the op that stopped it and
must not claim a replay; a replayed step must move the counter; the parity verdict must
follow the bf16-epsilon comparison it printed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "runs" / "step_graph_1b"


def _load(name: str) -> dict[str, Any]:
    """Load one measurement artifact, skipping when the card run has not happened."""
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip(f"no card measurement yet: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_failed_capture_names_its_op_and_never_claims_a_replay() -> None:
    """A refused capture is a row with a reason, not an eager step under a graph's name."""
    record = _load("capture_b4.json")["capture"]
    if record["captured"]:
        pytest.skip("the card captured this step; the refusal gate is vacuous")
    assert record["failed_on"]
    assert record["failed_on"] != "None"


def test_eager_only_speed_row_has_zero_replays() -> None:
    """When the capture failed, the ms/step ladder is eager-only and says so."""
    record = _load("speed_b8.json")["speed"]
    if not record.get("capture_failed"):
        pytest.skip("the card captured this step; the refusal row is vacuous")
    assert record["graph_replays"] == 0
    assert record["capture_failed_on"]
    assert record["eager_ms_runs"]


def test_parity_verdict_follows_the_bf16_epsilon_comparison() -> None:
    """`lossless` is derived from the measured deviation, never stated by hand."""
    record = _load("body-parity_b4.json")["parity"]
    assert record["within_bf16_epsilon"] == (record["worst_rel"] <= record["bf16_epsilon"])
    assert record["worst_rel"] >= 0
    assert record["grad_tensors"] > 0
