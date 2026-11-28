"""The A/B gate must measure zero before it is allowed to measure a delta."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from dormouse.metrics import LN2
from dormouse.train.runconfig import RunRecipe
from dormouse.train.streams import HeldOut, heldout_bytes_per_token

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ab_gate  # noqa: E402

CONFIG = ROOT / "tools" / "ab_gate.toml"


def _recipe() -> RunRecipe:
    return RunRecipe.read(CONFIG)


def test_control_is_bitwise_equal_at_twenty_steps() -> None:
    """Two identical 20-step runs must agree on every loss and on the eval BPB."""
    ab_gate.configure_cpu()
    recipe = _recipe()
    values = {
        "arm": "control",
        "seed": 7,
        "steps": 20,
        "batches": 1,
        "first": 0,
        "lr_scale": 0.1,
    }
    a = ab_gate.run_arm(recipe, **values)
    b = ab_gate.run_arm(recipe, **values)
    ab_gate.check_control(a, b, seed=7)
    assert torch.equal(a.losses, b.losses)
    assert torch.equal(a.bpb, b.bpb)
    assert float(a.bpb - b.bpb) == 0.0


def test_eval_window_uses_manifest_bytes_and_the_nats_formula() -> None:
    """The window is priced by the eval shard's 4.206775, not len(text)/tokens."""
    recipe = _recipe()
    bpt = heldout_bytes_per_token(recipe)
    assert bpt == 4.206775
    assert recipe.eval_path is not None
    stream = HeldOut(
        recipe.eval_path,
        bpt,
        batch=recipe.spec.batch,
        seq_len=recipe.spec.seq_len,
        probes=1,
    )
    window = stream.batch(0, recipe.spec.batch * recipe.spec.seq_len)
    assert window.payload_bytes == round(256 * 4.206775) == 1077
    nats = torch.tensor(9.0, dtype=torch.float64)
    run = ab_gate.ArmRun(
        torch.zeros(1, dtype=torch.float64), nats, window.payload_bytes, "ids", "eval"
    )
    assert torch.equal(run.bpb, nats / LN2 / window.payload_bytes)
    assert math.isclose(float(run.bpb), 9.0 / math.log(2) / 1077, rel_tol=1e-12)


def test_cuda_is_a_named_refusal() -> None:
    with pytest.raises(ab_gate.GateRefusal, match="B1 BANK_ATOMICS_NONDETERMINISTIC"):
        ab_gate.main(["--device", "cuda"])


def test_unknown_arm_is_refused_by_name() -> None:
    with pytest.raises(ab_gate.GateRefusal, match="unknown arm"), ab_gate._arm("not-an-arm"):
        pass


def test_control_floor_refuses_a_divergent_pair() -> None:
    a = ab_gate.ArmRun(
        torch.tensor([1.0]), torch.tensor(9.0, dtype=torch.float64), 1077, "same", "same"
    )
    b = ab_gate.ArmRun(
        torch.tensor([1.0, 2.0]), torch.tensor(9.0, dtype=torch.float64), 1077, "same", "same"
    )
    with pytest.raises(ab_gate.GateRefusal, match="CONTROL_NOT_BITWISE"):
        ab_gate.check_control(a, b, seed=0)
