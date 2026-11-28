"""Integration gates: the paired frozen-model protocol and the merged A4 dispatch.

The on-card numbers are in `tools/a4_integrate_gate.py` and `tools/a4_fused_ab.py`;
these tests pin the arithmetic and the refusal seams that make those numbers honest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model import quant
from dormouse.model.spec import StackSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.a4_integrate_gate import COUNTERS_8, _env, _set_a4, _summary


def test_summary_paired_ci_and_exact_repeatability_floor() -> None:
    """A 0.01 gate needs both the mean and its one-sided upper bound inside 0.01."""
    okay = _summary([0.001, 0.002, 0.003], [0.0, 0.0, 0.0])
    assert okay["verdict"] == "PASS" and okay["ci95"][1] <= 0.01
    biased = _summary([0.02, 0.02, 0.02], [0.0, 0.0, 0.0])
    assert biased["verdict"] == "RED" and abs(biased["mean_dbpb"] - 0.02) < 1e-12
    # A noisy reduction is red even if the point estimate is small: the from-scratch
    # protocol's floor was -2.56 BPB, which this check exists to refuse.
    assert _summary([0.001, 0.002, 0.003], [0.0, 0.5, 0.0])["verdict"] == "RED"


def test_env_restores_absent_and_present_flags() -> None:
    """A/B flags must not leak from one arm into the next."""
    before = os.environ.get(quant.FUSED_FLAG)
    with _env(DORMOUSE_A4_FUSED="1"):
        assert quant.fused_on()
    assert os.environ.get(quant.FUSED_FLAG) == before


def test_a4_toggle_touches_only_activation_boundaries() -> None:
    """The frozen arms differ only at the two norms per block, never in weights."""
    spec = StackSpec.from_mapping(
        {
            "d_model": 64,
            "n_layers": 4,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 32,
            "c_lat": 32,
            "d_ffn": 128,
            "n_experts": 2,
            "topk": 1,
            "expert_tsct_rank": 16,
            "attn_k": 16,
            "attn_window": 8,
            "seq_len": 32,
            "batch": 1,
            "torch_compile": False,
            "act_quant": "none",
            "weight_qat": "none",
        }
    )
    from dormouse.model.assemble import assemble

    model = assemble(spec, smoke=True, device="cpu").model
    weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert _set_a4(model, True) == 2 * spec.n_layers
    assert _set_a4(model, False) == 2 * spec.n_layers
    assert all(torch.equal(tensor, weights[name]) for name, tensor in model.state_dict().items())


def test_fused_dispatch_refuses_calibrations_it_does_not_implement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nearest-absmax only: SR and per-channel scale must not silently run eager."""
    monkeypatch.setenv(quant.FUSED_FLAG, "1")
    monkeypatch.setenv(quant.ROUND_FLAG, "sr")
    with pytest.raises(LoudError, match="fused kernel implements nearest-even"):
        quant.a4(torch.randn(1, 64))
    monkeypatch.setenv(quant.ROUND_FLAG, "near")
    monkeypatch.setenv(quant.SCALE_FLAG, "absmax")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(LoudError, match="Triton or CUDA is unavailable"):
        quant.a4(torch.randn(1, 64))


def test_eight_counter_receipts_are_distinct_and_named() -> None:
    """The counter gate is an explicit list, not a loose 'some counter moved' check."""
    assert len(COUNTERS_8) == 8 and len(set(COUNTERS_8)) == 8
    assert "bank_a4_bwd" in COUNTERS_8 and "tsct_ternary_bwd" in COUNTERS_8
