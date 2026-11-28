"""The core loop is visible: counters, parameter group, and a pre-loop readout."""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec


def _spec(**overrides: object) -> StackSpec:
    base = {
        "unit": "byte",
        "vocab": 256,
        "attn": "hybrid",
        "ffn": "dense",
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 8,
        "c_lat": 16,
        "d_ffn": 64,
        "n_experts": 0,
        "topk": 0,
        "n_shared": 0,
        "expert_tsct_rank": 0,
        "attn_k": 32,
        "attn_window": 8,
        "seq_len": 64,
        "batch": 2,
        "tokens": 128,
        "loop_passes": 2,
        "loop_start_frac": 0.25,
        "loop_end_frac": 0.75,
        "loop_rank": 4,
    }
    base.update(overrides)
    return StackSpec(**base)  # type: ignore[arg-type]


def test_loop_is_counted_and_reported_under_its_own_group() -> None:
    assembled = assemble(_spec(), device="cpu")
    ids = torch.randint(0, 256, (2, 64), generator=torch.Generator().manual_seed(7))
    counters = Counters()
    logits = assembled.model(ids, counters)
    assert logits.shape == (2, 64, 256)
    assert counters.counts["loop_fwd"] == 1
    report = assembled.model.param_report()
    assert report["head"] == 0
    assert report["loops"] > 0
    logits.sum().backward()
    assert counters.counts["loop_bwd"] == 1
