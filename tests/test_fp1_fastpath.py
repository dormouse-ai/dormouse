"""FP1 fast path: the dense ideal bank, its fused epilogue, and the refusals.

The card numbers live in ``research/perf/FP1-fastpath.md`` and in the harness
artifacts. These CPU gates hold the logic the measurement depends on: a bank
that builds and counts, an external fp64 anchor, and a graph that refuses an
epilogue it cannot capture instead of reporting an eager step as a replay.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_kernels.engine import mega_block
from torch import nn

from dormouse.model.assemble import assemble
from dormouse.model.moe import MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"


def _spec() -> StackSpec:
    """The shipped smoke stack, the smallest shape every arm accepts."""
    raw = tomllib.loads(SMOKE.read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    return StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(SMOKE))


def _bank() -> MoEFFN:
    """A tiny ideal bank whose every factor can be checked against fp64."""
    return MoEFFN(
        16,
        32,
        n_experts=4,
        topk=1,
        n_shared=0,
        d_ffn_gelu=48,
        expert_tsct_rank=8,
        ideal_bank=True,
        ideal_slots=4,
        min_expert_params=1,
    )


def test_ideal_bank_builds_counts_and_bills() -> None:
    """The dense arm has no routed leaves, and both receipts fire on one step."""
    bank = _bank()
    assert bank.ideal_width == 32
    assert list(bank.experts) == []
    assert bank.param_report()["stored"] > 0
    counters = Counters()
    x = torch.randn(5, 16, requires_grad=True)
    y = bank(x, counters)
    assert y.shape == (5, 16)
    y.sum().backward()
    snap = counters.snapshot()
    assert snap["ideal_bank_fwd"] == 5
    assert snap["ideal_bank_bwd"] == 5
    assert bank.ideal_u.grad is not None
    assert bank.ideal_down.grad is not None


def test_ideal_forward_matches_an_external_fp64_anchor() -> None:
    """The gate-masked big GEMMs are the same arithmetic in fp64, not a new model."""
    torch.manual_seed(11)
    bank = _bank().double().eval()  # no QB update between the forward and the anchor
    x = torch.randn(7, 16, dtype=torch.float64)
    y = bank(x)
    index, weight, _ = bank.router(x)
    rank = bank.expert_tsct_rank
    slots = index.unsqueeze(-1) * rank + torch.arange(rank)
    mask = torch.zeros(x.shape[0], bank.ideal_width, dtype=torch.float64)
    mask.scatter_(
        1,
        slots.reshape(x.shape[0], -1),
        weight.unsqueeze(-1).expand(-1, -1, rank).reshape(x.shape[0], -1),
    )
    expected = (((x @ bank.ideal_u) * mask) @ bank.ideal_v) @ bank.ideal_down
    assert torch.allclose(y, expected, rtol=1e-10, atol=1e-10)


def test_epilogue_reference_matches_an_external_fp64_anchor() -> None:
    """`V @ silu(U @ x) + residual` against fp64 torch, independent of Triton."""
    torch.manual_seed(12)
    x = torch.randn(9, 16, dtype=torch.float32)
    residual = torch.randn(9, 32)
    u = torch.randn(16, 32)
    v = torch.randn(32, 32)
    got = mega_block.epilogue_reference(x, residual, u, v)
    expected = torch.addmm(
        residual.double(), torch.nn.functional.silu(x.double() @ u.double()), v.double()
    )
    assert torch.allclose(got.double(), expected, rtol=2e-5, atol=2e-5)


def test_the_epilogue_never_silently_runs_eager() -> None:
    """Flag off is a named refusal; flag on without CUDA is another one."""
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    u = torch.randn(8, 8)
    v = torch.randn(8, 8)
    mega_block.set_epilogue(False)
    with pytest.raises(LoudError, match="arm is off"):
        mega_block.epilogue_apply(x, residual, u, v)
    mega_block.set_epilogue(True)
    try:
        with pytest.raises(LoudError, match="fused epilogue needs a CUDA"):
            mega_block.epilogue_apply(x, residual, u, v)
    finally:
        mega_block.set_epilogue(False)


def test_graph_epilogue_refuses_a_bank_it_cannot_fuse() -> None:
    """An epilogue over a routed/A4 bank is refused before any capture is attempted."""
    model = nn.Linear(4, 4)
    ids = torch.randint(0, 8, (2, 5))
    with pytest.raises(LoudError, match="dense ideal bank"):
        StepGraph(model, cast(Any, None), ids, epilogue=True)


def test_an_ideal_spec_refuses_a_slot_contradiction() -> None:
    """`topk > ideal_slots` is caught at spec construction, not inside a kernel."""
    with pytest.raises(LoudError, match="exceeds ideal_slots"):
        replace(_spec(), ideal_bank=True, ideal_slots=1, topk=2)
    with pytest.raises(LoudError, match="builds no dense bank"):
        replace(_spec(), ideal_slots=0)


def test_twenty_eager_steps_are_deterministic() -> None:
    """The fast path's eager arm is bit-repeatable; a graph is not needed for parity."""
    spec = replace(_spec(), ideal_bank=True, ideal_slots=4)
    losses: list[list[float]] = []
    for _ in range(2):
        torch.manual_seed(21)
        assembled = assemble(spec, device="cpu", smoke=True)
        optimizer = MixOptimizer(assembled.model)
        optimizer.set_lr(0.0)
        ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
        run: list[float] = []
        for _ in range(20):
            result = train_step(assembled.model, ids, optimizer=optimizer)
            assert not result.skipped
            assert torch.isfinite(torch.tensor(result.loss))
            run.append(result.loss)
        losses.append(run)
    assert losses[0] == losses[1]
