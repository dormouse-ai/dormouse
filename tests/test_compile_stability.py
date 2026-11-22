"""A compiled step must build one graph and keep it across steps.

The decisive benchmark was hitting ``config.recompile_limit`` because dynamo guarded on
``dormouse.kernels.bank._launches``: the fused kernel tallies a launch every step, so the
guard ``G['…bank']._launches == N`` failed on the next call and recompiled the whole step.
The compiled forward passes ``counters=None`` (``assemble.py``), so the tally is no longer
read there; the accessor is also disabled for dynamo. These gates keep both properties:
one graph for two consecutive steps, and eager parity of the forward and every gradient.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from dormouse.kernels import bank as fused_bank
from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.optim import MixOptimizer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"


def _spec(**overrides: object) -> StackSpec:
    """The CI smoke stack, with only the named fields replaced."""
    raw = tomllib.loads(SMOKE.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(SMOKE))
    return replace(spec, **overrides)  # type: ignore[arg-type]


def _two_steps(spec: StackSpec) -> int:
    """Build the stack, run two compiled steps, return the final dynamo graph count."""
    torch.manual_seed(0)
    assembled = assemble(spec, smoke=True)
    for bank in [m for m in assembled.model.modules() if isinstance(m, MoEFFN)]:
        bank.fixed_passes = MAX_DISPATCH_PASSES  # the static grid is the graph dispatch
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    assembled.step(ids, optimizer=optimizer)
    first = compiled.unique_graphs()
    # What the card does between steps: the fused kernel advances the launch tally.
    fused_bank._tally(2)
    assembled.step(ids, optimizer=optimizer)
    second = compiled.unique_graphs()
    assert first == second, f"the graph was recompiled: {first} -> {second}"
    return second


def test_one_graph_for_two_steps_despite_an_advancing_launch_tally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No mutable tally read may live in the compiled region, or step two recompiles.

    The spy turns any read of the launch tally inside the compiled forward into a hard
    failure: on the old code ``_dispatch`` read it on every step, which pinned a dynamo
    guard on its value and rebuilt the graph until ``recompile_limit`` (8).
    """

    def refuse() -> int:
        raise AssertionError("the compiled forward read the mutable launch tally")

    monkeypatch.setattr(fused_bank, "bank_launches", refuse)
    torch._dynamo.config.suppress_errors = False
    compiled.reset()
    fused_bank.reset_bank_launches()
    try:
        graphs = _two_steps(_spec(torch_compile=True))
    finally:
        compiled.reset()
    assert graphs == 1, f"expected one compiled graph for two steps, got {graphs}"


def test_compiled_forward_and_gradients_match_eager_within_fp32_epsilon() -> None:
    """The counter plumbing is side-effect-only: no number in the step may move."""

    def run(*, use_compile: bool) -> tuple[float, dict[str, torch.Tensor]]:
        torch.manual_seed(0)
        spec = _spec(torch_compile=use_compile)
        if use_compile:
            compiled.reset()
        assembled = assemble(spec, smoke=True)
        if use_compile:
            for bank in [m for m in assembled.model.modules() if isinstance(m, MoEFFN)]:
                bank.fixed_passes = MAX_DISPATCH_PASSES
        optimizer = MixOptimizer(assembled.model)
        ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
        result, _ = assembled.step(ids, optimizer=optimizer)
        grads = {
            name: p.grad.detach().clone().float()
            for name, p in assembled.model.named_parameters()
            if p.grad is not None
        }
        return float(result.loss), grads

    torch._dynamo.config.suppress_errors = False
    eager_loss, eager_grads = run(use_compile=False)
    compiled_loss, compiled_grads = run(use_compile=True)
    compiled.reset()
    assert set(eager_grads) == set(compiled_grads)
    assert eager_loss == pytest.approx(compiled_loss, rel=1e-5, abs=1e-6)
    for name in eager_grads:
        assert torch.allclose(compiled_grads[name], eager_grads[name], rtol=1e-5, atol=1e-6), name
