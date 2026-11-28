"""The compile lever: it engages, it says so, and a zero-graph "compiled" run fails.

The failure this file exists for is not a slow compile — it is a compile that never
happened. `torch_compile=true` on a stack whose forward dynamo refuses would run the eager
kernels under a compiled name, print a speed-up that came from somewhere else, and leave
every test green. So: a compiled step must report a non-zero graph count, and a run that
reports zero must raise rather than log.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import MissingReport

from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.optim import MixOptimizer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"


def _spec(**overrides: object) -> StackSpec:
    """The CI smoke stack — small enough that a compile is seconds, not minutes."""
    raw = tomllib.loads(SMOKE.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(SMOKE))
    return replace(spec, **overrides)  # type: ignore[arg-type]


def _one_step(spec: StackSpec, *, static_bank: bool = False) -> tuple[float, dict[str, int]]:
    """Build `spec`, run one step, and return its loss and counters.

    `static_bank` forces the fixed pass grid on the MoE bank. The anchor dispatch
    reads `counts.max()` into the host (`moe.py::_pass_plan`) and is explicitly not
    a graph path: under torch.compile that break left an `index_put_` scatter as an
    AOT output and Inductor raised `BackendCompilerFailed: Node index_put_3 was
    invalid, but is output`. The static grid is the dispatch that actually runs
    inside a graph, so a compiled step must exercise it.
    """
    torch.manual_seed(0)
    assembled = assemble(spec, smoke=True)
    if static_bank:
        banks = [m for m in assembled.model.modules() if isinstance(m, MoEFFN)]
        assert banks, "the smoke stack must contain an MoE bank to compile"
        for bank in banks:
            bank.fixed_passes = MAX_DISPATCH_PASSES
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    result, counters = assembled.step(ids, optimizer=optimizer)
    return result.loss, dict(counters.snapshot())


def test_compile_off_builds_no_graph_and_counts_nothing() -> None:
    """The recipe's default: no graph built, and no counter claims one was."""
    compiled.reset()
    _, snapshot = _one_step(_spec(torch_compile=False))
    assert compiled.unique_graphs() == 0
    assert snapshot.get(compiled.COUNTER, 0) == 0


def test_a_compiled_stack_builds_a_graph_and_the_step_reports_it() -> None:
    """`torch_compile=true` is a graph dynamo built, not a flag in a config file."""
    torch._dynamo.config.suppress_errors = False
    graphs, snapshot = 0, {}
    try:
        _, snapshot = _one_step(_spec(torch_compile=True), static_bank=True)
        graphs = compiled.unique_graphs()
    finally:
        compiled.reset()
    assert graphs > 0, (
        "torch_compile=true and dynamo built no graph: the eager kernels ran under a "
        "compiled name, which is the defect this counter exists to catch"
    )
    assert snapshot[compiled.COUNTER] > 0


def test_a_compiled_run_with_zero_graphs_fails_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter is the gate: compiled, zero graphs, no step."""
    monkeypatch.setattr(compiled, "enabled", lambda: True)
    monkeypatch.setattr(compiled, "unique_graphs", lambda: 0)
    with pytest.raises(MissingReport):
        _one_step(_spec(torch_compile=False))


def test_the_break_report_names_its_reasons() -> None:
    """`proof()` is what a report prints: graphs, frames, and breaks by name."""
    found = compiled.proof()
    assert set(found) == {"unique_graphs", "frames", "graph_break"}
    assert isinstance(found["unique_graphs"], int)
    assert all(isinstance(k, str) for k in found["graph_break"])
    assert "compile off" in compiled.line() or "compile on" in compiled.line()
