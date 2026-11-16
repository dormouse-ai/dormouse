"""The step graph: a replay that ran, a capture that names what stopped it, and parity.

Three gates, in the order the failure modes matter:

1. **The replay ran.** `replays()` and the `graph_replays` counter are the proof; a graph
   that was built and never replayed left the eager step in place.
2. **A capture that cannot run says why.** A step with a host read in it raises and names
   the op — the eager path under a graph's name is the defect class this project has paid
   for twice.
3. **The captured step is the step.** Same loss, same weights: the graph changes when the
   launches are issued, not what they compute.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"
REQUIRES_CARD = torch.cuda.is_available()
NO_CARD = "a CUDA graph is a card-side object; the eager step is what CI runs"


def _spec() -> StackSpec:
    """The CI smoke stack, at the smallest shape the arms accept."""
    raw = tomllib.loads(SMOKE.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    form = {k: v for k, v in raw.items() if k in keys}
    return StackSpec.from_mapping(form, source=str(SMOKE))


def _built() -> tuple[object, MixOptimizer, torch.Tensor]:
    """A seeded smoke stack on the card, its optimizer, and one batch of ids."""
    torch.manual_seed(3)
    spec = _spec()
    assembled = assemble(spec, device="cuda", smoke=True)
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda")
    return assembled, optimizer, ids


@pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)
def test_a_captured_step_is_the_eager_step() -> None:
    """Two steps through the graph are the two steps `train_step` would take."""
    assembled, optimizer, ids = _built()
    graph = assembled.capture(ids, optimizer=optimizer)
    assert isinstance(graph, StepGraph)

    torch.manual_seed(3)
    eager = assemble(_spec(), device="cuda", smoke=True).model
    eager.load_state_dict(assembled.model.state_dict())
    eager_optimizer = MixOptimizer(eager)

    for _ in range(2):
        result = assembled.step(ids, optimizer=optimizer, graph=graph)[0]
        expected = train_step(eager, ids, optimizer=eager_optimizer)
        assert result.loss == pytest.approx(expected.loss, rel=1e-6, abs=1e-6)
        assert result.skipped == expected.skipped
    assert graph.replays() == 2
    expected_params = dict(eager.named_parameters())
    for name, param in assembled.model.named_parameters():
        assert torch.allclose(param, expected_params[name], rtol=1e-6, atol=1e-6), name


@pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)
def test_the_graph_counts_its_replays() -> None:
    """The counter, not the object, proves the replay: `graph_replays` moves per step."""
    assembled, optimizer, ids = _built()
    graph = assembled.capture(ids, optimizer=optimizer)
    _, counters = assembled.step(ids, optimizer=optimizer, graph=graph)
    assert counters.snapshot()["graph_replays"] == 1
    assert graph.replays() == 1


@pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)
def test_a_replay_at_another_shape_is_refused() -> None:
    """A graph replays the shape it was captured at; another shape is a refusal."""
    assembled, optimizer, ids = _built()
    graph = assembled.capture(ids, optimizer=optimizer)
    with pytest.raises(LoudError, match="captured at"):
        graph.replay(ids[:1])


class _Syncs(torch.nn.Module):
    """A model that reads a device value to the host: a capture cannot record that."""

    def __init__(self) -> None:
        """Hold one projection so the optimizer has a parameter to own."""
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, counters: object = None) -> torch.Tensor:
        """`float(x.sum())` is the sync; the multiply is what a graph would have run."""
        return self.proj(x) * float(x.sum())


@pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)
def test_a_host_read_in_the_step_names_the_op_that_stopped_it() -> None:
    """A capture that cannot run raises — it never falls back to the eager step."""
    model = _Syncs().cuda()
    optimizer = MixOptimizer(model)
    ids = torch.zeros((8, 6), dtype=torch.long, device="cuda")
    graph = StepGraph.__new__(StepGraph)
    graph.model, graph.optimizer, graph.accum = model, optimizer, 1
    graph.rows, graph.seq = ids.shape[0], ids.shape[1] - 1
    graph.ids = ids.clone()
    graph._loss = torch.zeros((), device="cuda")
    graph._graph = None
    graph._replays = 0
    graph.failed_on = None
    with pytest.raises(LoudError, match="could not be captured"):
        graph._capture()
    assert graph.failed_on is not None
    assert graph.replays() == 0, "a failed capture must not report a replay"
