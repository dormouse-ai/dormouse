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

import copy
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import CHUNK_TOKENS, chunked_cross_entropy, train_step

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


# ----------------------------------------------------- the CPU half, without a capture
#
# The capture is a card-side object, but only the CAPTURE is: the body it records and the
# eager half `run` wraps around the replay are ordinary torch, and they are what a graph
# changes the timing of, not the result. So CI builds the object without a capture and
# holds those two to their claims: the body is a full step (zero_grad, forward, chunked
# CE, backward), and `run` fails closed — a non-finite loss or a vetoed pre-step skips
# the optimizer instead of writing a step of noise into the weights.


def _uncaptured() -> tuple[StepGraph, MixOptimizer, torch.Tensor]:
    """A `StepGraph` on the CPU: every field `_capture` would set, and `_graph=None`."""
    torch.manual_seed(3)
    spec = _spec()
    assembled = assemble(spec, device="cpu", smoke=True)
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    graph = StepGraph.__new__(StepGraph)
    graph.model, graph.optimizer, graph.accum = assembled.model, optimizer, 1
    graph.rows, graph.seq = ids.shape[0], ids.shape[1] - 1
    graph.ids = ids.clone()
    graph._loss = torch.zeros(())
    graph._graph = None
    graph._replays = 0
    graph.failed_on = None
    return graph, optimizer, ids


def test_the_constructor_refuses_a_shape_and_an_accum_it_cannot_split() -> None:
    """Both checks run before any card-side object exists, so CI can hold them."""
    model = torch.nn.Linear(4, 4)
    with pytest.raises(LoudError, match=r"ids \[batch, seq\+1\]"):
        StepGraph(model, object(), torch.zeros((8, 1), dtype=torch.long))  # type: ignore[arg-type]
    with pytest.raises(LoudError, match="does not split"):
        StepGraph(model, object(), torch.zeros((8, 6), dtype=torch.long), accum=3)  # type: ignore[arg-type]


def test_the_body_is_the_eager_step_it_captures() -> None:
    """`_body` is `zero_grad → forward → chunked CE → backward`: same loss, same grads."""
    graph, _optimizer, ids = _uncaptured()
    twin = copy.deepcopy(graph.model)
    loss = graph._body()
    assert torch.isfinite(loss)

    twin_optimizer = MixOptimizer(twin)
    twin_optimizer.zero_grad(set_to_none=False)
    logits = twin(ids[:, :-1], None)
    expected = chunked_cross_entropy(logits, ids[:, 1:], chunk=CHUNK_TOKENS)
    expected.backward()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    for (name, mine), (_, theirs) in zip(
        graph.model.named_parameters(), twin.named_parameters(), strict=True
    ):
        if theirs.grad is None:
            assert mine.grad is None, f"{name} grew a gradient the eager step does not have"
            continue
        assert mine.grad is not None, name
        assert torch.allclose(mine.grad, theirs.grad, rtol=1e-6, atol=1e-8), name


def test_run_clips_and_steps_with_the_replay_stubbed() -> None:
    """The eager half `run` wraps around the replay: clip, both arms' counters, the step."""
    graph, optimizer, ids = _uncaptured()
    graph._body()
    before = [p.detach().clone() for p in graph.model.parameters()]
    counter = Counters()
    graph.replay = lambda ids, counters=None: counter.count("graph_replays")  # type: ignore[method-assign]
    result = graph.run(ids, counters=counter)
    assert not result.skipped
    assert counter.snapshot()["graph_replays"] == 1
    assert counter.snapshot()["clip"] == 1
    assert counter.snapshot()["opt_muon"] == optimizer.sizes["muon"]
    assert counter.snapshot()["opt_adamw"] == optimizer.sizes["adamw"]
    assert result.grad_norm > 0
    moved = [
        p
        for p, was in zip(graph.model.parameters(), before, strict=True)
        if not torch.equal(p, was)
    ]
    assert moved, "a completed step must move the weights"


def test_a_non_finite_loss_never_reaches_the_optimizer() -> None:
    """The nan guard drops the gradients and reports the skip; the weights are untouched."""
    graph, _optimizer, ids = _uncaptured()
    graph._body()
    before = [p.detach().clone() for p in graph.model.parameters()]
    graph.replay = lambda ids, counters=None: None  # type: ignore[method-assign]
    graph._loss = torch.tensor(float("nan"))
    counter = Counters()
    result = graph.run(ids, counters=counter)
    assert result.skipped
    assert result.grad_norm != result.grad_norm  # NaN, stated
    assert counter.snapshot()["nan_guard"] == 1
    for param, was in zip(graph.model.parameters(), before, strict=True):
        assert torch.equal(param, was), "a NaN step must not move a weight"


def test_a_vetoed_pre_step_is_a_skip_and_a_kept_gradient() -> None:
    """`pre_step` returning False skips the update, and the counter says which branch ran."""
    graph, _optimizer, ids = _uncaptured()
    graph._body()
    before = [p.detach().clone() for p in graph.model.parameters()]
    graph.replay = lambda ids, counters=None: None  # type: ignore[method-assign]
    result = graph.run(ids, pre_step=lambda loss: False)
    assert result.skipped
    for param, was in zip(graph.model.parameters(), before, strict=True):
        assert torch.equal(param, was)


def test_a_non_finite_gradient_is_a_skip_as_well() -> None:
    """The clip's norm is the second host read, and `inf` takes the same guard as NaN."""
    graph, _optimizer, ids = _uncaptured()
    graph._body()
    for param in graph.model.parameters():
        if param.grad is not None:
            param.grad.fill_(float("inf"))
    before = [p.detach().clone() for p in graph.model.parameters()]
    graph.replay = lambda ids, counters=None: None  # type: ignore[method-assign]
    counter = Counters()
    result = graph.run(ids, counters=counter)
    assert result.skipped
    assert result.grad_norm == float("inf")
    assert counter.snapshot()["nan_guard"] == 1
    for param, was in zip(graph.model.parameters(), before, strict=True):
        assert torch.equal(param, was), "an inf-norm step must not move a weight"


def test_replay_copies_replays_and_counts() -> None:
    """The replay's own bookkeeping, against a recorder standing in for the graph handle."""
    graph, _optimizer, ids = _uncaptured()
    seen: list[torch.Tensor] = []

    class _Recorder:
        def replay(self) -> None:
            seen.append(graph.ids.clone())

    graph._graph = _Recorder()  # type: ignore[assignment]
    counter = Counters()
    fresh = torch.zeros_like(ids)
    graph.replay(fresh, counters=counter)
    assert graph.replays() == 1
    assert counter.snapshot()["graph_replays"] == 1
    assert torch.equal(seen[0], fresh), "the replay must read the ids it was handed"


def test_replay_at_another_shape_and_the_readers_are_pure() -> None:
    """The shape refusal is before the copy; `loss`, `replays` and `__repr__` are reads."""
    graph, _optimizer, ids = _uncaptured()
    with pytest.raises(LoudError, match="captured at"):
        graph.replay(ids[:1])
    graph._loss = torch.tensor(2.5)
    assert graph.loss() == pytest.approx(2.5)
    assert graph.replays() == 0
    assert "captured=False" in repr(graph) and "rows=8" in repr(graph)


def test_a_declared_graph_with_zero_replays_fails_the_gate() -> None:
    """`graph_replays` at zero under an enabled graph is MissingReport, not silence."""
    counter = Counters()
    counter.enable("graph", counter="graph_replays", why="a step must replay a graph")
    with pytest.raises(MissingReport, match="graph"):
        counter.check()


def test_a_replayed_overflow_is_refused_in_the_eager_half() -> None:
    """The fixed capture loop cannot raise on device, so `run` reads its flag and refuses."""
    graph, _optimizer, ids = _uncaptured()
    bank = next(m for m in graph.model.modules() if isinstance(m, MoEFFN))
    bank.fixed_passes = MAX_DISPATCH_PASSES
    bank._overflow.fill_(True)
    graph._banks = [bank]
    graph.replay = lambda ids, counters=None: None  # type: ignore[method-assign]
    with pytest.raises(LoudError, match=r"more than .* dispatch passes"):
        graph.run(ids)
