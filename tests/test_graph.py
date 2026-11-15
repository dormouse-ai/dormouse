"""The graphed step: what it costs before capture, what it names when it fails, parity.

Three gates, because the old engine failed in three ways and each one was invisible:

1. **Memory.** The private pool a graph needs is counted from a real step before the
   capture, and a card that cannot hold it is refused with the shortfall in MiB. The
   alternative is an OOM inside the graph's allocator, naming an op that is not the
   cause.
2. **Failure.** A capture that stops is a `LoudError` naming the op it stopped at, and
   `graph_replays` stays zero. There is no eager fallback to report a number for, so
   "the graph gave +0 %" cannot be a reading of the eager path.
3. **Parity.** Twenty steps through `device_step` — captured — and twenty through
   `train_step` must leave the weights within 1e-6. That is also the gate that stops the
   two step bodies from drifting apart, and it runs on CPU as well: the two paths differ
   only in the host reads, so a CPU comparison is a real comparison of the arithmetic.
"""

from __future__ import annotations

import pytest
import torch
from stacks import reference_stack

from dormouse.train.graph import StepGraph, device_step, preflight, state_bytes
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

REQUIRES_CARD = torch.cuda.is_available()
NO_CARD = "a CUDA graph needs a CUDA device; the CPU parity gate covers the arithmetic"

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}

MIB = 2**20


def _build(device: str = "cpu") -> tuple[torch.nn.Module, MixOptimizer, torch.Tensor]:
    """A tiny stack, its optimizer, and one fixed batch — seeded, not random."""
    torch.manual_seed(23)
    model = reference_stack(**SMALL).to(device)
    ids = torch.randint(0, 16, (3, 17), generator=torch.Generator().manual_seed(7))
    return model, MixOptimizer(model), ids.to(device)


# --------------------------------------------------------------- the preflight gate


def test_preflight_refuses_a_pool_the_card_cannot_hold() -> None:
    """A pool larger than the free memory minus the margin is refused, with the shortfall."""
    report = preflight(
        persistent_bytes=100 * MIB,
        pool_bytes=200 * MIB,
        free_bytes=900 * MIB,
        margin_bytes=512 * MIB,
    )
    assert report.total_bytes == 712 * MIB
    assert report.fits()

    with pytest.raises(Exception) as caught:
        preflight(
            persistent_bytes=100 * MIB,
            pool_bytes=200 * MIB,
            free_bytes=600 * MIB,
            margin_bytes=512 * MIB,
        )
    message = str(caught.value)
    assert "short by" in message
    assert "112 MiB" in message, message


def test_preflight_counts_the_state_from_the_tensors() -> None:
    """The resting set is weights, gradients and both arms' moments — counted, not assumed."""
    model, optimizer, ids = _build()
    before = state_bytes(model, optimizer)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    assert before == weights, "no gradients or moments exist before the first step"

    device_step(model, ids, optimizer)
    after = state_bytes(model, optimizer)
    # Gradients and the two arms' moments are more than the weights: 4 B/param of
    # gradient and at least one moment per parameter group.
    assert after > weights
    assert after >= 2 * weights


def test_the_graph_refuses_a_process_without_a_card() -> None:
    """No CUDA device is a refusal with a reason, not an AttributeError three lines in."""
    if REQUIRES_CARD:
        return
    model, optimizer, ids = _build()
    with pytest.raises(Exception) as caught:
        StepGraph(model, optimizer, ids=ids)
    assert "needs a CUDA device" in str(caught.value)


def test_the_failure_trace_names_the_op() -> None:
    """The recorder sees the dispatched op, which is how a capture failure is named."""
    from dormouse.train.graph import _DISPATCH

    if _DISPATCH is None:
        pytest.skip("this torch has no TorchDispatchMode")
    from dormouse.train.graph import _OpTrace

    left = torch.ones(4)
    with _OpTrace() as trace:
        torch.add(left, left)
    assert "add" in trace.last, trace.last


# ------------------------------------------------------------- the CPU parity gate


def test_device_step_and_train_step_agree_on_cpu() -> None:
    """Three steps of each body leave the same weights: the host reads were the only gap.

    This is the gate that keeps `device_step` from drifting away from the step the loop
    actually runs, and it runs in CI where there is no card.
    """
    eager_model, eager_optim, ids = _build()
    graph_model, graph_optim, _ = _build()
    torch.manual_seed(99)
    for _ in range(3):
        result = train_step(eager_model, ids, optimizer=eager_optim, clip=float("inf"))
        assert not result.skipped
        device_step(graph_model, ids, graph_optim)

    worst = 0.0
    for (name, left), (other, right) in zip(
        eager_model.named_parameters(), graph_model.named_parameters(), strict=True
    ):
        assert name == other
        worst = max(worst, float((left.detach() - right.detach()).abs().max()))
    assert worst <= 1e-6, f"the two step bodies disagree by {worst:.3e}"


def test_device_step_lowers_the_loss_on_one_batch() -> None:
    """A device-only body that does not descend is a body with the wrong sign somewhere."""
    model, optimizer, ids = _build()
    losses = []
    for _ in range(6):
        losses.append(float(device_step(model, ids, optimizer).detach()))
    assert losses[-1] < losses[0], losses


# ---------------------------------------------------------------- the capture itself


def test_capture_replay_and_twenty_step_parity() -> None:
    """A capture that succeeds replays, counts, and matches twenty eager steps to 1e-6."""
    if not REQUIRES_CARD:
        pytest.skip(NO_CARD)
    eager_model, eager_optim, ids = _build("cuda")
    graph_model, graph_optim, _ = _build("cuda")
    graph = StepGraph(graph_model, graph_optim, ids=ids)
    report = graph.capture()
    assert report.fits()
    assert report.pool_bytes > 0
    assert graph.pool_bytes > 0, "a captured graph takes pool memory; zero means no capture"
    assert graph.graph_replays == 0

    for _ in range(20):
        train_step(eager_model, ids, optimizer=eager_optim, clip=float("inf"))
        graph.replay(ids)
    assert graph.graph_replays == 20

    worst = 0.0
    for (name, left), (other, right) in zip(
        eager_model.named_parameters(), graph_model.named_parameters(), strict=True
    ):
        assert name == other
        worst = max(worst, float((left.detach() - right.detach()).abs().max()))
    assert worst <= 1e-6, f"graphed and eager weights disagree by {worst:.3e}"


def test_a_host_read_fails_the_capture_and_is_named() -> None:
    """The failure path: no graph, the op is named, and the replay counter stays zero."""
    if not REQUIRES_CARD:
        pytest.skip(NO_CARD)
    model, optimizer, ids = _build("cuda")
    graph = StepGraph(model, optimizer, ids=ids)
    original = graph.body

    def body_with_a_host_read() -> torch.Tensor:
        loss = original()
        float(loss)  # the read `train/step.py` makes, and the reason it cannot be captured
        return loss

    graph.body = body_with_a_host_read  # type: ignore[method-assign]
    with pytest.raises(Exception) as caught:
        graph.capture()
    assert graph.graph_replays == 0
    assert graph.failed_on is not None
    assert "local_scalar_dense" in graph.failed_on or "item" in graph.failed_on, graph.failed_on
    assert "capture stopped at" in str(caught.value)

    with pytest.raises(Exception) as replay:
        graph.replay(ids)
    assert "before a successful capture" in str(replay.value)


def test_replay_refuses_a_learning_rate_the_graph_was_not_captured_at() -> None:
    """The rate is baked into the recording; a changed rate must not be replayed silently."""
    if not REQUIRES_CARD:
        pytest.skip(NO_CARD)
    model, optimizer, ids = _build("cuda")
    graph = StepGraph(model, optimizer, ids=ids)
    graph.capture()
    optimizer.set_lr(0.5)
    with pytest.raises(Exception) as caught:
        graph.replay(ids)
    assert "captured at rates" in str(caught.value)
