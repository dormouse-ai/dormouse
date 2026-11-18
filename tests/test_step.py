"""One step: the order, the counters, the accumulation, and the firewall.

The four invariants here are the four defects the step exists to prevent: a
mechanism that is enabled and silent, a gradient that reaches no parameter of a
group, a batch that poisons the optimizer state, and a loss scaled by `accum` instead
of divided by it. The overfit run at the end is the only test that can see a step
whose arithmetic is wrong but whose shape is right: a step that does not descend on a
fixed batch is a step that is not descending at all.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport
from stacks import reference_stack

from dormouse.train import MixOptimizer, train_step
from dormouse.train.optim import MUON_LR
from dormouse.train.step import CHUNK_TOKENS, chunked_cross_entropy

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}


def _build(vocab: int = 16) -> tuple[torch.nn.Module, MixOptimizer, torch.Tensor]:
    """A fixed tiny stack, its optimizer, and one fixed batch — seeded, not random."""
    torch.manual_seed(23)
    model = reference_stack(**{**SMALL, "vocab": vocab})
    ids = torch.randint(0, vocab, (3, 17), generator=torch.Generator().manual_seed(7))
    return model, MixOptimizer(model), ids


# ------------------------------------------------------------------ the overfit run


def test_twenty_steps_reduce_the_loss_monotonically_on_one_batch() -> None:
    """A step that is wired right overfits three sequences in twenty steps.

    Measured curve (loss after each step): 4.85, 4.79, 4.73, 4.64, 4.53, 4.38, 4.19,
    3.95, 3.65, 3.31, 2.95, 2.61, 2.32, 2.09, 1.92, 1.81, 1.73, 1.68, 1.63, 1.58.
    Strictly descending — 4.85 to 1.58 — which is what a step whose sign, scale and
    routing are all right does to three sequences.

    Recalibrated 2026-11-16 after the tied-head 1/sqrt(d_model) scale (PR #74): the
    old curve started at 29.42 (unscaled logits) and ended at 1.92, a factor of 15.
    Scaling the head cuts the START to 4.85 while the overfit floor is unchanged (~1.6),
    so the ratio is necessarily smaller (0.325, was 0.065) — the same descent, measured
    from a correct init level. The 0.4 bar still fails a step that does not learn
    (ratio ~1) or descends too slowly; it has ~23% headroom over the measured ratio.
    """
    model, optimizer, ids = _build()
    losses = [train_step(model, ids, optimizer=optimizer).loss for _ in range(20)]
    assert all(later < earlier for earlier, later in itertools.pairwise(losses)), losses
    assert losses[-1] < losses[0] * 0.4, losses


# --------------------------------------------------------------------- the gradient


def test_every_parameter_gets_a_gradient_and_a_state_in_one_step() -> None:
    """Per group, not per model: the router's leftover group is the silent one.

    A parameter with no gradient leaves no optimizer state, so state is the receipt:
    `momentum` for the matrices, `exp_avg` for the vectors and the token table.
    """
    model, optimizer, ids = _build()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    train_step(model, ids, optimizer=optimizer)
    state = dict(optimizer.muon.state) | dict(optimizer.adamw.state)
    for name, param in model.named_parameters():
        assert param in state, f"{name} ({optimizer.table[name]}) never reached the optimizer"
        assert not torch.equal(param, before[name]), f"{name} did not move"
    assert {optimizer.table[name] for name in optimizer.table} == {"muon", "adamw"}


def test_the_matrices_move_at_a_different_scale_than_the_vectors() -> None:
    """The two arms are the point: Muon's `√rows` step and AdamW's `sign` step differ."""
    model, optimizer, ids = _build()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    train_step(model, ids, optimizer=optimizer)
    muon_step = (model.blocks[0].ffn.down.weight - before["blocks.0.ffn.down.weight"]).norm()
    adamw_step = (model.norm.weight - before["norm.weight"]).norm()
    # Muon moves a sqrt(rows)-sized block: 32 rows x MUON_LR. AdamW moves at most
    # `lr * sqrt(numel)` and its rate is 0.15 of Muon's.
    assert muon_step.item() == pytest.approx(MUON_LR * 32**0.5, rel=1e-3)
    assert adamw_step.item() < muon_step.item()


# ---------------------------------------------------------------------- the counters


def test_the_counters_prove_every_mechanism_ran() -> None:
    model, optimizer, ids = _build()
    counters = Counters()
    train_step(model, ids, optimizer=optimizer, counters=counters)
    assert counters.silent() == []
    for key in ("fwd", "bwd", "clip", "opt_muon", "opt_adamw"):
        assert counters.counts[key] > 0, key
    assert counters.counts["fwd"] == ids[:, :-1].numel()
    assert counters.counts["opt_muon"] == optimizer.sizes["muon"]


def test_the_step_ends_with_a_check() -> None:
    """An enabled mechanism that never ran must fail the step, not the next log line."""
    model, optimizer, ids = _build()
    counters = Counters()
    counters.enable("kda", counter="kda_fwd", why="the wiring defect under test")
    with pytest.raises(MissingReport, match="kda"):
        train_step(model, ids, optimizer=optimizer, counters=counters)


# ----------------------------------------------------------------------- the firewall


def test_a_nonfinite_loss_skips_the_step_and_leaves_the_weights_alone() -> None:
    model, optimizer, ids = _build()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    with torch.no_grad():
        model.embed.weight[0, 0] = float("inf")
        before["embed.weight"] = model.embed.weight.detach().clone()
    counters = Counters()
    result = train_step(model, ids, optimizer=optimizer, counters=counters)
    assert result.skipped
    assert math.isnan(result.grad_norm)
    assert counters.counts["nan_guard"] == 1
    # The optimizer never ran, so it must not claim to have: only the mechanisms
    # that were reached are enabled.
    assert "opt_muon" not in counters.enabled
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name


def test_a_nonfinite_gradient_trips_the_firewall_on_a_finite_loss() -> None:
    """The gradient path, not the loss path: the loss is fine, one grad is not."""
    model, optimizer, ids = _build()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    poison = model.blocks[0].register_full_backward_hook(
        lambda _module, grad_input, _grad_output: (grad_input[0] * float("nan"), None)
    )
    try:
        counters = Counters()
        result = train_step(model, ids, optimizer=optimizer, counters=counters)
    finally:
        poison.remove()
    assert result.skipped
    assert math.isfinite(result.loss), "the loss was finite; the gradient was not"
    assert counters.counts["nan_guard"] == 1
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name


def test_a_firewalled_step_leaves_no_gradient_behind() -> None:
    """The gradients are cleared, so the next step starts from a clean slate."""
    model, optimizer, ids = _build()
    with torch.no_grad():
        model.norm.weight[0] = float("nan")
    train_step(model, ids, optimizer=optimizer)
    assert all(param.grad is None for param in model.parameters())


# ----------------------------------------------------------------- the accumulation


def test_grad_accumulation_equals_one_pass_over_the_same_rows() -> None:
    """Two micro-batches of three rows must equal one batch of six, weight for weight."""
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    ids = torch.randint(0, 16, (6, 17), generator=torch.Generator().manual_seed(7))
    twin = reference_stack(**SMALL)
    twin.load_state_dict(model.state_dict())
    train_step(model, ids, optimizer=MixOptimizer(model), accum=2)
    train_step(twin, ids, optimizer=MixOptimizer(twin), accum=1)
    twins = dict(twin.named_parameters())
    for name, param in model.named_parameters():
        assert torch.allclose(param, twins[name], atol=1e-6), name


# ------------------------------------------------------------------- clipping, chunks


def test_the_clip_bounds_the_global_gradient_norm() -> None:
    model, optimizer, ids = _build()
    result = train_step(model, ids, optimizer=optimizer, clip=1e-3)
    assert result.grad_norm > 1e-3
    total = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
    assert total.item() == pytest.approx(1e-3, rel=1e-3)


def test_an_unclipped_step_reports_the_same_norm_it_would_have_clipped() -> None:
    model, optimizer, ids = _build()
    result = train_step(model, ids, optimizer=optimizer, clip=math.inf)
    total = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
    assert result.grad_norm == pytest.approx(total.item(), rel=1e-5)


def test_chunked_cross_entropy_is_the_unchunked_mean() -> None:
    torch.manual_seed(24)
    logits = torch.randn(2, 9, 16)
    targets = torch.randint(0, 16, (2, 9))
    want = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    for chunk in (1, 7, 18, CHUNK_TOKENS):
        got = chunked_cross_entropy(logits, targets, chunk=chunk)
        assert got.item() == pytest.approx(want.item(), rel=1e-6), chunk


def test_a_chunk_of_no_tokens_is_refused() -> None:
    with pytest.raises(LoudError, match="holds no token"):
        chunked_cross_entropy(torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.long), chunk=0)


def test_logits_and_targets_that_disagree_are_refused() -> None:
    with pytest.raises(LoudError, match="disagree"):
        chunked_cross_entropy(torch.zeros(1, 2, 4), torch.zeros(1, 3, dtype=torch.long))


# ---------------------------------------------------------------------- the refusals


def test_a_batch_that_does_not_split_evenly_is_refused() -> None:
    model, optimizer, _ = _build()
    with pytest.raises(LoudError, match="does not split"):
        train_step(model, torch.zeros(3, 5, dtype=torch.long), optimizer=optimizer, accum=2)


def test_ids_that_are_not_one_batch_are_refused() -> None:
    model, optimizer, _ = _build()
    with pytest.raises(LoudError, match="takes ids"):
        train_step(model, torch.zeros(1, 2, 3, dtype=torch.long), optimizer=optimizer)


def test_a_batch_with_no_target_position_is_refused() -> None:
    model, optimizer, _ = _build()
    with pytest.raises(LoudError, match="no input has a target"):
        train_step(model, torch.zeros(2, 1, dtype=torch.long), optimizer=optimizer)


def test_a_clip_of_zero_is_refused() -> None:
    model, optimizer, ids = _build()
    with pytest.raises(LoudError, match="scales every gradient to zero"):
        train_step(model, ids, optimizer=optimizer, clip=0.0)
