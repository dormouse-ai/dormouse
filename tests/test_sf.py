"""The schedule-free pair, against the papers' equations and the arms they replace.

Every number here is either a closed form written in a docstring — `‖ΔW‖_F =
0.2·η·√(m·n)`, "every row of the first step has the same norm", `c₁ = 1`, the ramp
`(total - t)/(total - start)` — or an equality with something outside this code: our
`AdamW`, our `polar`. None is recorded from a run of the code under test, so a wrong
implementation cannot agree with itself.

The four defects these gates exist for: an SF wrapper whose average is dead (a real
schedule-free run that is just a constant-rate run), a normalisation that divides
nothing, an `eval_weights` that reports the training iterate as if it were the model,
and a schedule written on top of a schedule-free optimizer.
"""

from __future__ import annotations

import inspect
import itertools

import pytest
import torch
from dormouse_core.errors import LoudError
from stacks import reference_stack

from dormouse.train.muon import orthogonalise
from dormouse.train.optim import (
    MUON_LR,
    AdamW,
    MixOptimizer,
    Muon,
    polar,
)
from dormouse.train.sf import SF_ETA_SCALE, SFAdamW, SFNorMuon
from dormouse.train.step import chunked_cross_entropy

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}


# ------------------------------------------------------------------------ anchors


def test_sf_adamw_with_the_average_off_is_adamw_bit_for_bit() -> None:
    """`average=False` sets `c = 1`, so `x ≡ z` and (1)-(3) are AdamW's own step.

    Bit-for-bit, not approximately: the wrapper is allowed to add an average, not to
    change the arithmetic of the rule it wraps. Six steps with decay on so the
    decoupled-decay branch runs too.
    """
    torch.manual_seed(0)
    ours = torch.nn.Parameter(torch.randn(16))
    theirs = torch.nn.Parameter(ours.detach().clone())
    reference = AdamW([ours], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False)
    wrapped = SFAdamW(
        [theirs], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, average=False
    )
    for step in range(6):
        grad = torch.randn(16)
        ours.grad = grad.clone()
        theirs.grad = grad.clone()
        reference.step()
        wrapped.step()
        assert torch.equal(ours.detach(), theirs.detach()), step


def test_the_sf_matrix_direction_is_orthogonal_to_1e_5() -> None:
    """The SF branch's `polar`, on the same three fixtures `test_optim.py` pins.

    The polar factor is what (8) multiplies by `η̂`, so it is the only part of the
    step whose orthogonality is a property of the method rather than of the data.
    """
    for shape in ((64, 832), (832, 2048), (2048, 832)):
        torch.manual_seed(2)
        q = orthogonalise(torch.randn(*shape))
        gram = q @ q.T if shape[0] <= shape[1] else q.T @ q
        error = (gram - torch.eye(gram.shape[0])).abs().max().item()
        assert error <= 1e-5, f"{shape}: {error:.2e}"


def test_the_first_step_gives_every_row_the_same_norm() -> None:
    """(6)-(7): the row-wise second moment is what "Nor" adds, and it is measurable.

    `ΔW_row = -η̂·P_row/(√v_row + ε)` and `v ∝ mean(P_row²)`, so every row of the first
    step has norm `η̂·√n` — the same number — whatever `polar` gave that row. On
    `[256, 64]` the polar factor's row norms span 1.54×, so a deleted normalisation
    fails this. (On a wide matrix `polar`'s rows are already unit-norm and the
    normalisation is a no-op; that is a fact about the SVD, not a defect, and it is
    pinned in the second half.)
    """
    for shape, spread in (((256, 64), True), ((64, 256), False)):
        torch.manual_seed(7)
        param = torch.nn.Parameter(torch.randn(*shape))
        start = param.detach().clone()
        grad = torch.randn(*shape)
        param.grad = grad.clone()
        SFNorMuon([param], lr=0.01, weight_decay=0.0).step()
        rows = (param.detach() - start).norm(dim=1)
        polar_rows = polar(grad).norm(dim=1)
        assert rows.max().item() / rows.min().item() == pytest.approx(1.0, abs=1e-4), shape
        if spread:
            assert polar_rows.max().item() / polar_rows.min().item() > 1.1, shape
        else:
            assert polar_rows.max().item() / polar_rows.min().item() == pytest.approx(1.0, abs=1e-4)


def test_the_first_step_is_the_paper_scaling() -> None:
    """(8): `‖ΔW‖_F = eta_scale·η·√(m·n)`, whatever the shape — one rate for all of them."""
    for shape in ((32, 64), (64, 32), (64, 64)):
        torch.manual_seed(6)
        param = torch.nn.Parameter(torch.randn(*shape))
        start = param.detach().clone()
        param.grad = torch.randn(*shape)
        SFNorMuon([param], lr=0.01, weight_decay=0.0).step()
        want = SF_ETA_SCALE * 0.01 * param.numel() ** 0.5
        assert (param.detach() - start).norm().item() == pytest.approx(want, rel=1e-5), shape


# ------------------------------------------------------------- the average is real


def test_the_average_is_a_different_point_and_eval_restores_it_exactly() -> None:
    """(3): with a constant gradient `z` drifts at `-η` a step and `x` trails it.

    After 50 steps at η=0.01 the fast iterate is at -0.4999 and the average at -0.2550,
    and the parameter must be `(1-β₁)·z + β₁·x = -0.2795` — eq. 1, checked against the
    two iterates rather than recorded. An implementation whose average is dead has
    `x == y` and fails the first assert. The restore is `torch.equal`, because
    `train_weights` copies the saved `Y` rather than inverting (1) in floating point.
    """
    param = torch.nn.Parameter(torch.zeros(8))
    opt = SFAdamW([param], lr=0.01, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(50):
        param.grad = torch.ones(8)
        opt.step()
    trained = param.detach().clone()
    opt.eval_weights()
    averaged = param.detach().clone()
    opt.train_weights()
    fast = opt.state[param]["z"][0].item()
    assert not torch.equal(averaged, trained)
    assert fast < trained[0].item() < averaged[0].item() < 0.0, (fast, averaged[0], trained[0])
    # The reconstruction in §4 is two rounding steps away from the internal `x`, so
    # eq. 1 holds to fp32, not to the last bit.
    assert trained[0].item() == pytest.approx(0.1 * fast + 0.9 * averaged[0].item(), abs=1e-7)
    assert torch.equal(param.detach(), trained)
    assert not opt._saved


def test_the_first_step_leaves_the_parameter_on_the_fast_iterate() -> None:
    """(4) with one term: `w₁/Σw = 1`, so `X₁ = Z₁` and `Y₁ = Z₁` — no lag to start."""
    param = torch.nn.Parameter(torch.zeros(4))
    param.grad = torch.ones(4)
    opt = SFAdamW([param], lr=0.01, weight_decay=0.0)
    opt.step()
    assert torch.equal(opt.state[param]["z"], param.detach())


# ------------------------------------------------------------------ no horizon


def test_the_sf_pair_has_no_horizon() -> None:
    """No constructor argument and no state entry may name a training horizon.

    The gate is the signature and the state, because a horizon that is only read
    somewhere else is the defect that costs a decay phase when the corpus grows.
    `begin_final_decay` is the one exception and it is a method, not state at
    construction: it is the opt-in annealing lever, and without the call the rate is
    flat at every step, including the ones a longer run would have.
    """
    forbidden = {"total", "total_steps", "horizon", "max_steps", "t_max", "schedule"}
    for cls in (SFNorMuon, SFAdamW):
        assert forbidden.isdisjoint(inspect.signature(cls.__init__).parameters), cls
    param = torch.nn.Parameter(torch.zeros(4))
    opt = SFAdamW([param], lr=0.01, weight_decay=0.0)
    param.grad = torch.ones(4)
    opt.step()
    assert set(opt.state[param]) == {"z", "m", "v"}
    assert {opt.lr_at(step) for step in (1, 2, 10, 10_000, 10**6)} == {0.01}


def test_twenty_steps_reduce_the_loss_monotonically_on_one_batch() -> None:
    """It learns: 26.727, 26.041, 25.664, ... 18.284 over twenty fixed sequences.

    Strictly descending, and by the factor of 1/1.46 that a step whose sign, scale and
    routing are right produces here. Not the 5× that `test_step.py` records for the
    scheduled pair: there the parameter takes the full step every step, here it is the
    trailing average and `Z` is the iterate that moves at the full rate — the reason
    Schedule-Free is measured on long runs (arXiv:2605.19095 §9), and the reason this
    test's bar is monotone descent rather than a factor copied from a different method.
    """
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    optimizer = MixOptimizer(model, schedule="sf")
    ids = torch.randint(0, SMALL["vocab"], (3, 17), generator=torch.Generator().manual_seed(7))
    losses = []
    for _ in range(20):
        logits = model(ids[:, :-1])
        loss = chunked_cross_entropy(logits, ids[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(later < earlier for earlier, later in itertools.pairwise(losses)), losses
    assert losses[-1] < losses[0] * 0.75, losses


# --------------------------------------------------- the scheduled path is untouched


def test_cosine_is_still_the_default_and_eval_weights_moves_nothing() -> None:
    """`schedule="cosine"` builds the same two arms it built before this module.

    `eval_weights` is a no-op on the weights for a scheduled run, so a loop can wrap
    an eval in it without asking which mode it is in — and the table, the sizes and
    the head-split groups are the ones the router produced.
    """
    model = reference_stack(**SMALL)
    mix = MixOptimizer(model)
    assert mix.schedule == "cosine"
    assert type(mix.muon) is Muon
    assert type(mix.adamw) is AdamW
    assert mix.sizes == {"muon": 10, "adamw": 6}
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    with mix.eval_weights():
        assert not model.training
        for name, param in model.named_parameters():
            assert torch.equal(param, before[name]), name
    assert model.training
    mix.set_lr(0.5)
    assert mix.muon.param_groups[0]["lr"] == pytest.approx(MUON_LR / 2)


def test_the_sf_pair_routes_exactly_where_the_scheduled_pair_does() -> None:
    """Same routing, same head splits, same counters: only the update rule differs."""
    model = reference_stack(**SMALL)
    scheduled, free = MixOptimizer(model), MixOptimizer(model, schedule="sf")
    assert free.table == scheduled.table
    assert free.sizes == scheduled.sizes
    splits = [group["head_split"] for group in scheduled.muon.param_groups]
    assert [group["head_split"] for group in free.muon.param_groups] == splits


# ---------------------------------------------------------------- the final decay


def test_the_final_decay_ramps_the_last_five_percent_to_zero() -> None:
    """`(total - t)/(total - start)` over the last 5 % of 200 steps: 0.0093 ... 0.0."""
    param = torch.nn.Parameter(torch.zeros(4))
    opt = SFAdamW([param], lr=0.01, final_decay_frac=0.05)
    opt.begin_final_decay(200)
    tail = [opt.lr_at(step) for step in range(190, 201)]
    assert tail[0] == pytest.approx(0.01)
    assert all(later < earlier for earlier, later in itertools.pairwise(tail)), tail
    assert tail[-1] == 0.0
    assert opt.lr_at(195) == pytest.approx(0.005)


def test_a_final_decay_of_zero_leaves_the_rate_flat() -> None:
    param = torch.nn.Parameter(torch.zeros(4))
    opt = SFAdamW([param], lr=0.01, final_decay_frac=0.0)
    opt.begin_final_decay(200)
    assert {opt.lr_at(step) for step in range(1, 201)} == {0.01}


def test_a_final_decay_is_refused_before_a_horizon_and_behind_one() -> None:
    """The class validates the fraction; the horizon cannot be in the past."""
    with pytest.raises(LoudError, match="outside"):
        SFAdamW([torch.nn.Parameter(torch.zeros(2))], final_decay_frac=1.0)
    opt = SFAdamW([torch.nn.Parameter(torch.zeros(2))], final_decay_frac=0.05)
    with pytest.raises(LoudError, match="behind"):
        opt.begin_final_decay(0)


def test_a_final_decay_on_a_scheduled_optimizer_is_refused() -> None:
    with pytest.raises(LoudError, match="scheduled optimizer"):
        MixOptimizer(reference_stack(**SMALL)).begin_final_decay(100)


# --------------------------------------------------------------------- the wiring


def test_set_lr_is_refused_on_a_schedule_free_optimizer() -> None:
    """A schedule on top of SF applies its decay twice: in the step and in the average."""
    with pytest.raises(LoudError, match="schedule-free"):
        MixOptimizer(reference_stack(**SMALL), schedule="sf").set_lr(0.5)


def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(LoudError, match="not a mode"):
        MixOptimizer(reference_stack(**SMALL), schedule="linear")


def test_the_sf_state_dict_round_trips_and_refuses_the_other_mode() -> None:
    """A resume at step 2 must continue; a resume from the other mode has no `z` to load."""
    torch.manual_seed(10)
    model = reference_stack(**SMALL)
    twin = reference_stack(**SMALL)
    twin.load_state_dict(model.state_dict())
    first, second = MixOptimizer(model, schedule="sf"), MixOptimizer(twin, schedule="sf")
    ids = torch.randint(0, SMALL["vocab"], (2, 9))
    _backward(model, ids)
    first.step()
    twin.load_state_dict(model.state_dict())
    second.load_state_dict(first.state_dict())
    model.zero_grad(set_to_none=True)
    twin.zero_grad(set_to_none=True)
    _backward(model, ids)
    _backward(twin, ids)
    first.step()
    second.step()
    twins = dict(twin.named_parameters())
    for name, param in model.named_parameters():
        assert torch.allclose(param, twins[name]), name
    with pytest.raises(LoudError, match="checkpoint is"):
        MixOptimizer(reference_stack(**SMALL)).load_state_dict(first.state_dict())


def test_the_arms_refuse_a_momentum_or_a_chain_that_does_nothing() -> None:
    with pytest.raises(LoudError, match="momentum"):
        SFNorMuon([torch.nn.Parameter(torch.zeros(2, 2))], momentum=1.0)
    with pytest.raises(LoudError, match="ns steps"):
        SFNorMuon([torch.nn.Parameter(torch.zeros(2, 2))], steps=0)
    with pytest.raises(LoudError, match="beta1"):
        SFAdamW([torch.nn.Parameter(torch.zeros(2))], betas=(0.0, 0.95))
    with pytest.raises(LoudError, match="warmup"):
        SFAdamW([torch.nn.Parameter(torch.zeros(2))], warmup=-1)
    with pytest.raises(LoudError, match="the first step"):
        SFAdamW([torch.nn.Parameter(torch.zeros(2))]).lr_at(0)


def test_eval_weights_twice_is_refused() -> None:
    """The second swap would save the averaged weights as if they were the training ones."""
    param = torch.nn.Parameter(torch.zeros(4))
    opt = SFAdamW([param], lr=0.01)
    param.grad = torch.ones(4)
    opt.step()
    opt.eval_weights()
    try:
        with pytest.raises(LoudError, match="twice"):
            opt.eval_weights()
    finally:
        opt.train_weights()


def _backward(model: torch.nn.Module, ids: torch.Tensor) -> None:
    """The step's loss, without the step: CE against the one-token shift."""
    logits = model(ids[:, :-1])
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    ).backward()
