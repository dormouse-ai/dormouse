"""LightMTP-S: the budget, the stop-gradient, causality, the curve, and the collapse.

Five gates, each one a way the objective can be wired and quietly not work:

* the head costs at most 1% of the trunk, and the number is printed rather than assumed;
* with the head attached the token loss is *bit-identical* — `torch.equal` on the logits
  and `==` on the reported loss — so an A/B measures the objective and not a rounding;
* the target carries no gradient (stop-gradient), and the prediction at `t` cannot see
  anything after `t`, which is what makes the target the only channel to the future;
* 200 CPU steps on a period-`k` sequence drop the aux loss by ≥30% with the token loss
  finite — a head that cannot fit a trivial periodicity is wired wrong;
* VarReg collapses the target's shared direction, measured; the paper's cost for that is
  quoted in `train/aux.py`, and nothing here may claim it is free.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.model import Transformer
from dormouse.model.spec import StackSpec
from dormouse.train.aux import LightMTP, varreg
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

#: A stack small enough for a CPU test and deep enough that a 1%-of-trunk head exists:
#: at 4 layers and d=32 the budget caps the head's hidden width at 6 (0.92%).
TINY = {"vocab": 16, "d_model": 32, "n_layers": 4, "n_heads": 2, "head_dim": 16, "d_ffn": 64}

#: The head width TINY can afford; `d_model//2` is 16 and would be 3.3% of this trunk.
WIDTH = 6


def _tiny(*, seed: int = 23) -> Transformer:
    """A seeded 4-layer stack, positional-free so a periodic target stays predictable."""
    torch.manual_seed(seed)
    return Transformer(**TINY, max_seq_len=32, rope=False)


def _ids(seq: int = 17, batch: int = 3) -> torch.Tensor:
    """One fixed batch of token ids in the tiny stack's vocabulary."""
    return torch.randint(0, 16, (batch, seq), generator=torch.Generator().manual_seed(7))


# ------------------------------------------------------------------------ the budget


def test_the_head_is_under_a_percent_of_the_trunk_and_prints_its_size() -> None:
    """The A/B's own config: 692,224 parameters, +0.73% of a 94.8M trunk."""
    spec = StackSpec.from_toml(CONFIGS / "lightmtp.toml")
    assembled = assemble(spec, device="meta")
    head = assembled.aux
    assert head is not None, "aux=lightmtp must build a head"
    trunk = assembled.report()["stored"] - head.head_params
    print(f"\nlightmtp: head={head.head_params:,} of trunk={trunk:,} ({head.fraction:.2%})")
    assert head.head_params == 2 * spec.d_model * (spec.d_model // 2)
    assert head.fraction <= 0.01
    # The paper's d→d→d head is 1.2M on a 286.3M trunk (+0.4%); ours is d→d/2 on 94.8M.
    assert 0.004 <= head.fraction <= 0.008
    line = next(text for text in plan_lines(spec) if text.startswith("aux: lightmtp"))
    assert f"head={head.head_params:,}" in line and f"+{head.fraction:.2%}" in line


def test_a_head_over_a_percent_is_refused_with_both_numbers() -> None:
    """`width=d_model` is the paper's own form, and unaffordable on this trunk."""
    model = _tiny()
    with pytest.raises(LoudError) as caught:
        LightMTP(model, k=4, width=model.d_model)
    message = str(caught.value)
    assert "2,048" in message and "at most 1%" in message
    assert "width=d_model" in message


def test_the_head_width_has_a_floor() -> None:
    with pytest.raises(LoudError, match="projects into no dimension"):
        LightMTP(_tiny(), k=4, width=0)


# ------------------------------------------------------------- the flag-off identity


def test_the_token_loss_and_the_logits_are_bit_identical_with_the_head_attached() -> None:
    """The gate: `torch.equal` on the logits, `==` on the loss, head on or off."""
    plain, with_aux = _tiny(), _tiny()
    with_aux.load_state_dict(plain.state_dict())
    aux = LightMTP(with_aux, k=4, width=WIDTH)
    aux.attach(with_aux)
    ids = _ids()
    seen: list[torch.Tensor] = []
    handle = plain.register_forward_hook(lambda _m, _i, out: seen.append(out))
    unauxed = train_step(plain, ids, optimizer=MixOptimizer(plain))
    handle.remove()
    handle = with_aux.register_forward_hook(lambda _m, _i, out: seen.append(out))
    auxed = train_step(with_aux, ids, optimizer=MixOptimizer(with_aux), aux=aux)
    handle.remove()
    assert torch.equal(seen[0], seen[1]), "the head changed the logits"
    assert unauxed.loss == auxed.loss, "the head changed the reported token loss"
    assert unauxed.aux_loss is None
    assert auxed.aux_loss is not None


def test_without_the_head_nothing_is_counted_and_no_aux_is_reported() -> None:
    model = _tiny()
    counters = Counters()
    result = train_step(model, _ids(), optimizer=MixOptimizer(model), counters=counters)
    assert result.aux_loss is None
    assert "aux_fwd" not in counters.counts and "aux" not in counters.enabled


# ----------------------------------------------------------- stop-gradient, causality


def test_the_target_carries_no_gradient() -> None:
    """Eq. 1's `sg(·)`: gradient into the head's input, none into the target."""
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    source = torch.randn(2, 9, 32, requires_grad=True)
    target = torch.randn(2, 9, 32, requires_grad=True)
    loss = aux.objective(source, target)
    loss.backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert target.grad is None, "the target path let gradient through"
    with pytest.raises(RuntimeError):  # not in the graph at all
        torch.autograd.grad(loss, target)


def test_the_prediction_at_t_cannot_see_past_t_and_a_corrupt_target_only_moves_the_loss() -> None:
    """The shift is `source[:, :-k]` against `target[:, k:]`, and nothing else."""
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    source = torch.randn(2, 9, 32, requires_grad=True)
    clean = torch.randn(2, 9, 32)
    corrupt = clean.clone()
    corrupt[:, 4:] = torch.randn(2, 5, 32)
    before = aux.objective(source, clean).detach()
    after = aux.objective(source, corrupt).detach()
    assert not torch.isclose(before, after), "corrupting the target left the loss unchanged"
    # The equation, spelled out here, including the direction of the shift: the prediction
    # at t against position t+k, never t-k or t itself.
    predicted = torch.nn.functional.normalize(aux.head(source[:, : -aux.k]).detach(), dim=-1)
    wanted = torch.nn.functional.normalize(clean[:, aux.k :], dim=-1)
    expected = 1.0 - (predicted * wanted).sum(dim=-1).mean()
    assert torch.allclose(before, expected, atol=1e-6), (float(before), float(expected))
    prediction = aux.head(source[:, : -aux.k]).detach()
    assert torch.equal(aux.head(source[:, : -aux.k]).detach(), prediction)
    aux.objective(source, clean).backward()
    assert source.grad is not None
    assert source.grad[:, -aux.k :].abs().sum() == 0, "the future leaked into the prediction"
    assert source.grad[:, : -aux.k].abs().sum() > 0


# ------------------------------------------------------------------------ it learns


def test_two_hundred_cpu_steps_fit_a_period_k_pattern() -> None:
    """A period-4 sequence, k=4, 200 steps: the aux loss must fall by a third at least.

    Measured here (fp32, one CPU thread's worth of work): aux 0.99 → 0.01, token loss
    31.4 → 0.06, in 5 s. The floor is 30% because the curve is the gate's subject and
    this is the failure it exists for — a head wired to the wrong tensor or a target
    that never shifts still reports a loss, and that loss does not fall.
    """
    torch.manual_seed(11)
    model = Transformer(**TINY, max_seq_len=32, rope=False)
    aux = LightMTP(model, k=4, width=WIDTH)
    aux.attach(model)
    optimizer = MixOptimizer(model)
    ids = torch.arange(33).remainder(4)[None, :].repeat(2, 1)
    curve = [train_step(model, ids, optimizer=optimizer, aux=aux) for _ in range(200)]
    aux_curve = [result.aux_loss for result in curve]
    tokens = [result.loss for result in curve]
    assert all(value is not None for value in aux_curve)
    first, last = aux_curve[0], aux_curve[-1]
    assert first is not None and last is not None
    print(f"\naux {first:.4f} → {last:.4f}; ntp {tokens[0]:.4f} → {tokens[-1]:.4f}")
    assert last < 0.7 * first, f"the aux loss did not learn: {first} → {last}"
    assert all(torch.isfinite(torch.tensor(value)) for value in tokens)


# ------------------------------------------------------------------- the collapse


def test_varreg_collapses_the_shared_direction_it_penalises() -> None:
    """Eq. 2 driven to its floor: `‖mean_t norm(h)‖` → 0, per-position norms stay 1.

    This is the collapse to know about before enabling it: every position is pushed to
    be orthogonal to every other, so the target layer carries no shared direction. The
    per-position norms cannot collapse — the normalisation pins them at 1 — which is why
    the paper's measured damage is a language-modelling regression (Table 1) and not a
    divergence, and why `aux_varreg` defaults to false.
    """
    torch.manual_seed(3)
    h = (torch.ones(2, 16, 8) + 0.05 * torch.randn(2, 16, 8)).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=0.05)
    start = float(varreg(h.detach()))
    for _ in range(100):
        optimizer.zero_grad()
        loss = varreg(h)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    end = float(varreg(h.detach()))
    unit = torch.nn.functional.normalize(h.detach(), dim=-1)
    similar = unit @ unit.transpose(1, 2)
    off = similar[:, ~torch.eye(16, dtype=torch.bool)].mean()
    print(f"\nvarreg: {start:.4f} → {end:.4f}, cross-position cosine {float(off):.4f}")
    assert start > 0.95, start
    assert end < 0.1, f"the shared direction did not collapse: {end}"
    # The penalty is scale-invariant (it reads `norm(h)`), so the raw vectors' norms drift
    # with the optimizer's step; what is asserted is the direction: positions end up
    # mutually orthogonal, which is 1 - information the target carries across positions.
    assert abs(float(off)) < 0.1, off
    assert bool(torch.isfinite(h).all()), "a collapse, not a divergence"


# ------------------------------------------------------------------ the counters, the log


def test_the_aux_counters_and_its_own_log_line() -> None:
    """`aux_fwd` proves the head ran; `aux_loss` proves the term was added; both logged."""
    model = _tiny()
    aux = LightMTP(model, k=4, width=WIDTH)
    aux.attach(model)
    counters = Counters()
    ids = _ids()
    result = train_step(model, ids, optimizer=MixOptimizer(model), counters=counters, aux=aux)
    assert counters.silent() == []
    assert counters.counts["aux_fwd"] == ids.shape[0] * (ids.shape[1] - 1 - aux.k)
    assert counters.counts["aux_loss"] == 1
    assert result.aux_loss is not None and result.aux_loss >= 0.0
    assert counters.counts["bwd"] > 0


def test_the_step_logs_the_aux_loss_separately_from_the_token_loss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One line for the aux value, and the token loss is not folded into it."""
    model = _tiny()
    aux = LightMTP(model, k=4, width=WIDTH)
    aux.attach(model)
    with caplog.at_level(logging.INFO, logger="dormouse.train.step"):
        result = train_step(model, _ids(), optimizer=MixOptimizer(model), aux=aux)
    logged = [record.getMessage() for record in caplog.records]
    aux_lines = [line for line in logged if line.startswith("aux lightmtp loss=")]
    assert len(aux_lines) == 1
    assert f"loss={result.aux_loss:.6f}" in aux_lines[0]
    assert "weighted=" in aux_lines[0]


def test_varreg_adds_its_own_weighted_term() -> None:
    """Eq. 3 with `λ_var = 0.1`: the loss is exactly `L_MTP + 0.1·L_VarReg`."""
    model = _tiny()
    source, target = torch.randn(2, 9, 32), torch.randn(2, 9, 32)
    aux = LightMTP(model, k=4, width=WIDTH)
    plain = aux.objective(source, target)
    aux.varreg = True
    regularised = aux.objective(source, target)
    expected = plain + 0.1 * varreg(target)
    assert torch.allclose(regularised, expected, atol=1e-6)


def test_both_objects_print_the_numbers_a_log_line_repeats() -> None:
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    assert "width=6" in repr(aux.head)
    assert "k=4" in repr(aux) and "layer=4" in repr(aux)
    assert f"params={aux.head_params}" in repr(aux)


# ------------------------------------------------------------------- the firewall


def test_a_nonfinite_aux_loss_skips_the_step_like_any_other_poisoned_batch() -> None:
    """The aux term is added to the loss, so it can poison the weights on its own."""
    model = _tiny()
    aux = LightMTP(model, k=4, width=WIDTH)
    aux.attach(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    poison = aux.head.register_forward_hook(lambda _m, _i, out: out * float("nan"))
    try:
        counters = Counters()
        result = train_step(
            model, _ids(), optimizer=MixOptimizer(model), counters=counters, aux=aux
        )
    finally:
        poison.remove()
    assert result.skipped and result.aux_loss is None
    assert counters.counts["nan_guard"] == 1
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name


# ------------------------------------------------------------------- the wiring


def test_a_config_with_the_objective_trains_and_every_mechanism_reports() -> None:
    """From config to three steps: the head is attached, counted, and the loss is finite."""
    spec = StackSpec.from_toml(CONFIGS / "lightmtp.toml")
    spec = StackSpec.from_mapping(
        {
            **spec.as_dict(),
            "d_model": 64,
            "n_heads": 4,
            "head_dim": 16,
            "d_ffn": 128,
            "seq_len": 64,
            "batch": 2,
            "tokens": 1_000_000,
        }
    )
    assembled = assemble(spec)
    assert assembled.aux is not None
    assert assembled.model.aux is assembled.aux
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    for step in range(3):
        result, counters = assembled.step(ids, optimizer=optimizer)
        assert not result.skipped, f"step {step} tripped the NaN firewall"
        assert result.aux_loss is not None
        # `Assembled.step` opens its own counters per step, so each one is checked alone.
        assert counters.counts["aux_fwd"] > 0 and counters.counts["aux_loss"] == 1
        assert counters.silent() == []


# ------------------------------------------------------------------- the refusals


def test_a_horizon_of_zero_is_refused() -> None:
    with pytest.raises(LoudError, match="zero positions ahead"):
        LightMTP(_tiny(), k=0, width=WIDTH)


def test_a_weight_of_zero_is_refused() -> None:
    with pytest.raises(LoudError, match="zero \\(or reversed\\) gradient"):
        LightMTP(_tiny(), k=4, weight=0.0, width=WIDTH)


@pytest.mark.parametrize("layer", [5, -1])
def test_a_layer_that_is_not_a_block_is_refused(layer: int) -> None:
    with pytest.raises(LoudError, match="not a block of this 4-layer stack"):
        LightMTP(_tiny(), k=4, layer=layer, width=WIDTH)


def test_one_aux_slot_per_model() -> None:
    model = _tiny()
    LightMTP(model, k=4, width=WIDTH).attach(model)
    with pytest.raises(LoudError, match="already carries an aux head"):
        LightMTP(model, k=4, width=WIDTH).attach(model)


def test_a_forward_without_the_hook_is_refused_instead_of_reporting_zero() -> None:
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    with pytest.raises(LoudError, match="no hidden state was captured"):
        aux.loss({})


def test_two_states_of_different_shapes_are_refused() -> None:
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    with pytest.raises(LoudError, match="two hidden states"):
        aux.objective(torch.randn(2, 9, 32), torch.randn(2, 8, 32))
    with pytest.raises(LoudError, match="two hidden states"):
        aux.objective(torch.randn(18, 32), torch.randn(18, 32))


def test_a_window_with_no_room_for_the_horizon_is_refused() -> None:
    aux = LightMTP(_tiny(), k=4, width=WIDTH)
    with pytest.raises(LoudError, match="has no position 4 ahead"):
        aux.objective(torch.randn(2, 4, 32), torch.randn(2, 4, 32))


def test_an_unattached_head_is_refused_by_the_step() -> None:
    model = _tiny()
    aux = LightMTP(model, k=4, width=WIDTH)
    with pytest.raises(LoudError, match="not attached to this model"):
        train_step(model, _ids(), optimizer=MixOptimizer(model), aux=aux)
