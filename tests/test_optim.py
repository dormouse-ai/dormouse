"""The two update rules, the router, and the schedule — each against a closed form.

Every number below is computed outside this repository or from the formula in the
module's docstring: `sign(g)` for AdamW's first step, the docstring's recurrence for
its third, `√rows` for Muon's step norm, `‖QQᵀ - I‖` for the polar factor, and the
docstring's two closed forms for the schedule. No value is recorded from a run of the
code under test, so a wrong implementation cannot agree with itself.
"""

from __future__ import annotations

import math

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model import Transformer
from dormouse.train.optim import (
    COLROW_EPS,
    MUON_LR,
    AdamW,
    MixOptimizer,
    Muon,
    colrow,
    muon_direction,
    polar,
    route,
    route_all,
)
from dormouse.train.schedule import warmup_cosine, warmup_stable_decay

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}


# --------------------------------------------------------------------------- anchors


def test_adamw_first_step_is_lr_times_the_sign_of_the_gradient() -> None:
    """Bias correction makes step 1 `m̂/√v̂ = g/|g|`, so the step is `η·sign(g)`."""
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(64))
    start = param.detach().clone()
    grad = torch.randn(64)
    param.grad = grad.clone()
    AdamW([param], lr=3e-3, weight_decay=0.0).step()
    step = (param.detach() - start) / 3e-3
    assert torch.allclose(step, -torch.sign(grad), atol=1e-5)


def test_adamw_matches_its_documented_recurrence_after_three_steps() -> None:
    """The docstring's formula, written again here in fp64, step by step."""
    lr, beta1, beta2, eps, decay = 0.01, 0.9, 0.95, 1e-8, 0.1
    torch.manual_seed(1)
    param = torch.nn.Parameter(torch.randn(8, dtype=torch.float64))
    grads = [torch.randn(8, dtype=torch.float64) for _ in range(3)]
    opt = AdamW([param], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay)
    weight, m, v = param.detach().clone(), torch.zeros(8), torch.zeros(8)
    for t, grad in enumerate(grads, start=1):
        param.grad = grad.clone()
        opt.step()
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        weight = weight * (1 - lr * decay) - lr * m_hat / (v_hat.sqrt() + eps)
        assert torch.allclose(param.detach(), weight, atol=1e-12), f"step {t}"


def test_adamw_decay_is_decoupled_from_the_gradient() -> None:
    """With a zero gradient the weight still shrinks: `w ← w·(1 - η·λ)`."""
    param = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([param], lr=0.1, weight_decay=0.5)
    param.grad = torch.zeros(4)
    opt.step()
    assert torch.allclose(param.detach(), torch.full((4,), 1.0 - 0.1 * 0.5))


def test_ns_orthogonalises_the_shapes_the_model_actually_has() -> None:
    """`‖QQᵀ - I‖∞ ≤ 1e-5` on the live rectangles: qkv head, ffn up, ffn gate."""
    for shape in ((64, 832), (832, 2048), (2048, 832)):
        torch.manual_seed(2)
        q = polar(torch.randn(*shape))
        gram = q @ q.T if shape[0] <= shape[1] else q.T @ q
        error = (gram - torch.eye(gram.shape[0])).abs().max().item()
        assert error <= 1e-5, f"{shape}: {error:.2e}"


def test_ns_is_orthogonal_to_1e_5_when_the_spectrum_is_controlled() -> None:
    """A square matrix with σ in [0.8, 1.2]: the quintic's precision, not conditioning."""
    torch.manual_seed(3)
    left, _ = torch.linalg.qr(torch.randn(256, 256))
    right, _ = torch.linalg.qr(torch.randn(256, 256))
    spectrum = torch.linspace(0.8, 1.2, 256)
    q = polar(left @ torch.diag(spectrum) @ right.T)
    assert (q @ q.T - torch.eye(256)).abs().max().item() <= 1e-5


def test_five_iterations_cannot_lift_an_ill_conditioned_spectrum() -> None:
    """The ceiling §3 records, pinned so nobody widens the anchor to cover it.

    A square gaussian has σ_min/σ_max ≈ 0.006 and the chain lifts up, not sideways.
    """
    torch.manual_seed(4)
    g = torch.randn(64, 64)
    q = polar(g)
    assert (q @ q.T - torch.eye(64)).abs().max().item() > 1e-2


def test_the_batched_chain_is_the_per_matrix_chain() -> None:
    """A `[N, m, n]` state must not move a value relative to `N` separate calls."""
    torch.manual_seed(5)
    stacked = torch.randn(6, 64, 128)
    one_by_one = torch.stack([polar(stacked[i]) for i in range(stacked.shape[0])])
    assert (polar(stacked) - one_by_one).abs().max().item() <= 1e-6


def test_muon_step_norm_is_sqrt_rows_times_lr() -> None:
    """ColRow leaves every row unit-norm, so `‖ΔW‖_F = η·√rows·max(1, m/n)^½`."""
    for shape in ((64, 832), (832, 64), (128, 128)):
        torch.manual_seed(6)
        param = torch.nn.Parameter(torch.randn(*shape))
        start = param.detach().clone()
        param.grad = torch.randn(*shape)
        Muon([param], lr=MUON_LR).step()
        rows, cols = shape
        want = MUON_LR * rows**0.5 * max(1.0, rows / cols) ** 0.5
        got = (param.detach() - start).norm().item()
        assert got == pytest.approx(want, rel=1e-5), f"{shape}: {got} vs {want}"


def test_muon_direction_does_not_depend_on_the_gradients_scale() -> None:
    """Every stage of the chain is homogeneous of degree 0: 1000·g gives the same step."""
    torch.manual_seed(7)
    g = torch.randn(64, 256)
    assert torch.allclose(muon_direction(g), muon_direction(1000.0 * g), atol=1e-5)


def test_headwise_muon_changes_the_direction_and_not_the_norm() -> None:
    """§5: per-head is a direction arm at fixed step magnitude, so no LR retune."""
    torch.manual_seed(8)
    d_model, n_heads = 832, 13
    packed = torch.randn(3 * d_model, d_model)
    whole = muon_direction(packed)
    per_head = muon_direction(packed, head_split=(2 * n_heads, 2 * d_model))
    cosine = (whole * per_head).sum() / (whole.norm() * per_head.norm())
    assert per_head.norm().item() == pytest.approx(whole.norm().item(), rel=1e-4)
    assert cosine.item() < 0.999, "the per-head chain did not run"


def test_a_head_split_that_does_not_cut_equal_slices_is_refused() -> None:
    with pytest.raises(LoudError, match="equal slices"):
        muon_direction(torch.randn(96, 32), head_split=(5, 64))


# ------------------------------------------------------------------ the two rules


def test_colrow_makes_every_row_a_unit_vector() -> None:
    torch.manual_seed(9)
    out = colrow(torch.randn(16, 40))
    assert torch.allclose(out.norm(dim=1), torch.ones(16), atol=COLROW_EPS * 10)
    batched = colrow(torch.randn(4, 16, 40))
    assert torch.allclose(batched.norm(dim=2), torch.ones(4, 16), atol=COLROW_EPS * 10)


def test_muon_momentum_is_the_documented_ema() -> None:
    param = torch.nn.Parameter(torch.zeros(8, 8))
    param.grad = torch.ones(8, 8)
    opt = Muon([param], lr=0.01, momentum=0.95)
    opt.step()
    assert torch.allclose(opt.state[param]["momentum"], torch.full((8, 8), 0.05))


def test_muon_skips_a_parameter_without_a_gradient() -> None:
    frozen = torch.nn.Parameter(torch.ones(8, 8))
    opt = Muon([frozen], lr=0.01)
    opt.step()
    assert torch.equal(frozen.detach(), torch.ones(8, 8))
    assert not opt.state


def test_muon_applies_its_own_weight_decay() -> None:
    """`w ← w(1 - ηλ) - η·R`: a zero gradient leaves only the decay to move the weight."""
    param = torch.nn.Parameter(torch.ones(8, 8))
    param.grad = torch.zeros(8, 8)
    Muon([param], lr=0.1, weight_decay=0.5).step()
    assert torch.allclose(param.detach(), torch.full((8, 8), 1.0 - 0.1 * 0.5))


def test_muon_refuses_a_momentum_that_never_forgets() -> None:
    with pytest.raises(LoudError, match="momentum"):
        Muon([torch.nn.Parameter(torch.zeros(2, 2))], momentum=1.0)


def test_muon_refuses_zero_newton_schulz_iterations() -> None:
    with pytest.raises(LoudError, match="ns steps"):
        Muon([torch.nn.Parameter(torch.zeros(2, 2))], steps=0)


def test_muon_runs_a_closure() -> None:
    opt = Muon([torch.nn.Parameter(torch.zeros(4, 4))], lr=0.01)
    loss = opt.step(lambda: torch.tensor(2.5))
    assert loss is not None and float(loss) == 2.5


def test_polar_of_a_tiny_momentum_is_polar_of_a_large_one() -> None:
    """The defect this pins: the chain is scale-invariant, and a floor that bites
    breaks it. A momentum of 1e-5 has `‖Gv‖ ≈ 1e-10`, and an absolute floor at 1e-7
    in the estimate's normalisation made the prescale 1000× wrong — `polar` then ran
    past the quintic's basin to `inf`, which is a skipped training step.
    """
    torch.manual_seed(11)
    tiny = torch.randn(4, 16, 32) * 1e-5
    small = polar(tiny)
    large = polar(tiny * 1e6)
    assert bool(torch.isfinite(small).all())
    assert (small - large).abs().max().item() <= 1e-3


def test_the_spectral_estimate_is_within_ten_percent_on_the_live_shapes() -> None:
    """A gap of a factor 1.53 turns the open prescale into a divergent chain."""
    from dormouse.train.optim import spectral_bound

    for shape in ((64, 832), (832, 2048)):
        torch.manual_seed(12)
        g = torch.randn(*shape)
        assert spectral_bound(g).item() / torch.linalg.svdvals(g).max().item() >= 0.9


def test_polar_refuses_a_shape_that_is_not_a_matrix() -> None:
    with pytest.raises(LoudError, match="batch of matrices"):
        polar(torch.zeros(2, 3, 4, 5))


# ------------------------------------------------------------------------ routing


def test_every_parameter_of_the_stack_is_routed() -> None:
    table = route_all(Transformer(**SMALL))
    assert table["embed.weight"] == "adamw"
    assert table["norm.weight"] == "adamw"
    assert table["blocks.0.norm1.weight"] == "adamw"
    assert table["blocks.0.attn.qkv.weight"] == "muon"
    assert table["blocks.0.attn.proj.weight"] == "muon"
    assert table["blocks.0.ffn.down.weight"] == "muon"
    assert set(table.values()) == {"muon", "adamw"}


def test_a_shape_neither_rule_claims_is_refused() -> None:
    """The defect this router exists for: a tensor trained by a fallback nobody chose."""
    with pytest.raises(LoudError, match="neither arm claims"):
        route("engram.table", torch.zeros(4, 8, 16))


def test_the_packed_qkv_is_the_only_per_head_group() -> None:
    model = Transformer(**SMALL)
    mix = MixOptimizer(model)
    splits = {group["head_split"] for group in mix.muon.param_groups}
    assert splits == {None, (2 * SMALL["n_heads"], 2 * SMALL["d_model"])}
    assert mix.sizes == {"muon": 10, "adamw": 6}


def test_a_qkv_weight_without_a_head_count_is_refused() -> None:
    """A per-head split that cannot read `n_heads` must refuse, not silently degrade."""

    class NotAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = torch.nn.Linear(8, 24, bias=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = NotAttention()

    class Stack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    with pytest.raises(LoudError, match="no n_heads"):
        MixOptimizer(Stack())


def test_mixing_a_model_without_matrices_is_refused() -> None:
    class Vectors(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gain = torch.nn.Parameter(torch.ones(4))

    with pytest.raises(LoudError, match="routed to Muon"):
        MixOptimizer(Vectors())


def test_set_lr_scales_both_peak_rates() -> None:
    mix = MixOptimizer(Transformer(**SMALL), lr_muon=0.02, lr_adamw=0.003)
    mix.set_lr(0.5)
    assert mix.muon.param_groups[0]["lr"] == pytest.approx(0.01)
    assert mix.adamw.param_groups[0]["lr"] == pytest.approx(0.0015)


def test_a_negative_lr_fraction_is_refused() -> None:
    with pytest.raises(LoudError, match="negative"):
        MixOptimizer(Transformer(**SMALL)).set_lr(-0.1)


def test_route_sends_a_vector_and_the_token_table_to_adamw() -> None:
    assert route("norm.weight", torch.zeros(32)) == "adamw"
    assert route("embed.weight", torch.zeros(16, 32)) == "adamw"
    assert route("blocks.0.attn.proj.weight", torch.zeros(32, 32)) == "muon"


def test_state_dict_round_trips_between_two_optimizers() -> None:
    """A resume at step 1 must continue, not restart: the momentum is in the state."""
    torch.manual_seed(10)
    model = Transformer(**SMALL)
    twin = Transformer(**SMALL)
    twin.load_state_dict(model.state_dict())
    first, second = MixOptimizer(model), MixOptimizer(twin)
    ids = torch.randint(0, SMALL["vocab"], (2, 9))
    _backward(model, ids)
    first.step()
    # A resume is the weights AND the optimizer state; loading one without the other
    # is the bug this test would otherwise let through.
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


def _backward(model: torch.nn.Module, ids: torch.Tensor) -> None:
    """The step's loss, without the step: CE against the one-token shift."""
    logits = model(ids[:, :-1])
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    ).backward()


# ----------------------------------------------------------------------- schedule


def test_warmup_cosine_matches_its_closed_form_at_six_points() -> None:
    peak, floor, warmup, total = 1.0, 0.1, 10, 110
    for step in (0, 4, 9, 10, 60, 109, 110, 200):
        if step < warmup:
            want = peak * (step + 1) / warmup
        else:
            u = min(1.0, (step - warmup) / (total - warmup))
            want = floor + (peak - floor) / 2 * (1 + math.cos(math.pi * u))
        got = warmup_cosine(step, warmup=warmup, total=total, peak=peak, floor=floor)
        assert got == pytest.approx(want, rel=1e-12), step


def test_warmup_cosine_hits_its_three_named_values() -> None:
    """Step 0 is `peak/warmup` (not 0), the last warmup step is `peak`, the end is `floor`."""
    kwargs = {"warmup": 10, "total": 110, "peak": 1.0, "floor": 0.1}
    assert warmup_cosine(0, **kwargs) == pytest.approx(0.1)
    assert warmup_cosine(9, **kwargs) == pytest.approx(1.0)
    assert warmup_cosine(110, **kwargs) == pytest.approx(0.1)
    assert warmup_cosine(500, **kwargs) == pytest.approx(0.1)


def test_warmup_stable_decay_matches_its_closed_form_at_six_points() -> None:
    peak, floor, warmup, start, total = 1.0, 0.0, 5, 55, 105
    for step in (0, 4, 5, 30, 55, 80, 105, 300):
        if step < warmup:
            want = peak * (step + 1) / warmup
        elif step < start:
            want = peak
        else:
            u = min(1.0, (step - start) / (total - start))
            want = floor + (peak - floor) * (1 - math.sqrt(u))
        got = warmup_stable_decay(
            step, warmup=warmup, decay_start=start, total=total, peak=peak, floor=floor
        )
        assert got == pytest.approx(want, rel=1e-12), step


def test_warmup_stable_decay_is_constant_across_the_stable_stretch() -> None:
    rates = {
        warmup_stable_decay(step, warmup=5, decay_start=50, total=100, peak=2.0)
        for step in range(5, 50)
    }
    assert rates == {2.0}


def test_the_schedule_refuses_parameters_it_cannot_divide_by() -> None:
    good = {"warmup": 10, "total": 100, "peak": 1.0}
    with pytest.raises(LoudError, match="not a step"):
        warmup_cosine(-1, **good)
    with pytest.raises(LoudError, match="divides by zero"):
        warmup_cosine(0, warmup=0, total=100, peak=1.0)
    with pytest.raises(LoudError, match="not past warmup"):
        warmup_cosine(0, warmup=100, total=100, peak=1.0)
    with pytest.raises(LoudError, match="not a learning rate"):
        warmup_cosine(0, warmup=1, total=10, peak=0.0)
    with pytest.raises(LoudError, match="outside"):
        warmup_cosine(0, warmup=1, total=10, peak=1.0, floor=2.0)
    with pytest.raises(LoudError, match="decay_start"):
        warmup_stable_decay(0, warmup=10, decay_start=5, total=100, peak=1.0)
