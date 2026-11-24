"""ReWY's frozen contract: external fp64 complex-step anchor, counters and arity.

The anchor is the token-by-token KDA recurrence written independently in numpy and
differentiated by complex step. It is neither this module's forward twice nor an
autograd of it, so a shared algebra mistake cannot make both agree. The eager path
is the one every host can run; the fused schedule needs a card and is exercised
through its refusals here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.kernels import rewy


def _numpy_recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """eq. 1 token by token; complex128 input makes complex-step differentiate it."""
    seq, width = q.shape
    state = np.zeros((width, width), dtype=q.dtype)
    out = np.empty((seq, width), dtype=q.dtype)
    for t in range(seq):
        decayed = state * alpha[t][:, None]
        state = decayed - beta[t] * np.outer(k[t], k[t] @ decayed)
        state = state + beta[t] * np.outer(k[t], v[t])
        out[t] = state.T @ q[t]
    return out, state


def _inputs(seed: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, 2, 8, 3, dtype=dtype) for _ in range(3))
    alpha = torch.rand(1, 2, 8, 3, dtype=dtype) * 0.3 + 0.6
    beta = torch.rand(1, 2, 8, dtype=dtype)
    return q, k, v, alpha, beta


def _holomorphic_loss(value: np.ndarray, state: np.ndarray) -> complex:
    """No conjugation, so the complex-step derivative is analytic."""
    return np.sum(value**2) + 0.1 * np.sum(state**2)


def _complex_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    step: float = 1e-20,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Derivatives of the numpy recurrence by complex step, machine-precision accurate."""
    original = (q, k, v, alpha, beta)
    grads: list[np.ndarray] = []
    for slot in range(5):
        grad = np.zeros_like(original[slot], dtype=np.float64)
        for index in np.ndindex(original[slot].shape):
            perturbed = [a.astype(np.complex128, copy=True) for a in original]
            perturbed[slot][index] += 1j * step
            value, state = _numpy_recurrence(*perturbed)
            grad[index] = _holomorphic_loss(value, state).imag / step
        grads.append(grad)
    want_value, want_state = _numpy_recurrence(*original)
    return tuple(grads), want_value, want_state


def _relative(got: torch.Tensor, want: np.ndarray) -> float:
    a, w = got.detach().numpy(), want
    return float(np.abs(a - w).max() / max(np.abs(w).max(), 1e-30))


def test_gradients_match_external_numpy_complex_step() -> None:
    """Write, final state and all five grads against the independent fp64 recurrence."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        q, k, v, alpha, beta = _inputs(101)
        args = [t.requires_grad_(True) for t in (q, k, v, alpha, beta)]
        out, state = rewy.kda_rewy(*args, chunk=4)
        (out.pow(2).sum() + state.pow(2).sum() * 0.1).backward()
        want_grads, want_value, want_state = _complex_step(
            *(t.detach()[0, 0].numpy() for t in (q, k, v, alpha, beta))
        )
        assert _relative(out[0, 0], want_value) <= 1e-6
        assert _relative(state[0, 0], want_state) <= 1e-6
        for name, tensor, want in zip(
            ("dq", "dk", "dv", "dalpha", "dbeta"), args, want_grads, strict=True
        ):
            assert _relative(tensor.grad[0, 0], want) <= 1e-6, name
            assert bool(torch.isfinite(tensor.grad).all()), name
    finally:
        torch.set_num_threads(previous)


def test_a_second_seed_and_two_chunks_hold_the_anchor() -> None:
    """Two chunks, a different head/dim and a different seed still fit 1e-6."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(202)
        q, k, v = (torch.randn(1, 1, 12, 2, dtype=torch.float64) for _ in range(3))
        alpha = torch.rand(1, 1, 12, 2, dtype=torch.float64) * 0.3 + 0.6
        beta = torch.rand(1, 1, 12, dtype=torch.float64)
        args = [t.requires_grad_(True) for t in (q, k, v, alpha, beta)]
        out, state = rewy.kda_rewy(*args, chunk=4)
        (out.pow(2).sum() + state.pow(2).sum() * 0.1).backward()
        want_grads, _, _ = _complex_step(
            *(t.detach()[0, 0].numpy() for t in (q, k, v, alpha, beta))
        )
        for tensor, want in zip(args, want_grads, strict=True):
            assert _relative(tensor.grad[0, 0], want) <= 1e-6
    finally:
        torch.set_num_threads(previous)


def test_twenty_steps_are_deterministic_and_both_counters_fire() -> None:
    """Same seed twice bitwise; fwd and bwd counters advance 20 times together."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        def run(seed: int) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], Counters]:
            counters = Counters()
            counters.enable("kda", counter=rewy.BWD_COUNTER, why="backward is the proof")
            losses: list[torch.Tensor] = []
            grads: list[list[torch.Tensor]] = []
            for step in range(20):
                args = [t.requires_grad_(True) for t in _inputs(seed + step)]
                out, state = rewy.kda_rewy(*args, chunk=4, counters=counters)
                loss = out.pow(2).mean() + state.pow(2).mean() * 0.1
                loss.backward()
                losses.append(loss.detach())
                grads.append([t.grad.clone() for t in args])
            return losses, grads, counters

        first, first_grads, counters = run(23)
        second, second_grads, _ = run(23)
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
        assert all(
            torch.equal(a, b)
            for x, y in zip(first_grads, second_grads, strict=True)
            for a, b in zip(x, y, strict=True)
        )
        assert all(bool(torch.isfinite(loss)) for loss in first)
        assert counters.counts[rewy.FWD_COUNTER] == 20 * 8
        assert counters.counts[rewy.BWD_COUNTER] == counters.counts[rewy.FWD_COUNTER] > 0
        counters.check()
    finally:
        torch.set_num_threads(previous)


def test_backward_returns_one_gradient_per_schema_input() -> None:
    """A custom-op backward returns exactly one grad per input, not an extra None."""

    class FakeCtx:
        needs_input_grad = (True,) * 5 + (False, False)
        saved_tensors: tuple[torch.Tensor, ...] = ()

        def save_for_backward(self, *tensors: torch.Tensor) -> None:
            self.saved_tensors = tensors

    ctx = FakeCtx()
    args = [t.requires_grad_(True) for t in _inputs(31)]
    out, state = rewy.KDAReWY.forward(ctx, *args, 4, None)
    grads = rewy.KDAReWY.backward(ctx, torch.ones_like(out), torch.ones_like(state))
    arity = len(inspect.signature(rewy.KDAReWY.forward).parameters) - 1
    assert len(grads) == arity == 7
    for grad, tensor in zip(grads[:5], args, strict=True):
        assert grad is not None and grad.shape == tensor.shape
        assert bool(torch.isfinite(grad).all())
    assert grads[5:] == (None, None)


def test_gradcheck_against_the_forward() -> None:
    """Numerical differentiation agrees with the custom op's backward."""
    args = [t.requires_grad_(True) for t in _inputs(37)]
    assert torch.autograd.gradcheck(
        lambda q, k, v, alpha, beta: rewy.KDAReWY.apply(q, k, v, alpha, beta, 4, None),
        args,
    )


def test_a_forward_without_a_backward_is_a_missing_report() -> None:
    """fwd>0 with bwd==0 is the 3126-forwards/0-backwards defect, made loud."""
    counters = Counters()
    counters.enable("kda", counter=rewy.BWD_COUNTER)
    rewy.kda_rewy(*_inputs(41), chunk=4, counters=counters)
    assert counters.counts[rewy.FWD_COUNTER] == 8
    with pytest.raises(MissingReport):
        counters.check()


def test_three_products_replace_seven() -> None:
    """The reduced graph is the one the custom op actually runs."""
    assert rewy.matmul_count() == rewy.REWY_PRODUCTS == 3
    assert rewy.matmul_count("expanded") == rewy.EXPANDED_PRODUCTS == 7
    assert rewy.MAX_SMEM == 101376
    assert 0 <= rewy.smem_bytes() <= rewy.MAX_SMEM
    with pytest.raises(KeyError):
        rewy.matmul_count("bogus")


def test_both_local_pullbacks_agree_on_identical_tensors() -> None:
    """3-product ReWY and 7-product expanded graphs return the same three tensors."""
    q, k, v, alpha, beta = _inputs(53)
    _, _, cache = rewy._forward(q, k, v, alpha, beta, 4)
    correction_bar = torch.randn(1, 2, 4, 3, dtype=q.dtype)
    for index in range(2):
        reduced = rewy.backward_local(cache, index, correction_bar, mode="rewy")
        expanded = rewy.backward_local(cache, index, correction_bar, mode="expanded")
        for got, want in zip(reduced, expanded, strict=True):
            np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12, rtol=1e-12)
    with pytest.raises(LoudError, match="unknown ReWY pullback mode"):
        rewy.backward_local(cache, 0, correction_bar, mode="bogus")


def test_inputs_are_validated_at_the_kernel_boundary() -> None:
    """Shape, dtype, device and tiling mistakes are refused by name, not silently."""
    q, k, v, alpha, beta = _inputs(43)
    with pytest.raises(LoudError, match="does not tile seq=8"):
        rewy.kda_rewy(q, k, v, alpha, beta, chunk=3)
    with pytest.raises(LoudError, match="expects q/k/v/alpha"):
        rewy.kda_rewy(q[0], k, v, alpha, beta, chunk=4)
    with pytest.raises(LoudError, match="inputs disagree on shape"):
        rewy.kda_rewy(q, k[..., :2], v, alpha, beta, chunk=4)
    with pytest.raises(LoudError, match="inputs disagree on dtype"):
        rewy.kda_rewy(q, k.float(), v, alpha, beta, chunk=4)
    with pytest.raises(LoudError, match="unknown ReWY fused mode"):
        rewy.set_fused("bogus")
    rewy.set_fused("auto")


def test_no_requested_grads_returns_only_nones() -> None:
    """A backward nobody asked for does no derivative work and returns seven Nones."""

    class FakeCtx:
        needs_input_grad = (False,) * 5 + (False, False)
        saved_tensors: tuple[torch.Tensor, ...] = ()

        def save_for_backward(self, *tensors: torch.Tensor) -> None:
            self.saved_tensors = tensors

    ctx = FakeCtx()
    args = _inputs(59)
    out, state = rewy.KDAReWY.forward(ctx, *args, 4, None)
    assert rewy.KDAReWY.backward(ctx, torch.ones_like(out), torch.ones_like(state)) == (None,) * 7
    # The reverse pass also accepts an absent upstream gradient from either output.
    gradients = rewy._backward(ctx.cache, None, None, args)
    assert len(gradients) == 5
    assert all(bool(torch.isfinite(g).all()) for g in gradients)


def test_fused_schedule_is_a_bool_that_tracks_the_host() -> None:
    """The fused path is card-only: off without a card, and still honest with one."""
    previous = rewy._MODE
    try:
        rewy.set_fused("off")
        assert rewy.fused_available() is False
        rewy.set_fused("auto")
        assert isinstance(rewy.fused_available(), bool)
        if not torch.cuda.is_available():
            assert rewy.fused_available() is False
    finally:
        rewy.set_fused(previous)


def test_a_later_token_cannot_change_an_earlier_write() -> None:
    """Causality: moving a token's value leaves every earlier write bitwise alone."""
    q, k, v, alpha, beta = _inputs(47)
    first, _ = rewy.kda_rewy(q, k, v, alpha, beta, chunk=4)
    moved = v.clone()
    moved[..., 6, :] += 1.0
    second, _ = rewy.kda_rewy(q, k, moved, alpha, beta, chunk=4)
    assert torch.equal(first[..., :6, :], second[..., :6, :])
    assert not torch.equal(first[..., 6, :], second[..., 6, :])
