"""KDA's four properties that a configuration test cannot reach.

(a) The chunkwise form is an optimisation of eq. 1, not a second definition of
    it: it must equal the token recurrence on the module's own projections, to
    machine precision in float64. A wrong Γ, a wrong triangular solve or a
    chunk-boundary mistake survives every shape test and dies here.
(b) The delta rule must actually be in the output: a plain linear attention fed
    the same projections differs, or the "delta" is a no-op.
(c) The gradient reaches EVERY parameter, and both counters fire. The project
    once ran 3126 KDA forwards and zero backwards; `kda_bwd` is the counter that
    makes that visible, and `MissingReport` is what turns it into a failure.
(d) The state is bounded: its norm must not grow with the window. Measured at
    256 / 1024 / 4096 tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.model.attn_variants.kda import KDA

D_MODEL = 32
N_HEADS = 2
HEAD_DIM = 8


def _kernel_args(kda: KDA, seq: int, *, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Project a batch through the module's own weights, in kernel layout."""
    torch.manual_seed(seed)
    x = torch.randn(2, seq, kda.d_model, dtype=kda.qkv.weight.dtype)
    q, k, v, alpha, beta = kda._project(x)
    return (*(z.transpose(1, 2) for z in (q, k, v, alpha)), beta.transpose(1, 2))


def test_the_chunk_path_is_eq_1_token_by_token() -> None:
    """Float64 equality with the recurrence, across several chunk boundaries."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=7).double()
    args = _kernel_args(kda, 35)
    fast, state = kda._chunkwise(*args)
    slow, slow_state = kda.recurrent(*args)
    assert torch.allclose(fast, slow, atol=1e-12, rtol=0)
    assert torch.allclose(state, slow_state, atol=1e-12, rtol=0)


def test_the_delta_rule_changes_the_answer() -> None:
    """Remove `I - βkkᵀ` and the output must move: the delta rule is not scenery."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8).double()
    q, k, v, alpha, beta = _kernel_args(kda, 16)
    delta, _ = kda._chunkwise(q, k, v, alpha, beta)
    state = q.new_zeros(*q.shape[:2], HEAD_DIM, HEAD_DIM)
    plain = []
    for t in range(q.shape[2]):
        state = state * alpha[:, :, t].unsqueeze(-1)
        state = state + beta[:, :, t].view(*q.shape[:2], 1, 1) * k[:, :, t].unsqueeze(-1) * v[
            :, :, t
        ].unsqueeze(-2)
        plain.append(torch.einsum("bhdv,bhd->bhv", state, q[:, :, t]))
    without = torch.stack(plain, dim=2)
    assert not torch.allclose(delta, without, atol=1e-6)
    assert (delta - without).abs().max() > 1e-3


def test_a_later_token_cannot_change_an_earlier_output() -> None:
    """The causal property, bit for bit, through the conv and the chunk carry."""
    torch.manual_seed(3)
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    x = torch.randn(1, 32, D_MODEL)
    y = kda(x)
    moved = x.clone()
    moved[:, 24] += 1.0
    assert torch.equal(y[:, :24], kda(moved)[:, :24])
    assert not torch.equal(y[:, 24], kda(moved)[:, 24])


def test_every_parameter_gets_a_gradient_and_both_counters_fire() -> None:
    """Three steps: non-zero grads everywhere, `kda_fwd` and `kda_bwd` non-zero."""
    torch.manual_seed(4)
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    counters = Counters()
    counters.enable("kda", counter="kda_bwd", why="the backward pass is the proof")
    trainable = torch.optim.AdamW(kda.parameters(), lr=1e-2)
    for _ in range(3):
        y = kda(torch.randn(2, 16, D_MODEL), counters)
        loss = y.pow(2).mean()
        trainable.zero_grad()
        loss.backward()
        trainable.step()
        assert torch.isfinite(loss)
    assert counters.counts["kda_fwd"] > 0
    assert counters.counts["kda_bwd"] == counters.counts["kda_fwd"]
    cold = [n for n, p in kda.named_parameters() if p.grad is None or not p.grad.any()]
    assert cold == []
    counters.check()


def test_a_forward_without_a_backward_is_a_missing_report() -> None:
    """The counter pair exists to fail loudly when only half the mechanism ran."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    counters = Counters()
    counters.enable("kda", counter="kda_bwd")
    kda(torch.randn(2, 8, D_MODEL), counters)
    with pytest.raises(MissingReport) as e:
        counters.check()
    assert "kda" in str(e.value)


@pytest.mark.parametrize("seq", [256, 1024, 4096])
def test_the_state_norm_does_not_grow_with_the_window(seq: int) -> None:
    """The recurrent state is a rolling memory, not a sum over the window."""
    torch.manual_seed(5)
    kda = KDA(64, 4, 16, chunk=64)
    _, state = kda.run(torch.randn(1, seq, 64))
    assert float(state.detach().norm()) < 10.0


def test_the_state_norm_at_4096_is_not_a_multiple_of_the_norm_at_256() -> None:
    """The bound above is one number; this is the shape of the curve."""
    torch.manual_seed(6)
    kda = KDA(64, 4, 16, chunk=64)
    norms = []
    for seq in (256, 1024, 4096):
        torch.manual_seed(7)
        norms.append(float(kda.run(torch.randn(1, seq, 64))[1].detach().norm()))
    assert max(norms) / min(norms) < 1.05, norms


def test_the_delta_rule_cannot_be_switched_off() -> None:
    """`delta=False` is plain linear attention wearing KDA's name."""
    with pytest.raises(LoudError) as e:
        KDA(D_MODEL, N_HEADS, HEAD_DIM, delta=False)
    assert "delta rule is not a switch" in str(e.value)


def test_the_decay_cannot_be_switched_off() -> None:
    """Kimi's diagonal gate is the mechanism, not an option."""
    with pytest.raises(LoudError) as e:
        KDA(D_MODEL, N_HEADS, HEAD_DIM, decay=False)
    assert "Diag" in str(e.value)


def test_the_decay_starts_just_below_one() -> None:
    """`alpha = exp(-softplus(b)) ∈ (0, 1)`, and the init is a ~38-token half-life."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    _, _, _, alpha, _ = kda._project(torch.zeros(1, 4, D_MODEL))
    assert float(alpha.detach().min()) > 0.0
    assert float(alpha.detach().max()) < 1.0
    assert float(alpha.detach().mean()) == pytest.approx(0.982, abs=5e-3)


def test_a_window_that_does_not_tile_the_chunk_is_refused() -> None:
    """A partial chunk has no state to carry; it is refused, not padded silently."""
    kda = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    with pytest.raises(LoudError) as e:
        kda(torch.zeros(1, 12, D_MODEL))
    assert "chunk=8" in str(e.value)


def test_the_coeffects_say_no_rope_and_a_recurrent_state() -> None:
    declared = KDA(D_MODEL, N_HEADS, HEAD_DIM).coeffects()
    assert declared.requires == {"attn.pos": "none", "attn.rule": "delta"}
    assert declared.provides["attn.state"] == "recurrent"


def test_widths_and_kernel_sizes_are_validated() -> None:
    """Three sizes a caller can get wrong, three refusals with a way out."""
    with pytest.raises(LoudError, match="positive widths"):
        KDA(0, N_HEADS, HEAD_DIM)
    with pytest.raises(LoudError, match="holds no positions"):
        KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=0)
    with pytest.raises(LoudError, match="one tap is a scale"):
        KDA(D_MODEL, N_HEADS, HEAD_DIM, conv_width=1)


def test_the_repr_names_the_kernel() -> None:
    assert "chunk=8" in repr(KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8))


def _retained(layer: KDA, x: torch.Tensor) -> int:
    """Bytes of unique storage kept alive for the backward pass of one forward."""
    kept: dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        kept[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        layer.run(x)
    return sum(kept.values())


def _pairwise_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 2-9 with the Γ ratio written pairwise: the anchor for the batched form.

    `ratio[t, i, d] = a_t,d / a_i,d` is materialised, multiplied and reduced once per
    term — the shape the batched form replaces with two matmuls. It is here, not in the
    package, because it is the arithmetic being replaced and a fast kernel is not
    evidence that it computes the same number.
    """
    batch, heads, seq, width = q.shape
    incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    strict = torch.tril(incl, -1)
    state = q.new_zeros(batch, heads, width, width)
    out = []
    for start in range(0, seq, chunk):
        window = slice(start, start + chunk)
        qi, ki, vi, ai = (tile[..., window, :] for tile in (q, k, v, alpha))
        step = beta[..., window].unsqueeze(-1)
        a = torch.cumprod(ai, dim=-2)
        log_a = torch.log(a.clamp_min(1e-12))
        ratio = (log_a.unsqueeze(-2) - log_a.unsqueeze(-3)).exp()
        ratio = ratio.masked_fill(~incl.view(1, 1, chunk, chunk, 1), 0.0)
        keyed = a * ki
        gram = (ratio * ki.unsqueeze(-2) * ki.unsqueeze(-3)).sum(-1).masked_fill(~strict, 0.0)
        qk = (ratio * qi.unsqueeze(-2) * ki.unsqueeze(-3)).sum(-1).masked_fill(~incl, 0.0)
        pseudo = torch.linalg.solve_triangular(
            step * gram, step * (vi - keyed @ state), upper=False, unitriangular=True
        )
        out.append((a * qi) @ state + qk @ pseudo)
        carried = torch.exp(log_a[..., -1:, :] - log_a)
        state = a[..., -1, :].unsqueeze(-1) * state + torch.einsum(
            "bhtd,bhtv->bhdv", carried * ki, pseudo
        )
    return torch.cat(out, dim=2), state


def test_the_batched_chunks_are_the_pairwise_ratio_form() -> None:
    """The matmul Γ is the pairwise Γ: same write, same state, same every gradient.

    In float64, where the two forms differ only by rounding: `(a ⊙ k)·(k/a)ᵀ` is
    term-by-term `Σ_d (a_t,d/a_i,d) k_t,d k_i,d`. The gradients are compared too, because
    a form that agrees in the forward can disagree through the solve.
    """
    torch.manual_seed(17)
    layer = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8, n_kv_heads=1, c_lat=16).double()

    def run(batched: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        for parameter in layer.parameters():
            parameter.grad = None
        #: The projections run inside the measured graph: a form that matches in the
        #: kernel but not through `project` would pass a kernel-only comparison.
        torch.manual_seed(23)
        x = torch.randn(2, 32, D_MODEL, dtype=torch.float64)
        q, k, v, alpha, beta = layer._project(x)
        #: GQA: `run` broadcasts the shared key/value head to its query heads before the
        #: kernel, so the parity case does the same rather than relying on broadcasting.
        k, v = k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2)
        args = (*[tile.transpose(1, 2) for tile in (q, k, v, alpha)], beta.transpose(1, 2))
        write, state = layer._chunkwise(*args) if batched else _pairwise_chunkwise(*args, chunk=8)
        (write.square().mean() + state.square().mean()).backward()
        grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        return write, state, grads

    fast_write, fast_state, fast_grads = run(True)
    slow_write, slow_state, slow_grads = run(False)
    assert torch.allclose(fast_write, slow_write, atol=1e-12, rtol=0)
    assert torch.allclose(fast_state, slow_state, atol=1e-12, rtol=0)
    assert sorted(fast_grads) == sorted(slow_grads)
    for name, gradient in fast_grads.items():
        assert torch.allclose(gradient, slow_grads[name], atol=1e-12, rtol=0), (
            name,
            float((gradient - slow_grads[name]).abs().max()),
        )


def test_the_chunk_graph_is_recomputed_not_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpointing the chunk loop: the same forward, the same gradients, less kept.

    The chunk graph is `O(T/chunk)` tensors of `[batch, heads, chunk, chunk, head_dim]` —
    measured 987 MiB per layer at b1 x s1024, 8.7 GiB over the nine KDA layers, which is
    the second allocation that does not fit on a 16 GiB card. Recomputing each chunk in
    the backward keeps the five inputs instead.

    NOTHING ELSE MOVES: the loop is the same arithmetic on the same chunks in the same
    order, so the write, the final state and every gradient are `torch.equal` to the arm
    without the checkpoint — this is a recomputation, not a rearrangement.
    """
    from dormouse.model.attn_variants import kda as module

    torch.manual_seed(13)
    layer = KDA(D_MODEL, N_HEADS, HEAD_DIM, chunk=8)
    x = torch.randn(2, 128, D_MODEL)

    def run(checkpointed: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        monkeypatch.setattr(module, "CHECKPOINT_CHUNKS", checkpointed)
        for parameter in layer.parameters():
            parameter.grad = None
        y, state = layer.run(x)
        (y.square().mean() + state.square().mean() * 0.1).backward()
        gradients = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        return y.detach(), state.detach(), gradients

    monkeypatch.setattr(module, "CHECKPOINT_CHUNKS", False)
    kept = _retained(layer, x)
    monkeypatch.setattr(module, "CHECKPOINT_CHUNKS", True)
    recomputed = _retained(layer, x)
    without = run(False)
    with_ = run(True)
    print(f"\nretained for the backward: {kept / 2**20:.2f} MiB -> {recomputed / 2**20:.2f} MiB")
    assert torch.equal(without[0], with_[0])
    assert torch.equal(without[1], with_[1])
    assert sorted(without[2]) == sorted(with_[2])
    for name, gradient in without[2].items():
        assert torch.equal(gradient, with_[2][name]), name
    #: Per-chunk checkpoints keep the carry and each chunk's output too; the graph is
    #: what must go. The margin is narrower than it was when this test was written
    #: (1.03 -> 0.36 MiB, a 5-D `[b, h, chunk, chunk, head_dim]` graph per chunk removed):
    #: the batched core retains `[b, h, chunks, chunk, chunk]`, so what the checkpoint
    #: still buys is the six intermediates against the five inputs — strictly less, which
    #: is the claim, at the measured 0.56 -> 0.34 MiB here and ~0.4 GiB at b8 x s1024.
    assert recomputed < kept, (recomputed, kept)


# --- numerical anchor: Kimi eq. 1 written out in numpy fp64 -------------------


def _numpy_recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The published recurrence, one token at a time, over raw fp64 arrays.

        S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ
        o_t = S_tᵀ q_t

    Nothing here calls a `dormouse` function: the delta term, the diagonal decay
    and the write are the three moves this anchor pins, written from the paper.
    """
    width = q.shape[1]
    state = np.zeros((width, width), dtype=np.float64)
    written = []
    for t in range(q.shape[0]):
        state = state * alpha[t][:, None]
        state = (np.eye(width) - beta[t] * np.outer(k[t], k[t])) @ state
        state = state + beta[t] * np.outer(k[t], v[t])
        written.append(state.T @ q[t])
    return np.stack(written), state


def _as_kernel(value: np.ndarray) -> torch.Tensor:
    """A hand-set array as the kernel's `[batch=1, heads=1, seq, width]` input."""
    return torch.tensor(value, dtype=torch.float64).unsqueeze(0).unsqueeze(0)


def anchor_kda_recurrence() -> dict[str, float]:
    """Compare KDA to the numpy fp64 recurrence on 3 tokens, d=2.

    Inputs are hand-set, not drawn from this repo's projections: a change to the
    delta term, the decay or the write moves the module away from a number that
    does not depend on it. The chunk path is anchored to the same reference, so
    the optimisation cannot drift from eq. 1 either.
    """
    q = np.array([[0.5, -0.3], [1.0, 0.2], [-0.7, 0.9]])
    k = np.array([[0.4, 0.8], [-0.6, 0.5], [0.9, -0.2]])
    v = np.array([[1.0, 0.5], [-0.5, 2.0], [0.3, -1.0]])
    alpha = np.array([[0.9, 0.8], [0.95, 0.7], [0.85, 0.6]])
    beta = np.array([0.5, 0.7, 0.3])

    want_out, want_state = _numpy_recurrence(q, k, v, alpha, beta)
    # Numbers from the numpy computation above, printed so the anchor is visible.
    assert want_out[1, 0] == pytest.approx(0.4301, abs=1e-12)
    assert want_out[2, 1] == pytest.approx(0.97233079, abs=1e-8)
    assert want_state[0, 0] == pytest.approx(0.34851363, abs=1e-8)
    assert want_state[1, 1] == pytest.approx(0.52130987, abs=1e-8)

    layer = KDA(8, 1, 2, chunk=3).double()
    args = (_as_kernel(q), _as_kernel(k), _as_kernel(v), _as_kernel(alpha), _as_kernel(beta))
    slow_out, slow_state = layer.recurrent(*args)
    slow_err = float(
        max(
            np.abs(slow_out.squeeze(0).squeeze(0).detach().numpy() - want_out).max(),
            np.abs(slow_state.squeeze(0).squeeze(0).detach().numpy() - want_state).max(),
        )
    )
    assert slow_err < 1e-12, f"recurrent drifted from eq. 1 by {slow_err:.3e}"

    fast_out, fast_state = layer._chunkwise(*args)
    fast_err = float(
        max(
            np.abs(fast_out.squeeze(0).squeeze(0).detach().numpy() - want_out).max(),
            np.abs(fast_state.squeeze(0).squeeze(0).detach().numpy() - want_state).max(),
        )
    )
    assert fast_err < 1e-12, f"chunkwise drifted from eq. 1 by {fast_err:.3e}"
    return {"recurrent": slow_err, "chunkwise": fast_err}


def test_kda_recurrence_matches_an_independent_numpy_fp64_anchor() -> None:
    """3 tokens, d=2: eq. 1 by hand in numpy is the number the module must return."""
    anchor_kda_recurrence()
