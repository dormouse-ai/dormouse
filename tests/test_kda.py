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
    #: what must go. At this width that is 3.0x less retained (measured: 1.03 -> 0.36 MiB);
    #: at b1 x s1024 the same setting is 4.6 GiB of chunk graph against 0.
    assert recomputed < kept / 2, (recomputed, kept)
