"""The GQA + latent projection block: the degeneracy gate and the saving.

Two claims, and neither is a shape test:

(a) WITH THE COMPRESSION OFF the projections are the layer that existed before this
    module did — one packed `[3·d_kv, d_model]` matrix, no bottleneck — so its
    parameter inventory is pinned here and two constructions that differ only in how
    the setting is spelled build the same tensors and the same output, bit for bit.
    A checkpoint from the dense recipe loads and runs without a translation, which is
    what "off" has to mean for the setting to be a degeneracy and not a second arm.
(b) WITH IT ON the projection block is 39.6% smaller at live widths, and the number
    is printed for both sides from `projections.bill` — the arithmetic, not a guess.

The kernel order is separate: `test_kda.py` anchors the chunkwise form to the
recurrence, and `test_dsa.py` anchors the selected-attention path to the dense one.
This file is only about the block that reads and writes the stream.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.attn_variants.dsa import DSA
from dormouse.model.attn_variants.kda import KDA
from dormouse.model.attn_variants.projections import bill

D, H, HD = 32, 2, 8

#: The parameter inventory of the layer before `projections.py` existed, pinned.
PRE_CHANGE = {
    "a_bias": (2, 8),
    "head_norm": (8,),
    "qkv.weight": (48, 32),
    "conv.weight": (48, 1, 4),
    "conv.bias": (48,),
    "a_down.weight": (8, 32),
    "a_up.weight": (16, 8),
    "beta.weight": (2, 32),
    "beta.bias": (2,),
    "gate.weight": (16, 32),
    "proj.weight": (32, 16),
}


def test_gqa_and_latent_off_is_the_pre_change_layer_bit_for_bit() -> None:
    """`n_kv_heads=n_heads, c_lat=d_model` is the old layer, not a new one."""
    torch.manual_seed(0)
    before = KDA(D, H, HD, chunk=8)
    after = KDA(D, H, HD, chunk=8, n_kv_heads=H, c_lat=D)
    assert not after.latent
    assert {name: tuple(p.shape) for name, p in after.named_parameters()} == PRE_CHANGE
    after.load_state_dict(before.state_dict())
    x = torch.randn(1, 8, D)
    assert torch.equal(after(x), before(x))


def test_the_two_dsa_spellings_are_the_same_layer() -> None:
    """The same gate for the other mechanism: one packed qkv, no bottleneck."""
    torch.manual_seed(1)
    before = DSA(D, H, HD, k=4, window=2, d_idx=8, chunk=4)
    after = DSA(D, H, HD, k=4, window=2, d_idx=8, chunk=4, n_kv_heads=H, c_lat=D)
    assert not after.latent
    after.load_state_dict(before.state_dict())
    x = torch.randn(1, 8, D)
    assert torch.equal(after(x), before(x))


def test_the_projection_block_saves_about_forty_percent_at_live_widths() -> None:
    """`bill` prints both sides; the saving is the difference, not a claim in prose."""
    dense = bill(d_model=832, n_heads=13, head_dim=64, n_kv_heads=13, c_lat=832)
    live = bill(d_model=832, n_heads=8, head_dim=64, n_kv_heads=2, c_lat=512)
    saving = 1.0 - live["params"] / dense["params"]
    print(
        f"\nprojection block: dense={dense['params']:,} p / {dense['macs']:,} mac/token, "
        f"gqa+latent={live['params']:,} p / {live['macs']:,} mac/token, "
        f"saving={saving:.1%}"
    )
    assert 0.37 <= saving <= 0.42, saving
    assert live["params"] == live["read"] + live["write"]
    assert live["params"] == live["macs"]


def test_the_bottleneck_is_what_pays_not_the_head_count() -> None:
    """GQA alone saves less than the pair: the comparison is the pair's."""
    gqa_only = bill(d_model=832, n_heads=8, head_dim=64, n_kv_heads=2, c_lat=832)
    both = bill(d_model=832, n_heads=8, head_dim=64, n_kv_heads=2, c_lat=512)
    print(f"\ngqa only={gqa_only['params']:,}, qga+latent={both['params']:,}")
    assert both["params"] < gqa_only["params"]


def test_gqa_really_shares_the_key_value_heads() -> None:
    """The kernel sees `n_kv_heads` keys and `n_heads` queries, and broadcasts."""
    layer = KDA(D, H, HD, chunk=8, n_kv_heads=1, c_lat=8)
    assert layer.latent
    q, k, v, _alpha, _beta = layer._project(torch.randn(1, 8, D))
    assert q.shape == (1, 8, H, HD)
    assert k.shape == v.shape == (1, 8, 1, HD)
    assert layer.d_kv == H * HD
    assert layer.d_kv_kv == HD


def test_a_head_count_that_does_not_tile_the_key_value_heads_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        KDA(D, H, HD, n_kv_heads=3, c_lat=8)
    assert "does not tile" in str(caught.value)


def test_a_bottleneck_of_no_channels_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        KDA(D, H, HD, n_kv_heads=1, c_lat=0)
    assert "not a width" in str(caught.value)


def test_the_latent_matrices_are_the_only_thing_the_compression_adds() -> None:
    """Off and on differ by the bottleneck pair, and by nothing else."""
    dense = {n for n, _ in KDA(D, H, HD, chunk=8).named_parameters()}
    latent = {n for n, _ in KDA(D, H, HD, chunk=8, n_kv_heads=1, c_lat=8).named_parameters()}
    assert dense - latent == {"qkv.weight"}
    assert latent - dense == {"q_lat.weight", "q_up.weight", "kv_lat.weight", "kv_up.weight"}


# --- numerical anchor: the explicit GQA group map -----------------------------


#: Two groupings, so `h // groups` is pinned for more than one group size.
GQA_CASES = ((4, 2), (6, 2))
#: The map written out by hand, not recomputed from the formula under test.
GQA_MAPS = {4: [0, 0, 1, 1], 6: [0, 0, 0, 1, 1, 1]}


def _final_state(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Final state of Kimi eq. 1 for one head's hand-set sequence, in numpy fp64."""
    width = q.shape[1]
    state = np.zeros((width, width), dtype=np.float64)
    for t in range(q.shape[0]):
        state = state * alpha[t][:, None]
        state = (np.eye(width) - beta[t] * np.outer(k[t], k[t])) @ state
        state = state + beta[t] * np.outer(k[t], v[t])
    return state


def _fake_project(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> Callable[[torch.Tensor], tuple[torch.Tensor, ...]]:
    """Return a `_project` that emits the hand-set tensors regardless of `x`."""

    def project(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return q, k, v, alpha, beta

    return project


def _capturing_chunkwise(
    real: Callable[..., tuple[torch.Tensor, torch.Tensor]], captured: dict[str, torch.Tensor]
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Wrap a chunk kernel so the broadcast k/v it receives are recorded."""

    def capture(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["k"], captured["v"] = k.detach(), v.detach()
        return real(q, k, v, alpha, beta)

    return capture


def anchor_gqa_group_map() -> dict[str, int]:
    """Query head `h` reads key/value head `h // (n_heads // n_kv_heads)`.

    The key/value tensors are hand-set so every kv head is distinct; the broadcast
    is captured where `run` hands it to the kernel, and each query head's final
    state is recomputed in numpy from the kv head the explicit map names. A
    different grouping (or a `h % groups` map) swaps heads and fails both checks.
    """
    for n_heads, n_kv_heads in GQA_CASES:
        groups = n_heads // n_kv_heads
        kv_for_query = [h // groups for h in range(n_heads)]
        assert kv_for_query == GQA_MAPS[n_heads], kv_for_query

        q = np.stack(
            [
                np.array([[0.5 + 0.1 * h, -0.2 + 0.03 * h], [0.9 - 0.05 * h, 0.4 + 0.02 * h]])
                for h in range(n_heads)
            ]
        )
        k = np.stack(
            [
                np.array([[0.4 + 0.2 * j, 0.8 - 0.1 * j], [-0.6 + 0.1 * j, 0.5 + 0.15 * j]])
                for j in range(n_kv_heads)
            ]
        )
        v = np.stack(
            [
                np.array([[1.0 + 0.5 * j, 0.5 - 0.2 * j], [-0.5 + 0.3 * j, 2.0 - 0.4 * j]])
                for j in range(n_kv_heads)
            ]
        )
        alpha = np.full((n_heads, 2, 2), 0.8)
        beta = np.full((n_heads, 2), 0.6)

        # `_project`'s contract: q/k/v/alpha [b, s, heads, d], beta [b, s, heads].
        proj_q = torch.tensor(q.transpose(1, 0, 2)[None], dtype=torch.float64)
        proj_k = torch.tensor(k.transpose(1, 0, 2)[None], dtype=torch.float64)
        proj_v = torch.tensor(v.transpose(1, 0, 2)[None], dtype=torch.float64)
        proj_alpha = torch.tensor(alpha.transpose(1, 0, 2)[None], dtype=torch.float64)
        proj_beta = torch.tensor(beta.T[None], dtype=torch.float64)

        layer = KDA(8, n_heads, 2, chunk=2, n_kv_heads=n_kv_heads, c_lat=8).double()

        captured: dict[str, torch.Tensor] = {}
        layer._project = _fake_project(  # type: ignore[method-assign]
            proj_q, proj_k, proj_v, proj_alpha, proj_beta
        )
        layer._chunkwise = _capturing_chunkwise(layer._chunkwise, captured)  # type: ignore[method-assign]
        _, state = layer.run(torch.zeros(1, 2, 8, dtype=torch.float64))

        assert captured["k"].shape == (1, n_heads, 2, 2)
        for h in range(n_heads):
            source = kv_for_query[h]
            assert torch.equal(captured["k"][0, h], proj_k[0, :, source]), (n_heads, h)
            assert torch.equal(captured["v"][0, h], proj_v[0, :, source]), (n_heads, h)
            want = _final_state(q[h], k[source], v[source], alpha[h], beta[h])
            got = state[0, h].detach().numpy()
            assert np.abs(got - want).max() < 1e-12, (n_heads, h, np.abs(got - want).max())
    return {"cases": len(GQA_CASES)}


def test_gqa_group_map_matches_an_explicit_h_over_groups() -> None:
    """The broadcast query heads must read, by hand, their `h // groups` kv head."""
    anchor_gqa_group_map()
