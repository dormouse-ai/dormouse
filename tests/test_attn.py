"""Attention checked against a hand-written reference, and its two order properties.

(a) A ten-line numpy attention — `softmax(qkᵀ/√hd, causal)v`, then the projection —
    is the anchor for the module's arithmetic.
(b) The causal property: perturbing position j must leave every output at i < j
    bit-identical. This is the test that fails when a mask is applied to the wrong
    axis or `is_causal` is dropped.
(c) The order properties, which say WHY the rotations are in the recipe: without
    RoPE and without a mask, attention is permutation-equivariant (a bag of
    positions); with the causal mask it is not, because the mask is a function of
    index; with RoPE it is not even without a mask, because the rotation is a
    function of absolute position. Three claims, three assertions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from reference_attn import Attention


def _reference(x: torch.Tensor, attn: Attention, *, causal: bool) -> np.ndarray:
    """The documented formula in float64, from the module's own weights, no RoPE."""
    qkv = attn.qkv.weight.detach().numpy().astype(np.float64)
    proj = attn.proj.weight.detach().numpy().astype(np.float64)
    t, d = x.shape[1], x.shape[2]
    q, k, v = np.split(qkv @ x[0].detach().numpy().astype(np.float64).T, 3, axis=0)

    def heads(z: np.ndarray) -> np.ndarray:
        return z.T.reshape(t, attn.n_heads, attn.head_dim).transpose(1, 0, 2)

    q, k, v = heads(q), heads(k), heads(v)
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(attn.head_dim)
    if causal:
        future = np.triu(np.ones((t, t), dtype=bool), 1)
        scores = np.where(future, -np.inf, scores)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = (p @ v).transpose(1, 0, 2).reshape(t, d)
    return (proj @ out.T).T


def test_matches_the_numpy_reference_on_three_tokens() -> None:
    """3 tokens, 2 heads of width 4: the mask, the scale and the blend all matter."""
    torch.manual_seed(0)
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=False)
    x = torch.randn(1, 3, 8)
    assert np.allclose(attn(x)[0].detach().numpy(), _reference(x, attn, causal=True), atol=1e-6)


def test_matches_the_numpy_reference_without_a_mask() -> None:
    """The non-causal arm is the same formula with the mask removed, not a copy."""
    torch.manual_seed(1)
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=False)
    x = torch.randn(1, 4, 8)
    got = attn(x, causal=False)[0].detach().numpy()
    assert np.allclose(got, _reference(x, attn, causal=False), atol=1e-6)


def test_the_rope_table_is_the_closed_form() -> None:
    """θ_j = 10000^(-2j/head_dim), angle = position·θ_j — Su et al. eq. 1."""
    attn = Attention(d_model=8, n_heads=2, head_dim=4, max_seq_len=16)
    for pos in (0, 1, 7, 15):
        for j in range(attn.head_dim // 2):
            theta = pos * 10_000.0 ** (-2.0 * j / attn.head_dim)
            assert attn.cos[pos, j].item() == pytest.approx(math.cos(theta), abs=1e-6)
            assert attn.sin[pos, j].item() == pytest.approx(math.sin(theta), abs=1e-6)


def test_rotation_matches_a_hand_written_loop() -> None:
    """Each channel pair (2j, 2j+1) rotated by its own angle, computed by hand."""
    attn = Attention(d_model=4, n_heads=1, head_dim=4, max_seq_len=8)
    x = torch.randn(1, 1, 3, 4)
    got = Attention._rotate(x, attn.cos[:3], attn.sin[:3])
    want = x.detach().numpy().astype(np.float64)
    for pos in range(3):
        for j in range(2):
            theta = pos * 10_000.0 ** (-2.0 * j / 4)
            a, b = want[0, 0, pos, 2 * j], want[0, 0, pos, 2 * j + 1]
            want[0, 0, pos, 2 * j] = a * math.cos(theta) - b * math.sin(theta)
            want[0, 0, pos, 2 * j + 1] = a * math.sin(theta) + b * math.cos(theta)
    assert np.allclose(got.detach().numpy(), want, atol=1e-6)


def test_a_later_token_cannot_change_an_earlier_output() -> None:
    """The causal property, bit for bit: i never reads j > i."""
    torch.manual_seed(2)
    attn = Attention(d_model=8, n_heads=2, head_dim=4)
    x = torch.randn(1, 5, 8)
    y = attn(x)
    moved = x.clone()
    moved[:, 4] += 1.0
    y2 = attn(moved)
    assert torch.equal(y[:, :4], y2[:, :4])
    assert not torch.equal(y[:, 4], y2[:, 4])


def test_without_rope_or_mask_attention_is_permutation_equivariant() -> None:
    """A bag of positions: permuting the input permutes the output, nothing else.

    The reference is the permutation itself. The band is 1e-6 because the softmax's
    reduction runs in the permuted order; the measured rounding floor on this case is
    1.2e-7, so 1e-6 is the arithmetic's noise and not a tolerance for a difference.
    """
    torch.manual_seed(3)
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=False)
    x = torch.randn(1, 4, 8) * 100.0
    perm = torch.tensor([2, 0, 3, 1])
    plain = attn(x, causal=False)
    shuffled = attn(x[:, perm], causal=False)
    assert torch.allclose(plain[:, perm], shuffled, atol=1e-6, rtol=0)


def test_the_causal_mask_alone_breaks_that_equivariance() -> None:
    """The mask is a function of index, so order now matters even without RoPE."""
    torch.manual_seed(4)
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=False)
    x = torch.randn(1, 4, 8) * 100.0
    perm = torch.tensor([2, 0, 3, 1])
    delta = (attn(x)[:, perm] - attn(x[:, perm])).abs().max().item()
    assert delta > 0.1  # measured 0.90: the mask, three orders above the noise


def test_rope_alone_breaks_it_too() -> None:
    """Rotations are a function of absolute position, so even unmasked it differs."""
    torch.manual_seed(5)
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=True)
    x = torch.randn(1, 4, 8) * 100.0
    perm = torch.tensor([2, 0, 3, 1])
    delta = (attn(x, causal=False)[:, perm] - attn(x[:, perm], causal=False)).abs().max()
    assert delta > 0.1  # measured 0.99 without a mask in sight


def test_it_reports_attn_and_rope() -> None:
    counters = Counters()
    attn = Attention(d_model=8, n_heads=2, head_dim=4)
    attn(torch.zeros(2, 3, 8), counters)
    assert counters.snapshot() == {"attn_fwd": 6, "rope_fwd": 6}


def test_rope_off_reports_only_attn() -> None:
    counters = Counters()
    attn = Attention(d_model=8, n_heads=2, head_dim=4, rope=False)
    attn(torch.zeros(1, 3, 8), counters)
    assert counters.snapshot() == {"attn_fwd": 3}


def test_a_window_past_the_table_is_refused_not_extrapolated() -> None:
    attn = Attention(d_model=8, n_heads=2, head_dim=4, max_seq_len=4)
    with pytest.raises(LoudError) as e:
        attn(torch.zeros(1, 5, 8))
    assert "max_seq_len=4" in str(e.value)


def test_an_odd_head_dim_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Attention(d_model=6, n_heads=2, head_dim=3)
    assert "odd" in str(e.value)


def test_heads_must_tile_the_width() -> None:
    with pytest.raises(LoudError) as e:
        Attention(d_model=8, n_heads=3, head_dim=4)
    assert "tile" in str(e.value)
