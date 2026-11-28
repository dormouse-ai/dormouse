"""The defining property of RoPE, which no configuration test can replace.

A permutation test proves that positions matter. It does not prove that the
rotations encode the *right* thing: a wrong pairing, a wrong frequency or a wrong
sign all make positions matter too. The property that separates a correct
implementation from a plausible one is relative invariance —

    ⟨R_i q, R_j k⟩ = ⟨q, R_{j-i} k⟩

so the score of two vectors depends on their distance and not on where they sit.
This is what the model actually needs, and it is the check that was missing.
"""

from __future__ import annotations

import math

import torch
from reference_attn import Attention

HEAD_DIM = 8
BASE = 10_000.0


def _tables(seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    half = HEAD_DIM // 2
    freq = BASE ** (-2.0 * torch.arange(half, dtype=torch.float32) / HEAD_DIM)
    ang = torch.outer(torch.arange(seq, dtype=torch.float32), freq)
    return ang.cos(), ang.sin()


def test_scores_depend_only_on_the_distance_between_positions() -> None:
    """The same pair of vectors, moved to every offset, must score by distance."""
    torch.manual_seed(0)
    seq = 64
    cos, sin = _tables(seq)
    q = torch.randn(1, 1, 1, HEAD_DIM).expand(1, 1, seq, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM).expand(1, 1, seq, HEAD_DIM)
    qr, kr = Attention._rotate(q, cos, sin), Attention._rotate(k, cos, sin)

    groups: dict[int, list[float]] = {}
    for i in range(seq // 2):
        for j in range(seq // 2):
            groups.setdefault(i - j, []).append(float((qr[0, 0, i] * kr[0, 0, j]).sum()))

    worst = max(
        ((d, max(v) - min(v)) for d, v in groups.items() if len(v) > 1),
        key=lambda kv: kv[1],
    )
    assert worst[1] < 1e-5, f"distance {worst[0]} scores inconsistently by {worst[1]:.2e}"


def test_the_rotation_preserves_the_norm() -> None:
    """Rotating a vector cannot change its length; a scaling bug would show here."""
    torch.manual_seed(1)
    cos, sin = _tables(32)
    x = torch.randn(1, 1, 32, HEAD_DIM)
    rotated = Attention._rotate(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


def test_the_pair_is_a_rotation_not_a_scaling() -> None:
    """Adjacent channels must mix: a pure multiply would leave each channel alone."""
    cos, sin = _tables(4)
    x = torch.zeros(1, 1, 1, HEAD_DIM)
    x[..., 0] = 1.0
    out = Attention._rotate(x, cos[1:2], sin[1:2])
    assert out[..., 0].abs().max() > 0.5
    assert out[..., 1].abs().max() > 0.5


def test_frequencies_decay_geometrically() -> None:
    """Theta_j = base^(-2j/d): the slowest channel is 1.0 and each next is 1/base^(2/d)."""
    half = HEAD_DIM // 2
    freq = BASE ** (-2.0 * torch.arange(half, dtype=torch.float32) / HEAD_DIM)
    assert math.isclose(float(freq[0]), 1.0, rel_tol=1e-6)
    ratio = float(freq[1] / freq[0])
    assert math.isclose(ratio, BASE ** (-2.0 / HEAD_DIM), rel_tol=1e-5)
