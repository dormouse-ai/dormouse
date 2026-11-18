"""Deterministic CPU fixtures for the three demonstrated mechanisms.

A fixture holds only arrays and scalars drawn from a seed. Two different seeds
give two different inputs, which is what refutes a captured golden at check
time. All arrays are fp64 so the anchors and the degenerations meet on the same
arithmetic.
"""

from __future__ import annotations

import torch

from .fixture import Fixture

#: Small widths that still cross chunk (KDA) and gather (DSA) boundaries.
KDA_SHAPE = (1, 2, 16, 8)  # batch, heads, seq, head_dim
ATTN_D_MODEL = 16  # 2 heads x 8
SEQ = 16
KEEP = 16  # k = window = seq: the degenerate, dense-equivalent configuration
TSCT_SHAPE = (4, 16, 8, 12)  # rows, in, rank, out


def _rng(seed: int) -> torch.Generator:
    """Return a CPU generator; no global RNG state is touched by a proof."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def kda_fixture(seed: int) -> Fixture:
    """Kernel-layout queries, keys, values and gates for a 16-token KDA write."""
    rng = _rng(seed)
    batch, heads, seq, width = KDA_SHAPE
    shape = (batch, heads, seq, width)
    q, k, v = (torch.randn(shape, generator=rng, dtype=torch.float64) for _ in range(3))
    alpha = torch.rand(shape, generator=rng, dtype=torch.float64) * 0.1 + 0.9
    beta = torch.rand((batch, heads, seq), generator=rng, dtype=torch.float64) * 0.8 + 0.1
    return Fixture(
        seed,
        {"q": q, "k": k, "v": v, "alpha": alpha, "beta": beta},
        {"chunk": 8, "head_dim": width},
    )


def dsa_fixture(seed: int) -> Fixture:
    """Build queries, keys, values and the dense-equivalent read set of a DSA layer."""
    rng = _rng(seed)
    q, k, v = (torch.randn((1, SEQ, 2, 8), generator=rng, dtype=torch.float64) for _ in range(3))
    proj = torch.randn((ATTN_D_MODEL, ATTN_D_MODEL), generator=rng, dtype=torch.float64)
    at = torch.arange(SEQ).view(1, 1, SEQ).expand(1, SEQ, KEEP)
    limit = torch.arange(SEQ).view(1, SEQ, 1)
    visible = at <= limit
    return Fixture(
        seed,
        {"q": q, "k": k, "v": v, "proj": proj, "at": at, "visible": visible},
        {"head_dim": 8, "k": KEEP, "window": SEQ, "seq": SEQ},
    )


def tsct_fixture(seed: int) -> Fixture:
    """Factors, their dense product and the input of one rank-`r` expert matrix."""
    rng = _rng(seed)
    rows, width, rank, out = TSCT_SHAPE
    x = torch.randn(rows, width, generator=rng, dtype=torch.float64)
    sigma = 0.02**0.5 * rank**-0.25
    u = torch.randn(width, rank, generator=rng, dtype=torch.float64) * sigma
    v = torch.randn(rank, out, generator=rng, dtype=torch.float64) * sigma
    return Fixture(
        seed,
        {"x": x, "u": u, "v": v},
        {"rows": rows, "in": width, "rank": rank, "out": out, "sigma": sigma},
    )
