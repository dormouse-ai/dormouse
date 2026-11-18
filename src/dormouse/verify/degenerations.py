"""Flag-off callables: each returns `(flag_on, flag_off)` on a CPU fixture.

The off arm is the stack the mechanism replaced — the recurrent recurrence for
KDA's chunk algebra, full causal attention for DSA's selection, the dense
`nn.Linear` for TSCT's factors. Model imports are local so importing a contract
never imports the mechanism it checks.
"""

from __future__ import annotations

import torch
from torch import nn

from .fixture import Fixture


def kda_degeneration(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk form (on) against the token recurrence (off), in fp64."""
    from dormouse.model.attn_variants.kda import KDA

    layer = KDA(16, 2, 8, chunk=int(fixture.number("chunk"))).double()
    args = tuple(fixture.tensor(name) for name in ("q", "k", "v", "alpha", "beta"))
    on, _ = layer._chunkwise(*args)
    off, _ = layer.recurrent(*args)
    return on, off


def dsa_degeneration(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected-key attention (on) against full causal attention (off), in fp64."""
    from dormouse.model.attn_variants.dsa import DSA

    q, key, value, proj = (fixture.tensor(name) for name in ("q", "k", "v", "proj"))
    at, visible = fixture.tensor("at"), fixture.tensor("visible")
    layer = DSA(
        16,
        2,
        8,
        k=int(fixture.number("k")),
        window=int(fixture.number("window")),
        d_idx=16,
        chunk=4,
    ).double()
    with torch.no_grad():
        layer.proj.weight.copy_(proj)
    on = layer._attend(q, key, value, at, visible)
    scale = fixture.number("head_dim") ** -0.5
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True, scale=scale
    )
    off = dense.transpose(1, 2).reshape(q.shape[0], q.shape[1], layer.d_kv) @ proj.T
    return on, off


def tsct_degeneration(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-`r` factors (on) against the dense matrix they replace (off), in fp64."""
    from dormouse.model.tsct import FactorLinear

    x, u, v = (fixture.tensor(name) for name in ("x", "u", "v"))
    rank, std = int(fixture.number("rank")), 0.02
    factors = FactorLinear(int(fixture.number("in")), int(fixture.number("out")), rank, std=std)
    factors.double()
    with torch.no_grad():
        factors.u.copy_(u)
        factors.v.copy_(v)
    dense = nn.Linear(x.shape[-1], v.shape[-1], bias=False).double()
    with torch.no_grad():
        dense.weight.copy_((u @ v).T)
    return factors(x), dense(x)
