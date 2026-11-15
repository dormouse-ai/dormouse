"""TSCT: a matrix's forward through its rank-`r` spectrum, the masters kept dense.

THE FACTORISATION. For a bias-free `nn.Linear` with weight `W [out, in]`, the map is
`y = x·Wᵀ`. Write `Wᵀ = u·diag(s)·v` with `u [in, r]`, `s [r]`, `v [r, out]`; then

    y = ((x · u) ⊙ s) · v

costs `in·r + r + r·out` multiplies instead of `in·out`. At `in=832, out=6144, r=256`
that is 1.79M against 5.11M — the FLOP argument for TSCT (burn-spectral's
`SpectralLinear`, `docs/papers/tsct.md`).

WHY THE MASTERS STAY DENSE. Two distinct things are called "TSCT" in this project:
a parameterisation (the factors ARE the parameters, `r·(in+out+1)` stored) and a
compute path (the parameters stay `in·out` and the forward runs the truncated
spectrum). This is the second: the expert bank's stored and active bills are the
dense ones (`3·d·f` per expert), and only the multiplies change. That is the arm the
1B/46M pair was priced on, and the only one whose checkpoint is a plain `nn.Linear`.

THE GATE THAT MAKES IT SAFE. At `r = min(in, out)` there is nothing to truncate, and
`factors` returns the exact identity basis (`u = I, s = 1, v = Wᵀ`, or `u = Wᵀ,
s = 1, v = I` when the short side is `out`). Both products then reduce to one matmul
with an identity, which IEEE arithmetic evaluates exactly — so the factorised expert
is `torch.equal` to the dense one, not merely close. A test pins that, because a
"factorisation" that is only approximately the dense layer is a different model
wearing the same name.

Ceiling: the SVD runs in the forward, per matrix, per call — `ponytail:`, the honest
cost of masters that keep training. Make the factors parameters (or refresh them
every N steps) when the projection becomes a training-time arm rather than a compute
path; until then the numbers this module prints are FLOPs, not step time.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

#: The rank the one architecture's experts are factorised at.
TSCT_RANK = 256


def factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(u, s, v)` with `u·diag(s)·v == weightᵀ`, truncated to `rank`.

    `rank >= min(in, out)` is the whole matrix and returns the identity basis, which
    makes the factorised product exact (see the module docstring). Below that it is
    the truncated SVD — the best rank-`rank` approximation in Frobenius norm.
    """
    if weight.dim() != 2:
        raise LoudError(
            f"TSCT factorises a matrix, got shape {tuple(weight.shape)}",
            "a matrix is [out, in]; flatten or drop the extra axes",
        )
    transposed = weight.transpose(0, 1)
    in_features, out_features = transposed.shape
    short = min(in_features, out_features)
    if rank < 1:
        raise LoudError(
            f"TSCT rank={rank} keeps no direction",
            f"use a rank in [1, {short}]; {TSCT_RANK} is the recipe, and `r = {short}` "
            "is the whole matrix",
        )
    if rank >= short:
        unit = torch.ones(short, dtype=weight.dtype, device=weight.device)
        eye = torch.eye(
            in_features if in_features <= out_features else out_features,
            dtype=weight.dtype,
            device=weight.device,
        )
        return (
            (eye, unit, transposed)
            if in_features <= out_features
            else (
                transposed,
                unit,
                eye,
            )
        )
    u, s, vh = torch.linalg.svd(transposed.float(), full_matrices=False)
    #: fp32 in, the weight's dtype out (`train/precision.py`): bf16 has no LAPACK SVD path,
    #: and a bf16 basis is a different rank-r approximation of the same matrix. For fp32
    #: weights `.float()` and `.to()` are the identity, so this is the same call it was.
    return u[:, :rank].to(weight.dtype), s[:rank].to(weight.dtype), vh[:rank].to(weight.dtype)


def low_rank(
    linear: nn.Linear,
    x: torch.Tensor,
    rank: int,
    counters: Counters | None = None,
) -> torch.Tensor:
    """`linear(x)` computed through the rank-`rank` factors of its current weight."""
    if rank < 1:
        return cast(torch.Tensor, linear(x))
    u, s, v = factors(linear.weight, rank)
    y: torch.Tensor = ((x @ u) * s) @ v
    if counters is not None:
        counters.count("tsct_fwd", x.shape[0])
    return y + linear.bias if linear.bias is not None else y
