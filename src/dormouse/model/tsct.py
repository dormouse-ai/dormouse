"""TSCT: an expert matrix IS its rank-`r` factors, and the forward is two matmuls.

THE STORED OBJECT. For a bias-free map `y = x·Wᵀ` with `W [out, in]`, TSCT keeps the
two factors

    u [in, r],   v [r, out]         y = (x·u)·v

so the matrix stores `r·(in + out)` parameters where the dense draw stored `in·out`,
and runs `in·r + r·out` multiply-accumulates per row where the dense one ran `in·out`.
At the live expert width (`in=832, out=6144, r=256`) that is 1 785 856 parameters against
5 111 808 and 2.862x fewer MACs (34.9% of the dense cost) — the FLOP argument for TSCT
(burn-spectral's `SpectralLinear`, `docs/papers/tsct.md`).

WHY THE FACTORS ARE THE PARAMETERS. An earlier build kept a dense master and ran a
truncated SVD of it *inside the forward, on every call*. That is rank compression that
saves no FLOPs (the SVD is `O(in·out·min)`), it forces the dense `[out, in]` matrix to
exist — measured on the card as 5 265 MiB for one shared-bank call against 541 MiB with
an identity basis, dtype-independent, which is what OOM'd the 1B config at s1024 — it
keeps a `u, s, v` triple alive per matrix for the backward, and it never trains the
factors as factors. Here the SVD is deleted from the model, not moved.

INIT, AND WHY `σ = √std·r^{-1/4}`. Both factors are drawn iid `N(0, σ²)` with

    σ_u = σ_v = √std · r^{-1/4}   ⟹   Var(W'_jk) = r·σ⁴ = std²   ⟹   Var(y_k) = std²·Σ_j x_j²

(`W' = u·v`, each entry a sum of `r` products of two `N(0, σ²)` draws) — the same
variance as the dense draw `W ~ N(0, std²)` it replaces, so the residual stream's scale
(and every activation statistic downstream of it) is unchanged. The formula is derived
once in `factor_std` and pinned, with the output variance, in `tests/test_tsct.py`.

THE DEGENERACY. At `r = min(in, out)` nothing is truncated and the class can express any
dense matrix exactly: `u = I, v = Wᵀ` (or `u = Wᵀ, v = I` when the short side is `out`)
makes the product one matmul by the identity, which IEEE arithmetic evaluates exactly —
so the full-rank `FactorLinear` is `torch.equal` to the dense layer, not merely close.
That is the gate which says a factorised expert is the same function as the dense one.

`rank=0` is the dense control arm (`project` hands back an `nn.Linear`); a rank above the
short side is refused, because it would store more directions than the matrix has.
"""

from __future__ import annotations

import math

import torch
from dormouse_core.errors import LoudError
from torch import nn

from dormouse.verify import register as register_contract
from dormouse.verify import require as require_contract
from dormouse.verify.contracts import TSCT_CONTRACT

#: The counter, degeneration, external anchor and invariants travel with the class.
register_contract(TSCT_CONTRACT)

#: The rank the one architecture's experts are factorised at.
TSCT_RANK = 256


def factor_std(std: float, rank: int) -> float:
    """Return `√std · r^{-1/4}`: the per-factor std whose product has the dense draw's std.

    `W' = u·v` has entries `Σ_i u_ji·v_ik`, `r` products of two `N(0, σ²)` draws, so
    `Var(W'_jk) = r·σ⁴`; setting that to the dense draw's `std²` gives `σ² = std/√r`,
    i.e. `σ = √std·r^{-1/4}` — and `Var(y_k) = std²·Σ_j x_j²`, the dense draw's own.
    """
    return float(math.sqrt(std) * rank**-0.25)


class FactorLinear(nn.Module):
    """`y = (x·u)·v` — a bias-free linear whose rank-`r` factors are the parameters.

    `u` is `[in_features, rank]` and `v` is `[rank, out_features]`, so the pair stores
    `rank·(in + out)` parameters against the dense matrix's `in·out`. The module is
    callable exactly like the `nn.Linear` it replaces, because the arms that hold it
    should not care which storage they got.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        std: float,
        device: torch.device | str | None = None,
    ) -> None:
        """Validate the rank and draw both factors at `√std·r^{-1/4}`."""
        require_contract("tsct")  # no proof, no allocation
        super().__init__()
        short = min(in_features, out_features)
        if rank < 1:
            raise LoudError(
                f"TSCT rank={rank} keeps no direction",
                f"use a rank in [1, {short}]; {TSCT_RANK} is the recipe, and rank=0 is "
                "the dense arm (`project` builds an nn.Linear for it)",
            )
        if rank > short:
            raise LoudError(
                f"TSCT rank={rank} is wider than the short side {short} of "
                f"{in_features}x{out_features}",
                f"a rank above min(in, out)={short} stores directions the matrix does not "
                "have; the dense arm has no rank at all",
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        scale = factor_std(std, rank)
        self.u = nn.Parameter(torch.randn(in_features, rank, device=device) * scale)
        self.v = nn.Parameter(torch.randn(rank, out_features, device=device) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x [..., in_features]` → `[..., out_features]` through the two factors."""
        return (x @ self.u) @ self.v

    def extra_repr(self) -> str:
        """Show the shapes the bill is quoted for."""
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


def project(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    std: float,
    device: torch.device | str | None = None,
) -> nn.Linear | FactorLinear:
    """Build the dense `nn.Linear` at `rank=0`, the `FactorLinear` above it.

    One function, because the arm that reads the result must not branch on storage: the
    dense arm is the control the factorised one is measured against, and both answer
    `module(x)` with the same shape and the same initial output variance.
    """
    if rank == 0:
        linear = nn.Linear(in_features, out_features, bias=False, device=device)
        nn.init.normal_(linear.weight, std=std)
        return linear
    return FactorLinear(in_features, out_features, rank, std=std, device=device)
