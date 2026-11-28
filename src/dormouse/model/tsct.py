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
import os
from typing import Any, cast

import torch
from dormouse_core.errors import LoudError
from torch import nn

#: The rank the one architecture's experts are factorised at.
TSCT_RANK = 256

__concepts__ = ("FactorLinear", "NativeFactorLinear")


#: Set by `StackSpec` from the config's `weight_qat`; the env flag is an A/B override.
_CONFIG_NATIVE = False


def request_native_tern(on: bool) -> None:
    """Record the config's choice; `weight_qat="ternary_native"` is self-sufficient."""
    global _CONFIG_NATIVE
    _CONFIG_NATIVE = bool(on)


def native_tern() -> bool:
    """Whether this process must train the factors as ternary from the first step.

    The flag is not a hint: with it on, a forward that reads the fp32 factors is a
    LoudError, never a silent fallback. The config arm `weight_qat="ternary_native"`
    sets this itself; the env flag is an explicit A/B override and still refuses a
    mismatch (`spec.validate`).
    """
    raw = os.environ.get("DORMOUSE_TSCT_NATIVE_TERN", "").strip().lower()
    if raw in ("0", "off", "false"):
        #: An explicit veto wins: an A/B arm must be able to turn the arm off.
        return False
    return _CONFIG_NATIVE or raw != ""


#: STE backwards issued by the ternary forward. Read and zeroed once per step.
_ternary_bwd = 0


def take_ternary_bwd() -> int:
    """Return and reset the STE backwards since the previous read."""
    global _ternary_bwd
    seen = _ternary_bwd
    _ternary_bwd = 0
    return seen


def factor_std(std: float, rank: int) -> float:
    """Return `√std · r^{-1/4}`: the per-factor std whose product has the dense draw's std.

    `W' = u·v` has entries `Σ_i u_ji·v_ik`, `r` products of two `N(0, σ²)` draws, so
    `Var(W'_jk) = r·σ⁴`; setting that to the dense draw's `std²` gives `σ² = std/√r`,
    i.e. `σ = √std·r^{-1/4}` — and `Var(y_k) = std²·Σ_j x_j²`, the dense draw's own.
    """
    return float(math.sqrt(std) * rank**-0.25)


def _threshold(shadow: torch.Tensor) -> torch.Tensor:
    """Return the code boundary, from the shadow's magnitude and never from the scale.

    Keeping the boundary detached means the learnable scale controls only the
    magnitude of `{-1, 0, +1}` — growing it cannot silently change the codes.
    """
    return shadow.detach().abs().mean().clamp_min(1e-8)


def scale_for_rms(shadow: torch.Tensor) -> torch.Tensor:
    """Scale whose ternary read has the shadow's fp32 RMS at init.

    A plain `mean|x|` init shrinks a Gaussian factor ~6x after rounding, which moves
    every activation statistic on step 0; restoring the RMS keeps `factor_std`'s
    variance contract while the scale stays a learnable parameter.
    """
    with torch.no_grad():
        t = (shadow / _threshold(shadow)).round().clamp(-1, 1)
        code_rms = t.square().mean().sqrt().clamp_min(1e-8)
        return (shadow.square().mean().sqrt() / code_rms).reshape(()).clone()


class _TernarySTE(torch.autograd.Function):
    """Forward `scale·{-1,0,+1}`; straight-through gradient to the shadow, real one to scale."""

    @staticmethod
    def forward(
        ctx: Any, shadow: torch.Tensor, scale: torch.Tensor, threshold: torch.Tensor
    ) -> torch.Tensor:
        """Return the ternary read; no fp32 shadow value reaches the matmul."""
        t = (shadow / threshold).round().clamp(-1, 1)
        ctx.save_for_backward(t)
        return scale * t

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Pass `grad` to the shadow unchanged and `Σ grad·t` to the learnable scale."""
        global _ternary_bwd
        _ternary_bwd += 1
        (t,) = ctx.saved_tensors
        return grad, (grad * t).sum(), None


def ternarized(shadow: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return the only forward read of a factor: `scale·{-1,0,+1}` with STE."""
    return cast("torch.Tensor", _TernarySTE.apply(shadow, scale, _threshold(shadow)))


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
        #: Muon's aspect scale is a property of the LOGICAL matrix, not of a
        #: factor's shape: `[in, r]` and `[r, out]` are two halves of one
        #: `[in, out]` update. Without this the same LR moves a rank-64 pair
        #: 2.3x harder than a rank-256 pair and the rank A/B measures the scale.
        aspect = max(1.0, in_features / out_features) ** 0.5
        self.u.muon_scale = aspect  # type: ignore[attr-defined]
        self.v.muon_scale = aspect  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x [..., in_features]` → `[..., out_features]` through the two factors."""
        if native_tern() and type(self) is not NativeFactorLinear:
            raise LoudError(
                "DORMOUSE_TSCT_NATIVE_TERN=1 but a TSCT forward read the fp32 factors",
                "build with weight_qat='ternary_native' (which swaps in NativeFactorLinear), "
                "or unset the flag; a flag that runs the fp32 read reports a ternary model "
                "that does not exist",
            )
        return (x @ self.u) @ self.v

    def extra_repr(self) -> str:
        """Show the shapes the bill is quoted for."""
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


class NativeFactorLinear(FactorLinear):
    """A factorised linear trained as ternary from step 0, not converted afterwards.

    `u`/`v` stay fp32 ONLY as the optimizer's shadow state: every forward reads
    `scale·round(u/threshold)` and `scale·round(v/threshold)`, so no matmul ever sees
    the fp32 factor. The scale is a learnable parameter; the straight-through
    estimator sends the shadow's gradient through unchanged and a real gradient to
    the scale. Both the eager expert read and the bank's `stack_matrices` go through
    `u_code()`/`v_code()`, so the GPU kernel's operands are ternarised before launch.
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
        """Draw the shadows as before, then size the two learnable scales to restore RMS."""
        super().__init__(in_features, out_features, rank, std=std, device=device)
        with torch.no_grad():
            self.scale_u = nn.Parameter(scale_for_rms(self.u))
            self.scale_v = nn.Parameter(scale_for_rms(self.v))

    def u_code(self) -> torch.Tensor:
        """Return `scale_u·{-1,0,+1}` — the only `u` value a forward may read."""
        return ternarized(self.u, self.scale_u)

    def v_code(self) -> torch.Tensor:
        """Return `scale_v·{-1,0,+1}` — the only `v` value a forward may read."""
        return ternarized(self.v, self.scale_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two matmuls on ternary codes, one STE backward to the fp32 shadows."""
        return (x @ self.u_code()) @ self.v_code()


def native_ternarize(model: nn.Module) -> int:
    """Swap every built `FactorLinear` to the native arm; return matrices swapped.

    A class swap, not a rebuild: the factors keep the draw (and a checkpoint's loaded
    values), and only the two scale parameters are added. Called after the bank is
    built, so both the routed experts and the shared one are covered.
    """
    swapped = 0
    for module in model.modules():
        if type(module) is FactorLinear:
            module.__class__ = NativeFactorLinear
            with torch.no_grad():
                module.scale_u = nn.Parameter(scale_for_rms(module.u))
                module.scale_v = nn.Parameter(scale_for_rms(module.v))
            swapped += 1
    return swapped


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
