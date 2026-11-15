"""Schedule-free training: SF-NorMuon on matrices, SF-AdamW on vectors and tables.

A schedule-free optimizer keeps three iterates and no horizon:

    Y_t = (1 - β₁)·Z_t + β₁·X_t                                              (1)
    Z_{t+1} = Z_t - η_t·(direction at Y_t) - η_t·λ·Z_t                       (2)
    X_{t+1} = (1 - c_{t+1})·X_t + c_{t+1}·Z_{t+1}                            (3)
    c_{t+1} = w_t / Σ_{i≤t} w_i,       w_t = η_t²                            (4)

with `X_1 = Z_1` (Apte et al., arXiv:2605.23061 §1 eqs. 1-3, Algorithm 1 lines 3,
17-18; Defazio et al. 2024 for the general form). The parameter holds `Y_t`, the point
the gradient is taken at; `Z` is the fast iterate on a constant step; `X` is the average.
**`X_T` is the model and `Y_T` is the one being trained**, and (3) is why a growing
corpus needs no restart: `η_t` never falls, and the implicit decay comes from `X` lagging
`Z` (Defazio, arXiv:2605.19095 §2).

§1. THE SPECTRAL ARM. Under the spectral norm the steepest-descent direction is the
polar factor, `ΔW* = -η·polar(∇f)` (2605.23061 §2 eq. 8), since `‖UVᵀ‖_op = 1` and every
other direction spends more operator norm for less first-order decrease. Three
refinements on top (§2.2):

    M_t = μ·M_{t-1} + (1-μ)·G_t,        P_t = polar(M_t)                 (5)
    v_t = β₂·v_{t-1} + (1-β₂)·mean_rows(P_t ⊙ P_t),   v_t ∈ R^m          (6)
    P̂_t = P_t ⊘ (√v_t + ε)                                              (7)
    η̂_t = eta_scale·η_t·√(m·n) / ‖P̂_t‖_F                                (8)
    Z_{t+1} = Z_t·(1 - η_t·λ) - η̂_t·P̂_t                                  (9)

(5) is the explicit momentum buffer: ablating it (μ=0, the polar factor of the raw
gradient) costs 3.14 → 3.28 nats (Fig. 3 left). (6)-(7) is the "Nor": a row-wise second
moment, one scale per output neuron, because the polar factor's rows differ in
magnitude; ablating it costs ~12% tokens to the same loss (Fig. 3 right). (8) is the
Adam-comparable rescale, which makes `‖ΔW‖_F = eta_scale·η_t·√(m·n)` for every shape —
what lets one η and one λ serve both arms (§2.2). Algorithm 1 line 13 prints `max(m,n)`;
the paper's Appendix G computes `sqrt(m*n)`, which is what is implemented here. (9)
decays `Z`, not `Y`: §3 measures decay at `Y` diverging after ~30B tokens where decay at
`Z` stays stable.

§2. THE ADAMW ARM (Algorithm 2 + 2605.19095 §3).
    m_t = β₁·m_{t-1} + (1-β₁)·G_t,      v_t = β₂·v_{t-1} + (1-β₂)·G_t²   (10)
    Z_{t+1} = Z_t·(1 - η_t·λ) - η_t·m̂_t/(√v̂_t + ε),   m̂_t = m_t/(1-β₁^t) (11)
with (1)-(4) around it and `v̂` bias-corrected the same way. The inner momentum in (10)
is a deliberate reinstatement: Algorithm 2 has none, and 2605.19095 §3 measures that
without it SF-AdamW cliffs at 2M tokens where AdamW reaches 4M (Fig. 2, β₁ = 0.75-0.9)
and diverges at rates AdamW survives (Fig. 3a). It is also what makes the AdamW anchor
mean something: average off, the wrapper must be bit-for-bit `adamw.AdamW`.

§3. WHERE THIS DEVIATES, AND WHY. (a) The chain is `muon.polar`'s quintic, not Appendix
G's Jordan triple, which is not the polar factor (`‖QQᵀ-I‖∞ ≈ 3.2e-1`) and fails
`muon.py` §1's 1e-5 anchor. (b) No ColRow here: (6)-(7) already divide each row by its
own scale, so ColRow's unit row norm would leave nothing to divide; (8) is what makes
the step shape-independent instead. (c) `w_t = η_t²` (their eq. 14) rather than
2605.19095 §8 eq. 16's `t^r·max η²`: `r = 1` wins past ~30B tokens and `r = 0` below it,
and a 50 GB corpus at ~4 bytes/token is under 13B. (d) No learning-rate-free rule
(2605.19095 §6): one rate is given; what is dropped is the schedule, not the number.
(e) `β₂` stays 0.95 where the paper's AdamW arm tuned 0.99 — a window, not a mechanism.

§4. EVALUATION IS A SWAP, NOT A LOSS. The parameters hold `Y_t`; the model is
`X_t = (Y_t - (1-β₁)Z_t)/β₁`. Reporting a loss on `Y` is the silent failure of every
schedule-free implementation: the number is worse than the model, by an amount that
starts at zero. So `eval_weights()` writes `X_t` in and `train_weights()` puts `Y_t`
back **by copying the saved tensor**, not by inverting (1) — that inverse is not
bit-exact, and the test asserts `torch.equal`. `optim.py` wraps both so an eval cannot
reach `model.eval()` without the swap.

§5. MEMORY. Per matrix: the parameter, `z`, `mom`, and `v` of `m` entries — 3mn + m, their
Remark 1, the same as AdamW. `X_t` is reconstructed from (1) when needed instead of
being stored, so it is one temporary inside a step, not a buffer for the whole run.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from dormouse_core.errors import LoudError
from torch.optim import Optimizer

from .muon import NS_STEPS, orthogonalise

__all__ = ["SFAdamW", "SFNorMuon"]

__concepts__ = ("SFAdamW", "SFNorMuon")

#: The one base rate the paper tunes at 2× Chinchilla and reuses for both arms
#: (2605.23061 §4: η = 0.01 for SF-AdamW, reused by SF-NorMuon through eq. 8).
SF_LR = 0.01

#: `β₁` in (1): the interpolation, and on the AdamW arm the inner momentum's β₁.
SF_BETA = 0.9

#: `μ` in (5): the paper's momentum buffer before the polar factor.
SF_MOMENTUM = 0.8

#: `eta_scale` in (8), Adam's RMS-equivalent.
SF_ETA_SCALE = 0.2

#: ε in (7) and (11).
SF_EPS = 1e-8

#: λ for both arms: the paper tunes 0.05 and reuses it (§4), and their §3 is the
#: measurement that decay must be at `Z`.
SF_WEIGHT_DECAY = 0.05

#: Default length of the opt-in final decay, as a fraction of the run.
SF_FINAL_DECAY = 0.05


def _anneal(step: int, total: int, fraction: float) -> float:
    """1.0 until the last `fraction` of `total`, then a linear ramp to 0 (see `_rate`)."""
    if total <= 0 or fraction == 0.0:
        return 1.0
    start = int(total * (1.0 - fraction))
    return 1.0 if step <= start else max(0.0, (total - step) / (total - start))


class _ScheduleFree(Optimizer):
    """The wrapper both arms share: (1)-(4), the rate, and the swap in §4."""

    def __init__(
        self,
        params: Any,
        defaults: dict[str, Any],
        *,
        warmup: int,
        final_decay_frac: float,
        average: bool,
    ) -> None:
        """Validate the wrapper's three numbers and put them on every group."""
        beta = defaults["betas"][0]
        if not 0.0 < beta <= 1.0:
            raise LoudError(
                f"sf beta1={beta} is not in (0, 1]",
                "eq. 1 divides by beta1 to reconstruct the average; 0.9 is the paper's",
            )
        if warmup < 0:
            raise LoudError(
                f"sf warmup={warmup} steps is negative",
                "warmup multiplies the rate by min(1, t/warmup); 0 means no ramp",
            )
        if not 0.0 <= final_decay_frac < 1.0:
            raise LoudError(
                f"sf final_decay_frac={final_decay_frac} is outside [0, 1)",
                "the ramp covers the last fraction of a horizon set by begin_final_decay()",
            )
        super().__init__(
            params,
            {
                **defaults,
                "warmup": warmup,
                "final_decay_frac": final_decay_frac,
                "average": average,
                "k": 0,
                "weight_sum": 0.0,
                "total": 0,
            },
        )
        self._saved: dict[torch.Tensor, torch.Tensor] = {}

    def _rate(self, group: dict[str, Any], step: int) -> float:
        """Return `η_t = η·min(1, t/warmup)·anneal(t)` — no term holds a horizon (§3).

        `total == 0` means no horizon has been declared and the rate is flat, which is
        the state a schedule-free run lives in until someone decides to stop.
        """
        warmup = int(group["warmup"])
        rate = float(group["lr"]) * (min(1.0, step / warmup) if warmup else 1.0)
        return rate * _anneal(step, int(group["total"]), float(group["final_decay_frac"]))

    def lr_at(self, step: int) -> float:
        """Return the rate `step` (1-based) will use, from the first group's base rate."""
        if step < 1:
            raise LoudError(
                f"lr_at({step}): the first step is 1",
                "the counter starts at one so min(1, t/warmup) is not zero at the ramp",
            )
        return self._rate(self.param_groups[0], step)

    def begin_final_decay(self, total: int) -> None:
        """Declare the horizon to anneal over, when a run decides to stop.

        This is the annealing lever the data research pays for, and it is the only
        method here that knows a horizon exists: not calling it leaves the rate flat
        forever, which is what lets a corpus grow without a restart.
        """
        current = int(self.param_groups[0]["k"])
        if total <= current:
            raise LoudError(
                f"begin_final_decay(total={total}) is behind step {current}",
                "pass the total step count of the run you are about to finish",
            )
        for group in self.param_groups:
            group["total"] = total

    def _average(self, group: dict[str, Any], rate: float) -> float:
        """`c_{t+1}` in (4), or 1.0 when the average is switched off."""
        weight = rate * rate
        group["weight_sum"] += weight
        if not group["average"] or not group["weight_sum"]:
            return 1.0
        return float(weight / group["weight_sum"])

    def _x_before(
        self, param: torch.Tensor, z: torch.Tensor, beta: float, c: float
    ) -> torch.Tensor | None:
        """`X_t` reconstructed from `Y_t` and `Z_t` before (2) overwrites `Z_t`.

        None when `c == 1`, where `X_t == Z_t` by construction and reconstructing it
        would cost two rounding steps to arrive at a tensor already in hand.
        """
        if c >= 1.0:
            return None
        return (param - (1.0 - beta) * z).div_(beta)

    def _y_after(
        self,
        param: torch.Tensor,
        z: torch.Tensor,
        beta: float,
        c: float,
        x: torch.Tensor | None,
    ) -> None:
        """Write `Y_{t+1}` from the new `Z_{t+1}` and `X_{t+1}` (eq. 1, eq. 3)."""
        if x is None:
            param.copy_(z)
        else:
            param.copy_(x.lerp_(z, c)).lerp_(z, 1.0 - beta)

    @torch.no_grad()
    def eval_weights(self) -> None:
        """Swap `X_t` into the parameters, saving `Y_t` (§4)."""
        if self._saved:
            raise LoudError(
                "eval_weights() was entered twice",
                "take the averaged weights once: train_weights() restores what it saved",
            )
        for group in self.param_groups:
            beta = group["betas"][0]
            for param in group["params"]:
                state = self.state[param]
                if "z" not in state:
                    continue
                self._saved[param] = param.detach().clone()
                param.copy_((param - (1.0 - beta) * state["z"]).div_(beta))

    @torch.no_grad()
    def train_weights(self) -> None:
        """Put `Y_t` back, exactly — a copy, never the inverse of eq. 1 (§4)."""
        for param, y in self._saved.items():
            param.copy_(y)
        self._saved.clear()


class SFAdamW(_ScheduleFree):
    """SF-AdamW: (10)-(11) inside (1)-(4) — vectors, norms and the token table."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float = SF_LR,
        betas: tuple[float, float] = (SF_BETA, 0.95),
        eps: float = SF_EPS,
        weight_decay: float = SF_WEIGHT_DECAY,
        warmup: int = 0,
        final_decay_frac: float = SF_FINAL_DECAY,
        average: bool = True,
    ) -> None:
        """`lr` is η, `betas[0]` is β₁ in both (1) and (10)."""
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
            warmup=warmup,
            final_decay_frac=final_decay_frac,
            average=average,
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Advance `z`, `x` and the parameter one step; return the closure's loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, decay = group["eps"], group["weight_decay"]
            group["k"] += 1
            step = group["k"]
            rate = self._rate(group, step)
            c = self._average(group, rate)
            correction1 = 1.0 - beta1**step
            correction2 = (1.0 - beta2**step) ** 0.5
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "z" not in state:
                    state["z"] = param.detach().clone()
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                z, m, v = state["z"], state["m"], state["v"]
                x = self._x_before(param, z, beta1, c)
                # The same five operations, in the same order, as AdamW's own step:
                # that is what makes §3's anchor a bit-for-bit equality.
                m.lerp_(param.grad, 1.0 - beta1)
                v.mul_(beta2).addcmul_(param.grad, param.grad, value=1.0 - beta2)
                if decay:
                    z.mul_(1.0 - rate * decay)
                z.addcdiv_(m, (v.sqrt() / correction2).add_(eps), value=-rate / correction1)
                self._y_after(param, z, beta1, c, x)
        return loss


class SFNorMuon(_ScheduleFree):
    """SF-NorMuon: (5)-(9) inside (1)-(4) — every matrix parameter."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float = SF_LR,
        betas: tuple[float, float] = (SF_BETA, 0.95),
        momentum: float = SF_MOMENTUM,
        eta_scale: float = SF_ETA_SCALE,
        eps: float = SF_EPS,
        weight_decay: float = SF_WEIGHT_DECAY,
        steps: int = NS_STEPS,
        warmup: int = 0,
        final_decay_frac: float = SF_FINAL_DECAY,
        average: bool = True,
    ) -> None:
        """`momentum` is μ in (5), `eta_scale` the 0.2 in (8), `steps` Newton–Schulz."""
        if not 0.0 <= momentum < 1.0:
            raise LoudError(
                f"sf momentum={momentum} is not in [0, 1)",
                "0.8 is the paper's; 0 takes the polar factor of the raw gradient, "
                "which their Fig. 3 measures at 3.28 against 3.14 nats",
            )
        if steps < 1:
            raise LoudError(
                f"sf ns steps={steps} runs no iteration",
                "5 is the recipe's; see muon.py §3 for what fewer reaches",
            )
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "momentum": momentum,
                "eta_scale": eta_scale,
                "eps": eps,
                "weight_decay": weight_decay,
                "steps": steps,
            },
            warmup=warmup,
            final_decay_frac=final_decay_frac,
            average=average,
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Advance `z`, `x` and the parameter one step; return the closure's loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            momentum, eta_scale = group["momentum"], group["eta_scale"]
            steps, split = group["steps"], group.get("head_split")
            eps, decay = group["eps"], group["weight_decay"]
            group["k"] += 1
            step = group["k"]
            rate = self._rate(group, step)
            c = self._average(group, rate)
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "z" not in state:
                    state["z"] = param.detach().clone()
                    state["mom"] = torch.zeros_like(param)
                    state["v"] = torch.zeros(
                        param.shape[0], dtype=torch.float32, device=param.device
                    )
                z, mom, v = state["z"], state["mom"], state["v"]
                x = self._x_before(param, z, beta1, c)
                mom.mul_(momentum).add_(param.grad, alpha=1.0 - momentum)
                direction = orthogonalise(mom, head_split=split, steps=steps).to(param.dtype)
                scale = (direction * direction).mean(dim=1, dtype=torch.float32)
                v.mul_(beta2).add_(scale, alpha=1.0 - beta2)
                phat = direction / (v.sqrt() + eps).unsqueeze(1)
                if decay:
                    z.mul_(1.0 - rate * decay)
                eta = eta_scale * rate * math.sqrt(param.numel())
                z.add_(phat, alpha=-float(eta / phat.norm().clamp_min(1e-12)))
                self._y_after(param, z, beta1, c, x)
        return loss
