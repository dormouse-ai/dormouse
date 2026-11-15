"""AdamW, and the one method a schedule needs.

    ADAMW (Loshchilov & Hutter 2019, arXiv:1711.05101 Alg. 2, decoupled decay)
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        m̂_t = m_t/(1-β₁^t),  v̂_t = v_t/(1-β₂^t)
        W_t = W_{t-1}·(1 - η·λ) - η·m̂_t/(√v̂_t + ε)

Torch's implementation is the formula: decoupled decay, bias-corrected moments. Its
first step with `λ=0` is `-η·g/(|g|+ε)`, i.e. `η·sign(g)` to within ε, and
`tests/test_optim.py` pins that as the anchor.

`set_lr` is the seam a schedule writes to: `warmup_cosine` and `warmup_stable_decay`
return a rate, the loop calls `set_lr`, and no other method of this class has state
that knows what step it is on.

§1. THE TWO STATE WIDTHS, AND WHICH ONE RUNS. `state_bits` is `v` and `first_bits` is
`m`; both default to 32, and a run at 32 is `torch.optim.AdamW.step` itself — the
inherited kernel, untouched, which is what makes "the bits knob off is the optimiser
that was here" a property of the code rather than a hope. With either moment at 8 the
class runs `_step_lowbit`, which is the same five operations in the same order on the
same values (the test pins it bit-for-bit against the inherited path) with
`LowBitState` in place of the tensors: dequantise a window, `lerp_`/`addcmul_` in fp32,
re-quantise. `MixOptimizer` holds the token table at 32 bits whatever is asked — the
paper's stable-embedding rule, see `lowbit.py` §3.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from dormouse_core.errors import LoudError

from .lowbit import BLOCK, LowBitState, adopt, check, plain_state_dict, revive_state_dict

__all__ = ["ADAMW_LR", "AdamW"]

#: AdamW's rate on the embedding table and the norms (the recipe's `×0.15`).
ADAMW_LR = 3e-3


class AdamW(torch.optim.AdamW):
    """AdamW with the formula above in this docstring and one knob for the schedule."""

    def __init__(
        self,
        params: Any,
        lr: float = ADAMW_LR,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        state_bits: int = 32,
        first_bits: int = 32,
        block: int = BLOCK,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Set the defaults every group inherits, including the two state widths (§1).

        The inherited keywords are forwarded unchanged. The four that change what the
        update *is* — `amsgrad`, `maximize`, `capturable`, `differentiable` — are
        refused with an 8-bit moment rather than ignored: the low-bit loop below is
        `torch.optim`'s algebra, not its every branch. `foreach` and `fused` are how the
        elementwise calls are batched, which the loop decides for itself.
        """
        check(state_bits, block)
        check(first_bits, block)
        if 8 in (state_bits, first_bits):
            for name, asked in (
                ("amsgrad", amsgrad),
                ("maximize", maximize),
                ("capturable", capturable),
                ("differentiable", differentiable),
            ):
                if asked:
                    raise LoudError(
                        f"{name}=True with an 8-bit moment",
                        "this arm runs torch's algebra with a quantised buffer, not every "
                        "branch of it; run the arm at 32 bits, or drop the keyword",
                    )
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        for group in self.param_groups:
            group.setdefault("state_bits", state_bits)
            group.setdefault("first_bits", first_bits)
            group.setdefault("block", block)

    def set_lr(self, lr: float) -> None:
        """Set one base rate on every group, scaled by the group's μP multiplier."""
        for group in self.param_groups:
            group["lr"] = lr * group.get("lr_mult", 1.0)

    def step(self, closure: Any = None) -> Any:
        """Advance one step: torch's kernel at 32 bits, the same algebra at 8 (§1)."""
        if all(
            group.get("state_bits", 32) == 32 and group.get("first_bits", 32) == 32
            for group in self.param_groups
        ):
            return super().step(closure=closure)
        return self._step_lowbit(closure)

    def state_dict(self) -> dict[str, Any]:
        """The moments as plain tensors, so `weights_only=True` can read them back."""
        return plain_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a checkpoint into the container, on each parameter's own device."""
        super().load_state_dict(revive_state_dict(state_dict))
        adopt(self.state)

    @torch.no_grad()
    def _step_lowbit(self, closure: Any = None) -> Any:
        """AdamW's update with a moment in `LowBitState`, op for op with `torch.optim`."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, lr, decay = group["eps"], group["lr"], group["weight_decay"]
            #: `get`: a checkpoint written before the knob existed restores its own
            #: `param_groups`, and the only state it can hold is the dense one.
            bits = group.get("state_bits", 32)
            first = group.get("first_bits", 32)
            block = group.get("block", BLOCK)
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = LowBitState(param, bits=first, block=block)
                    state["exp_avg_sq"] = LowBitState(param, bits=bits, block=block)
                state["step"] += 1
                step = state["step"]
                exp_avg: LowBitState = state["exp_avg"]
                exp_avg_sq: LowBitState = state["exp_avg_sq"]
                exp_avg.apply(partial(_first_moment, weight=1.0 - beta1), grad)
                exp_avg_sq.apply(partial(_second_moment, keep=beta2, weight=1.0 - beta2), grad)
                if decay:
                    param.mul_(1.0 - lr * decay)
                correction1 = 1.0 - beta1**step
                correction2 = (1.0 - beta2**step) ** 0.5
                #: The casts are no-ops at 32 bits (the state already has the parameter's
                #: dtype) and are what keeps the 8-bit arithmetic from promoting the
                #: parameter on the way back in.
                numerator = exp_avg.dense().to(param.dtype)
                denominator = (exp_avg_sq.dense().sqrt() / correction2).add_(eps).to(param.dtype)
                param.addcdiv_(numerator, denominator, value=-lr / correction1)
        return loss


def _first_moment(moment: torch.Tensor, grad: torch.Tensor, *, weight: float) -> None:
    """`m ← lerp(m, g, 1-β₁)`, torch's `exp_avg.lerp_(grad, 1 - β₁)`."""
    moment.lerp_(grad, weight)


def _second_moment(moment: torch.Tensor, grad: torch.Tensor, *, keep: float, weight: float) -> None:
    """`v ← β₂·v + (1-β₂)·g²`, torch's `mul_` then `addcmul_`, in that order."""
    moment.mul_(keep).addcmul_(grad, grad, value=weight)
