"""AdamW, and the one method a schedule needs.

    ADAMW (Loshchilov & Hutter 2019, arXiv:1711.05101 Alg. 2, decoupled decay)
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        m̂_t = m_t/(1-β₁^t),  v̂_t = v_t/(1-β₂^t)
        W_t = W_{t-1}·(1 - η·λ) - η·m̂_t/(√v̂_t + ε)

Torch's implementation is the formula: decoupled decay, bias-corrected moments. Its
first step with `λ=0` is `-η·g/(|g|+ε)`, i.e. `η·sign(g)` to within ε, and
`tests/test_optim.py` pins that as the anchor.

A bf16 RUN KEEPS ITS MOMENTS IN FP32, which is why `step` is overridden. Torch allocates
`exp_avg`/`exp_avg_sq` with `zeros_like(param)`, so a bf16 parameter would get bf16 moments
and be updated in bf16 — the silent quality leak `train/precision.py` refuses. The mixed
path below is the same arithmetic in fp32: moments are created fp32, the gradient is cast
up, and the new weight is computed in fp32 and written back into the parameter. An fp32
model takes the inherited `step` unchanged, so the reference run stays bit-identical.

`set_lr` is the seam a schedule writes to: `warmup_cosine` and `warmup_stable_decay`
return a rate, the loop calls `set_lr`, and no other method of this class has state
that knows what step it is on.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["ADAMW_LR", "AdamW"]

#: AdamW's rate on the embedding table and the norms (the recipe's `×0.15`).
ADAMW_LR = 3e-3


class AdamW(torch.optim.AdamW):
    """AdamW with the formula above in this docstring and one knob for the schedule."""

    def set_lr(self, lr: float) -> None:
        """Set one rate on every group, which is what a schedule produces."""
        for group in self.param_groups:
            group["lr"] = lr

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Advance every parameter; an fp32 model takes torch's own step."""
        mixed = any(
            param.dtype != torch.float32 for group in self.param_groups for param in group["params"]
        )
        if not mixed:
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps, decay = group["eps"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                state["step"] += 1
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                moment, second = state["exp_avg"], state["exp_avg_sq"]
                grad = param.grad.to(torch.float32)
                moment.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denominator = (second / correction2).sqrt_().add_(eps)
                weight = param.detach().float()
                weight.mul_(1.0 - lr * decay)
                weight.addcdiv_(moment, denominator, value=-lr / correction1)
                param.copy_(weight)
        return loss
