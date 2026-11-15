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
"""

from __future__ import annotations

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
