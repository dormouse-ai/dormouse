"""The EMA: one extra copy of the weights, updated once per checkpoint.

Worth about 0.002 BPB and free at the end of a run — which is why it is on by default
rather than an experiment. The horizon is a count of checkpoints, not steps: a run that
checkpoints every 500 steps and one that checkpoints every 5000 both want "the last six",
and `decay = 1 - 1/N` gives a weight whose mean age is `N-1` checkpoints.

One copy of the model, not N. Six snapshots of a 1B stack is 24 GB to compute the same
number, and a run that cannot hold six copies silently keeps fewer — which is a different
average reported under the same name.

The average is taken over the *evaluable* weights (`optimizer.eval_weights()`), because a
schedule-free arm holds `Y_t` in its parameters and the model worth averaging is `X_t`.
Averaging the wrong one produces a number that looks like an EMA and is not.
"""

from __future__ import annotations

import torch
from torch import nn

from .optim import MixOptimizer

__all__ = ["apply", "decay", "update"]


def decay(checkpoints: int) -> float:
    """Return the per-checkpoint weight of the past: `1 - 1/N`."""
    return 1.0 - 1.0 / checkpoints


def update(
    average: dict[str, torch.Tensor] | None,
    model: nn.Module,
    optim: MixOptimizer,
    *,
    checkpoints: int,
) -> dict[str, torch.Tensor]:
    """Fold `model`'s evaluable weights into `average`; return the new average.

    Kept in fp32 whatever the run's dtype: an average of bf16 weights accumulates the
    rounding of every checkpoint into the number the run reports at the end.
    """
    keep = decay(checkpoints)
    out: dict[str, torch.Tensor] = {} if average is None else average
    with optim.eval_weights():
        for name, param in model.named_parameters():
            live = param.detach().to(torch.float32)
            previous = out.get(name)
            out[name] = live.clone() if previous is None else keep * previous + (1.0 - keep) * live
    return out


def apply(model: nn.Module, average: dict[str, torch.Tensor] | None) -> nn.Module:
    """Write the average into `model`'s parameters and return it; the live one if empty."""
    if not average:
        return model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in average:
                param.copy_(average[name])
    return model
