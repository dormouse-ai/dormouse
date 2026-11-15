"""The three guards that decide whether a step may be taken, and whether the box keeps up.

Each is a pure function over state the loop carries, so the state it depends on is
visible at the call site and checkpointed with everything else rather than hidden in an
object:

`adaptive_clip` — per-tensor gradient clipping at `z` robust deviations (2504.02507,
2502.11034). Global clipping divides every tensor by one factor, so a spike in one
matrix scales every other tensor's step down with it; that is the measured failure it
is here to replace. The reference is an exponential mean of that tensor's own norms,
seeded by the first step it sees: there is nothing to compare a run's opening step to,
and inventing a reference there clips against a number nobody measured. The reference is
updated with the *clipped* norm, so one huge gradient cannot become the next step's
ceiling.

`is_spike` — a loss above `median + k·MAD` of the last `window` steps. The band is built
from observed losses only, so a skipped step cannot widen the band that would have
skipped it. `floor` is the least scale the band may have, as a fraction of the median: a
converged run's MAD is the batch's own noise, and a band built from it alone calls a
2% fluctuation a spike.

`throughput_drop` — the message for a step 20% below the median of the opening window,
naming the cause. The old project lost hours to a stalled stream nobody noticed, so the
message separates a data stall (the fetch took longer than the compute) from compute
slowing down.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from torch import nn

__all__ = ["CLIP_BETA", "SPIKE_FLOOR", "adaptive_clip", "is_spike", "throughput_drop"]

#: The clip reference's memory, in steps: 0.99 is ~100 steps, so one spike cannot
#: become the reference.
CLIP_BETA = 0.99

#: The least scale the spike band may have, as a fraction of the median loss. 5% with
#: the recipe's k=6 is "30% above the recent median", which is the least a spike is worth.
SPIKE_FLOOR = 0.05


def adaptive_clip(model: nn.Module, running: dict[str, float], z: float) -> tuple[int, int]:
    """Clip every gradient to `z`× its own running norm; return (examined, scaled)."""
    examined = scaled = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        examined += 1
        norm = float(param.grad.detach().norm())
        previous = running.get(name)
        if previous is None:
            running[name] = norm
            continue
        ceiling = z * previous
        if norm > ceiling:
            param.grad.detach().mul_(ceiling / norm)
            norm = ceiling
            scaled += 1
        running[name] = CLIP_BETA * previous + (1.0 - CLIP_BETA) * norm
    return examined, scaled


def is_spike(loss: float, window: Sequence[float], *, k: float, size: int) -> bool:
    """Return whether `loss` is above `median + k·MAD` of a full window of past losses."""
    if len(window) < size:
        return False
    median = statistics.median(window)
    mad = statistics.median([abs(value - median) for value in window])
    scale = max(mad, SPIKE_FLOOR * abs(median), 1e-9)
    return loss > median + k * scale


def throughput_drop(
    rate: float,
    baseline: float,
    *,
    window: int,
    drop: float,
    data_s: float,
    step_s: float,
) -> str | None:
    """Return the warning for a step `drop` below `baseline` tok/s, or None.

    No throttle: a stall warns on every step, and every one of those lines is evidence.
    Suppressing the second one is how a run stays quiet while it burns hours.
    """
    if not math.isfinite(rate) or rate >= (1.0 - drop) * baseline:
        return None
    cause = "data stall" if data_s > step_s else "compute"
    return (
        f"throughput {rate:.0f} tok/s is {100.0 * (1.0 - rate / baseline):.0f}% below the "
        f"{baseline:.0f} tok/s median of the first {window} steps; likely cause: {cause} "
        f"(data {data_s:.3f}s, compute {step_s:.3f}s in this step)"
    )
