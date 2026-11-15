"""The learning rate as two closed forms, warmup plus cosine and warmup-stable-decay.

Both take a step index and return the rate for that step. The formulas, exactly:

    WARMUP + COSINE (`warmup_cosine`)
        t < warmup:      η(t) = peak·(t+1)/warmup
        t ≥ warmup:      η(t) = floor + (peak-floor)/2·(1 + cos(π·u))
                         u = min(1, (t - warmup)/(total - warmup))

    WARMUP-STABLE-DECAY (Hägele et al. 2024, arXiv:2404.10102 §3, the 1-√ decay)
        t < warmup:      η(t) = peak·(t+1)/warmup
        warmup ≤ t < d:  η(t) = peak
        t ≥ d:           η(t) = floor + (peak-floor)·(1 - √u)
                         u = min(1, (t - d)/(total - d))

`(t+1)/warmup` AND NOT `t/warmup`. The second form is exactly zero at step 0, so the
first optimizer step would move nothing and the run's first loss would be its second
loss. The `+1` puts the first step at `peak/warmup` and the last warmup step at
exactly `peak`, which is what the old stack measured its step-time budget against.

THE FLOOR IS PART OF THE FORM, NOT A CLAMP. A cosine that reaches zero reaches a
learning rate at which no parameter moves; every run that used one here ended its last
few hundred steps having written nothing to the journal that said so. `floor` defaults
to 0.0 because the formula is the formula — pass 0.1·peak to keep the tail alive.
"""

from __future__ import annotations

import math

from dormouse_core.errors import LoudError

__all__ = ["warmup_cosine", "warmup_stable_decay"]


def _check(step: int, warmup: int, total: int, peak: float, floor: float) -> None:
    """Refuse the parameter sets for which the closed forms are undefined."""
    if step < 0:
        raise LoudError(
            f"step={step} is not a step",
            "the schedule starts at 0; a negative index is a counter that never initialised",
        )
    if warmup < 1:
        raise LoudError(
            f"warmup={warmup} divides by zero",
            "at least one warmup step; 0 would make the first rate infinite",
        )
    if total <= warmup:
        raise LoudError(
            f"total={total} is not past warmup={warmup}",
            "the schedule needs a step to decay over: total > warmup",
        )
    if peak <= 0.0:
        raise LoudError(
            f"peak={peak} is not a learning rate",
            "a run at rate 0 is a run that does nothing and reports a flat loss",
        )
    if not 0.0 <= floor <= peak:
        raise LoudError(
            f"floor={floor} is outside [0, peak={peak}]",
            "the floor is the rate the tail settles at; it cannot exceed the peak",
        )


def _warmup(step: int, warmup: int, peak: float) -> float:
    """`peak·(step+1)/warmup` — the ramp both schedules share."""
    return peak * (step + 1) / warmup


def warmup_cosine(step: int, *, warmup: int, total: int, peak: float, floor: float = 0.0) -> float:
    """Warmup then a half-cosine from `peak` to `floor` at `total`, clamped after it."""
    _check(step, warmup, total, peak, floor)
    if step < warmup:
        return _warmup(step, warmup, peak)
    u = min(1.0, (step - warmup) / (total - warmup))
    return floor + (peak - floor) / 2.0 * (1.0 + math.cos(math.pi * u))


def warmup_stable_decay(
    step: int,
    *,
    warmup: int,
    decay_start: int,
    total: int,
    peak: float,
    floor: float = 0.0,
) -> float:
    """Warmup, a constant stretch, then `1-√u` decay from `peak` to `floor`."""
    _check(step, warmup, total, peak, floor)
    if not warmup <= decay_start < total:
        raise LoudError(
            f"decay_start={decay_start} is not in [{warmup}, total={total})",
            "the stable stretch begins at the end of warmup and ends before the run does",
        )
    if step < warmup:
        return _warmup(step, warmup, peak)
    if step < decay_start:
        return peak
    u = min(1.0, (step - decay_start) / (total - decay_start))
    return floor + (peak - floor) * (1.0 - math.sqrt(u))
