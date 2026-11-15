"""The run's recipe: what a step consumes, which stages feed it, and every refusal.

Eight items, each with the measurement that put it here. None is a preference:

1. **Warmup of ≈ N_active tokens.** `warmup_steps=1900` at 24,576 tokens/step is 46.7M
   tokens. 2406.19146 sweeps a 64× warmup range and the whole span is 0.026 BPB, so the
   value is not where the risk is: it is a config value with a measured default, and a
   run that omits it gets the plateau without the ramp.
2. **Per-tensor adaptive clipping at z=2.5, not global norm.** Global clipping divides
   every tensor by one factor, so one layer's spike suppresses every other layer's step;
   2504.02507 and 2502.11034 measure that as a failure to stop spikes. The mechanism is
   in `guards.py`; the knob is here.
3. **EMA over the last 6 checkpoints**, worth ≈0.002 BPB and free. It lives beside the
   optimizer state, is checkpointed, and `ema=False` disables it with a zero counter.
4. **Stages with an anneal.** The stage list comes from the mixture and must cover
   exactly `[0, steps)`: an uncovered tail is how an anneal silently does not happen, so
   the loop refuses it at construction instead of running the tail at full rate. The last
   ~22% of steps anneal to zero and carry the clean-data stage.
5. **Spikes are skipped, not rolled back** — `guards.py` owns the band, `spike_k` and
   `max_skips` are here.
6. **Byte accounting from bytes.** Every batch declares the positions the loss covers and
   the payload bytes they stand for, and `check_batch` refuses one whose two numbers
   disagree with each other or with the stream's manifest. A BPB computed from a token
   count times an assumed ratio is not a measurement.
7. **The journal is the run**; `loop.py` writes it.
8. **A throughput guard** after 20% below the median of the opening `guard_window` steps.

A stage list that tiles the budget is the one thing this module cannot check late: by
the time a run discovers the hole it is 3000 steps in, so `LoopConfig.__post_init__` is
where it fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import torch
from dormouse_core.errors import LoudError

from .schedule import warmup_stable_decay

__all__ = [
    "ANNEAL_FLOOR",
    "Batch",
    "BatchStream",
    "LoopConfig",
    "Stage",
    "annealed_schedule",
    "check_batch",
    "check_cover",
    "config_from_dict",
]

__concepts__ = ("Batch", "BatchStream", "LoopConfig", "Stage")

#: The anneal is the last 20–25% of a run; 0.22 is the middle of that band.
ANNEAL_FRACTION = 0.22

#: The last step's rate, as a fraction of peak, that still counts as an anneal. The
#: recipe's 22% window lands at 0.02–0.03 here; a fifth of peak is the most a tail can
#: still be at and be called one, and a schedule that ignores its horizon sits at 1.0.
ANNEAL_FLOOR = 0.2


class BatchStream(Protocol):
    """What the loop needs from a mixture: batches by step, and a byte ratio.

    `batch(step, tokens)` is a pure function of `step`. That is the whole resume
    contract: `data/bin.py` positions by arithmetic, so restoring the step restores the
    stream, and a stream that carries a buffer of its own is a stream whose resume
    cannot be checked against a single run.
    """

    bytes_per_token: float

    def batch(self, step: int, tokens: int) -> Batch:
        """Return the batch that belongs to global `step`."""
        ...


@dataclass(frozen=True)
class Batch:
    """One batch, with what it cost: the positions the loss covers and their bytes."""

    ids: torch.Tensor
    tokens: int
    payload_bytes: int
    stage: str


@dataclass(frozen=True)
class Stage:
    """A stretch of the step budget fed by one mixture: `[start, end)`."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class LoopConfig:
    """The run's recipe. Every default is a measured number, named in the docstring."""

    steps: int
    stages: tuple[Stage, ...]
    warmup_steps: int = 1900
    clip_z: float = 2.5
    clip: bool = True
    ema: bool = True
    ema_checkpoints: int = 6
    anneal_frac: float = ANNEAL_FRACTION
    spike_k: float = 6.0
    spike_window: int = 50
    max_skips: int = 5
    tokens_per_step: int = 24_576
    accum: int = 1
    guard_window: int = 200
    guard_drop: float = 0.20
    eval_batches: int = 8

    def __post_init__(self) -> None:
        """Refuse a recipe that cannot anneal, cannot warm up, or cannot be resumed."""
        if self.steps < 1:
            raise LoudError(
                f"steps={self.steps} is not a run",
                "the budget is the number of batches the mixture must cover",
            )
        if not 0.20 <= self.anneal_frac <= 0.25:
            raise LoudError(
                f"anneal_frac={self.anneal_frac} is outside [0.20, 0.25]",
                "the anneal is the last 20-25% of a run; a shorter one is not an anneal "
                "and a longer one is training on the clean stage only",
            )
        if not self.stages:
            raise LoudError(
                "no stages: the mixture has no schedule",
                f"one stage covering [0, {self.steps}) is the minimum",
            )
        check_cover(self.stages, self.steps)
        if self.warmup_steps < 1:
            raise LoudError(
                f"warmup_steps={self.warmup_steps} divides by zero",
                "the schedule's ramp needs at least one step",
            )
        if self.warmup_steps >= self.anneal_start:
            raise LoudError(
                f"warmup_steps={self.warmup_steps} overlaps the anneal at {self.anneal_start}",
                "the recipe is a ramp, a stable stretch, then the anneal; a run that "
                "never leaves warmup is a run that never reaches its peak rate",
            )
        if self.ema_checkpoints < 2:
            raise LoudError(
                f"ema_checkpoints={self.ema_checkpoints} averages nothing",
                "the EMA horizon is a count of checkpoints; 6 is the measured default",
            )
        for name, value in (
            ("clip_z", self.clip_z),
            ("spike_k", self.spike_k),
            ("guard_window", self.guard_window),
            ("max_skips", self.max_skips),
        ):
            if value <= 0:
                raise LoudError(
                    f"{name}={value} is not a positive number",
                    "a zero threshold disables the mechanism it is supposed to calibrate",
                )

    @property
    def anneal_start(self) -> int:
        """The first step of the anneal: the last ~22% of the budget."""
        return self.steps - round(self.anneal_frac * self.steps)


def check_cover(stages: tuple[Stage, ...], steps: int) -> None:
    """Refuse a stage list that does not tile `[0, steps)` exactly."""
    if stages[0].start != 0:
        raise LoudError(
            f"stages start at {stages[0].start}, not 0",
            "the first stage owns step 0; the mixture has no batch before it",
        )
    for previous, following in pairwise(stages):
        if previous.end != following.start:
            gap = (previous.end, following.start)
            raise LoudError(
                f"stages {previous.name!r} and {following.name!r} leave "
                f"[{gap[0]}, {gap[1]}) uncovered",
                "the mixture's stages must tile the budget: a gap is a range of steps no "
                "stage feeds, and it is discovered as a crash 3000 steps in",
            )
    if stages[-1].end != steps:
        raise LoudError(
            f"the stages end at {stages[-1].end} but the budget is {steps}: "
            f"[{stages[-1].end}, {steps}) has no stage",
            f"extend the last stage to {steps} or add one; an uncovered tail is the "
            "anneal that never happens",
        )


def annealed_schedule(cfg: LoopConfig) -> Callable[[int], float]:
    """Return the recipe's rate, as a fraction of peak: warmup, stable, then the anneal.

    The stable stretch ends where the anneal begins, so the decay and the clean-data
    stage start on the same step — the two halves of item 4 are one window.
    """

    def rate(step: int) -> float:
        """Return `step`'s rate as a fraction of peak."""
        return float(
            warmup_stable_decay(
                step,
                warmup=cfg.warmup_steps,
                decay_start=cfg.anneal_start,
                total=cfg.steps,
                peak=1.0,
            )
        )

    return rate


def check_batch(batch: Batch, *, stage: Stage | None, stream: BatchStream, step: int) -> None:
    """Refuse a batch whose tokens or bytes disagree with what it carries.

    The byte count is the measurement, so it is checked against the ids it came from
    and against the stream's manifest before it reaches the journal.
    """
    ids = batch.ids
    if ids.dim() != 2 or ids.shape[1] < 2:
        raise LoudError(
            f"batch ids {tuple(ids.shape)} is not [rows, seq+1]",
            "the +1 is the target shift: inputs ids[:, :-1], targets ids[:, 1:]",
        )
    predicted = ids.shape[0] * (ids.shape[1] - 1)
    if batch.tokens != predicted:
        raise LoudError(
            f"the batch declares {batch.tokens} tokens and carries {predicted}",
            "tokens are the positions the loss covers, rows x seq - not the padded "
            "width and not rows x (seq+1)",
        )
    expected = batch.tokens * float(stream.bytes_per_token)
    if abs(batch.payload_bytes - expected) > 1.0:
        raise LoudError(
            f"the batch declares {batch.payload_bytes} B for {batch.tokens} tokens, "
            f"but the stream's manifest says {stream.bytes_per_token} B/token "
            f"= {expected:.1f} B",
            "the manifest and the data disagree: fix the manifest beside the stream, "
            "or the BPB this run reports will be wrong by that ratio",
        )
    if stage is not None and batch.stage != stage.name:
        raise LoudError(
            f"step {step} is in stage {stage.name!r} and the mixture returned {batch.stage!r}",
            "the stage schedule and the mixture's own boundaries disagree; the anneal's "
            "clean-data stage would silently keep the bulk data",
        )


def config_from_dict(raw: dict[str, Any]) -> LoopConfig:
    """Rebuild the config a checkpoint was written under, stages included."""
    data = dict(raw)
    data["stages"] = tuple(Stage(**stage) for stage in raw["stages"])
    return LoopConfig(**data)
