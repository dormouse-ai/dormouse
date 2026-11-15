"""What a resume needs, in one file: weights, moments, EMA, clip state, and the step.

A checkpoint is a claim that the next batch is the one that would have come. Every
piece a run carries between steps is in it, because the first thing a partial
checkpoint breaks is the first post-resume step, silently:

* the weights, as the arms left them (under `sf` that is `Y_t`, not the eval `X_t`);
* both arms' moments, and the mode — `MixOptimizer.state_dict` refuses a mode mismatch
  rather than restoring `z` into `momentum`;
* the EMA, which is the model `final_model()` returns;
* the per-tensor clip estimates, without which the first resumed step clips against a
  reference of zero and is not the step the run would have taken;
* the robust-loss window and the throughput baseline, for the same reason;
* torch's RNG state, so a stack with dropout resumes the same stream;
* the step itself. The data position is not stored: it *is* the step, because
  `data/bin.py` reads by arithmetic on the step index.

`MIRROR` is the single list of the loop's own bookkeeping, used by both `capture` and
`restore`, so a field cannot be saved and not restored. The file is written to a
temporary name and renamed, so a killed run keeps its last good checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from dormouse_core.errors import LoudError
from torch import nn

from .dist import is_main
from .optim import MixOptimizer
from .recipe import LoopConfig

__all__ = ["CKPT", "MIRROR", "Resumable", "Snapshot", "capture", "load", "restore", "save"]

#: The single file a run keeps: the last checkpoint, not a rolling set. A rolling set
#: is what a resume has to choose from, and picking the wrong one is a silent replay.
CKPT = "ckpt.pt"

#: The keys a checkpoint must carry. A missing one is a version skew, and a version
#: skew defaulted is a run that resumes from a step it already took.
KEYS = ("step", "config", "model", "optim", "ema", "rng", "state")

#: The loop's own bookkeeping, by attribute name. One list, read by both halves.
MIRROR = (
    "tokens",
    "payload_bytes",
    "nats",
    "evals",
    "skips",
    "spike_skips",
    "consecutive",
    "clip_events",
    "throughput_warnings",
    "clip_norms",
    "losses",
    "rates",
    "baseline",
)


class Resumable(Protocol):
    """The loop, structurally: what `capture` reads and `restore` writes.

    Structural rather than imported, so this module never imports the loop and the
    dependency runs one way. `bytes` is the payload byte count — the number BPB is
    computed from, not the size of a batch in the model's dtype.
    """

    step: int
    tokens: int
    payload_bytes: int
    nats: float
    evals: int
    skips: int
    spike_skips: int
    consecutive: int
    clip_events: int
    throughput_warnings: int
    clip_norms: dict[str, float]
    losses: list[float]
    rates: list[float]
    baseline: float | None
    ema: dict[str, torch.Tensor] | None
    cfg: LoopConfig
    model: nn.Module
    optim: MixOptimizer


@dataclass(frozen=True)
class Snapshot:
    """One resumable point: the step, and everything the next batch depends on."""

    step: int
    config: dict[str, Any]
    model: dict[str, torch.Tensor]
    optim: dict[str, Any]
    ema: dict[str, torch.Tensor] | None
    rng: torch.Tensor
    state: dict[str, Any]


def capture(run: Resumable) -> Snapshot:
    """Read a run into a snapshot: the weights, the moments, the EMA and the bookkeeping."""
    return Snapshot(
        step=run.step,
        config=asdict(run.cfg),
        model=run.model.state_dict(),
        optim=run.optim.state_dict(),
        ema=None if run.ema is None else dict(run.ema),
        rng=torch.get_rng_state(),
        state={name: getattr(run, name) for name in MIRROR},
    )


def restore(run: Resumable, snapshot: Snapshot) -> None:
    """Write a snapshot back into a run, refusing a checkpoint that is missing a field."""
    missing = [name for name in MIRROR if name not in snapshot.state]
    if missing:
        raise LoudError(
            f"{snapshot.step}: the checkpoint does not carry {', '.join(missing)}",
            "an older checkpoint than this loop writes; resuming without a field is "
            "resuming from a step the run already took",
        )
    run.model.load_state_dict(snapshot.model)
    run.optim.load_state_dict(snapshot.optim)
    run.ema = snapshot.ema
    run.step = snapshot.step
    for name in MIRROR:
        setattr(run, name, snapshot.state[name])
    run.losses = list(run.losses)
    run.rates = list(run.rates)
    torch.set_rng_state(snapshot.rng)


def save(run_dir: str | Path, snap: Snapshot) -> Path:
    """Write `run_dir/ckpt.pt` atomically and return the path, on rank 0 only.

    Every rank holds the same state — DDP keeps the weights identical and the shards
    agree on the step — so one writer is enough. Two ranks writing the same temporary
    file is a corrupted checkpoint that still loads, which is worse than a missing one.
    """
    path = Path(run_dir) / CKPT
    if not is_main():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": snap.step,
            "config": snap.config,
            "model": snap.model,
            "optim": snap.optim,
            "ema": snap.ema,
            "rng": snap.rng,
            "state": snap.state,
        },
        tmp,
    )
    os.replace(tmp, path)  # noqa: PTH105 - atomic rename, not a path rewrite
    return path


def load(run_dir: str | Path) -> Snapshot:
    """Read `run_dir/ckpt.pt`, refusing a file that cannot be a resume point."""
    path = Path(run_dir) / CKPT
    if not path.exists():
        raise LoudError(
            f"no checkpoint at {path}",
            "point at the run directory that holds ckpt.pt, or start a new run: a "
            "resume without weights is a fresh run wearing a step number",
        )
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise LoudError(f"{path}: not a checkpoint mapping", "write it with checkpoint.save")
    missing = [key for key in KEYS if key not in raw]
    if missing:
        raise LoudError(
            f"{path}: missing {', '.join(missing)}",
            "the checkpoint came from an older loop; a resume that defaults a field "
            "resumes from a step the run already took",
        )
    step = raw["step"]
    if not isinstance(step, int):
        raise LoudError(f"{path}: step={step!r} is not an int", "a resume starts at a step")
    return Snapshot(
        step=step,
        config=raw["config"],
        model=raw["model"],
        optim=raw["optim"],
        ema=raw["ema"],
        rng=raw["rng"],
        state=raw["state"],
    )
