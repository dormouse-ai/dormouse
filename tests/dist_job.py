"""The job the distribution gates run: one function, one card or N.

`tests/test_dist.py` starts this file once per rank as a subprocess, with `RANK` and
`WORLD_SIZE` in the environment — the way a launcher supplies them. Every mode goes
through the *same* `Trainer`, `train_step` and `MixOptimizer`; the only things that know
the world are `dist.init`, `dist.wrap`, `dist.shard_stream` and `dist.RankJournal`, which
is the claim the gates check.

    python tests/dist_job.py --run-dir DIR --out FILE [--mode train|resume] [--bug]

`--bug` injects the defect the loss normalisation exists to prevent: each micro-batch
contributes its own mean (`accum` times the correct gradient), which is the standard
per-micro-batch-mean bug. It is one multiply on the loss, in `step.py`'s own function, so
the code path under test is the real one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from dormouse_core.journal import Journal
from stacks import reference_stack

from dormouse.train import dist, step
from dormouse.train.loop import Trainer
from dormouse.train.optim import MixOptimizer
from dormouse.train.recipe import Batch, LoopConfig, Stage, annealed_schedule

#: The *global* batch: at world 2 each rank holds half of these rows. The config does not
#: change with the world — that is the whole claim.
ROWS, SEQ = 8, 16
STEPS = 20
ACCUM = 2
#: `steps + 1`, so the last step still has a stage and a resumed run can take one more.
BUDGET = STEPS + 1
SMALL = {"vocab": 16, "d_model": 16, "n_layers": 1, "n_heads": 2, "head_dim": 8, "d_ffn": 32}


class ToyStream:
    """Ids are a pure function of the step, so every rank slices the same batch."""

    bytes_per_token = 4.0

    def batch(self, step_index: int, tokens: int) -> Batch:
        """Return the step's whole batch; the shard cuts it."""
        generator = torch.Generator().manual_seed(1000 + step_index)
        ids = torch.randint(0, 4, (ROWS, SEQ + 1), generator=generator)
        return Batch(
            ids=ids,
            tokens=ROWS * SEQ,
            payload_bytes=ROWS * SEQ * 4,
            stage="bulk",
        )


class Recorder:
    """A stream that remembers the batch each rank actually consumed."""

    def __init__(self, stream: Any) -> None:
        """Wrap a stream; `seen` collects `(step, ids, tokens, bytes)`."""
        self.stream = stream
        self.seen: list[dict[str, Any]] = []

    @property
    def bytes_per_token(self) -> float:
        """The manifest's ratio."""
        return float(self.stream.bytes_per_token)

    def batch(self, step_index: int, tokens: int) -> Batch:
        """Record, then delegate."""
        batch: Batch = self.stream.batch(step_index, tokens)
        self.seen.append(
            {
                "step": step_index,
                "ids": batch.ids.tolist(),
                "tokens": batch.tokens,
                "bytes": batch.payload_bytes,
            }
        )
        return batch


def config() -> LoopConfig:
    """The recipe: one stage, a short warmup, the anneal on the last steps."""
    return LoopConfig(
        steps=BUDGET,
        stages=(Stage("bulk", 0, BUDGET),),
        warmup_steps=2,
        tokens_per_step=ROWS * SEQ,
        accum=ACCUM,
        spike_window=10,
    )


def break_normalisation(accum: int) -> None:
    """Inject the per-micro-batch mean: the loss is not divided by `accum` on the way out."""
    original = step.chunked_cross_entropy

    def summed(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return original(logits, targets) * accum

    step.chunked_cross_entropy = summed


def main() -> int:
    """Run one rank of one job and write what the gates compare."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("train", "resume"), default="train")
    parser.add_argument("--bug", action="store_true")
    args = parser.parse_args()

    if args.bug:
        break_normalisation(ACCUM)
    world = dist.init()
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    optim = MixOptimizer(model)
    cfg = config()
    journal = dist.RankJournal(
        Journal.open(Path(args.run_dir) / "run.jsonl"),
        topo=world,
    )
    recorder = Recorder(dist.shard_stream(ToyStream()))
    wrapped = dist.wrap(model)
    trainer = (
        Trainer.load(
            args.run_dir,
            model=wrapped,
            stream=recorder,
            optim=optim,
            schedule=annealed_schedule(cfg),
            journal=journal,
        )
        if args.mode == "resume"
        else Trainer(wrapped, recorder, optim, annealed_schedule(cfg), cfg, journal)
    )
    trainer.train(steps=BUDGET if args.mode == "resume" else STEPS)
    dist.barrier()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    steps = [record for record in journal.raw if "loss" in record]
    out.write_text(
        json.dumps(
            {
                "rank": world.rank,
                "world": world.world,
                "step": trainer.step,
                # The un-reduced records: what *this* rank said the step was.
                "local": steps,
                "seen": recorder.seen,
                "loss": trainer.nats / trainer.tokens if trainer.tokens else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
