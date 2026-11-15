"""Data-parallel training: one code path on one card and on N.

The claim under test is a price, not a feature: bytes per FLOP is only a cheaper run if
the same recipe runs on more cards without being rewritten. So this module is not a
second trainer — `loop.py` walks the steps, `step.py` takes them — it supplies the four
things that genuinely differ with the world:

* **The world comes from the environment** (`RANK`, `WORLD_SIZE`), never from the
  config. A config that names a `world_size` is refused by `refuse_world_size`: a file
  and a launcher that disagree start a run that is not the run that was described, and
  nothing in the journal would say which one ran.
* **The batch is cut, not repeated.** `shard_stream` gives each rank a disjoint slice of
  the step's rows and keeps `tokens`, `payload_bytes` and `stage` as the *step's*, so
  byte accounting, the throughput guard, `report.audit` and the checkpoint's ledger stay
  global. A global batch that does not divide by the world is refused (see
  `shard_stream`).
* **The loss is a mean over the global token count.** `step.py` divides every
  micro-batch by `accum` and DDP averages the ranks, so the divisor is
  `accum × world × micro` — the whole step's tokens. A rank that normalised by its own
  shard instead would be training a different model, which is the defect
  `tests/test_dist.py::test_per_microbatch_mean_fails_the_equivalence_gate` injects and
  the equivalence gate catches.
* **One writer.** Metrics are reduced by `all_reduce_metrics` so the journal records the
  global numbers once and rank 0 is the only process that touches the run directory. Two
  journals for one run is how a run becomes unreproducible, and a checkpoint carrying
  DDP's `module.` prefix is how a two-card file stops loading on one card; `wrap` keeps
  the bare model's state dict for exactly that reason.

The world is not a mask over the single-card code: `init`, `wrap`, `shard_stream` and
`RankJournal` are no-ops at `world == 1`, so one card runs the same run with the same
config.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from dormouse_core.errors import LoudError
from dormouse_core.journal import KINDS, Journal
from torch import nn

if TYPE_CHECKING:
    from .recipe import Batch, BatchStream

__all__ = [
    "RANK",
    "WORLD",
    "RankJournal",
    "Topology",
    "all_reduce_metrics",
    "barrier",
    "init",
    "is_main",
    "refuse_world_size",
    "shard_stream",
    "topology",
    "wrap",
]

__concepts__ = ("RankJournal", "Topology")

#: The two variables the world is read from. `MASTER_ADDR`/`MASTER_PORT` are torch's,
#: and they are the launcher's too: `init` passes `init_method="env://"`.
RANK = "RANK"
WORLD = "WORLD_SIZE"

#: How each journal field becomes the global number.
#:
#: * `STEP`: the step's size. Every rank carries the same number of rows, so the step's
#:   tokens and bytes are a rank's times the world — and a rank that disagrees is a
#:   shard that is not a partition, refused rather than averaged into a plausible number
#:   (the loss normalisation is only the global token mean if the shards are equal).
#: * `SUM`: work the whole job did this step — clip events, tokens/s.
#: * `MAX`: the worst rank, for a duration or a norm that must already be identical.
#: * `MEAN`: a per-token density, so the ranks' values average to the job's (the losses
#:   are means over equal shards, the BPBs are nats/byte).
#: * Any other field is rank 0's: a step index, a stage name or a config value is the
#:   same everywhere, and averaging a step number is how a journal stops being readable.
STEP_FIELDS = ("bytes", "tokens")
SUM_FIELDS = ("clip_events", "tokens_per_s")
MAX_FIELDS = ("data_s", "grad_norm", "step_s")
MEAN_FIELDS = ("bpb", "loss")

#: The events whose `tokens`/`bytes` are one rank's shard. An eval's window is the same
#: window on every rank, so multiplying it by the world would invent data nobody read.
SHARDED = ("step", "end")


@dataclass(frozen=True)
class Topology:
    """A process's place in the world, as the launcher declared it."""

    rank: int
    world: int

    @property
    def main(self) -> bool:
        """Whether this is the rank that writes the run's artifacts."""
        return self.rank == 0


def topology() -> Topology:
    """Read `RANK` and `WORLD_SIZE` from the environment.

    Never from the config: the launcher knows how many processes it started and a config
    file does not, so a `world_size` in a config is a claim about the launcher that
    nothing checks.
    """
    try:
        world = int(os.environ.get(WORLD, "1") or 1)
        rank = int(os.environ.get(RANK, "0") or 0)
    except ValueError as exc:
        raise LoudError(
            f"{RANK}={os.environ.get(RANK)!r} / {WORLD}={os.environ.get(WORLD)!r} is not a rank",
            "both are integers; torchrun sets them for every process it starts",
        ) from exc
    if world < 1:
        raise LoudError(
            f"{WORLD}={world} has no process in it",
            "a run is at least one rank; unset the variable to mean one",
        )
    if not 0 <= rank < world:
        raise LoudError(
            f"{RANK}={rank} is outside [0, {world})",
            "ranks are 0..world-1; a duplicate or skipped rank deadlocks the first "
            "collective and looks like a slow step",
        )
    return Topology(rank=rank, world=world)


def _collective() -> bool:
    """Whether a reduction across processes is possible and needed."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def init(*, backend: str | None = None) -> Topology:
    """Join the process group the environment describes; a no-op at one rank.

    A single card is not a degenerate case to be special-cased upwards: it is the world
    of size one, and every other function here is already a no-op for it.

    `gloo` is the default because it is the backend that works for every device this
    stack is tested on, and a process group whose backend does not know the model's
    device fails inside DDP's constructor with "no backend type associated with device
    type cpu". A CUDA launcher that wants NCCL's faster collectives passes it.
    """
    world = topology()
    if world.world == 1 or (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        return world
    if not torch.distributed.is_available():
        raise LoudError(
            f"{WORLD}={world.world} but this torch has no distributed support",
            "install a torch built with distributed, or run one card with WORLD_SIZE unset",
        )
    name = backend or "gloo"
    torch.distributed.init_process_group(
        backend=name, init_method="env://", rank=world.rank, world_size=world.world
    )
    return world


def barrier() -> None:
    """Wait until every rank reaches this point, when there is more than one."""
    if _collective():
        torch.distributed.barrier()


def is_main() -> bool:
    """Whether this process owns the run directory (the writer, and only it)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return topology().main


class _Parallel(nn.Module):
    """A DDP module that is a *view* of the bare model: same names, same state dict.

    DDP prefixes every parameter and every key with `module.`. Both are load-bearing
    outside this file: `guards.adaptive_clip` keys its running norms by parameter name and
    `ema.apply` looks parameters up by name, so a checkpoint written through a prefixed
    view has clip estimates and an EMA that a one-card model cannot find — it resumes
    with a clip that never fires and an EMA that never applies, silently. Portability is
    the whole point of a distributed checkpoint: the file must not know how many cards
    wrote it. The parameters are the same tensors either way, so an optimizer built
    before wrapping keeps working.
    """

    def __init__(self, ddp: nn.parallel.DistributedDataParallel) -> None:
        """Hold the DDP wrapper as a submodule, so `train()`, `eval()` and grads follow."""
        super().__init__()
        self.ddp = ddp

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped forward; DDP's hooks live in this call."""
        return self.ddp(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """Return the bare model's names, which is what the clip and the EMA are keyed by."""
        return self.ddp.module.named_parameters(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return the inner module's state dict: no `module.` prefix in the file."""
        return self.ddp.module.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load into the inner module, from a file written on any number of cards."""
        return self.ddp.module.load_state_dict(state, strict=strict, assign=assign)


def wrap(model: nn.Module) -> nn.Module:
    """Put `model` under DDP at more than one rank; return it untouched at one.

    DDP *averages* the ranks' gradients, which is the second half of the global-token
    normalisation: each rank's micro-batches are means over its own tokens, and their
    mean over equal shards is the mean over the step's tokens.
    """
    world = topology()
    if world.world == 1:
        return model
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise LoudError(
            f"{WORLD}={world.world} but the process group is not initialised",
            "call dist.init() before wrapping the model: DDP broadcasts the weights in "
            "its constructor, and that collective needs its peers",
        )
    device_ids = None
    first = next(iter(model.parameters()), None)
    if first is not None and first.is_cuda:
        device_ids = [first.device.index]
    return _Parallel(nn.parallel.DistributedDataParallel(model, device_ids=device_ids))


class _Shard:
    """One rank's rows of a stream, and that rank's share of the step's cost.

    `ids`, `tokens` and `payload_bytes` are this rank's slice: `recipe.check_batch`
    measures a batch by the rows it carries, and a rank that declared the whole step's
    tokens would be claiming a loss over positions it never read. The slice's tokens are
    `rows x seq` and its bytes come from the manifest ratio, so the journal can add the
    ranks' shares back into the step's number and the byte/token ratio survives the cut.
    """

    def __init__(self, stream: BatchStream, rank: int, world: int) -> None:
        """Bind one rank of a stream; the split itself is decided per batch."""
        self.stream, self.rank, self.world = stream, rank, world
        self.bytes_per_token = float(stream.bytes_per_token)

    def batch(self, step: int, tokens: int) -> Batch:
        """Return this rank's contiguous rows of the step's batch."""
        whole = self.stream.batch(step, tokens)
        rows = int(whole.ids.shape[0])
        if rows % self.world:
            raise LoudError(
                f"step {step} has {rows} rows, which does not split across {self.world} ranks",
                "the loss is a mean over the global token count, and that is only the "
                "token mean when every rank holds the same number of tokens; give the "
                "mixture a batch size that divides, or use fewer cards",
            )
        local = rows // self.world
        if local < 1:
            raise LoudError(
                f"step {step} has {rows} rows for {self.world} ranks: a rank gets none",
                "at least one row per micro-batch per rank; raise the batch size",
            )
        start = self.rank * local
        ids = whole.ids[start : start + local]
        carried = local * (int(ids.shape[1]) - 1)
        return replace(
            whole,
            ids=ids,
            tokens=carried,
            payload_bytes=round(carried * self.bytes_per_token),
        )


def shard_stream(
    stream: BatchStream, rank: int | None = None, world: int | None = None
) -> BatchStream:
    """Cut `stream`'s steps across ranks: disjoint rows, one global batch per step.

    The cut is contiguous, so the ranks' slices concatenated in rank order are the
    single-card batch exactly — the equivalence gate compares that byte for byte rather
    than statistically. Uneven splits are refused, not reweighted: `step.py`'s `1/accum`
    and DDP's `1/world` only compose into the global token mean when the shards are the
    same size.
    """
    world_size = topology() if rank is None or world is None else Topology(rank, world)
    if world_size.world == 1:
        return stream
    return _Shard(stream, world_size.rank, world_size.world)


def _combine(digests: list[str]) -> str:
    """Hash the ranks' batch digests into one name for the step's whole batch."""
    joined = "|".join(sorted(digests)).encode()
    return hashlib.blake2b(joined, digest_size=8).hexdigest()


def _merge(records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Fold the ranks' versions of one record into the global one."""
    keys = set(records[0])
    if any(set(record) != keys for record in records[1:]):
        raise LoudError(
            "the ranks wrote different journal fields in the same step",
            "every rank takes the same step, so the record is the same shape; a "
            "difference means two runs are writing to one journal",
        )
    merged: dict[str, Any] = {}
    for key in sorted(keys):
        values = [record[key] for record in records]
        if key in STEP_FIELDS and kind in SHARDED:
            if len(set(values)) > 1:
                raise LoudError(
                    f"the ranks disagree on the step's {key}: {values}",
                    "the shards are not one partition of the batch: check that every "
                    "rank reads the same stream with a different rank number",
                )
            merged[key] = int(values[0]) * len(values)
        elif key == "batch":
            merged[key] = _combine([str(value) for value in values])
        elif key == "skipped":
            merged[key] = any(bool(value) for value in values)
        elif key in SUM_FIELDS or key in MAX_FIELDS or key in MEAN_FIELDS:
            numbers = [float(value) for value in values if value is not None]
            if not numbers:
                merged[key] = None
            elif key in SUM_FIELDS:
                merged[key] = sum(values)
            elif key in MAX_FIELDS:
                merged[key] = max(numbers)
            else:
                merged[key] = sum(numbers) / len(numbers)
        else:
            merged[key] = values[0]
    return merged


def all_reduce_metrics(fields: Mapping[str, Any], *, kind: str = "step") -> dict[str, Any]:
    """Reduce one journal record across ranks; every rank must call this.

    The rules are the table above, and `kind` decides whether `tokens`/`bytes` are one
    rank's shard (a step) or the whole window (an eval). It is a collective, so a rank
    that skips a write hangs the others — which is why `RankJournal` calls it from
    `write`, where the loop already calls every rank.
    """
    if not _collective():
        return dict(fields)
    gathered: list[dict[str, Any]] = [{} for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, dict(fields))
    return _merge(gathered, kind)


class RankJournal(Journal):
    """The run's journal: the global numbers once, written by rank 0 only.

    A journal per rank is how a run becomes unreproducible — two files, two step counts,
    neither the job's. Every rank appends to its own `records` so that `last()` answers
    the same everywhere (a resume's "has this run started?" must not be decided
    differently per rank, or one rank writes a `start` event the others do not and the
    next collective hangs), but only rank 0 opens a file.

    `raw` keeps each rank's un-reduced record, in order: the reduction is a claim, and a
    post-mortem needs to see what the ranks actually said.
    """

    def __init__(self, journal: Journal, *, topo: Topology | None = None) -> None:
        """Wrap an opened journal; non-main ranks get a sink path and their own records."""
        world = topology() if topo is None else topo
        super().__init__(
            path=journal.path if world.main else Path(os.devnull),
            records=journal.records,
        )
        self._main = world.main
        self.raw: list[dict[str, Any]] = []

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Reduce the record over the ranks, then append it once."""
        self.raw.append(dict(fields))
        merged = all_reduce_metrics(fields, kind=kind)
        if self._main:
            return super().write(kind, **merged)
        if kind not in KINDS:
            raise LoudError(
                f"unknown journal event {kind!r}",
                f"use one of: {', '.join(sorted(KINDS))}",
            )
        record: dict[str, Any] = {"kind": kind, "t": round(time.time(), 3), **merged}
        self.records.append(record)
        return record


def refuse_world_size(config: Mapping[str, Any]) -> None:
    """Refuse a config that names the world; it is the launcher's, not the recipe's.

    A `world_size` in a file is a second, unchecked opinion about how many processes are
    running. It cannot be verified at load time (the process group is not up yet), and
    when it is wrong the run trains a model that is not the one the config describes —
    silently, because every number in it stays plausible.
    """
    if "world_size" in config:
        raise LoudError(
            f"the config names world_size={config['world_size']!r}",
            "the world comes from the environment (RANK, WORLD_SIZE): delete the key, "
            "and launch with torchrun --nproc-per-node N — a config that names it "
            "silently disagrees with the launcher",
        )
