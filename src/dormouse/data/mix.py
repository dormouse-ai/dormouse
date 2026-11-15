"""The mixture the trainer reads: exact shares, arithmetic resume, no silent end.

A run that samples each source independently at its weight gets the *expected* share
and the wrong *actual* one: over 200 steps the realised share of a 30 % source drifts
by whole percent, and the drift is invisible in the loss. So the plan is not sampled,
it is scheduled: `plan_cycle` apportions draws by largest deficit, which keeps every
source's count within `1 + max_share` draws of its share at *every* prefix — not on
average. At the batch a run uses (24 rows), 200 steps is 4800 draws, so the bound is
0.03 % of the mixture; it is a bound on a prefix, not a random walk.

WHY A CYCLE AND NOT A CLOSED FORM. Exact prefix quotas and monotone counts cannot both
hold for every weight vector (Balinski-Young: no apportionment satisfies the quota rule
and house monotonicity at once — the Alabama paradox), and monotonicity is the one that
cannot be given up here: a count that ever falls would have to *unread* tokens. The plan
is therefore one cycle of `CYCLE` draws whose prefixes are exact, repeated; the cycle's
own remainder is the only drift, at most one draw per `CYCLE` draws (0.01 % of the
corpus), and it does not grow with the run.

Position is arithmetic, not state, for the same reason `bin.Stream` says it: a resume
reads the plan's counts at `start_step`, multiplies by the window, and lands where the
run stopped. Nothing is stored between steps, so nothing can drift.

A STAGE is a name, an ordered list of `(source, weight)` — derived from the `Source`
values so the two cannot disagree — and a step range ending at `until`. The boundary
is a journal event (`swap`), written once per change, with the stage's weights; a
resumed run re-announces the stage it starts in rather than assuming the reader knows.

Tensors are `numpy.int64`, not torch: `torch.from_numpy(x)` is a view, so the trainer
pays no copy, and the data layer imports without torch. The two arrays are row-strided
views of one `[batch, seq_len + 1]` buffer — the shift is the point; copying them apart
costs 400 kB of traffic per step and buys nothing a strided tensor cannot do. Reading is
`mmap` only (`bin.Stream`): a batch of 24 x 1024 tokens is copied, never a corpus.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

from .bin import Stream

__all__ = ["CYCLE", "MixtureStream", "Source", "Stage", "plan_cycle"]

#: Three concepts, declared: a source with its weight, the stage that orders them,
#: and the stream that reads them. Every one of them is a value a caller names.
__concepts__ = ("MixtureStream", "Source", "Stage")

#: Draws per plan cycle. Prime, so the pattern cannot line up with a stage boundary or
#: a batch size and turn into a shortcut a run can learn. Bigger is a smaller cycle
#: remainder and a slower first step; 10 007 is ~7 ms to build and 0.01 % of draws.
CYCLE = 10_007


def _shares(weights: Sequence[float] | np.ndarray) -> list[float]:
    """Validate a weight vector and return it normalised, as the plan reads it."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise LoudError(
            f"a stage needs a 1-D weight vector, got shape {w.shape}",
            "one weight per source; pass the stage's list of (source, weight)",
        )
    shares = w.tolist()
    if any(not math.isfinite(v) or v < 0.0 for v in shares):
        raise LoudError(
            f"weights must be finite and non-negative, got {shares}",
            "a negative or NaN weight has no share to hit; drop the source instead",
        )
    total = sum(shares)
    if total <= 0.0:
        raise LoudError(
            "every weight in this stage is 0",
            "a stage with no weight is a stage that reads nothing; remove it or weight it",
        )
    return [v / total for v in shares]


def plan_cycle(weights: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one `CYCLE` of the plan: `(prefix_counts[CYCLE + 1, sources], slot_per_draw)`.

    Row `n` of the counts is how many draws each source has after `n` draws. Every row
    is monotone in the one above it (a draw is a count going up) and within
    `1 + max_share` draws of its exact share — the tightest bound a monotone, exact-size
    plan admits (see the module docstring), and the reason this is a plan, not a sample.
    The slot array is the same plan in draw order: an interleave, never a block.
    """
    shares = _shares(weights)
    counts = [0] * len(shares)
    slots = np.empty(CYCLE, dtype=np.int64)
    for draw in range(CYCLE):
        best, gap = 0, shares[0] * (draw + 1) - counts[0]
        for k in range(1, len(shares)):
            k_gap = shares[k] * (draw + 1) - counts[k]
            if k_gap > gap:
                best, gap = k, k_gap
        counts[best] += 1
        slots[draw] = best
    prefix = np.zeros((CYCLE + 1, len(shares)), dtype=np.int64)
    prefix[np.arange(CYCLE) + 1, slots] = 1
    return np.cumsum(prefix, axis=0), slots


@dataclass(frozen=True)
class Source:
    """One token stream, its weight inside its stage, and the stage it belongs to."""

    path: Path | str
    weight: float
    stage: str = "train"

    def __post_init__(self) -> None:
        """Refuse a source that cannot be given a share."""
        if not self.stage:
            raise LoudError(
                f"{self.path}: a source with no stage",
                "name the stage it belongs to; 'train' is the single-stage default",
            )
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise LoudError(
                f"{self.path}: weight={self.weight!r} carries no data",
                "a source at weight 0 is a source that never trains; drop it or weight it",
            )


@dataclass(frozen=True)
class Stage:
    """A named mix and the step it ends at: `[previous.until, until)`, or to the end."""

    name: str
    until: int | None = None

    def covers(self, step: int) -> bool:
        """Whether `step` falls in this stage's range."""
        return self.until is None or step < self.until


class MixtureStream:
    """Weighted batches over several token streams, positioned by the step alone."""

    def __init__(
        self,
        sources: Sequence[Source],
        *,
        stages: Sequence[Stage] = (),
        seed: int = 0,
        tokenizer_sha: str = "",
        wrap: bool = False,
    ) -> None:
        """Open the sources, check the stage table, and refuse a tokenizer mismatch."""
        if not sources:
            raise LoudError(
                "a mixture with no sources reads nothing",
                "pass Source(path, weight, stage); one source is a legal mixture",
            )
        self.sources = list(sources)
        if seed < 0:
            raise LoudError(
                f"seed={seed} is negative",
                "the seed picks the order inside a batch; 0 is a seed, a negative one is a "
                "counter that never initialised",
            )
        self.seed = seed
        self.wrap = wrap
        self._streams = [Stream(s.path) for s in self.sources]
        self._tokens = [stream.tokens for stream in self._streams]
        self._check_tokenizer(tokenizer_sha)
        self.stages = self._stage_table(stages)
        self._members = self._stage_members()
        self._weights = self._stage_weights()
        self._cycles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._recorded: str | None = None

    # ------------------------------------------------------------------ checks
    def _check_tokenizer(self, expected: str) -> None:
        """One tokenizer per mixture: refuse a manifest that disagrees with the others."""
        shas = {stream.manifest.tokenizer_sha for stream in self._streams}
        if len(shas) > 1:
            listed = ", ".join(
                f"{s.path}={stream.manifest.tokenizer_sha or '(none)'}"
                for s, stream in zip(self.sources, self._streams, strict=True)
            )
            raise LoudError(
                f"the sources were tokenized by different tokenizers: {listed}",
                "retokenize every source with one tokenizer, then rebuild the manifests",
            )
        actual = shas.pop()
        if expected and actual != expected:
            raise LoudError(
                f"the data's tokenizer is {actual or '(none)'}, the run's is {expected}",
                "tokenize the corpus with this run's tokenizer, or load the tokenizer "
                "the data was written with; ids from two vocabularies are noise",
            )

    def _stage_table(self, stages: Sequence[Stage]) -> list[Stage]:
        """Return the declared stages, or one implicit stage when all sources agree."""
        names = list(dict.fromkeys(s.stage for s in self.sources))
        if not stages:
            if len(names) > 1:
                raise LoudError(
                    f"sources name {len(names)} stages ({', '.join(names)}) and no stage "
                    "table says which comes first",
                    f"pass stages=[Stage({names[0]!r}, until=N), Stage({names[1]!r})]",
                )
            return [Stage(names[0])]
        declared = [s.name for s in stages]
        if declared != names:
            raise LoudError(
                f"stages {declared} do not match the sources' stages {names}",
                "declare every stage the sources name, in the order they run",
            )
        end: int | None = 0
        for stage in stages:
            if stage.until is not None:
                if end is None:
                    raise LoudError(
                        f"stage {stage.name!r} has a range after a stage that has none",
                        "only the last stage may omit until=N",
                    )
                if stage.until <= end:
                    raise LoudError(
                        f"stage {stage.name!r} ends at {stage.until}, not after {end}",
                        "each stage's until must be greater than the previous one's",
                    )
            end = stage.until
        return list(stages)

    def _stage_members(self) -> list[np.ndarray]:
        """Return each stage's source indices, in declaration order."""
        return [
            np.array([i for i, s in enumerate(self.sources) if s.stage == stage.name])
            for stage in self.stages
        ]

    def _stage_weights(self) -> list[np.ndarray]:
        """Return each stage's normalised weights, in its members' order."""
        out = []
        for members in self._members:
            w = np.array([self.sources[i].weight for i in members], dtype=np.float64)
            out.append(w / w.sum())
        return out

    # ------------------------------------------------------------------ the plan
    def stage_at(self, step: int) -> Stage:
        """Return the stage `step` reads from; a step past the last range is an error."""
        if step < 0:
            raise LoudError(
                f"step={step} is negative",
                "steps start at 0; a negative index is a counter that never initialised",
            )
        for stage in self.stages:
            if stage.covers(step):
                return stage
        raise LoudError(
            f"step {step} is past the last stage {self.stages[-1].name!r} "
            f"(ends at {self.stages[-1].until})",
            "extend the last stage's until, or stop the run at that step",
        )

    def shares(self, stage: str | None = None) -> dict[str, float]:
        """Return the target share of every source in `stage`; omit it if there is one."""
        if stage is None:
            if len(self.stages) > 1:
                raise LoudError(
                    f"this mixture has {len(self.stages)} stages, so 'the share' is ambiguous",
                    f"name one: {', '.join(s.name for s in self.stages)}",
                )
            stage = self.stages[0].name
        for i, name in enumerate(s.name for s in self.stages):
            if name == stage:
                return {
                    str(self.sources[int(k)].path): float(w)
                    for k, w in zip(self._members[i], self._weights[i], strict=True)
                }
        raise LoudError(
            f"no stage named {stage!r}",
            f"declared: {', '.join(s.name for s in self.stages)}",
        )

    def _step_plan(self, step: int, batch: int) -> tuple[int, np.ndarray, np.ndarray]:
        """Return the stage index, the step's stage-local slots, and the counts before it."""
        if batch <= 0:
            raise LoudError(f"batch={batch} asks for no rows", "pass the trainer's batch size")
        index = self.stages.index(self.stage_at(step))
        return index, self._slots(index, step, batch), self._counts_at(index, step * batch)

    def _counts_at(self, index: int, draws: int) -> np.ndarray:
        """Return the plan's counts after `draws`: whole cycles plus the cycle's prefix."""
        prefix = self._plan(index)[0]
        whole, rest = divmod(draws, CYCLE)
        total: np.ndarray = whole * prefix[CYCLE] + prefix[rest]
        return total

    def _plan(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the stage's cycle, built on first use: 7 ms a single-stage run pays once."""
        if index not in self._cycles:
            self._cycles[index] = plan_cycle(self._weights[index])
        return self._cycles[index]

    def draws(self, step: int, batch: int) -> np.ndarray:
        """Return the stage-local source index for each of the step's `batch` rows.

        The slots are the cycle's own draw order for `[step*batch, (step+1)*batch)`, so
        the plan is a function of the step and not a walk from step 0: `start_step`
        costs the same as step 0. The rotation in `_slots` is the seed's only job.
        """
        return self._step_plan(step, batch)[1]

    def _slots(self, index: int, step: int, batch: int) -> np.ndarray:
        """Return the step's stage-local slots, in the plan's own interleaved order.

        The order is the cycle's, rotated by the seed: a rotation keeps every count and
        moves which row of the batch each draw lands on, which is the seed's whole job.
        """
        order = self._plan(index)[1]
        at = np.arange(step * batch, step * batch + batch) % CYCLE
        slots = order[at]
        offset = (self.seed * 0x9E3779B1 + index * 7919) % batch
        return np.roll(slots, int(offset)) if offset else slots

    # ------------------------------------------------------------------ reading
    def _windows(self, k: int, start: int, count: int, seq_len: int, step: int) -> np.ndarray:
        """`count` consecutive `seq_len + 1` windows starting at token `start`."""
        need = count * (seq_len + 1)
        if start + need > self._tokens[k] and not self.wrap:
            raise LoudError(
                f"{self.sources[k].path}: out of tokens at step {step}",
                f"the source has {self._tokens[k]} tokens; step {step} asked for "
                f"[{start}, {start + need}). Reweight the mixture, shorten the run, or "
                "pass wrap=True to train on a looped source",
            )
        return self._read(k, start, need).reshape(count, seq_len + 1)

    def _read(self, k: int, start: int, need: int) -> np.ndarray:
        """Read `need` tokens from `start`, wrapping only when the caller asked."""
        if not self.wrap:
            return self._streams[k].take(start, need)
        if need > self._tokens[k]:
            raise LoudError(
                f"{self.sources[k].path}: a step needs {need} tokens, the source has "
                f"{self._tokens[k]}",
                "wrap repeats the source, it cannot make it longer than a step",
            )
        out = np.empty(need, dtype=self._streams[k].dtype)
        pos, filled = start % self._tokens[k], 0
        while filled < need:
            chunk = min(need - filled, self._tokens[k] - pos)
            out[filled : filled + chunk] = self._streams[k].take(pos, chunk)
            filled, pos = filled + chunk, 0
        return out

    def step(self, step: int, *, batch: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the `(inputs, targets)` of one step, as `int64` arrays the trainer views."""
        if seq_len < 1:
            raise LoudError(f"seq_len={seq_len} predicts nothing", "pass at least one token")
        index, slots, before = self._step_plan(step, batch)
        buf = np.empty((batch, seq_len + 1), dtype=np.int64)
        for slot, k in enumerate(self._members[index]):
            rows = np.flatnonzero(slots == slot)
            if rows.size == 0:
                continue
            start = int(before[slot]) * (seq_len + 1)
            buf[rows] = self._windows(int(k), start, int(rows.size), seq_len, step)
        return buf[:, :-1], buf[:, 1:]

    def batches(
        self, *, batch: int, seq_len: int, start_step: int = 0
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(x, y)` from `start_step` on, each `[batch, seq_len]` of `int64`.

        A resumed run passes the step it stopped at and sees exactly the batches it
        would have seen: nothing is carried between steps.
        """
        step = start_step
        while True:
            yield self.step(step, batch=batch, seq_len=seq_len)
            step += 1

    # ------------------------------------------------------------------ journal
    def record(self, journal: Journal, step: int) -> dict[str, object] | None:
        """Write the stage boundary as a `swap` event when `step` changes stage."""
        stage = self.stage_at(step)
        if stage.name == self._recorded:
            return None
        index = self.stages.index(stage)
        weights = {
            str(self.sources[int(k)].path): round(float(w), 6)
            for k, w in zip(self._members[index], self._weights[index], strict=True)
        }
        record = journal.write(
            "swap", step=step, what="mixture", stage=stage.name, weights=weights, seed=self.seed
        )
        self._recorded = stage.name
        return record
