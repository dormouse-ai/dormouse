"""The training loop: the run, the journal, and a resume that is exact.

`recipe.py` owns what a step consumes and every refusal about the shape of the run;
`guards.py` owns the three mechanisms that decide whether a step may be taken;
`report.py` owns the numbers, BPB included; `checkpoint.py` owns the resume point. This
module is the thing that walks the steps between them, and it exists because each of
those decisions has to happen at a particular moment in a step:

    fetch a batch (a pure function of the step)
    → train_step (forward, chunked CE, backward, the global norm)
    → the loop's policy: per-tensor clip, then the spike judgement — before the moments move
    → optimizer step, or a skip that leaves every parameter bit-identical
    → one journal line: loss, rate, norm, clip events, tokens, bytes, BPB, throughput
    → a checkpoint every `ckpt_every`, an eval every `eval_every`, a swap on a stage change

Two details are the difference between a run and a run that looks like one. **The anneal
is checked before the first step**: a stage list that does not tile `[0, steps)` refuses
at construction, and so does a schedule still at full rate on its last step — the same
defect from two sides, and unfixable once the last steps have been spent. **The resume is
exact, not plausible**: the clip estimates are the easy field to forget and impossible to
notice, because without them the first resumed step clips against a reference of zero and
the run continues smoothly from a step it would otherwise not have taken.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal
from torch import nn

from ..metrics import BitsPerByte
from . import checkpoint, ema, guards, report
from .graph import StepGraph
from .optim import MixOptimizer
from .recipe import (
    ANNEAL_FLOOR,
    BatchStream,
    LoopConfig,
    Stage,
    check_batch,
    check_cover,
    config_from_dict,
)
from .step import StepResult, train_step

__all__ = ["Trainer"]

_LOG = logging.getLogger(__name__)


class Trainer:
    """One run: batches in, a journal out, and a checkpoint that resumes exactly."""

    def __init__(
        self,
        model: nn.Module,
        stream: BatchStream,
        optim: MixOptimizer,
        schedule: Callable[[int], float],
        cfg: LoopConfig,
        journal: Journal,
        eval_stream: BatchStream | None = None,
        use_graph: bool = False,
        heartbeat: Callable[[str], None] | None = None,
    ) -> None:
        """Wire the run and refuse a schedule whose last step is not the anneal's."""
        self.model, self.stream, self.optim = model, stream, optim
        self.schedule, self.cfg, self.journal = schedule, cfg, journal
        self.eval_stream = eval_stream
        self.use_graph = use_graph
        self.heartbeat = heartbeat
        self.graph: StepGraph | None = None
        self.log_every = 1
        self._last_norm = 0.0
        self.run_id = journal.path.parent.name or "run"
        last = float(schedule(cfg.steps - 1))
        if last > ANNEAL_FLOOR:
            raise LoudError(
                f"the schedule is still at {last:.3f}x peak on the last step {cfg.steps - 1}",
                "pass a schedule that anneals to <=0.2 by the budget "
                "(recipe.annealed_schedule(cfg) is the recipe); a flat tail is a run "
                "that ends without its anneal",
            )
        self.step = 0
        self.tokens = 0
        self.payload_bytes = 0
        self.nats = 0.0
        self.evals = 0
        self.skips = 0
        self.spike_skips = 0
        self.consecutive = 0
        self.clip_events = 0
        self.throughput_warnings = 0
        self.stage = cfg.stages[0].name
        self.ema: dict[str, torch.Tensor] | None = {} if cfg.ema else None
        self.clip_norms: dict[str, float] = {}
        self.losses: list[float] = []
        self.rates: list[float] = []
        self.baseline: float | None = None
        self.counters = Counters()

    # ------------------------------------------------------------------- the loop

    def train(
        self,
        *,
        steps: int | None = None,
        eval_every: int | None = None,
        ckpt_every: int | None = None,
        log_every: int = 1,
    ) -> dict[str, float]:
        """Run to the **absolute** horizon `steps` (default: the config's), journaling.

        `log_every` is the host-read cadence. Every read of a step scalar goes through
        `step.host_scalar`; the step's own arithmetic and its skip decision stay on the
        device. At the default of one the journal is unchanged; a larger value keeps the
        norm on the device between log lines. The loss is read every step: the spike
        gate and a skipped step's receipt need the real value, not a sentinel.

        `steps` is the total, not an increment: a run resumed at step 60 with
        `train(steps=100)` takes 40 more steps, which is what makes the resume gate a
        comparison with one 100-step run instead of two different budgets. A horizon past
        the config's needs stages for the extra steps; stopping short is a pause.
        """
        if log_every < 1:
            raise LoudError(
                f"log_every={log_every} journalises no step",
                "one reads every step; a larger value reads at that cadence",
            )
        self.log_every = log_every
        total = self.cfg.steps if steps is None else steps
        if total < self.step:
            raise LoudError(
                f"steps={total} is behind the resumed step {self.step}",
                f"pass the absolute horizon (>={self.step}), not the number of steps to add",
            )
        if eval_every is not None and self.eval_stream is None:
            raise LoudError(
                "eval_every is set and there is no eval_stream",
                "evaluation needs its own data: pass eval_stream to the Trainer, or drop "
                "eval_every — an eval on the training stream measures memorisation",
            )
        if total > self.cfg.steps:
            check_cover(self.cfg.stages, total)
        if self.journal.last("start") is None:
            report.note(
                self.journal,
                "start",
                config=self.run_id,
                steps=total,
                warmup_steps=self.cfg.warmup_steps,
                anneal_start=self.cfg.anneal_start,
                clip_z=self.cfg.clip_z if self.cfg.clip else 0.0,
                ema=self.cfg.ema,
            )
        while self.step < total:
            self._step(total, eval_every=eval_every, ckpt_every=ckpt_every)
        self._checkpoint()
        if self.step >= self.cfg.steps:
            # A stop short of the budget is a pause, not an end: `dorm status` reads the
            # last `end` event, and a partial call writing one would show a live run as
            # finished until its next step line.
            report.note(
                self.journal,
                "end",
                step=self.step,
                tokens=self.tokens,
                bytes=self.payload_bytes,
                bpb=report.total_bpb(self.nats, self.payload_bytes),
            )
        return {
            "steps": float(self.step),
            "tokens": float(self.tokens),
            "bytes": float(self.payload_bytes),
            "bpb": float(report.total_bpb(self.nats, self.payload_bytes) or 0.0),
            "skips": float(self.skips),
            "clip_events": float(self.clip_events),
        }

    def _step(self, total: int, *, eval_every: int | None, ckpt_every: int | None) -> None:
        """One global step: fetch, clip and judge, step or skip, journal, maybe checkpoint."""
        cfg = self.cfg
        step = self.step
        stage = self._stage(step)
        if stage.name != self.stage:
            report.note(self.journal, "swap", step=step, stage_from=self.stage, stage_to=stage.name)
            self.stage = stage.name
        counters = Counters()
        self.counters = counters
        # A step announces itself before the forward: an expensive step then leaves a
        # heartbeat on stdout instead of a journal that stops mid-step with no explanation.
        counters.enable(
            "heartbeat",
            counter="step_heartbeat",
            why="a step reports that it started before the forward runs",
        )
        counters.count("step_heartbeat")
        if self.heartbeat is not None:
            self.heartbeat(f"step {step}/{total}: computing ({stage.name})")
        started = time.perf_counter()
        batch = self.stream.batch(step, cfg.tokens_per_step)
        fetched = time.perf_counter()
        check_batch(batch, stage=stage, stream=self.stream, step=step)
        rate = self._set_rate(step)
        self._capture_graph(step, batch.ids)
        # Defer only the norm's host read: the loss is crossed every step because the
        # spike gate and the skip receipt decide on it. ### ponytail: the journal's norm
        # between log lines is the last read one; a device-resident norm window is a
        # follow-up once a deferred-read run is measured on the card.
        read = self.log_every <= 1 or (step + 1) % self.log_every == 0

        # The step's own global clip is off (`clip=inf` leaves the norm unscaled and still
        # reports it): the loop's per-tensor policy replaces it inside `pre_step`.
        def pre_step(loss: float) -> bool:
            return self._pre_step(loss, counters)

        if self.graph is not None:
            result: StepResult = self.graph.run(
                batch.ids, counters=counters, clip=math.inf, pre_step=pre_step
            )
        else:
            result = train_step(
                self.model,
                batch.ids,
                optimizer=self.optim,
                accum=cfg.accum,
                clip=math.inf,
                counters=counters,
                pre_step=pre_step,
                read=read,
            )
        loss = result.loss
        norm = result.grad_norm if read else self._last_norm
        if read:
            self._last_norm = norm
        finished = time.perf_counter()
        elapsed = max(finished - started, 1e-9)
        self.step = step + 1
        self.tokens += batch.tokens
        self.payload_bytes += batch.payload_bytes
        self._guard(batch.tokens / elapsed, fetched - started, finished - fetched)
        if result.skipped:
            self.skips += 1
            self.consecutive += 1
            # V4/N4: the batch's bytes were already added to the ledger; its
            # nats must be too, or BPB is computed over fewer nats than bytes
            # (measured -0.0443). NaN loss stays non-finite on purpose: the
            # receipt then refuses BPB instead of printing a flattering value.
            self.nats += loss * batch.tokens
            counters.count("skipped")
        else:
            self.consecutive = 0
            self.losses = [*self.losses, loss][-cfg.spike_window :]
            self.nats += loss * batch.tokens
            counters.count("opt_steps")
        report.note(
            self.journal,
            "step",
            step=step,
            stage=stage.name,
            loss=report.finite(loss),
            lr=report.finite(rate),
            grad_norm=report.finite(norm),
            clip_events=counters.counts.get("clip_events", 0),
            skipped=result.skipped,
            **(
                {"graph_replays": counters.counts.get("graph_replays", 0)}
                if self.graph is not None
                else {}
            ),
            tokens=batch.tokens,
            bytes=batch.payload_bytes,
            bpb=report.step_bpb(result.loss, batch.tokens, batch.payload_bytes),
            tokens_per_s=round(batch.tokens / elapsed, 1),
            data_s=round(fetched - started, 4),
            step_s=round(finished - fetched, 4),
            batch=report.digest(batch.ids),
        )
        if self.consecutive > cfg.max_skips:
            raise LoudError(
                f"{self.consecutive} consecutive skipped steps by step {step}: that is a "
                "divergence, not a spike",
                "stop here: lower the peak rate, or find the shard that started it — the "
                "journal has the loss, the norm and the batch hash for every step",
            )
        if ckpt_every is not None and self.step % ckpt_every == 0:
            self._checkpoint()
        if eval_every is not None and self.step % eval_every == 0:
            self._evaluate(self.step)
        if not result.skipped:
            # A skipped step ran no mechanism (a NaN loss returns before the policy), so
            # the counter discipline has nothing to prove; `skipped` is the receipt.
            counters.check()

    def _capture_graph(self, step: int, ids: torch.Tensor) -> None:
        """Capture one whole-step graph after warm-up; a failed capture is LOUD."""
        if not self.use_graph or self.graph is not None or step < self.cfg.warmup_steps:
            return
        try:
            self.graph = StepGraph(self.model, self.optim, ids, accum=self.cfg.accum)
        except LoudError as exc:
            raise LoudError(
                f"--graph requested, but capture refused at step {step}: {exc}",
                "a graph that replayed nothing is not an eager fallback; fix the op the "
                "capture names, or drop --graph — never let a refused capture run eager",
            ) from exc

    def _pre_step(self, loss: float, counters: Counters) -> bool:
        """Apply the loop's policy between the backward pass and the optimizer step.

        Both halves belong here because both must happen *before* the moments move: the
        per-tensor clip (the step's global form is off) and the spike judgement, whose
        answer is "do not take this step at all". A policy applied after `optimizer.step`
        is a rollback, and a rollback leaves the moments poisoned.
        """
        if self.cfg.clip:
            # Enabled here, not before the forward: the stack's own `Counters.check()` runs
            # inside its forward, so a counter this step produces must not be declared
            # until the step is ready to produce it (the same rule `step.py` follows).
            counters.enable(
                "adaptive_clip",
                counter="clip_tensors",
                why="per-tensor clipping is the default (recipe item 2)",
            )
            examined, scaled = guards.adaptive_clip(self.model, self.clip_norms, self.cfg.clip_z)
            counters.count("clip_tensors", examined)
            self.clip_events += scaled
            if scaled:
                counters.count("clip_events", scaled)
        if guards.is_spike(loss, self.losses, k=self.cfg.spike_k, size=self.cfg.spike_window):
            self.spike_skips += 1
            counters.count("spike")
            return False
        return True

    def _stage(self, step: int) -> Stage:
        """Return the stage that owns `step`; the config guarantees one exists."""
        for stage in self.cfg.stages:
            if stage.start <= step < stage.end:
                return stage
        raise LoudError(
            f"step {step} belongs to no stage",
            "the stage list was validated at construction; this is a counter that moved",
        )

    def _set_rate(self, step: int) -> float:
        """Write the schedule's fraction through `set_lr`; return the absolute rate."""
        fraction = float(self.schedule(step))
        self.optim.set_lr(fraction)
        return float(self.optim.base_lr[0]) * fraction

    def _guard(self, rate: float, data_s: float, step_s: float) -> None:
        """Warn when throughput falls `guard_drop` below the median of the opening window."""
        cfg = self.cfg
        if self.baseline is None:
            self.rates.append(rate)
            if len(self.rates) >= cfg.guard_window:
                self.baseline = statistics.median(self.rates)
            return
        message = guards.throughput_drop(
            rate,
            self.baseline,
            window=cfg.guard_window,
            drop=cfg.guard_drop,
            data_s=data_s,
            step_s=step_s,
        )
        if message is not None:
            self.throughput_warnings += 1
            _LOG.warning("%s", message)

    # ------------------------------------------------------------- EMA, audit, resume

    def _evaluate(self, step: int) -> BitsPerByte:
        """Write one eval event for the training weights over the held-out window."""
        bpb = report.eval_event(
            self.journal,
            step,
            self.model,
            self.optim,
            self.eval_stream,
            batches=self.cfg.eval_batches,
            tokens_per_step=self.cfg.tokens_per_step,
            first=self.evals * self.cfg.eval_batches,
        )
        self.evals += 1
        return bpb

    def final_model(self) -> nn.Module:
        """Return the model worth evaluating: the EMA when on, the live weights when off."""
        return ema.apply(self.model, self.ema)

    def audit(self) -> dict[str, int]:
        """Refuse a run whose journal cannot explain the tokens and bytes it reports."""
        return report.audit(
            self.journal,
            steps=self.step,
            tokens=self.tokens,
            payload=self.payload_bytes,
            bytes_per_token=float(self.stream.bytes_per_token),
        )

    def _checkpoint(self) -> Path:
        """Write the whole resume point: weights, moments, EMA, clip state, step."""
        if self.ema is not None:
            self.ema = ema.update(
                self.ema, self.model, self.optim, checkpoints=self.cfg.ema_checkpoints
            )
            self.counters.enable("ema", counter="ema", why="the EMA is updated at checkpoints")
            self.counters.count("ema")
        path = checkpoint.save(self.journal.path.parent, checkpoint.capture(self))
        report.note(self.journal, "checkpoint", step=self.step, path=path.name, tokens=self.tokens)
        return path

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        *,
        model: nn.Module,
        stream: BatchStream,
        optim: MixOptimizer,
        schedule: Callable[[int], float],
        journal: Journal,
        eval_stream: BatchStream | None = None,
        use_graph: bool = False,
        heartbeat: Callable[[str], None] | None = None,
    ) -> Trainer:
        """Restore a run from `run_dir` so that the next batch is the one that would have come.

        The objects are passed rather than built here: the architecture comes from
        `model/assemble.py` and the mixture from `data/mix.py`, and a loop that built
        either would be a second place that decides what a run is. What is restored is
        everything the loop owns — the config (from the checkpoint, so a resume cannot
        quietly change its anneal), the weights, both arms' moments, the EMA, the clip
        estimates, the loss window, the throughput baseline and the step.
        """
        snapshot = checkpoint.load(run_dir)
        trainer = cls(
            model,
            stream,
            optim,
            schedule,
            config_from_dict(snapshot.config),
            journal,
            eval_stream,
            use_graph=use_graph,
            heartbeat=heartbeat,
        )
        checkpoint.restore(trainer, snapshot)
        trainer.stage = trainer._stage(min(snapshot.step, trainer.cfg.steps - 1)).name
        return trainer
