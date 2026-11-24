"""The run: one command, one journal, one artifact — and no second recipe.

Every part of a run was already a module. `model/assemble.py` builds the stack and prices
it, `data/mix.py` schedules the batches by step, `train/optim.py` routes the parameters
under the schedule the config names, `train/loop.py` walks the steps and writes the journal,
`train/checkpoint.py` owns the resume point, `eval/` scores the weights. What no module
owned was the **order** they are built in and the refusals at the seams: a model built at
one width and a mixture priced at another, a plan whose step cost the run never measured, a
resume whose config quietly moved. That is this file's whole job, so it is deliberately the
only place that decides them.

THE PLAN IS AN ASSERTION, NOT A SUMMARY. `dorm plan` prices a step from the architecture's
bill; the run measures the same step from the counters it moved (`6 x active x tokens`) and
refuses to start when the two differ by more than `PLAN_TOLERANCE`. A plan that disagrees
with the run is the defect class this project keeps hitting: a config that priced 24x1024
and trained 8x512 has a cost estimate nobody can compare to anything, including the next
one.

THE RUN LEAVES FIVE THINGS *and every one of them is a file a reader can re-derive:

    runs/<id>/run.jsonl        the journal: one line a step, plus start, swap, checkpoint,
                               eval and end — with the counters of the last step in it
    runs/<id>/config.toml      the text that produced the run, and config.json beside it
    runs/<id>/ckpt.pt          the resume point: weights, moments, EMA, clip state, rng
    runs/<id>/final.pt         the final weights, loadable by the same code as the export
    runs/<id>/export/model.pt  the artifact: spec, weights, tokenizer, and the config

WHAT `--smoke` IS FOR. Twenty steps on whatever the config says, with the journal, the
checkpoint, the eval, the harness and the artifact all still running. It is the CI seam
test, not a measurement: the numbers in a smoke journal are not a result, and the config it
runs says so in its own comment. A paused run (`--steps N` below the budget) writes no
`final.pt` and no export on purpose — a pause is not a result either.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal
from torch import nn

from ..data.mix import MixtureStream
from ..eval import harness
from ..eval import report as evals
from ..eval.tasks import suite
from ..model.assemble import Assembled, assemble, plan_lines
from ..model.spec import StackSpec
from . import report
from .loop import Trainer
from .optim import MixOptimizer
from .recipe import LoopConfig, annealed_schedule
from .runconfig import RunRecipe, fresh, refuse_drift
from .schedule import warmup_cosine
from .streams import HeldOut, Mixture, heldout_bytes_per_token
from .talker import Talker, vocabulary

__all__ = ["SMOKE_STEPS", "evaluate", "load_artifact", "train"]

#: `--smoke`: the CI seam test's horizon. Every other number stays the config's.
SMOKE_STEPS = 20

#: How far a run's measured step may drift from the plan's before it refuses to start.
PLAN_TOLERANCE = 0.01

#: A plan longer than this needs `--yes`. The README's one-hour rule, in seconds.
BUDGET_SECONDS = 3600.0

#: The counters a finished run must report. The stack's own mechanisms come from
#: `Assembled.mechanisms()`; these are the step's, and none of them may be silent.
STEP_COUNTERS = frozenset({"embed_fwd", "block_fwd", "norm_fwd", "head_fwd", "tied_head"})


def _resolve_device(choice: str) -> torch.device:
    """Resolve the spec's device: `auto` picks CUDA when present, else CPU.

    An explicit `cuda` on a card-less host refuses out loud — a step that silently
    trained on CPU would report card speed for arithmetic that never ran on a card.
    """
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise LoudError(
                "device=cuda, but torch.cuda.is_available() is False",
                "set device=auto (or cpu) in the config; a cuda request that quietly "
                "trained on cpu would report card speed from a cpu run",
            )
        return torch.device("cuda")
    raise LoudError(f"unknown device {choice!r}", "use auto, cpu or cuda")


class _NoSwap:
    """`eval_weights()` that does nothing, for a model that is already the eval weights.

    Under `sf` the arms hold `Y_t` and the evaluable weights are `X_t`; the EMA averages
    `X_t` and the artifact holds it. Swapping `Y_t` over it for an eval would report the
    wrong model, so the final measurement takes the weights as they are.
    """

    def eval_weights(self) -> Any:
        """Return a context manager that changes nothing."""
        return nullcontext()


def _plan(spec: StackSpec, *, smoke: bool) -> tuple[float, int, str]:
    """Return the plan's duration and step cost, and which of the plan's faces produced it.

    Parsed rather than recomputed on purpose: a second copy of `plan_lines`' arithmetic is
    a second answer, and the point of the assertion is that the number the reader sees is
    the number the run spends. A `--smoke` stack is the one exception, and it is named: it
    sits below the expert floor `plan_lines` prices at (`MoEFFN` refuses a routed expert
    under 10M parameters when the stack is built on the meta device outside a smoke arm),
    so the smoke number is the same formula from the same bill at the same widths.
    """
    if smoke:
        meta = assemble(spec, device="meta", smoke=True).report()
        return 0.0, 6 * meta["active"] * spec.batch * spec.seq_len, "bill"
    lines = plan_lines(spec)
    for line in lines:
        flops = re.search(r"flops/step=([\d,]+)", line)
        if flops:
            total = next(
                (found for item in lines if (found := re.search(r"phase\(s\), (\S+) total", item))),
                None,
            )
            if total is None:
                break
            return _seconds(total.group(1)), int(flops.group(1).replace(",", "")), "plan"
    raise LoudError(
        "the plan printed no step cost",
        "plan_lines is the one place a step is priced; a run with nothing to check itself "
        "against is a run whose cost nobody measured",
    )


def _seconds(text: str) -> float:
    """Return `90s`, `24min`, `4.6h` in seconds, the way `Plan.render` writes a duration."""
    for suffix, scale in (("min", 60.0), ("h", 3600.0), ("s", 1.0)):
        if text.endswith(suffix):
            return float(text.removesuffix(suffix)) * scale
    raise LoudError(
        f"the plan's total {text!r} is not a duration this reads",
        "Plan.human() writes s, min or h; fix the reader or the renderer, not the run",
    )


def _assert_plan(where: str, priced: int, measured: int) -> None:
    """Refuse a step whose measured cost is not the plan's within `PLAN_TOLERANCE`."""
    drift = measured / priced - 1.0
    if abs(drift) > PLAN_TOLERANCE:
        raise LoudError(
            f"{where}: the plan prices a step at {priced:,} FLOP and the run measures "
            f"{measured:,} ({drift:+.2%})",
            "the config prices one shape and the run takes another; `dorm plan` prints "
            "batch x seq_len x 6 x active, so a run that disagrees has no cost estimate "
            "that means anything",
        )


def _mixture(recipe: RunRecipe, device: torch.device) -> Mixture:
    """Open the mixture on the run's device and the ratio the accounting divides by."""
    expected = ""
    if recipe.spec.tokenizer:
        from ..data.tokenizer import sha_file

        expected = sha_file(recipe.spec.tokenizer)
    mix = MixtureStream(
        recipe.sources, stages=recipe.stages, seed=recipe.seed, tokenizer_sha=expected
    )
    return Mixture(
        mix,
        heldout_bytes_per_token(recipe),
        batch=recipe.spec.batch,
        seq_len=recipe.spec.seq_len,
        device=device,
    )


def _optimizer(
    model: nn.Module, recipe: RunRecipe, cfg: LoopConfig
) -> tuple[MixOptimizer, Callable[[int], float]]:
    """Build the arms and the rate the `schedule` key names, routed by `train/optim.py`."""
    optim = MixOptimizer(
        model,
        schedule=recipe.schedule,
        lr_muon=recipe.lr_muon,
        lr_adamw=recipe.lr_adamw,
        muon_ns_dtype=recipe.muon_ns_dtype,
        sf_warmup=cfg.warmup_steps,
    )
    if recipe.schedule != "cosine":
        # `wsd` is the recipe's own shape; `sf` derives its rate from its own counter, so
        # this function is only the loop's anneal check — the arms never see a multiplier.
        return optim, annealed_schedule(cfg)

    def rate(step: int) -> float:
        """Return the cosine's rate for `step`, as a fraction of peak."""
        return float(warmup_cosine(step, warmup=cfg.warmup_steps, total=cfg.steps, peak=1.0))

    return optim, rate


def _counts(trainer: Trainer, assembled: Assembled) -> dict[str, int]:
    """Return the last step's counters, requiring every mechanism the run enabled."""
    counts = {name: int(value) for name, value in trainer.counters.counts.items() if value}
    expected = set(STEP_COUNTERS)
    expected |= {
        counter for mechanism, counter in assembled.mechanisms().items() if mechanism != "ntp"
    }
    silent = sorted(name for name in expected if counts.get(name, 0) == 0)
    if silent:
        raise LoudError(
            f"enabled but never reported: {', '.join(silent)}",
            "either the flag did not reach the code or the mechanism did not run; a run "
            "whose journal cannot show its own mechanisms is the defect counters exist for",
        )
    return dict(sorted(counts.items()))


def _heldout(recipe: RunRecipe, steps: int, device: torch.device) -> HeldOut:
    """Return the held-out window, refusing one the training arithmetic would reach into."""
    if recipe.eval_path is None:
        raise LoudError(
            f"{recipe.path}: no eval stream",
            'add eval = "heldout.bin": BPB and the harness are measured on data the run '
            "did not train on, and an eval over the training stream measures memorisation",
        )
    probes, _ = recipe.probes(steps)
    window = HeldOut(
        recipe.eval_path,
        heldout_bytes_per_token(recipe),
        batch=recipe.spec.batch,
        seq_len=recipe.spec.seq_len,
        probes=probes,
        device=device,
    )
    read = steps * recipe.spec.batch * (recipe.spec.seq_len + 1)
    for source in recipe.sources:
        if Path(source.path) == recipe.eval_path and read > window.start:
            raise LoudError(
                f"{recipe.eval_path}: the run reads {read} tokens and the held-out window "
                f"starts at {window.start}",
                "shorten the run, or point eval at a separate shard: a window the training "
                "stream reaches is a training number reported as a held-out one",
            )
    return window


def train(
    config: str | Path,
    *,
    runs: str | Path = "runs",
    resume: bool = False,
    smoke: bool = False,
    until: int | None = None,
    approve: bool = False,
    graph: bool = False,
    muon_ns_dtype: str | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run one config end to end and return its receipt: journal, numbers, artifact.

    `graph` replays a captured whole step after warm-up. It is OFF until the live-shape
    parity gate passes; a capture that refuses is LOUD, never an eager fallback.
    """
    recipe = RunRecipe.read(config)
    if muon_ns_dtype is not None:
        recipe = replace(recipe, muon_ns_dtype=muon_ns_dtype)
    target = _resolve_device(recipe.spec.device)
    budget = SMOKE_STEPS if smoke else recipe.steps
    steps = budget if until is None else min(until, budget)
    if until is not None and until > budget:
        raise LoudError(
            f"--steps {until} is past this run's {budget} steps",
            f"raise tokens in {recipe.path} (one step is {recipe.tokens_per_step} tokens); "
            "the stage table has to cover the run it feeds, so a longer horizon is a "
            "different recipe",
        )
    directory = recipe.directory(runs)
    if resume:
        refuse_drift(recipe, directory)
    else:
        fresh(recipe, directory)

    seconds, priced, plan_source = _plan(recipe.spec, smoke=smoke)
    if seconds * steps / max(recipe.steps, 1) > BUDGET_SECONDS and not approve:
        raise LoudError(
            f"the plan needs {seconds * steps / recipe.steps / 3600:.1f}h, the budget is "
            f"{BUDGET_SECONDS / 3600:.0f}h",
            "re-run with --yes to accept the cost, or shrink the run",
        )

    cfg = recipe.loop(budget)
    if smoke:
        # A 47k-parameter step is dispatch-bound: ten threads spent nine cores to be
        # slower (9.4 s of CPU for 1.7 s of work, measured), so the seam test runs on one
        # and its timing says something about the code rather than the core count.
        torch.set_num_threads(1)
    torch.manual_seed(recipe.seed)
    assembled = assemble(recipe.spec, device=target, smoke=smoke)
    per_step = 6 * assembled.report()["active"] * cfg.tokens_per_step
    _assert_plan("start", priced, per_step)

    stream = _mixture(recipe, target)
    window = _heldout(recipe, budget, target)
    journal = Journal.open(directory / "run.jsonl")
    optim, schedule = _optimizer(assembled.model, recipe, cfg)
    if resume:
        trainer = Trainer.load(
            directory,
            model=assembled.model,
            stream=stream,
            optim=optim,
            schedule=schedule,
            journal=journal,
            eval_stream=window,
            use_graph=graph,
            heartbeat=echo,
        )
    else:
        trainer = Trainer(
            assembled.model,
            stream,
            optim,
            schedule,
            cfg,
            journal,
            window,
            use_graph=graph,
            heartbeat=echo,
        )
        stream.mix.record(journal, 0)
    if journal.last("start") is None:
        report.note(
            journal,
            "start",
            config=trainer.run_id,
            steps=budget,
            warmup_steps=cfg.warmup_steps,
            anneal_start=cfg.anneal_start,
            clip_z=cfg.clip_z if cfg.clip else 0.0,
            ema=cfg.ema,
            schedule=recipe.schedule,
            config_hash=recipe.digest,
            plan_flops=priced,
            plan_source=plan_source,
            step_flops=per_step,
            graph=graph,
            device=str(target),
        )
    resumed_from = trainer.step
    echo(f"run {directory}: {steps} steps from {resumed_from}, {recipe.schedule} schedule")
    done = trainer.train(
        steps=steps,
        eval_every=recipe.eval_every or None,
        ckpt_every=recipe.ckpt_every or None,
    )
    trainer.audit()
    counts = _counts(trainer, assembled)
    _assert_plan(
        f"step {trainer.step - 1}",
        priced,
        6 * assembled.report()["active"] * int(trainer.counters.counts.get("fwd", 0)),
    )
    receipt: dict[str, Any] = {
        "directory": str(directory),
        "steps": trainer.step,
        "bpb": done["bpb"],
        "counters": counts,
        "plan_flops": priced,
        "plan_source": plan_source,
        "step_flops": per_step,
        "resumed_from": resumed_from,
        "graph": graph,
        "graph_replays": int(trainer.counters.counts.get("graph_replays", 0)),
    }
    if trainer.step >= budget:
        receipt.update(_finish(recipe, trainer, window, directory, counts, smoke, echo))
    echo(
        f"{trainer.step} steps, {int(done['tokens']):,} tokens, bpb {done['bpb']:.4f} "
        f"-> {directory}"
    )
    return receipt


def _finish(
    recipe: RunRecipe,
    trainer: Trainer,
    window: HeldOut,
    directory: Path,
    counts: dict[str, int],
    smoke: bool,
    echo: Callable[[str], None],
) -> dict[str, Any]:
    """Evaluate the final weights, write `final.pt` and the export, and journal the sum."""
    final = trainer.final_model()
    _, first = recipe.probes(trainer.cfg.steps)
    arms = trainer.optim if recipe.schedule != "sf" else _NoSwap()
    bpb = report.evaluate(
        final,
        cast(MixOptimizer, arms),
        window,
        batches=recipe.eval_batches,
        tokens_per_step=recipe.tokens_per_step,
        first=first,
    )
    artifact, blob = _export(recipe, final, trainer.step, directory, smoke)
    torch.save({**blob, "config": recipe.canonical(), "counters": counts}, directory / "final.pt")
    scores: dict[str, float] = {}
    if recipe.harness_on:
        encode, decode = vocabulary(recipe.spec)
        rows = harness.run(
            suite(recipe.seed), Talker(final, recipe.spec, encode=encode, decode=decode)
        )
        scores = {row.task: row.score for row in rows}
        table = evals.render(rows, bpb)
        (directory / f"eval-{trainer.step}.txt").write_text(table + "\n", encoding="utf-8")
        echo(table)
    report.note(
        trainer.journal,
        "eval",
        step=trainer.step,
        bpb=bpb.bpb,
        bytes=bpb.bytes_evaluated,
        weights="final",
        harness=scores,
        artifact=str(artifact.relative_to(directory)),
        counters=counts,
    )
    return {
        "bpb": bpb.bpb,
        "bytes": bpb.bytes_evaluated,
        "harness": scores,
        "artifact": str(artifact),
    }


def _export(
    recipe: RunRecipe, model: nn.Module, step: int, directory: Path, smoke: bool
) -> tuple[Path, dict[str, Any]]:
    """Write the artifact: the spec, the weights, and the tokenizer that reads them."""
    export = directory / "export"
    export.mkdir(exist_ok=True)
    tokenizer = ""
    if recipe.spec.tokenizer:
        source = Path(recipe.spec.tokenizer)
        shutil.copy2(source, export / source.name)
        tokenizer = source.name
    blob: dict[str, Any] = {
        "spec": recipe.spec.as_dict(),
        "state": model.state_dict(),
        "step": step,
        "smoke": smoke,
        "tokenizer": tokenizer,
    }
    path = export / "model.pt"
    torch.save(blob, path)
    shutil.copy2(recipe.path, export / "config.toml")
    return path, blob


def load_artifact(path: str | Path, *, device: str = "cpu") -> tuple[StackSpec, nn.Module]:
    """Read an exported artifact back: the spec it was built at and the weights it holds.

    Artifacts are portable and load on CPU by default; pass `device="cuda"` to place
    one on a card. The train path picks its own device from the spec.
    """
    return _build(_read(path), _resolve_device(device))


def _read(path: str | Path) -> dict[str, Any]:
    """Read an artifact's mapping, refusing a file that is not one."""
    raw: Any = torch.load(_artifact_path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "spec" not in raw or "state" not in raw:
        raise LoudError(
            f"{path}: not an artifact (no spec and state)",
            "point at runs/<id>/export/model.pt, or at runs/<id>/final.pt, which carries "
            "the same two",
        )
    return raw


def _build(raw: dict[str, Any], device: torch.device) -> tuple[StackSpec, nn.Module]:
    """Build the stack an artifact names on `device` and load its weights into it."""
    spec = cast(StackSpec, StackSpec.from_mapping(raw["spec"]))
    model = assemble(spec, device=device, smoke=bool(raw["smoke"])).model
    model.load_state_dict(raw["state"])
    model.eval()
    return spec, model


def evaluate(checkpoint_path: str | Path, *, echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Score a checkpoint: the five-task table and the held-out BPB with its byte window."""
    path = _artifact_path(checkpoint_path)
    raw = _read(path)
    if bool(raw.get("smoke", False)):
        torch.set_num_threads(1)
    spec, model = _build(raw, torch.device("cpu"))
    directory = path.parent
    stored = directory / "config.json"
    if not stored.exists():
        raise LoudError(
            f"{path}: no config.json in {directory}",
            "point at runs/<id>/final.pt: the eval window and the cadence are the run's, "
            "and a checkpoint without them has no comparable BPB",
        )
    recipe = RunRecipe.from_canonical(
        json.loads(stored.read_text(encoding="utf-8")), path=directory / "config.toml"
    )
    if recipe.eval_path is None:
        raise LoudError(
            f"{directory}: the run recorded no eval stream",
            'add eval = "heldout.bin" to the config and retrain: BPB without a window is '
            "not a number",
        )
    probes, first = recipe.probes(recipe.steps)
    window = HeldOut(
        recipe.eval_path,
        heldout_bytes_per_token(recipe),
        batch=spec.batch,
        seq_len=spec.seq_len,
        probes=probes,
        device=torch.device("cpu"),
    )
    bpb = report.evaluate(
        model,
        cast(MixOptimizer, _NoSwap()),
        window,
        batches=recipe.eval_batches,
        tokens_per_step=spec.batch * spec.seq_len,
        first=first,
    )
    encode, decode = vocabulary(spec)
    rows = harness.run(suite(recipe.seed), Talker(model, spec, encode=encode, decode=decode))
    echo(evals.render(rows, bpb))
    return {
        "bpb": bpb.bpb,
        "bytes": bpb.bytes_evaluated,
        "harness": {row.task: row.score for row in rows},
    }


def _artifact_path(path: str | Path) -> Path:
    """Return `runs/<id>` → `runs/<id>/final.pt`; a file is taken as it is."""
    out = Path(path)
    if out.is_dir():
        out = out / "final.pt"
    if not out.exists():
        raise LoudError(
            f"no checkpoint at {out}",
            "point at runs/<id>/final.pt or runs/<id>: a finished run writes both the "
            "export and the final weights",
        )
    return out
