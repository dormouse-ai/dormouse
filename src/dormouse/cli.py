"""The `dorm` command: one entry point, no silent exits.

Three commands are real today — `doctor`, which says what this machine can do, `plan`,
which prices a config on the `meta` device without allocating it, and `status`, which
reads a run journal. The rest refuse with a named escape instead of pretending: a
command that is not wired must not look like one that ran.

The CLI's contract, in four lines:

* every failure exits non-zero and names the way out;
* no command writes to a log file — runs write to a journal, the CLI prints;
* nothing long-running starts without a plan and, past the budget, `--yes`;
* every command has two faces: the screen (`tui`) and `--json`, over the same data.
"""

from __future__ import annotations

import argparse
import glob
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .model.spec import StackSpec

from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

from . import tui

#: Commands that exist and do something. Everything else refuses.
READY = frozenset({"doctor", "status", "plan"})

#: Commands that are planned; the escape tells the reader what to use instead.
PLANNED = {
    "train": "use the current port/ trainer until the model layer moves; see PLAN.md step 4",
    "run": "same as train: the pipeline moves after the model layer",
    "tokenize": "scripts/tokenize_corpus.py in port/ until the data layer moves",
    "eval": "port/scripts for now; the harness moves in PLAN.md step 4",
    "serve": "port/ serve for now; the Rust binary arrives in step 7",
    "export": "port/ export for now; moves with the model layer",
    "audit": "the invariants live in dormouse-core's tests today",
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch, and return the process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    ui = tui.UI()
    try:
        return _dispatch(args, ui)
    except LoudError as exc:
        ui.error(exc)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dorm",
        description="One entry point for training, evaluation, serving and export.",
    )
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="report what this machine can do")
    doctor.add_argument("--json", action="store_true", help="the same report, machine-readable")

    status = sub.add_parser("status", help="read a run journal")
    status.add_argument("journal", nargs="?", default="runs/*/run.jsonl")
    status.add_argument("--json", action="store_true", help="the same rows, machine-readable")

    plan = sub.add_parser("plan", help="what a config will run, and what it will cost")
    plan.add_argument("config", help="path to a stack .toml, e.g. configs/kitchen_sink.toml")
    plan.add_argument("--json", action="store_true", help="the same plan, machine-readable")

    for name in sorted(PLANNED):
        cmd = sub.add_parser(name, help=f"not wired yet: {PLANNED[name]}")
        cmd.add_argument("target", nargs="?", default="")
    return parser


def _dispatch(args: argparse.Namespace, ui: tui.UI) -> int:
    if args.command == "doctor":
        report = doctor()
        if args.json:
            print(tui.as_json(report))
        else:
            ui.doctor(report)
        return 0
    if args.command == "status":
        return _status(args, ui)
    if args.command == "plan":
        return _plan(args, ui)
    raise LoudError(
        f"{args.command} is not wired yet",
        PLANNED.get(args.command, "see PLAN.md for the order of work"),
    )


def doctor() -> dict[str, Any]:
    """Report the versions, the card and the disk this machine offers."""
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "git_sha": _git_sha(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        report["torch"] = None
        report["cuda"] = None
        report["gpu"] = None
    report["disk_free_gb"] = round(shutil.disk_usage(Path.home()).free / 1e9, 1)
    report["nvidia_smi"] = _smi()
    report.update(_vram(report["nvidia_smi"]))
    return report


def _smi() -> str | None:
    """Ask the driver directly; torch can lie about a busy or missing card."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _vram(raw: object) -> dict[str, Any]:
    """Parse `nvidia-smi`'s first line into the name, total and used MiB.

    The driver is the only source that reports memory as well as the name, so this is
    what the doctor's GPU and VRAM rows are built from; torch's own answer is kept as
    a second opinion in `gpu`.
    """
    empty: dict[str, Any] = {"gpu_name": None, "vram_total_mib": None, "vram_used_mib": None}
    if not isinstance(raw, str) or not raw:
        return empty
    parts = [part.strip() for part in raw.splitlines()[0].split(",")]
    if len(parts) != 3:
        return {**empty, "gpu_name": parts[0] or None}
    return {
        "gpu_name": parts[0] or None,
        "vram_total_mib": _mib(parts[1]),
        "vram_used_mib": _mib(parts[2]),
    }


def _mib(text: str) -> int | None:
    """`16303 MiB` → 16303; anything else → None, because a guessed card is worse."""
    number = text.removesuffix("MiB").strip()
    return int(number) if number.isdigit() else None


def _git_sha() -> str | None:
    """Return the commit the CLI was launched from, or None outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _plan(args: argparse.Namespace, ui: tui.UI) -> int:
    """Print the plan for a config: the stack, its bill, its step, its mechanisms.

    Imported here rather than at module scope: `plan` is the only command that needs
    the model layer, and `dorm doctor` has to work on a machine where torch is not
    installed (it reports that instead of failing).
    """
    from .model.assemble import plan_data

    spec = _spec(args.config)
    with ui.progress(spec.n_layers, f"pricing {args.config} on the meta device") as bar:
        data = plan_data(
            spec,
            on_layer=lambda done, total, kind: bar.advance(f"{kind} {done}/{total}"),
            measured_tflops=_measured_tflops(),
        )
    data["config"] = args.config
    if args.json:
        print(tui.as_json(data))
    else:
        ui.plan(data)
    return 0


def _measured_tflops(pattern: str = "runs/*/run.jsonl") -> float | None:
    """Return the newest run's recorded sustained rate, if a journal carries one.

    A step that measures its own rate writes `sustained_tflops` into its record; until
    one does, this returns None and the plan falls back to the documented profile
    number, printing which source it used. Reading the journal (not a log) is the
    project's contract for a number a run produced.
    """
    paths = sorted(
        (Path(found) for found in glob.glob(pattern, recursive=True)),  # noqa: PTH207
        reverse=True,
    )
    for path in paths:
        if not path.is_file():
            continue
        for record in reversed(Journal.open(path).records):
            rate = record.get("sustained_tflops")
            if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
                return float(rate)
    return None


def _spec(config: str) -> StackSpec:
    """Read a stack config, annotated as what it is: `from_toml` returns the base."""
    from .model.spec import StackSpec

    return cast(StackSpec, StackSpec.from_toml(config))


def _status(args: argparse.Namespace, ui: tui.UI) -> int:
    """Print one row per journal found, newest last."""
    pattern = args.journal
    if any(ch in pattern for ch in "*?["):
        # `glob.glob` because `Path.glob` refuses an absolute pattern (PTH207 says the opposite).
        paths = sorted(Path(found) for found in glob.glob(pattern, recursive=True))  # noqa: PTH207
    else:
        paths = [Path(pattern)]
    found = [p for p in paths if p.is_file()]
    if not found:
        raise LoudError(
            f"no journal matched {pattern}",
            "pass a path, or run something first: runs/<id>/run.jsonl",
        )
    now = _now()
    runs = [journal_row(path, now) for path in found]
    if args.json:
        print(tui.as_json({"runs": runs}))
    else:
        ui.status(runs)
    return 0


def _now() -> float:
    """Return the clock `status` measures age against; a function so a test can freeze it."""
    return time.time()


def journal_row(path: Path, now: float) -> dict[str, Any]:
    """One `status` row, read from the journal and nothing else."""
    journal = Journal.open(path)
    start = journal.last("start") or {}
    step = journal.last("step") or {}
    ended = journal.last("end") is not None
    phase = "ended" if ended else ("running" if step else "not started")
    stamp = journal.records[-1].get("t") if journal.records else None
    age = max(now - stamp, 0.0) if isinstance(stamp, (int, float)) else None
    return {
        "id": _run_id(path),
        "config": start.get("config"),
        "phase": phase,
        "step": step.get("step", 0),
        "metric": _metric(journal),
        "segments": len(journal.segments()),
        "age_seconds": age,
    }


def _run_id(path: Path) -> str:
    """`runs/<id>/run.jsonl` names the run after its directory; any other file names itself."""
    return path.parent.name if path.name == "run.jsonl" else path.stem


def _metric(journal: Journal) -> str | None:
    """Return the last number a run reported: the step's loss, else the first eval metric."""
    for kind in ("step", "eval"):
        record = journal.last(kind) or {}
        for name, value in sorted(record.items()):
            if name in {"kind", "t", "step", "config"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{name}={value:.4g}"
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            for name in sorted(metrics):
                value = metrics[name]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return f"{name}={value:.4g}"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
