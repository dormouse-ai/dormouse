"""The `dorm` command: one entry point, no silent exits.

Three commands are real today — `doctor`, which says what this machine can do, `plan`,
which prices a config on the `meta` device without allocating it, and `status`, which
reads a run journal. The rest refuse with a named escape instead of pretending: a
command that is not wired must not look like one that ran.

The CLI's contract, in three lines:

* every failure exits non-zero and names the way out;
* no command writes to a log file — runs write to a journal, the CLI prints;
* nothing long-running starts without a plan and, past the budget, `--yes`.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .model.spec import StackSpec

from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

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
    try:
        return _dispatch(args)
    except LoudError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dorm",
        description="One entry point for training, evaluation, serving and export.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="report what this machine can do")

    status = sub.add_parser("status", help="read a run journal")
    status.add_argument("journal", nargs="?", default="runs/*/run.jsonl")

    plan = sub.add_parser("plan", help="what a config will run, and what it will cost")
    plan.add_argument("config", help="path to a stack .toml, e.g. configs/kitchen_sink.toml")
    plan.add_argument(
        "--journal",
        default=None,
        help="a run journal whose eval events measure the quality/data elasticity",
    )

    for name in sorted(PLANNED):
        cmd = sub.add_parser(name, help=f"not wired yet: {PLANNED[name]}")
        cmd.add_argument("target", nargs="?", default="")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=1, sort_keys=True))
        return 0
    if args.command == "status":
        return _status(args.journal)
    if args.command == "plan":
        return _plan(args.config, args.journal)
    raise LoudError(
        f"{args.command} is not wired yet",
        PLANNED.get(args.command, "see PLAN.md for the order of work"),
    )


def doctor() -> dict[str, object]:
    """Report the versions, the card and the disk this machine offers."""
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        report["torch"] = None
    report["disk_free_gb"] = round(shutil.disk_usage(Path.home()).free / 1e9, 1)
    report["nvidia_smi"] = _smi()
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


def _plan(config: str, journal: str | None = None) -> int:
    """Print the plan for a config: the stack, its bill, its step, its levers.

    Imported here rather than at module scope: `plan` is the only command that needs
    the model layer, and `dorm doctor` has to work on a machine where torch is not
    installed (it reports that instead of failing).
    """
    from .model.assemble import plan_lines

    elasticity, source = _elasticity(journal)
    for line in plan_lines(_spec(config), elasticity=elasticity, elasticity_from=source):
        print(line)
    return 0


def _elasticity(journal: str | None) -> tuple[float, str]:
    """Return the elasticity the lever verdicts use, and where it came from."""
    from .economics import ELASTICITY, measured_elasticity

    if journal is None:
        return ELASTICITY, "constant"
    measured = measured_elasticity(Journal.open(journal))
    if measured is None:
        raise LoudError(
            f"{journal}: no two eval points to measure the elasticity from",
            "a journal needs two `eval` events carrying loss and tokens (or drop "
            "--journal to use the constant); silence here would price every lever "
            "against a number the run did not produce",
        )
    return measured, f"measured from {journal}"


def _spec(config: str) -> StackSpec:
    """Read a stack config, annotated as what it is: `from_toml` returns the base."""
    from .model.spec import StackSpec

    return cast(StackSpec, StackSpec.from_toml(config))


def _status(pattern: str) -> int:
    """Print one line per journal found, newest last."""
    paths = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)]
    found = [p for p in paths if p.is_file()]
    if not found:
        raise LoudError(
            f"no journal matched {pattern}",
            "pass a path, or run something first: runs/<id>/run.jsonl",
        )
    for path in found:
        print(f"{path}: {Journal.open(path).summary()}")
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
