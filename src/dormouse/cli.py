"""The `dorm` command: one entry point, no silent exits.

Two commands are real today — `doctor`, which says what this machine can do, and
`status`, which reads a run journal. The rest refuse with a named escape instead
of pretending: a command that is not wired must not look like one that ran.

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
from pathlib import Path

from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

#: Commands that exist and do something. Everything else refuses.
READY = frozenset({"doctor", "status"})

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


if __name__ == "__main__":
    raise SystemExit(main())
