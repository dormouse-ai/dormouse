"""Run the portable IQ harness and print one NDJSON row per task.

    uv run --no-sync python tools/iq_eval.py --model mock --suite smoke --json

`dorm iq-eval` is not wired yet; this tool is the whole CLI for now, and the
future verb will call ``main`` rather than grow a second implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dormouse.eval.iq import (
    SCHEMA_VERSION,
    MockJudge,
    build_suite,
    evaluate,
)
from dormouse.eval.model_adapter import (
    CheckpointAdapter,
    EchoAdapter,
    KnownAdapter,
    RandomAdapter,
)


def main(argv: list[str] | None = None) -> int:
    """Build a suite, score it with one model, emit rows; zero on success."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("mock", "echo", "known", "checkpoint"), default="mock")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="one JSON object per line")
    parser.add_argument("--checkpoint", type=Path, help="artifact for --model checkpoint")
    args = parser.parse_args(argv)

    suite = build_suite(args.suite, seed=args.seed)
    if args.model == "mock":
        model = RandomAdapter(args.seed)
    elif args.model == "echo":
        model = EchoAdapter()
    elif args.model == "known":
        model = KnownAdapter(suite.tasks)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --model checkpoint")
        model = CheckpointAdapter(args.checkpoint)  # pragma: no cover - needs a real artifact

    rows = evaluate(suite.tasks, model, MockJudge())
    if args.json:
        for row in rows:
            record = {
                "schema_version": SCHEMA_VERSION,
                "task": row.task,
                "score": row.score,
                "reason": row.reason,
            }
            print(json.dumps(record, ensure_ascii=False))
    else:
        total = sum(row.score for row in rows) / len(rows)
        print(f"suite {suite.name}: tasks={len(suite)} score={total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
