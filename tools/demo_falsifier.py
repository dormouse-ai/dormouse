"""F0 gate for the Friday demo: prove the OpenCode IQ bridge can fail.

RANDOM must score 0.0 and ORACLE 1.0 on the smoke suite, through the same
stub_server.py -> opencode_solver.py -> iq_eval scorer path a scored run uses.
The in-process floor/ceiling from tools/iq_eval.py is printed alongside it.

    PYTHONPATH=src python tools/demo_falsifier.py
    PYTHONPATH=src python tools/demo_falsifier.py --floor-only
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dormouse.eval.iq import MockJudge, build_suite, evaluate  # noqa: E402
from dormouse.eval.model_adapter import FunctionAdapter  # noqa: E402
from tools import opencode_solver  # noqa: E402


def wait_port(port: int, timeout: float = 10.0) -> None:
    """Wait until the stub listens, rather than racing the first OpenCode call."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"stub did not listen on {port}")


def score_via_opencode(port: int, seed: int, suite_name: str) -> float:
    """Score one suite through provider local/dormouse, bound to the stub's config."""
    suite = build_suite(suite_name, seed=seed)
    generate = opencode_solver.make_generate(port=port, seed=seed, suite=suite_name)
    adapter = FunctionAdapter(lambda prompt: generate(prompt, 96, ()))
    rows = evaluate(suite.tasks, adapter, MockJudge())
    for row in rows:
        record = {"task": row.task, "score": row.score, "reason": row.reason}
        print(json.dumps(record, ensure_ascii=False))
    return sum(row.score for row in rows) / len(rows)


def score_inprocess(model: str, seed: int, suite_name: str) -> float:
    """Report the iq_eval.py floor/ceiling for a same-run comparison."""
    suite = build_suite(suite_name, seed=seed)
    if model == "mock":
        from dormouse.eval.model_adapter import RandomAdapter

        adapter = RandomAdapter(seed)
    else:
        from dormouse.eval.model_adapter import KnownAdapter

        adapter = KnownAdapter(suite.tasks)
    rows = evaluate(suite.tasks, adapter, MockJudge())
    return sum(row.score for row in rows) / len(rows)


def main() -> int:
    """Run the OpenCode floor and ceiling, then refuse if either is wrong."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    p.add_argument("--floor-only", action="store_true", help="in-process scores, no OpenCode")
    args = p.parse_args()

    suite = build_suite(args.suite, seed=args.seed)
    print(f"{args.suite} tasks={len(suite)} seed={args.seed}")
    print(
        f"iq_eval in-process: mock={score_inprocess('mock', args.seed, args.suite):.3f} "
        f"known={score_inprocess('known', args.seed, args.suite):.3f}"
    )
    if args.floor_only:
        return 0

    scores: dict[str, float] = {}
    for mode in ("RANDOM", "ORACLE"):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["DORMOUSE_STUB_PORT"] = str(args.port)
        env["STUB_MODE"] = mode
        env["STUB_SEED"] = str(args.seed)
        env["STUB_SUITE"] = args.suite
        server = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tools" / "stub_server.py"),
                "--port",
                str(args.port),
                "--mode",
                mode,
                "--seed",
                str(args.seed),
                "--suite",
                args.suite,
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            wait_port(args.port)
            scores[mode] = score_via_opencode(args.port, args.seed, args.suite)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        print(f"OpenCode {mode}: {scores[mode]:.3f}")

    ok = scores["RANDOM"] == 0.0 and scores["ORACLE"] == 1.0
    print(
        f"F0 falsifier: {'PASS' if ok else 'REFUSE'} "
        f"RANDOM={scores['RANDOM']:.3f} ORACLE={scores['ORACLE']:.3f}"
    )
    if not ok:
        print("REFUSE: the bridge cannot fail; no scored run may start", file=sys.stderr)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
