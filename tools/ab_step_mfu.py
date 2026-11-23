"""Interleaved OFF/ON runs of the locked whole-step MFU harness.

Each arm runs in its own process: the CUDA-graph capture owns ~5.6 GiB of private
pool, so two captures cannot coexist on the 16 GiB card. Order is counterbalanced
(0,1,1,0 per round) so a warming/cooling card cannot favour one arm. The harness
is unchanged; only ``DORMOUSE_MFU40_KERNELS`` differs.

    flock -w 2400 /tmp/dormouse-gpu.lock env \
      PYTHONPATH=src:/home/sehaxe/dormouse2/core/src \
      /home/sehaxe/dormouse2/app/.venv-cuda/bin/python -u tools/ab_step_mfu.py \
      --rounds 2 --json runs/verify_step_mfu/ab_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "verify_step_mfu.py"
PY = os.environ.get("MFU_PY", "/home/sehaxe/dormouse2/app/.venv-cuda/bin/python")


def _run(flag: str, index: int) -> dict[str, Any]:
    """Run one harness arm in a fresh process and return its JSON artifact."""
    out = ROOT / "runs" / "verify_step_mfu" / f"ab_{flag}_{index}.json"
    env = dict(os.environ)
    env["DORMOUSE_MFU40_KERNELS"] = flag
    # SHARDS is a module int captured by the fake op: inductor's on-disk cache does
    # not key on it, so the two arms would otherwise reuse each other's compiled
    # backward and trip an 8-vs-4 metadata guard.
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/opencode/mfu40_inductor_{flag}"
    proc = subprocess.run(
        [PY, "-u", str(HARNESS), "--json", str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"[ab] arm={flag} run={index} rc={proc.returncode}", flush=True)
    if proc.stderr.strip():
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)
    if not out.exists():
        raise RuntimeError(f"arm={flag} run={index} wrote no artifact; rc={proc.returncode}")
    return json.loads(out.read_text(encoding="utf-8"))


def _summary(reports: list[dict[str, Any]], flag: str) -> dict[str, Any]:
    """Median scalars across the runs of one arm; FLOP bill is the harness's own."""
    ms = statistics.median(r["step_ms"] for r in reports)
    mfu = statistics.median(r["mfu_pct_at_92"] for r in reports)
    flops = statistics.median(r["executed_flops_per_step"] for r in reports)
    receipts = reports[0].get("mfu40_counters", {})
    return {
        "arm": f"MFU40-{'ON' if flag == '1' else 'OFF'}",
        "runs": len(reports),
        "step_ms": ms,
        "mfu_pct_at_92": mfu,
        "executed_flops_per_step": flops,
        "all_step_ms": [r["step_ms"] for r in reports],
        "all_mfu_pct": [r["mfu_pct_at_92"] for r in reports],
        "counters": reports[0]["counters"],
        "mfu40_counters": receipts,
        "target_met": any(r["gate"]["target_met"] for r in reports),
    }


def main() -> int:
    """Run the interleaved A/B, print the requested table, write a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--json", type=Path, default=ROOT / "runs" / "verify_step_mfu" / "ab_summary.json"
    )
    args = parser.parse_args()

    order = [flag for _ in range(args.rounds) for flag in ("0", "1", "1", "0")]
    reports: dict[str, list[dict[str, Any]]] = {"0": [], "1": []}
    for i, flag in enumerate(order):
        reports[flag].append(_run(flag, i))

    arms = [_summary(reports["0"], "0"), _summary(reports["1"], "1")]
    print("\n| arm | ms | MFU | FLOP | counter | verdict |")
    print("|---|---:|---:|---:|---|---|")
    for arm in arms:
        live = " ".join(
            f"{k}={arm['counters'].get(k, 0)}" for k in ("graph_replays", "epilogue_fwd", "ns_fp16")
        )
        ko = " ".join(f"{k}={v}" for k, v in arm["mfu40_counters"].items())
        verdict = "PASS" if arm["target_met"] else "REFUTE"
        print(
            f"| {arm['arm']} | {arm['step_ms']:.2f} | {arm['mfu_pct_at_92']:.2f}% | "
            f"{int(arm['executed_flops_per_step']):,} | {live} {ko} | {verdict} |"
        )

    delta_ms = arms[0]["step_ms"] - arms[1]["step_ms"]
    print(
        f"\nON minus OFF: {arms[1]['mfu_pct_at_92'] - arms[0]['mfu_pct_at_92']:+.3f} pp, "
        f"{-delta_ms:+.2f} ms; "
        f"40% line = {arms[0]['executed_flops_per_step'] / (0.40 * 92e12) * 1e3:.2f} ms"
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps({"arms": arms, "order": order}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
