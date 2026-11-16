"""Turn the A/B's measurement files into the tables and the pre-registered verdict.

Nothing here re-measures anything: every number is read from a `run.jsonl`, a
`run_summary.jsonl` or a profiler `profile.txt` written by the chain. The byte ratio
is the one recorded in the corpus manifest, not an assumed constant.

    .venv/bin/python scripts/ab_report.py runs/ab_top1_r192
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BPT = 4.717789094290279  # mix_smoke.bin.json: measured text bytes per token
SPEED_THRESHOLD = 1.30
BPB_LOSS_THRESHOLD = 0.03
PHASES = ("forward", "ce", "backward", "clip", "muon", "adamw", "other")


def quality(root: Path) -> list[dict[str, object]]:
    """Read each arm/seed's final-eval BPB from its own journal and receipt."""
    rows: list[dict[str, object]] = []
    for family in ("ctrl", "topk1"):
        for seed in (0, 1, 2):
            directory = root / f"{family}_s{seed}"
            if not (directory / "run.jsonl").exists():
                continue
            lines = (directory / "run.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            evals = [r for r in records if r.get("kind") == "eval" and r.get("weights") == "final"]
            if not evals:
                continue
            final = evals[-1]
            receipt = json.loads((directory / "run_summary.jsonl").read_text().strip())
            rows.append(
                {
                    "arm": family,
                    "seed": seed,
                    "bpb": final["bpb"],
                    "bytes": final["bytes"],
                    "steps": receipt["steps"],
                    "wall_s": receipt["wall_s"],
                    "gpu_before_mib": receipt["gpu_before_mib"],
                    "gpu_after_mib": receipt["gpu_after_mib"],
                    "journal": str(directory / "run.jsonl"),
                }
            )
    return rows


def speed(root: Path) -> list[dict[str, object]]:
    """Parse each profiler transcript for ms/step, launches/step and its phase table."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("speed_*_?/profile.txt")):
        arm = path.parent.name.removeprefix("speed_").rsplit("_", 1)[0]
        text = path.read_text(encoding="utf-8")
        total = re.search(r"^\s+total\s+([\d.]+)\s+([\d,]+)\s+tok/s", text, re.M)
        launches = re.search(r"([\d,]+)\s+kernel launches/step", text)
        if not total or not launches:
            continue
        ms = float(total.group(1))
        tok_s = float(total.group(2).replace(",", ""))
        phases: dict[str, float] = {}
        for name in PHASES:
            hit = re.search(rf"^\s+{name}\s+([\d.]+)\s+", text, re.M)
            if hit:
                phases[name] = float(hit.group(1))
        rows.append(
            {
                "arm": arm,
                "run": path.parent.name,
                "ms_per_step": ms,
                "tokens_per_s": tok_s,
                "bytes_per_s": round(tok_s * BPT),
                "launches_per_step": int(launches.group(1).replace(",", "")),
                "phases_ms": phases,
                "profile": str(path),
            }
        )
    return rows


def median(values: list[float]) -> float:
    """Return the middle value, the statistic used for the speed table."""
    return sorted(values)[len(values) // 2]


def main() -> int:
    """Write quality/speed/decision JSONLs and print the three tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="the run root, e.g. runs/ab_top1_r192")
    args = parser.parse_args()
    root = Path(args.root)

    q = quality(root)
    s = speed(root)
    for name, rows in (("quality", q), ("speed", s)):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )

    for row in q:
        print(f"BPB  {row['arm']:6} s{row['seed']}  {row['bpb']:.4f}  ({row['journal']})")
    ctrl = {int(r["seed"]): float(r["bpb"]) for r in q if r["arm"] == "ctrl"}
    arm = {int(r["seed"]): float(r["bpb"]) for r in q if r["arm"] == "topk1"}
    deltas = {seed: (arm[seed] / ctrl[seed] - 1.0) for seed in sorted(ctrl.keys() & arm.keys())}
    for seed, delta in deltas.items():
        print(f"DELTA s{seed}  {delta:+.2%}")

    ctrl_ms = median([float(r["ms_per_step"]) for r in s if r["arm"] == "ctrl"])
    arm_ms = median([float(r["ms_per_step"]) for r in s if r["arm"] == "topk1"])
    speedup = ctrl_ms / arm_ms if arm_ms else 0.0
    for row in s:
        print(
            f"SPEED {row['arm']:6} {row['run']:16} {row['ms_per_step']:8.1f} ms  "
            f"{row['tokens_per_s']:7,.0f} tok/s  {row['bytes_per_s']:6,} B/s  "
            f"{row['launches_per_step']:,} launches"
        )
    print(f"SPEEDUP median control/{arm} = {speedup:.3f}x  (threshold {SPEED_THRESHOLD}x)")
    max_loss = max(deltas.values()) if deltas else 1.0
    if not deltas:
        print("QUALITY: UNMEASURABLE -- see runs/ab_top1_r192/chain.log (router guard)")
    print(f"MAX BPB LOSS = {max_loss:+.2%}  (threshold {BPB_LOSS_THRESHOLD:.0%})")
    verdict = "TAKE" if speedup >= SPEED_THRESHOLD and max_loss <= BPB_LOSS_THRESHOLD else "REFUSE"
    decision = {
        "speedup": round(speedup, 4),
        "max_bpb_loss": round(max_loss, 6),
        "deltas": {str(k): round(v, 6) for k, v in deltas.items()},
        "median_ctrl_ms": ctrl_ms,
        "median_arm_ms": arm_ms,
        "verdict": verdict,
        "quality": "measured" if deltas else "unmeasurable (router-collapse guard)",
    }
    (root / "decision.jsonl").write_text(
        json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("DECISION", json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
