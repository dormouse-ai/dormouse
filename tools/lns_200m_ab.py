"""Deterministic on-card paired A/B for LNS on the 200M demo stack.

Every arm uses the SAME config, seed, batches and bf16 autocast; the only difference
is `DORMOUSE_LNS`. `DORMOUSE_BANK=0`, `DORMOUSE_SPARSE=0` and
`torch.use_deterministic_algorithms(True)` route the scattered reductions through
PyTorch's deterministic CUDA path: an unmodified shipped-stack control-vs-control
drifts by ~0.03 BPB in 10 steps (bank Triton atomics), which makes the 0.01 gate
unmeasurable. The cost gate is measured separately on the SHIPPED path with
`tools/profile_step.py`; see research/perf/LNS-200m.md.

    python tools/lns_200m_ab.py --steps 200 --seeds 11 12 \
        --config-dir /tmp/opencode/lns200m \
        --out /tmp/opencode/lns200m/ab.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

# These are read at MODULE IMPORT by the kernel flags. Setting them inside _run
# would leave the shipped atomics live and the control would drift.
os.environ["DORMOUSE_BANK"] = "0"
os.environ["DORMOUSE_SPARSE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dormouse.train import entry, streams  # noqa: E402
from dormouse.train.loop import Trainer  # noqa: E402

#: One-sided 95% Student t. df=1 (two seeds) is 6.314; df=2 (three seeds) is 2.920.
T95 = {1: 6.314, 2: 2.920}


def _on_cuda() -> None:
    """Only the two device seams: assemble on the card, move batch ids there."""
    original = entry.assemble

    def cuda_assemble(spec: Any, *, device: Any = None, smoke: bool = False) -> Any:
        return original(spec, device=("cuda" if device is None else device), smoke=smoke)

    entry.assemble = cuda_assemble
    for cls in (streams.Mixture, streams.HeldOut):
        original_batch = cls.batch

        def batch_on_cuda(self: Any, step: int, tokens: int, _orig: Any = original_batch) -> Any:
            item = _orig(self, step, tokens)
            return replace(item, ids=item.ids.to("cuda"))

        cls.batch = batch_on_cuda


def _no_checkpoint(self: Trainer) -> Path:
    """A/B runs end before the recipe's budget; a 3.6 GB artifact is not the gate."""
    return Path("/dev/null")


def _journal(path: Path) -> tuple[list[float], float | None]:
    """Return (step losses, final heldout BPB) from one run's own journal."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    losses = [float(r["loss"]) for r in records if r.get("kind") == "step" and "loss" in r]
    evals = [
        float(r["bpb"])
        for r in records
        if r.get("kind") == "eval" and r.get("weights") == "training"
    ]
    return losses, (evals[-1] if evals else None)


def _run(config: Path, runs: Path, tag: str, *, lns: bool, steps: int) -> dict[str, Any]:
    """One arm: env set, engine untouched, bf16 autocast, deterministic CUDA."""
    if lns:
        os.environ["DORMOUSE_LNS"] = "1"
    else:
        os.environ.pop("DORMOUSE_LNS", None)
    _on_cuda()
    Trainer._checkpoint = _no_checkpoint

    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        receipt = entry.train(
            str(config), runs=str(runs / tag), until=steps, approve=True, echo=lambda *a: None
        )
    directory = Path(receipt["directory"])
    losses, bpb = _journal(directory / "run.jsonl")
    if bpb is None:
        raise RuntimeError(f"{tag}: no eval line in {directory / 'run.jsonl'}")
    return {
        "tag": tag,
        "seed": int(config.stem.rsplit("_s", 1)[1]),
        "lns": lns,
        "bpb": bpb,
        "losses": losses,
        "counters": receipt["counters"],
        "wall_s": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    """Run every arm, require an exact control floor, print the paired verdict."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    p.add_argument("--config-dir", type=Path, default=Path("/tmp/opencode/lns200m"))
    p.add_argument("--runs", type=Path, default=Path("/tmp/opencode/lns200m/ab"))
    p.add_argument("--out", type=Path, default=Path("/tmp/opencode/lns200m/ab.json"))
    p.add_argument("--skip-cvc", action="store_true", help="already proved the control floor")
    args = p.parse_args(argv)

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for lns in (False, True):
            tag = f"{'lns' if lns else 'ctrl'}_s{seed}"
            config = args.config_dir / f"{tag}.toml"
            result = _run(config, args.runs, tag, lns=lns, steps=args.steps)
            runs.append(result)
            print(
                f"{tag:9s} heldout BPB {result['bpb']:.6f}  "
                f"lns_layers={result['counters'].get('lns_layers', 0)}  "
                f"{result['wall_s']}s",
                flush=True,
            )

    floor = 0.0
    if not args.skip_cvc:
        seed = args.seeds[0]
        repeat = _run(
            args.config_dir / f"ctrl_s{seed}.toml",
            args.runs,
            f"cvc_s{seed}",
            lns=False,
            steps=args.steps,
        )
        control = next(r for r in runs if r["tag"] == f"ctrl_s{seed}")
        if repeat["losses"] != control["losses"] or repeat["bpb"] != control["bpb"]:
            floor = abs(repeat["bpb"] - control["bpb"])
            raise RuntimeError(
                f"control-vs-control drift {floor:.9f} BPB at seed {seed}: "
                "the 0.01 gate is unmeasurable in this regime"
            )
        runs.append(repeat)

    deltas = []
    for seed in args.seeds:
        ctrl = next(r for r in runs if r["tag"] == f"ctrl_s{seed}")
        lns_run = next(r for r in runs if r["tag"] == f"lns_s{seed}")
        deltas.append(lns_run["bpb"] - ctrl["bpb"])
    mean = statistics.fmean(deltas)
    spread = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    hw = T95[len(deltas)] * spread / math.sqrt(len(deltas))
    upper, lower = mean + hw, mean - hw
    counts_ok = all(r["counters"].get("lns_layers", 0) > 0 for r in runs if r["lns"])
    verdict = "PASS" if upper <= 0.01 and counts_ok else "REFUSE"
    gate = {
        "steps": args.steps,
        "seeds": args.seeds,
        "control_floor_bpb": floor,
        "dBPB": {str(s): round(d, 6) for s, d in zip(args.seeds, deltas, strict=True)},
        "mean": round(mean, 6),
        "lower95": round(lower, 6),
        "upper95": round(upper, 6),
        "lns_layers_positive": counts_ok,
        "upper_gate": upper <= 0.01,
        "improvement_sign": lower < 0,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"gate": gate, "runs": runs}, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(gate, sort_keys=True), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
