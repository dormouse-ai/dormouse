"""Run one A/B arm on the card with bf16, reusing the engine untouched.

The CLI's `dorm train` builds on the default device (CPU) and has no dtype flag, so a
GPU measurement cannot call it as-is. This harness changes only two seams, neither in
`model/`: `entry.assemble` is asked for `device="cuda"`, and the two batch producers
move their `[rows, seq+1]` ids to the card. The loop, guards, mixture, journal, eval
and artifact are the engine's own; autocast wraps the whole call, so both arms see
the identical bf16 step the profiler times (and the optimizer runs in the same
autocast state on both arms, so the A/B stays paired).

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python scripts/ab_train.py \
        configs/ab/ctrl_s0.toml --runs runs/ab_top1_r192 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.train import entry, streams

GPU_LOCK = "/tmp/dormouse-gpu.lock"


def gpu_used_mib() -> int:
    """Return the driver's reported used VRAM in MiB, for the before/after receipt."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def _on_cuda() -> None:
    """Point the engine's two device seams at the card, patching nothing in `model/`."""
    original = entry.assemble

    def cuda_assemble(spec: Any, *, device: Any = None, smoke: bool = False) -> Any:
        """Build on the card unless a caller (the meta-device plan) named a device."""
        return original(spec, device=("cuda" if device is None else device), smoke=smoke)

    entry.assemble = cuda_assemble

    for cls in (streams.Mixture, streams.HeldOut):
        original_batch = cls.batch

        def batch_on_cuda(self: Any, step: int, tokens: int, _orig: Any = original_batch) -> Any:
            """Return the engine's batch with its ids moved to the card."""
            item = _orig(self, step, tokens)
            return replace(item, ids=item.ids.to("cuda"))

        cls.batch = batch_on_cuda


def main() -> int:
    """Train one config on the card in bf16 and leave a receipt beside its journal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a run .toml (stack keys + [source] + eval + seed)")
    parser.add_argument("--runs", default="runs/ab_top1_r192", help="run directory root")
    parser.add_argument("--steps", type=int, default=4000, help="absolute step horizon")
    parser.add_argument("--no-amp", action="store_true", help="train in fp32 (no autocast)")
    parser.add_argument(
        "--amp-forward-only",
        action="store_true",
        help="keep the optimizer outside autocast, as tools/profile_step.py does",
    )
    args = parser.parse_args()

    _on_cuda()
    if args.amp_forward_only:
        from dormouse.train.optim import MixOptimizer

        original_step = MixOptimizer.step

        def step_outside_autocast(self: Any) -> None:
            """Run Muon/AdamW in fp32, as the profiler's phase table does."""
            with torch.autocast("cuda", enabled=False):
                original_step(self)

        MixOptimizer.step = step_outside_autocast
    before = gpu_used_mib()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
        receipt = entry.train(
            args.config, runs=args.runs, until=args.steps, approve=True, echo=print
        )
    wall_s = time.perf_counter() - started
    after = gpu_used_mib()

    summary = {
        **receipt,
        "wall_s": round(wall_s, 1),
        "gpu_before_mib": before,
        "gpu_after_mib": after,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    directory = Path(receipt["directory"])
    (directory / "run_summary.jsonl").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
