"""Interleaved min-of-7 CUDA-event timing: full vs capped NS chain.

One process, one set of input tensors, alternating arms to cancel clock drift and
sibling load. Peak bytes are measured separately (ns_cap_shapes.py, one arm/process).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.train.muon import _polar_many

BUCKETS = (
    (65, 256, 6144),
    (65, 256, 8192),
    (65, 256, 7168),
    (1, 8192, 6144),
)


def main() -> None:
    """Alternate capped and full arms and print each arm's best chain milliseconds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ns_cap_ms: CUDA unavailable")
    with ExitStack() as stack:
        if args.lock:
            handle = stack.enter_context(Path("/tmp/dormouse-gpu.lock").open("w"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        torch.manual_seed(7)
        batch = [torch.randn(n, s, cols, device="cuda") for n, s, cols in BUCKETS]
        arms = {"full": None, "cap": 134217728}
        for cap in arms.values():
            for _ in range(2):
                _polar_many(batch, ns_dtype=args.ns_dtype, chunk_bytes=cap)
        torch.cuda.synchronize()
        best = {name: float("inf") for name in arms}
        order = list(arms)
        for round_index in range(args.rounds + 1):
            if round_index % 2:
                order.reverse()
            for name in order:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                _polar_many(batch, ns_dtype=args.ns_dtype, chunk_bytes=arms[name])
                end.record()
                torch.cuda.synchronize()
                if round_index:
                    best[name] = min(best[name], start.elapsed_time(end))
        print(
            json.dumps(
                {
                    "ns_dtype": args.ns_dtype,
                    **{f"{name}_ms": round(ms, 2) for name, ms in best.items()},
                    "cap_over_full": round(best["cap"] / best["full"], 4),
                }
            )
        )


if __name__ == "__main__":
    main()
