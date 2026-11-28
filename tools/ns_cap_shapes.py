"""NS-chain-only A/B on the live 401M/500M buckets, one arm per process.

The 500M stack cannot be assembled on this 16 GB card at this HEAD (its 808-expert
fp32 bank alone is ~11.8 GiB), so its NS bucket shapes are measured directly here.
Both arms allocate the same momentum-shaped inputs; only the chain cap differs.
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
    ("401m-expert", 65, 256, 6144),
    ("500m-ffn-expert", 65, 256, 8192),
    ("500m-model-expert", 65, 256, 7168),
    ("wide", 1, 8192, 6144),
)


def main() -> None:
    """Time one lockstep chain over all buckets and report its allocator peak."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-bytes", type=int, default=134217728)
    parser.add_argument("--ns-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ns_cap_shapes: CUDA unavailable")
    with ExitStack() as stack:
        if args.lock:
            handle = stack.enter_context(Path("/tmp/dormouse-gpu.lock").open("w"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _run(args)


def _run(args: argparse.Namespace) -> None:
    """Build the four buckets once, then time the chained arm."""
    torch.manual_seed(7)
    batch = [torch.randn(n, s, cols, device="cuda") for _, n, s, cols in BUCKETS]
    cap = args.cap_bytes or None
    for _ in range(2):
        _polar_many(batch, ns_dtype=args.ns_dtype, chunk_bytes=cap)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        out = _polar_many(batch, ns_dtype=args.ns_dtype, chunk_bytes=cap)
    end.record()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "cap_bytes": args.cap_bytes,
                "ns_dtype": args.ns_dtype,
                "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "ms_per_chain": round(start.elapsed_time(end) / args.iters, 2),
                "finite": all(bool(torch.isfinite(x).all()) for x in out),
            }
        )
    )


if __name__ == "__main__":
    main()
