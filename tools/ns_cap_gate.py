"""One-arm on-card gate for the Newton-Schulz cap (401M/500M b2 step).

One arm per fresh process: CUDA pages are retained across arms in one process and
fake the second arm's peak. Pair two runs with `--cap-bytes 0` (full) and
`--cap-bytes 134217728` (capped):

    .venv-cuda/bin/python tools/ns_cap_gate.py --lock --config configs/dormouse_1b.toml \
        --batch 2 --ns-dtype fp32 --cap-bytes 0 --steps 1
    .venv-cuda/bin/python tools/ns_cap_gate.py --lock --config configs/dormouse_1b.toml \
        --batch 2 --ns-dtype fp32 --cap-bytes 134217728 --steps 1

Prints one JSON line: peak MiB (torch allocator, this process only), CUDA-event
ms/step, finite-loss flag and the cap actually used. Refuses without CUDA.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import tomllib
from contextlib import ExitStack
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy

LOCK = Path("/tmp/dormouse-gpu.lock")


def parse_args() -> argparse.Namespace:
    """Read the one-arm config: stack, cap, dtype, step count and lock choice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--ns-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--cap-bytes", type=int, default=134217728)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lock", action="store_true")
    return parser.parse_args()


def build(
    config: str, batch: int, seq: int, ns_dtype: str, cap_bytes: int
) -> tuple[object, MixOptimizer]:
    """Assemble the config's stack and point its Muon arm at this arm's cap."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    spec = replace(spec, batch=batch, seq_len=seq or spec.seq_len)
    assembled = assemble(spec, device="cuda")
    optimizer = MixOptimizer(assembled.model, muon_ns_dtype=ns_dtype)
    optimizer.muon.chunk_bytes = cap_bytes or None
    return assembled, optimizer


def main() -> None:
    """Run one warm step, then measure the next one's peak bytes and GPU time."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ns_cap_gate: CUDA unavailable; this is a card gate")
    with ExitStack() as stack:
        if args.lock:
            handle = stack.enter_context(LOCK.open("w"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        assembled, optimizer = build(
            args.config, args.batch, args.seq, args.ns_dtype, args.cap_bytes
        )
        spec = assembled.spec  # type: ignore[attr-defined]
        torch.manual_seed(7)
        ids = torch.randint(
            0,
            spec.vocab,
            (args.steps + 1, args.batch, spec.seq_len),
            device="cuda",
        )
        inputs, targets = ids[:, :-1], ids[:, 1:]
        start = end = None
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if step == 1:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
                logits = assembled.model(inputs[step])  # type: ignore[attr-defined]
                loss = chunked_cross_entropy(logits, targets[step])
            loss.backward()
            optimizer.muon.step()
            optimizer.adamw.step()
            if step == args.steps and start is not None and end is not None:
                end.record()
                torch.cuda.synchronize()
        assert start is not None and end is not None
        print(
            json.dumps(
                {
                    "config": Path(args.config).name,
                    "batch": args.batch,
                    "seq": spec.seq_len,
                    "ns_dtype": args.ns_dtype,
                    "cap_bytes": args.cap_bytes,
                    "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "ms_per_step": round(start.elapsed_time(end) / args.steps, 2),
                    "finite": bool(torch.isfinite(loss)),
                    "loss": round(float(loss.detach()), 5),
                }
            )
        )


if __name__ == "__main__":
    main()
