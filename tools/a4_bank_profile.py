"""Locate the bank's real peak allocation under a training step, on the card.

The shipped dormouse_1b spec at b2 exits before the anchor's backward: its 64-expert
router needs more than the four static dispatch slots. This tool keeps the real FFN
widths (d_ffn=6144, rank) but shrinks the expert count and depth, so the
`[group, capacity, d_ffn]` slab is still the largest thing the step builds and a
two-pass A/B can attribute it.

    PYTHONPATH=src UV_PROJECT_ENVIRONMENT=/home/sehaxe/dormouse2/app/.venv-cuda \
      uv run --no-sync python tools/a4_bank_profile.py --batch 8 --seq 1024 --top 20
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

ROOT = str(Path(__file__).resolve().parents[1])


def spec(batch: int, seq: int, *, n_experts: int = 8, rank: int = 64) -> StackSpec:
    """Build a bank-heavy stack: real d_ffn, a dispatch that fits four passes."""
    raw = tomllib.loads((Path(ROOT) / "configs" / "dormouse_1b.toml").read_text())
    known = {f.name for f in fields(StackSpec) if f.init}
    values = {key: value for key, value in raw.items() if key in known}
    values.update(
        n_layers=4,
        n_experts=n_experts,
        topk=1,
        expert_tsct_rank=rank,
        seq_len=seq,
        batch=batch,
        torch_compile=False,
    )
    return StackSpec.from_mapping(values)


def _site(frames: list[dict[str, Any]]) -> str:
    """Return the innermost frame of ours that asked for a block."""
    for frame in frames:
        name = str(frame.get("filename", ""))
        if ROOT in name and "/tools/" not in name:
            return f"{Path(name).relative_to(ROOT)}:{frame.get('line')}"
    return "torch-internal" if frames else "unknown"


def peak_rows(trace: list[dict[str, Any]]) -> tuple[int, list[tuple[int, str]]]:
    """Replay the allocator trace, returning peak bytes and live blocks at it."""
    live: dict[int, tuple[int, str]] = {}
    total = peak = 0
    at_peak: dict[int, tuple[int, str]] = {}
    for event in trace:
        action = event.get("action")
        if action == "alloc":
            block = (int(event.get("size", 0)), _site(event.get("frames") or []))
            live[int(event["addr"])] = block
            total += block[0]
            if total > peak:
                peak, at_peak = total, dict(live)
        elif action in ("free_completed", "free"):
            block = live.pop(int(event["addr"]), None)
            if block is not None:
                total -= block[0]
    grouped: dict[str, int] = {}
    for _addr, (size, site) in at_peak.items():
        grouped[site] = grouped.get(site, 0) + size
    return peak, sorted(((size, site) for site, size in grouped.items()), reverse=True)


def main() -> int:
    """Print the peak allocation and the blocks alive at it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    built = assemble(spec(args.batch, args.seq), smoke=True, device="cuda")
    optimizer = MixOptimizer(built.model, steps=1)
    ids = torch.randint(0, built.spec.vocab, (args.batch, args.seq + 1), device="cuda")
    for _ in range(2):  # pay JIT before the measured step
        built.step(ids, optimizer=optimizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
    try:
        built.step(ids, optimizer=optimizer)
    finally:
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
        trace = torch.cuda.memory._snapshot()["device_traces"][0]
        torch.cuda.memory._record_memory_history(enabled=None)
    traced, rows = peak_rows(trace)
    print(
        f"b{args.batch} s{args.seq}: peak {peak / 2**20:,.1f} MiB  trace {traced / 2**20:,.1f} MiB"
    )
    for size, site in rows[: args.top]:
        print(f"  {size / 2**20:9.1f} MiB  {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
