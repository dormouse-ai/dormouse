"""One real step, and the blocks that were alive at its peak — by name.

`tools/preflight.py` answers "which batch fits". This answers "what is inside it":
the step is run once under `torch.cuda.memory._record_memory_history`, the alloc/free
trace is replayed, and every block that was LIVE at the peak instant is reported with
its size and the line of this repository that asked for it.

    flock /tmp/dormouse-gpu.lock uv run --no-sync python tools/alloc_profile.py \
        configs/dormouse_1b.toml --batch 1 --seq 1024 --bf16

The two numbers a fix is judged by:

* `peak`      — `torch.cuda.max_memory_allocated`, the ladder's own number;
* `retained`  — bytes still live after `backward()` (parameters, grads, moments; and
  anything the graph did not release), i.e. what the NEXT step starts from.

The composition at the peak is what names the tensor: a gather, a `cat`, an `einsum`
output. An OOM is a ROW here too: the trace up to the refusal still holds the peak.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

#: Frames are reported from this repository only: torch's own internals are the same
#: frame for every allocation and name nothing.
ROOT = str(Path(__file__).resolve().parents[1])


def _args() -> tuple[str, int, int, bool, int]:
    """Return (config, batch, seq, bf16, top) from the command line."""
    argv = sys.argv[1:]
    config = argv[0] if argv and not argv[0].startswith("--") else "configs/dormouse_1b.toml"

    def flag(name: str, default: int) -> int:
        return int(argv[argv.index(name) + 1]) if name in argv else default

    return config, flag("--batch", 1), flag("--seq", 0), "--bf16" in argv, flag("--top", 12)


def _site(frames: list[dict[str, Any]]) -> str:
    """Return the innermost frame of ours that requested the block, as `file:line`."""
    for frame in frames:
        name = str(frame.get("filename", ""))
        if ROOT in name and "/tools/" not in name:
            return f"{Path(name).relative_to(ROOT)}:{frame.get('line')}"
    return "torch-internal" if frames else "unknown"


def _peak_rows(trace: list[dict[str, Any]]) -> tuple[int, list[tuple[int, str]]]:
    """Replay the trace; return (peak bytes, [(bytes, site)] live at that instant)."""
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


def _largest(trace: list[dict[str, Any]], top: int) -> list[tuple[int, str, int]]:
    """Return the biggest SINGLE blocks alive at the peak: (bytes, site, frames captured)."""
    live: dict[int, tuple[int, str, int]] = {}
    total = peak = 0
    at_peak: dict[int, tuple[int, str, int]] = {}
    for event in trace:
        action = event.get("action")
        if action == "alloc":
            frames = event.get("frames") or []
            block = (int(event.get("size", 0)), _site(frames), len(frames))
            live[int(event["addr"])] = block
            total += block[0]
            if total > peak:
                peak, at_peak = total, dict(live)
        elif action in ("free_completed", "free"):
            block = live.pop(int(event["addr"]), None)
            if block is not None:
                total -= block[0]
    return sorted(at_peak.values(), reverse=True)[:top]


def main() -> int:
    """Run one step under the allocator's own trace and print what the peak was made of."""
    config, batch, seq_override, bf16, top = _args()
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    if seq_override:
        spec = replace(spec, seq_len=seq_override)
    seq = spec.seq_len
    assembled = assemble(spec, device="cuda")
    optimizer = MixOptimizer(assembled.model)
    report = assembled.report()
    print(
        f"params stored={report['stored']:,} seq={seq} batch={batch} "
        f"dtype={'bf16 autocast' if bf16 else 'fp32'}"
    )
    ids = torch.randint(0, spec.vocab, (batch, seq + 1), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
    refused = ""
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            assembled.step(ids, optimizer=optimizer)
    except torch.OutOfMemoryError as exc:
        refused = next(
            (line.strip() for line in str(exc).splitlines() if "Tried to allocate" in line),
            "out of memory",
        )
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    retained = int(torch.cuda.memory_allocated())
    trace = torch.cuda.memory._snapshot()["device_traces"][0]
    torch.cuda.memory._record_memory_history(enabled=None)
    traced, rows = _peak_rows(trace)
    print(
        f"{'OOM  ' if refused else 'ok   '}peak {peak / 2**20:,.0f} MiB  "
        f"retained {retained / 2**20:,.0f} MiB  (before the step {before / 2**20:,.0f} MiB)"
        + (f"  {refused}" if refused else "")
    )
    print(f"trace peak {traced / 2**20:,.0f} MiB; the blocks alive at it:")
    for size, site in rows[:top]:
        print(f"  {size / 2**20:9.1f} MiB  {site}")
    print("the biggest single blocks:")
    for size, site, frames in _largest(trace, 6):
        print(f"  {size / 2**20:9.1f} MiB  {site}  (frames={frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
