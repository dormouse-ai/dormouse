"""Forward + backward only profile of the 404M dense stack (no optimizer noise).

`tools/profile_step.py` prices the WHOLE step; its top-kernel table is dominated by
Muon's Newton-Schulz chain, which hides the forward/backward kernels this task owns.
This tool times only forward -> CE -> backward, so the phase table and the profiler
table attribute directly to KDA / DSA / CE / norm / FFN.

    python fcntl_lock.py PYTHONPATH=... python tools/profile_fwdbwd.py --batch 2

Numbers are CUDA-event GPU ms (median over --steps), one warm step before timing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.step import chunked_cross_entropy


class Events:
    """A named pair of CUDA events, elapsed on demand."""

    def __init__(self) -> None:
        """Allocate two events."""
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> Events:
        """Record start."""
        self.start.record()
        return self

    def __exit__(self, *exc: object) -> None:
        """Record end."""
        self.end.record()

    def ms(self) -> float:
        """Elapsed GPU ms."""
        return float(self.start.elapsed_time(self.end))


def run_fwdbwd(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    amp: bool,
    timeit: bool = True,
):
    """One forward+CE+backward; return per-phase GPU ms and the loss."""
    marks = {name: Events() for name in ("forward", "ce", "backward")}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        with marks["forward"]:
            logits = model(inputs)
        with marks["ce"]:
            loss = chunked_cross_entropy(logits, targets)
        with marks["backward"]:
            loss.backward()
    if not timeit:
        return {}, loss.detach()
    torch.cuda.synchronize()
    return {name: e.ms() for name, e in marks.items()}, loss.detach()


def main() -> int:
    """Build the stack, print fwd/bwd phases and the self-device-time kernel table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_500m_dense_2b.toml")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    amp = args.dtype == "bf16"

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=args.config)
    spec = replace(spec, batch=args.batch, seq_len=args.seq or spec.seq_len)
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    bill = assembled.report()
    print(
        f"stack: {spec.n_layers}L d={spec.d_model} f={spec.d_ffn} attn={spec.attn} "
        f"ffn={spec.ffn} active={bill['active']:,} b{spec.batch}xs{spec.seq_len} {args.dtype}",
        flush=True,
    )
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda")
    inputs, targets = ids[:, :-1], ids[:, 1:]

    for _ in range(2):
        model.zero_grad(set_to_none=True)
        run_fwdbwd(model, inputs, targets, amp=amp)
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {k: [] for k in ("forward", "ce", "backward")}
    totals: list[float] = []
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        marks, loss = run_fwdbwd(model, inputs, targets, amp=amp)
        torch.cuda.synchronize()
        for k, v in marks.items():
            samples[k].append(v)
        totals.append(sum(marks.values()))
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss")
    print(f"\nphases, GPU ms (median of {args.steps}):")
    for k in ("forward", "ce", "backward"):
        print(f"  {k:10} {statistics.median(samples[k]):9.2f}")
    print(
        f"  {'fwd+ce+bwd':10} {statistics.median(totals):9.2f}  (sum of medians "
        f"{sum(statistics.median(samples[k]) for k in samples):.2f})"
    )
    print(f"  peak VRAM {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB")

    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        run_fwdbwd(model, inputs, targets, amp=amp, timeit=False)
    torch.cuda.synchronize()
    by_time = sorted(
        ((e.self_device_time_total / 1e3, e.count, str(e.key)) for e in prof.key_averages()),
        reverse=True,
    )[: args.top]
    print(f"\ntop {args.top} by self device time (fwd+ce+bwd only):")
    print(f"  {'ms':>8} {'count':>7}  op")
    for ms, count, name in by_time:
        print(f"  {ms:8.2f} {count:7d}  {name[:110]}")
    by_count = sorted(
        ((e.count, e.self_device_time_total / 1e3, str(e.key)) for e in prof.key_averages()),
        reverse=True,
    )[:20]
    print("\ntop 20 by CPU op count:")
    print(f"  {'count':>8} {'dev ms':>8}  op")
    for count, ms, name in by_count:
        print(f"  {count:8d} {ms:8.2f}  {name[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
