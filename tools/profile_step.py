"""Where a step's 8.7 s goes: the phase table, the op table, and the module bill.

WHY THIS FILE EXISTS. The bank's fused chain is 99 ms of fwd+bwd for 16 384 rows, under
2 % of the step, yet the step is still seconds long. Nobody has profiled the whole step
since the bank was rewritten, and "the bank is fast now" is not a measurement of the step.
So this prints three tables from ONE run of the real stack at the live shape:

* **phases**, from CUDA events inside the step's own statements: forward, chunked CE,
  backward, clip, Muon, AdamW — and `other`, the part of the wall clock no phase claimed;
* **ops**, `torch.profiler`'s top 15 by self device time and by CPU op count, i.e. what
  the GPU actually ran and how many times the host asked;
* **modules**, each component built at its live shape and timed alone, so the forward is
  broken down by embedding / attention type / bank / norm / head rather than by op name.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/profile_step.py
    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/profile_step.py --batch 8

Phase timings are events recorded in-stream, so a phase's time is GPU time between its
markers; `other` is therefore honest CPU/launch headroom, not hidden kernel work.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy

#: The card's practical dense bf16 peak, in TFLOP/s. The 350 kB/s projection is written
#: against this number and it is quoted here so the MFU printed below can be checked.
PRACTICAL_PEAK_TFLOPS = 52.0


def build(config: str, batch: int, seq: int | None) -> tuple[StackSpec, object, MixOptimizer]:
    """Read the config, override the batch, and build the stack the run would build."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    spec = replace(spec, batch=batch, seq_len=seq or spec.seq_len)
    assembled = assemble(spec, device="cuda")
    return spec, assembled, MixOptimizer(assembled.model)


class _Call(torch.nn.Module):
    """A callable wrapped as a module, so `module_ms` can zero its grads."""

    def __init__(self, fn: object) -> None:
        """Hold the callable."""
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call it."""
        return self.fn(x)  # type: ignore[operator]


class Events:
    """A named pair of CUDA events, elapsed on demand."""

    def __init__(self, name: str) -> None:
        """Allocate the two events."""
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> Events:
        """Record the start marker in the stream."""
        self.start.record()
        return self

    def __exit__(self, *exc: object) -> None:
        """Record the end marker in the stream."""
        self.end.record()

    def ms(self) -> float:
        """GPU milliseconds between the markers."""
        return float(self.start.elapsed_time(self.end))


def phases(
    assembled: object, optimizer: MixOptimizer, ids: torch.Tensor, steps: int, amp: bool
) -> dict[str, float]:
    """Time the step's own statements, in the step's order, and return ms per step."""
    model = assembled.model
    inputs, targets = ids[:, :-1], ids[:, 1:]
    names = ("forward", "ce", "backward", "clip", "muon", "adamw")
    marks = {name: Events(name) for name in names}
    total = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            with marks["forward"]:
                logits = model(inputs)
            with marks["ce"]:
                loss = chunked_cross_entropy(logits, targets)
            with marks["backward"]:
                loss.backward()
        with marks["clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with marks["muon"]:
            optimizer.muon.step()
        with marks["adamw"]:
            optimizer.adamw.step()
        torch.cuda.synchronize()
        total.append((time.perf_counter() - started) * 1e3)
        if not torch.isfinite(loss):
            raise SystemExit("the loss is not finite; the profile would measure a broken step")
    return {**{name: marks[name].ms() for name in names}, "total": sum(total) / steps}


def module_ms(fn: torch.nn.Module, x: torch.Tensor, *, backward: bool, amp: bool) -> float:
    """Median of three device-timed calls of `fn(x)` (plus a backward when asked)."""
    runs = []
    for _ in range(3):
        x = x.detach().clone().requires_grad_(backward)
        fn.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        event = Events("module")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp), event:
            out = fn(x)
            if backward:
                out.float().square().mean().backward()
        torch.cuda.synchronize()
        runs.append(event.ms())
    return sorted(runs)[1]


def modules(
    assembled: object, spec: StackSpec, ids: torch.Tensor, amp: bool
) -> list[tuple[str, int, float, float]]:
    """Time each component at its live shape: (name, count, ms fwd, ms fwd+bwd)."""
    model = assembled.model
    x = torch.randn(spec.batch, spec.seq_len, spec.d_model, device="cuda", dtype=torch.float32)
    kinds = [type(block.attn).__name__ for block in model.blocks]
    rows = []
    rows.append(("embedding", 1, module_ms(model.embed, ids[:, :-1], backward=False, amp=amp), 0.0))
    per_kind: dict[str, torch.nn.Module] = {}
    for kind, block in zip(kinds, model.blocks, strict=True):
        per_kind.setdefault(kind, block.attn)
    for kind, layer in per_kind.items():
        rows.append(
            (
                f"attn {kind}",
                kinds.count(kind),
                module_ms(layer, x, backward=False, amp=amp),
                module_ms(layer, x, backward=True, amp=amp),
            )
        )
    bank = model.blocks[0].ffn
    rows.append(
        (
            "ffn bank",
            len(model.blocks),
            module_ms(bank, x, backward=False, amp=amp),
            module_ms(bank, x, backward=True, amp=amp),
        )
    )
    norms = 2 * len(model.blocks) + 1
    rows.append(
        (
            "rmsnorm (12x2 + final)",
            norms,
            module_ms(model.norm, x, backward=False, amp=amp),
            module_ms(model.norm, x, backward=True, amp=amp),
        )
    )
    rows.append(
        (
            "head (tied)",
            1,
            module_ms(_Call(model.head), x, backward=False, amp=amp),
            module_ms(_Call(model.head), x, backward=True, amp=amp),
        )
    )
    return rows


def op_tables(
    steps: int, run
) -> tuple[list[tuple[float, int, str]], list[tuple[int, float, str]], int]:
    """Profile `run` for `steps` and return (top by device ms, top by op count, launches)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        run()
    torch.cuda.synchronize()
    by_time = sorted(
        ((e.self_device_time_total / 1e3, e.count, str(e.key)) for e in profiler.key_averages()),
        reverse=True,
    )[:15]
    by_count = sorted(
        ((e.count, e.self_device_time_total / 1e3, str(e.key)) for e in profiler.key_averages()),
        reverse=True,
    )[:15]
    launches = sum(1 for e in profiler.events() if e.name == "cudaLaunchKernel")
    return by_time, by_count, launches // steps


def mfu(tokens_per_s: float, active: int) -> float:
    """Model FLOPs utilisation from `6·N·tokens/s` against the practical peak."""
    return 6.0 * active * tokens_per_s / (PRACTICAL_PEAK_TFLOPS * 1e12)


def main() -> int:
    """Build the stack, print the step's phase table, op table and module bill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help="learning-rate fraction held for every step; 0 freezes the weights",
    )
    args = parser.parse_args()
    amp = args.dtype == "bf16"
    spec, assembled, optimizer = build(args.config, args.batch, args.seq or None)
    #: Weights are frozen by default, and it is not a shortcut: with random token ids the
    #: router concentrates past the dispatch capacity after two Muon steps, and the
    #: dropless second pass (`moe.py: _passes`) would then be timed rather than the step
    #: the profile is about. A frozen step runs the same kernels on the same shapes; only
    #: the values stop moving.
    optimizer.set_lr(args.lr)
    bill = assembled.report()
    tokens_per_step = spec.batch * spec.seq_len
    print(
        f"stack: 12 layers, {spec.d_model}/{spec.d_ffn}, E={spec.n_experts} top-{spec.topk}, "
        f"active={bill['active']:,} stored={bill['stored']:,}, "
        f"b{spec.batch} x s{spec.seq_len} = {tokens_per_step:,} tokens, {args.dtype}"
    )
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda")

    for _ in range(2):
        phases(assembled, optimizer, ids, 1, amp)
    torch.cuda.synchronize()

    table = phases(assembled, optimizer, ids, args.steps, amp)
    per_step_ms = table["total"]
    rate = tokens_per_step / (per_step_ms * 1e-3)
    print(
        f"\npeak VRAM over the timed steps: {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB "
        f"of {torch.cuda.get_device_properties(0).total_memory / 2**20:,.0f} MiB, "
        f"{torch.cuda.max_memory_reserved() / 2**20:,.0f} MiB reserved"
    )
    print(f"\nphases, GPU ms per step ({args.steps} steps, CUDA events):")
    print(f"  {'phase':10} {'ms':>9} {'% of step':>10}")
    accounted = 0.0
    for name in ("forward", "ce", "backward", "clip", "muon", "adamw"):
        accounted += table[name]
        print(f"  {name:10} {table[name]:9.1f} {100 * table[name] / per_step_ms:9.1f}%")
    print(
        f"  {'other':10} {table['total'] - accounted:9.1f} "
        f"{100 * (table['total'] - accounted) / per_step_ms:9.1f}%  (host, launch, sync)"
    )
    print(
        f"  {'total':10} {per_step_ms:9.1f}  {rate:,.0f} tok/s  {rate / 1000:,.1f} kB/s  "
        f"MFU {100 * mfu(rate, bill['active']):.1f}%"
    )

    print("\nmodules, device ms at the live shape (fwd / fwd+bwd, one unit each):")
    print(f"  {'component':24} {'x':>3} {'fwd':>8} {'fwd+bwd':>9} {'step ms':>9}")
    fwd_sum = bwd_sum = 0.0
    for name, count, fwd, both in modules(assembled, spec, ids, amp):
        fwd_sum += count * fwd
        bwd_sum += count * both
        print(f"  {name:24} {count:3d} {fwd:8.1f} {both:9.1f} {count * both:9.1f}")
    print(f"  {'sum of components':24} {'':3} {fwd_sum:8.1f} {bwd_sum:9.1f}")

    def one_run() -> None:
        phases(assembled, optimizer, ids, 1, amp)

    by_time, by_count, launches = op_tables(args.steps, one_run)
    print(f"\ntop 15 by self device time (whole step, {launches:,} kernel launches/step):")
    print(f"  {'ms':>8} {'count':>7}  op")
    for ms, count, name in by_time:
        print(f"  {ms:8.1f} {count:7d}  {name}")
    print("\ntop 15 by CPU op count:")
    print(f"  {'count':>8} {'dev ms':>8}  op")
    for count, ms, name in by_count:
        print(f"  {count:8d} {ms:8.1f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
