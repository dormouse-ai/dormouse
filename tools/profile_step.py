"""One training step, timed on the card, phase by phase, against the card's own peak.

WHAT THIS ANSWERS, AND IN WHAT UNIT. A step's cost is either a matmul or it is not, and
only the first kind can be moved by a faster GEMM. So the step is timed with CUDA events
between six boundaries — data, forward, backward, clip, optimizer, other — and each
phase is priced twice: milliseconds of the step it is, and TFLOP/s it achieved against
`PEAK_TFLOPS`. A phase with no multiply-adds of its own prints `-` rather than a zero
TFLOP/s, because a zero is a number someone averages.

WHERE THE PEAK COMES FROM. 92.0 TFLOP/s is the bf16-with-fp32-accumulate peak this card
was measured at, and it is the denominator every MFU number in this project is read
against. The old engine quoted 47.7 in the same slot, which is 52 % of 92.0 — that
constant was a *sustained* figure wearing a *peak* label, so every MFU computed from it
was 1.93x too high. A peak is a property of the card; a sustained number is a property
of a kernel, and they do not go in the same row. `--peak` exists so a different card can
be priced without editing this file, and `matmul_peak()` measures what this card reaches
so the report can say both.

WHERE THE FLOPs COME FROM. From the modules, not from a constant: `macs_per_token()` is
summed over the blocks and the tied head, so a width that changed moves the number. One
token through the forward is `2·macs` FLOPs, through the backward `4·macs`, so a step is
`6·macs`, which is the same accounting `assemble.plan_data` prices a run with.

WHAT IT REFUSES. The card is shared. A run here takes the same lock `tools/gpu_lock.sh`
takes, and refuses:
  * if another process is already holding that lock;
  * if a foreign process is *computing* on the card (>= `CONTENTION`) — two kernels on
    one SM budget is not a slow measurement, it is a measurement of the contention;
  * if the device's free memory is below what the step's weights, gradients, optimizer
    moments and one activation pass need, counted rather than guessed.
A foreign process that holds memory and is idle (a desktop service) is *printed* and does
not stop the run: it costs headroom, which the third check already prices, and it does
not touch the timing.

THE WIDTHS THE DEFAULT RUNS. The live config is 1B stored; with Muon's momentum and
AdamW's two moments that is ~16 B/parameter, which does not fit a 16 GiB card next to its
activations. So the default narrows `--layers` and `--experts` — the per-token matmul
shapes (`d_model=832`, `d_ffn=6144`, rank-256 experts, `seq_len=1024`) are the recipe's,
so the *ratio* of matmul to non-matmul is the live stack's even though the parameter
count is not. Every number printed is labelled with the widths it was taken at; quote the
full config's MFU from a run that had the memory for it.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.graph import state_bytes
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy

#: bf16 dense-with-fp32-accumulate peak of this card in TFLOP/s, measured. The one
#: number an MFU is a fraction of; see the module docstring for why not 47.7.
PEAK_TFLOPS = 92.0

#: The six boundaries the step is broken at, in the order they happen.
PHASES: tuple[str, ...] = ("data", "forward", "backward", "clip", "optimizer", "other")

#: Median GPU utilization, in percent, at which a foreign process is computing.
#: A desktop compositor repaints in the 1-10 % band and a training kernel holds the SMs
#: above 90 %, so 15 % is above the repaint band and below any real co-tenant.
CONTENTION = 15

#: Utilization samples per check; the median, not the maximum: one repaint must not
#: refuse a run, and one quiet moment must not hide a co-tenant.
SAMPLES = 5

#: The lock `tools/gpu_lock.sh` takes. Same path, so either can hold the card.
LOCK = Path(os.environ.get("DORMOUSE_GPU_LOCK", "/tmp/opencode/dormouse-gpu.lock"))

MIB = 2**20

#: The config the report is about; `--config` for another.
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dormouse_1b.toml"


@dataclass(frozen=True)
class PhaseTime:
    """One phase: how long it took, and what it moved while it did."""

    name: str
    ms: float
    flops: float

    def tflops(self, ms: float) -> float | None:
        """Return the phase's achieved TFLOP/s, or None when it has no multiply-adds."""
        if self.flops <= 0.0 or ms <= 0.0:
            return None
        return self.flops / (ms * 1e-3) / 1e12


def foreign() -> list[tuple[int, str, int]]:
    """Return `(pid, name, MiB)` for every compute app on the card that is not us."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"nvidia-smi is not answering ({exc}): no card, no profile") from exc
    apps = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, name, mib = (part.strip() for part in line.rsplit(",", 2))
        if int(pid) != os.getpid():
            apps.append((int(pid), name, int(mib)))
    return apps


def utilization(samples: int = SAMPLES) -> int:
    """Return the median of `samples` utilization readings, in percent."""
    readings = []
    for _ in range(samples):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        readings.append(int(out.strip().splitlines()[0]))
        time.sleep(0.05)
    return int(statistics.median(readings))


def hold_card() -> IO[bytes] | None:
    """Take the GPU lock, or refuse naming what holds it.

    Returns None when `tools/gpu_lock.sh` already holds it for us: an flock is per open
    file description, so a child that takes the same lock blocks on its own parent.
    """
    if os.environ.get("DORMOUSE_GPU_LOCK_HELD"):
        return None
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        holder = LOCK.with_suffix(".lock.holder")
        who = holder.read_text().strip() if holder.exists() else "unknown holder"
        raise SystemExit(
            f"the card is held: {who} (lock {LOCK}); wait for it, or run under "
            "tools/gpu_lock.sh --wait"
        ) from None
    return handle


def matmul_peak(size: int = 8192) -> float:
    """Return the TFLOP/s an `size`-cubed bf16 matmul reaches, as a ceiling check."""
    if not torch.cuda.is_available():
        return 0.0
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        a @ b
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(3):
        start = time.perf_counter()
        a @ b
        torch.cuda.synchronize()
        best = max(best, 2 * size**3 / (time.perf_counter() - start) / 1e12)
    return best


def activation_floor(spec: StackSpec) -> int:
    """Return a floor on the bytes one step's activations need, from the widths.

    A floor, not a peak: it counts the four buffers a step cannot avoid holding at once
    — the `[batch, seq, vocab]` logits, the attention arms' `[batch, heads, seq, seq]`
    scores, and the FFN's two `[batch, seq, d_ffn]` halves — in fp32. The real peak is a
    multiple of this, which is why the refusal below keeps a margin on top of it; the
    point of the floor is that a run that cannot hold even these is refused by a number
    instead of dying inside KDA's chunkwise solve.
    """
    logits = spec.batch * spec.seq_len * spec.vocab * 4
    scores = spec.batch * spec.n_heads * spec.seq_len * spec.seq_len * 4
    ffn = 2 * spec.batch * spec.seq_len * spec.d_ffn * 4
    return logits + scores + ffn


def spec_for(
    config: Path, *, attn: str, layers: int, experts: int, batch: int, seq_len: int
) -> StackSpec:
    """Load the live config and narrow it to widths the card can hold."""
    base = StackSpec.from_toml(config)
    return dataclasses.replace(
        base, attn=attn, n_layers=layers, n_experts=experts, batch=batch, seq_len=seq_len
    )


def _events(count: int) -> list[torch.cuda.Event]:
    """Return `count` fresh CUDA events, timing-enabled."""
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def measure(
    model: torch.nn.Module,
    ids_cpu: torch.Tensor,
    ids_gpu: torch.Tensor,
    optimizer: MixOptimizer,
    *,
    steps: int,
    warmup: int,
    clip: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return per-iteration ms per phase and per-iteration wall ms of the step."""
    times: dict[str, list[float]] = {name: [] for name in PHASES}
    walls: list[float] = []
    inputs, targets = ids_gpu[:, :-1], ids_gpu[:, 1:]
    for iteration in range(warmup + steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        e = _events(6)
        started = time.perf_counter()
        e[0].record()
        ids_gpu.copy_(ids_cpu, non_blocking=True)
        e[1].record()
        logits = model(inputs, None)
        loss = chunked_cross_entropy(logits, targets)
        e[2].record()
        loss.backward()  # type: ignore[no-untyped-call]
        e[3].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        e[4].record()
        optimizer.step()
        e[5].record()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - started)
        if iteration < warmup:
            continue
        spans = [e[index].elapsed_time(e[index + 1]) for index in range(5)]
        spans.append(max(walls[-1] * 1e3 - sum(spans), 0.0))
        for name, value in zip(PHASES, spans, strict=True):
            times[name].append(value)
    return times, {"wall": walls[warmup:]}


def kernel_count(
    model: torch.nn.Module,
    ids_gpu: torch.Tensor,
    optimizer: MixOptimizer,
    *,
    clip: float,
) -> tuple[int, int]:
    """Return `(launches, distinct)` for one profiled step."""
    inputs, targets = ids_gpu[:, :-1], ids_gpu[:, 1:]
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        logits = model(inputs, None)
        loss = chunked_cross_entropy(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    return len(names), len(set(names))


def phase_rows(times: dict[str, list[float]], tokens: int, macs_per_token: int) -> list[PhaseTime]:
    """Turn the measured spans into phases with their own FLOP bills."""
    flops = {
        "data": 0.0,
        "forward": 2.0 * macs_per_token * tokens,
        "backward": 4.0 * macs_per_token * tokens,
        "clip": 0.0,
        "optimizer": 0.0,
        "other": 0.0,
    }
    return [
        PhaseTime(name=name, ms=statistics.median(values), flops=flops[name])
        for name in PHASES
        if (values := times[name])
    ]


def render(
    rows: list[PhaseTime],
    *,
    wall_ms: float,
    tokens: int,
    macs_per_token: int,
    peak: float,
    measured_peak: float,
    launches: int,
    distinct: int,
    reserved: float,
    allocated: float,
    state: float,
    free: float,
) -> None:
    """Print the table, the MFU line and the counters the next decision needs."""
    step = sum(row.ms for row in rows)
    print()
    print(f"{'phase':<12}{'ms':>9}{'% step':>9}{'TFLOP/s':>10}  {'share of the bill':<20}")
    print("-" * 64)
    for row in rows:
        tflops = row.tflops(row.ms)
        share = f"{row.flops / (6.0 * macs_per_token * tokens):5.1%}" if row.flops else "-"
        print(
            f"{row.name:<12}{row.ms:>9.2f}{row.ms / step:>8.1%}"
            f"{'  -' if tflops is None else f'{tflops:>10.2f}'}  {share:<20}"
        )
    print("-" * 64)
    flops = 6.0 * macs_per_token * tokens
    achieved = flops / (step * 1e-3) / 1e12
    print(
        f"step {step:.2f} ms (wall {wall_ms:.2f} ms), {tokens:,} tokens, "
        f"{flops / 1e12:.3f} TFLOP/step"
    )
    print(
        f"MFU {achieved / peak:6.1%} of {peak:.1f} TFLOP/s peak "
        f"({achieved:.2f} TFLOP/s achieved); an {8192}-cubed bf16 matmul reaches "
        f"{measured_peak:.1f} TFLOP/s here"
    )
    print(
        f"kernels {launches:,} launches/step, {distinct:,} distinct; "
        f"peak VRAM {allocated:.0f} MiB allocated / {reserved:.0f} MiB reserved"
    )
    print(
        f"state {state:.0f} MiB (weights+grads+moments), free {free:.0f} MiB, "
        f"FLOP/token {2 * macs_per_token / 1e9:.3f} G forward "
        f"({6 * macs_per_token / 1e9:.3f} G per step) from the module bill"
    )
    print(
        "'other' is everything the events do not cover: zero_grad, the loss' host read "
        "and launch gaps between phases"
    )


def main(argv: list[str] | None = None) -> int:
    """Measure one step on the card and print the breakdown."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--attn", default="kda", choices=["hybrid", "kda", "dsa"], help="attention arm"
    )
    parser.add_argument("--layers", type=int, default=2, help="n_layers (4k for hybrid)")
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--peak", type=float, default=PEAK_TFLOPS)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup non-negative")
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device in this process: the profile is CUDA-event timing of a step "
            "on the card, and there is nothing to measure without one"
        )

    lock = hold_card()
    try:
        apps = foreign()
        busy = utilization()
        if busy >= CONTENTION:
            raise SystemExit(
                f"the card is computing for someone else ({busy}% utilisation); the "
                "timings would be a measurement of the contention"
            )
        if apps:
            print("foreign compute apps on the card (memory only, not timed):")
            for pid, name, mib in apps:
                print(f"  {pid:>8} {name} {mib} MiB")

        spec = spec_for(
            args.config,
            attn=args.attn,
            layers=args.layers,
            experts=args.experts,
            batch=args.batch,
            seq_len=args.seq_len,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = assemble(spec, smoke=True).model.cuda()
        optimizer = MixOptimizer(model)
        tokens = spec.batch * spec.seq_len
        state = state_bytes(model, optimizer) / MIB
        free, _total = torch.cuda.mem_get_info()
        need = (state_bytes(model, optimizer) + activation_floor(spec)) / MIB
        if free / MIB < need:
            raise SystemExit(
                f"free {free / MIB:.0f} MiB is below the {need:.0f} MiB this step needs: "
                f"{state:.0f} MiB of weights, gradients and moments plus a counted floor "
                "on the activations; narrow --batch/--seq-len/--experts, or free the card"
            )
        macs = model.macs_per_token(spec.seq_len)  # type: ignore[attr-defined]
        ids_cpu = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), pin_memory=True)
        ids_gpu = ids_cpu.to("cuda")
        print(
            f"profile: d_model={spec.d_model} d_ffn={spec.d_ffn} n_layers={spec.n_layers} "
            f"experts={spec.n_experts} topk={spec.topk} seq={spec.seq_len} "
            f"batch={spec.batch} -> {sum(p.numel() for p in model.parameters()):,} params"
        )

        torch.cuda.reset_peak_memory_stats()
        try:
            times, walls = measure(
                model,
                ids_cpu,
                ids_gpu,
                optimizer,
                steps=args.steps,
                warmup=args.warmup,
                clip=args.clip,
            )
        except torch.OutOfMemoryError as exc:
            raise SystemExit(
                f"the step does not fit: {exc}. The widths above are the ones asked for; "
                "narrow --batch/--layers/--experts until it does — a profiled step that "
                "OOMs has no breakdown to report"
            ) from exc
        launches, distinct = kernel_count(model, ids_gpu, optimizer, clip=args.clip)
        rows = phase_rows(times, tokens, macs["total"])
        render(
            rows,
            wall_ms=statistics.median(walls["wall"]) * 1e3,
            tokens=tokens,
            macs_per_token=macs["total"],
            peak=args.peak,
            measured_peak=matmul_peak(),
            launches=launches,
            distinct=distinct,
            reserved=torch.cuda.max_memory_reserved() / MIB,
            allocated=torch.cuda.max_memory_allocated() / MIB,
            state=state,
            free=free / MIB,
        )
        print(
            "blocks "
            f"{macs['blocks'] / 1e6:.1f} M MAC/token, head {macs['head'] / 1e6:.1f} M, "
            f"total {macs['total'] / 1e6:.1f} M"
        )
        return 0
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
