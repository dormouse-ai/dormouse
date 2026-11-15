"""What the fp4 rung costs and buys, measured — and the one thing it does not prove.

Three instruments, each named next to its number:

* MEMORY — the bytes an activation of the live shape (`24 x 1024 x 832`) occupies, and
  the process RSS high-water for the matmul that reads it, per rung. Peak RSS is
  measured in an isolated child per rung, so one rung's high-water cannot leak into
  another's. This is the number fp4 is actually for: it buys memory, not speed.
* TIME — one `[tokens, 832] @ [6144, 832]ᵀ` forward per rung, on `--device`. The
  default is the CPU: `card_state()` prints what the card is doing and whether this
  wheel can pack the format at all, and a CPU number is labelled a CPU number rather
  than reported as a GPU one. On the card, run it through `scripts/gpu_lock.sh`.
* LOSS — `--steps` steps of fp32 training on `tests/fixtures/corpus_2mb.bin` with
  `weight_qat`/`act_quant` at `none`, `ternary` and `fp4`, identical seeds. This is a
  SMOKE, not a verdict: a 200-step run of a 2M-parameter arm on 2 MB of text cannot
  rank two quantizers, and the line that reports it says so.

    python scripts/bench_fp4.py                    # defaults: 200 steps, b24 x s1024
    python scripts/bench_fp4.py --steps 50 --time-reps 1
    python scripts/bench_fp4.py --skip-loss        # memory + time only
    GPU_TIMEOUT=300 bash scripts/gpu_lock.sh python scripts/bench_fp4.py --device cuda

The measured facts this script exists to keep honest (they are why the rung is
forward-only today): on sm_120 `torch._scaled_mm` takes NVFP4 operands in the forward
and has no backward at all, so a TRAINING run is QAT — the reference fake-quant with a
straight-through estimator — and `fp4.tally()` will report `fp4_scaled_mm=0` for every
step of it. A number that claims otherwise would be a claim about a kernel nobody ran.
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from dormouse.model import fp4
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "corpus_2mb.bin"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

#: The one architecture at widths a CPU can train for 200 steps; every slot is kept.
SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "n_layers": 4,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 32,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}

RUNGS: tuple[dict[str, str], ...] = (
    {"weight_qat": "none", "act_quant": "none"},
    {"weight_qat": "ternary", "act_quant": "int4"},
    {"weight_qat": "fp4", "act_quant": "fp4"},
)


def rss_mib() -> float:
    """Process RSS high-water mark in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def card_state() -> str:
    """Say what the card is doing and what this wheel can do with it."""
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        used = "unreadable"
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    ).stdout.strip()
    verdict = fp4.verdict()
    return (
        f"card: {used} MiB in use; compute apps:\n{apps}\n"
        f"wheel: torch {torch.__version__}, cuda available={torch.cuda.is_available()}\n"
        f"fp4 kernel on {('cuda' if torch.cuda.is_available() else 'cpu')}: "
        f"{verdict or 'available'}"
    )


def memory(tokens: int, width: int) -> None:
    """Bytes per rung at the live activation shape, and the peak of reading it."""
    print(f"\n== memory: activations [{tokens}, {width}] ==")
    values = tokens * width
    rows = {
        "fp32": values * 4,
        "bf16": values * 2,
        "fp4": values // 2 + values // fp4.BLOCK,
    }
    for name, size in rows.items():
        print(f"  {name:5s} {size / 2**20:9.2f} MiB  ({size / values:.2f} bytes/value)")
    print("  fp4 is 4 bits/value (measured as numel/2) plus one e4m3 scale per 16")
    print("  (measured as numel/16); the packed dtype cannot be allocated on this build.")
    if torch.cuda.is_available():
        print("\n  peak allocation on the card (torch.cuda.max_memory_allocated):")
        device = torch.device("cuda")
        weight = torch.randn(width * 2, width)
        for label, body in (
            ("bf16", lambda x: x.to(torch.bfloat16) @ weight.to(device, torch.bfloat16).T),
            ("fp4 ", lambda x: fp4.linear(x, weight.to(device))),
        ):
            with torch.no_grad():
                x = torch.zeros(tokens, width, device=device)
                torch.cuda.reset_peak_memory_stats()
                body(x)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**20
                del x
                torch.cuda.empty_cache()
            print(f"    {label} peak {peak:9.1f} MiB (device total for the call)")
        return

    print("\n  peak RSS of one matmul, isolated child per rung:")
    baseline = _child_peak(f"import torch; x = torch.empty({tokens}, {width})")[0]
    for label, body in (
        ("bf16", "y = x.to(torch.bfloat16) @ w.to(torch.bfloat16).T"),
        ("fp4 ", "from dormouse.model import fp4\n    y = fp4.linear(x, w)"),
    ):
        peak = _child_peak(
            f"import torch\n"
            f"x = torch.empty({tokens}, {width})\n"
            f"w = torch.empty({width * 2}, {width})\n"
            f"def run():\n"
            f"    {body}\n"
            f"    return y\n"
            f"run()"
        )[0]
        print(f"    {label} peak {peak:8.1f} MiB, +{peak - baseline:8.1f} MiB over baseline")


def _child_peak(code: str) -> tuple[float, int]:
    """Run `code` in a child interpreter; return its RSS high-water and exit status."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import resource\n{code}\n"
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if done.returncode != 0:
        print(f"    (child failed: {done.stderr.strip().splitlines()[-1:]})")
        return (0.0, done.returncode)
    return (float(done.stdout.strip().splitlines()[-1]), 0)


def timing(tokens: int, reps: int, device: str) -> None:
    """One linear forward per rung at the live shape, on `device`, with the verdict."""
    width, ffn = 832, 6144
    where = torch.device(device)
    print(f"\n== time: [{tokens}, {width}] @ [{ffn}, {width}]ᵀ, {reps} reps, {device} ==")
    generator = torch.Generator(device=where)
    generator.manual_seed(0)
    x = torch.randn(tokens, width, generator=generator, device=where)
    weight = torch.randn(ffn, width, generator=generator, device=where)
    with torch.no_grad():
        quantized = fp4.quantize(weight)
        scale = weight.abs().mean().clamp_min(1e-8)
        ternary = ((weight / scale).round().clamp(-1, 1) * scale).contiguous()
        for label, forward in (
            ("none   ", lambda: functional.linear(x, weight)),
            ("ternary", lambda: functional.linear(x, ternary)),
            ("fp4    ", lambda: fp4.linear(x, weight)),
            (
                "fp4 grid, bf16 matmul",
                lambda: functional.linear(x.to(torch.bfloat16), quantized.to(torch.bfloat16)),
            ),
        ):
            _sync(where)
            forward()
            _sync(where)
            start = time.perf_counter()
            for _ in range(reps):
                forward()
            _sync(where)
            elapsed = (time.perf_counter() - start) / reps
            gflop = 2 * tokens * ffn * width / 1e9
            print(f"  {label:22s} {elapsed * 1e3:9.2f} ms  {gflop / elapsed:8.1f} GFLOP/s")
    print(f"  paths after the timing block: {fp4.tally(where)}")
    print(
        "  (the bf16 row is NOT an fp4 kernel: it dequantises to bf16 and uses tensor"
        " cores, so its speed is bf16's, and its memory is the grid's)"
    )


def _sync(device: torch.device) -> None:
    """Wait for the device when it is an accelerator, so a timing is a timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def loss(steps: int) -> None:
    """Identical-seed training per rung on the 2 MB fixture; a smoke, not a verdict."""
    text = torch.frombuffer(bytearray(FIXTURE.read_bytes()[:2_000_000]), dtype=torch.uint8)
    base = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    spec = _replace(base, **SMOKE)
    print(f"\n== loss smoke: {steps} fp32 CPU steps on {FIXTURE.name}, identical seeds ==")
    print("   A 200-step run of a 2M-parameter arm on 2 MB cannot rank two quantizers.")
    for rung in RUNGS:
        torch.manual_seed(0)
        stack: Any = assemble(_replace(spec, **rung), smoke=True)
        optimizer = MixOptimizer(stack.model, steps=3, lr_muon=0.02, lr_adamw=0.02)
        torch.manual_seed(1)
        window = steps * spec.batch * spec.seq_len
        offset = int(torch.randint(0, text.numel() - window - 1, (1,)))
        data = text[offset : offset + window + spec.seq_len].long().reshape(spec.batch, -1)
        losses, start = [], time.perf_counter()
        for step in range(steps):
            ids = data[:, step * spec.seq_len : (step + 1) * spec.seq_len + 1]
            result, _ = stack.step(ids, optimizer=optimizer)
            losses.append(result.loss)
        tail = sum(losses[-20:]) / len(losses[-20:])
        head = sum(losses[:20]) / len(losses[:20])
        print(
            f"  {rung['weight_qat']:>7s}+{rung['act_quant']:<4s} "
            f"first20={head:.4f} last20={tail:.4f} "
            f"({tail - head:+.4f})  {time.perf_counter() - start:.1f}s  {fp4.tally()}"
        )


def _replace(spec: StackSpec, **changes: object) -> StackSpec:
    """`dataclasses.replace` for the spec, spelled once."""
    from dataclasses import replace

    return replace(spec, **changes)


def main() -> int:
    """Run the three instruments and print the verdict lines, labelled as such."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--time-reps", type=int, default=2)
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--skip-time", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda when free")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # a long run must be watchable
    print(card_state())
    tokens = args.batch * args.sequence
    if not args.skip_memory:
        memory(tokens, 832)
    if not args.skip_time:
        timing(tokens, args.time_reps, args.device)
    if not args.skip_loss:
        loss(args.steps)
    print(f"\nprocess RSS high-water: {rss_mib():.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
