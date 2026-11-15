"""What the fp4 rung actually buys on the card, per live shape — kernel included.

The measurement this script exists to make impossible: a "cost of FP4" that never ran
FP4. `require_kernel` is called at the end, so a run where every matmul fell back to the
reference fake-quant exits non-zero instead of printing a time.

Four numbers per shape, on `--device`:

* BF16 — `a·wᵀ` at the shape the model actually uses, the baseline;
* KERNEL — the `_scaled_mm` call alone, both operands already packed: the format's
  compute, which is what a serving path with static weights pays;
* ACT-QUANT — the activation packed per call, the weight kept packed: what a step pays;
* END TO END — both operands quantised per call, which is what `fp4.linear` does today.

And one memory pair: the live activation as bf16 against the same activation packed to
4.5 bits, plus the peak the matmul needs on each rung.

    tools/gpu_lock.sh --wait -- python scripts/bench_fp4.py
    tools/gpu_lock.sh --wait -- python scripts/bench_fp4.py --reps 50

The card is shared by several lanes. Run it under `tools/gpu_lock.sh`: two processes
timing the same GPU is one measurement of the contention.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import partial

import torch

from dormouse.model import fp4

#: The two contractions the model runs: attention's projections and the experts' FFN.
SHAPES: tuple[tuple[int, int, int], ...] = ((24576, 832, 2048), (24576, 832, 832))


def card_state() -> str:
    """Report what the card is doing and whether this wheel can run the kernel on it."""
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    ).stdout.strip()
    used = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    ).stdout.strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        f"card: {used}\ncompute apps:\n{apps}\n"
        f"torch {torch.__version__}, fp4 kernel on {device}: {fp4.verdict(device) or 'available'}"
    )


def timed(body: Callable[[], object], reps: int, warmup: int = 3) -> float:
    """Return the median wall time of `body`, in milliseconds, with the card synchronised."""
    for _ in range(warmup):
        body()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        body()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def flops(rows: int, out_features: int, width: int) -> float:
    """Count multiply-adds twice, which is what GFLOP/s means everywhere else."""
    return 2.0 * rows * out_features * width


def kernel(tensors: tuple[torch.Tensor, ...]) -> None:
    """Run the matmul alone, operands packed: the format's compute, nothing else."""
    left, right, scale_a, scale_b, tensor_a, tensor_b = tensors
    out = fp4._scaled_mm(left, right, scale_a, scale_b)
    torch.mul(out, tensor_a * tensor_b)


def act_quant(tensors: tuple[torch.Tensor, ...]) -> None:
    """Run the matmul with the activation packed from scratch on every call."""
    a, right, scale_b, tensor_b = tensors
    left, scale_a, tensor_a = fp4.packed(a)
    out = fp4._scaled_mm(left, right, scale_a, scale_b)
    torch.mul(out, tensor_a * tensor_b)


def speed(device: torch.device, reps: int) -> None:
    """Time bf16 against the three fp4 paths on every live shape, and the ratios."""
    print(f"\n== compute: GFLOP/s and ms, median of {reps} ==")
    print(f"  {'shape':26s} {'rung':12s} {'ms':>9s} {'GFLOP/s':>9s} {'vs bf16':>8s}")
    for rows, width, out_features in SHAPES:
        a = torch.randn(rows, width, device=device)
        w = torch.randn(out_features, width, device=device)
        a16, w16 = a.to(torch.bfloat16), w.to(torch.bfloat16)
        packed_a, scale_a, tensor_a = fp4.packed(a)
        packed_w, scale_w, tensor_w = fp4.packed(w)
        work = flops(rows, out_features, width)
        baseline = timed(partial(torch.mm, a16, w16.T), reps)
        for label, milliseconds in (
            ("bf16", baseline),
            (
                "fp4 kernel",
                timed(
                    partial(kernel, (packed_a, packed_w, scale_a, scale_w, tensor_a, tensor_w)),
                    reps,
                ),
            ),
            ("fp4 act-quant", timed(partial(act_quant, (a, packed_w, scale_w, tensor_w)), reps)),
            ("fp4 e2e", timed(partial(fp4.fp4_mm, a, w), reps)),
        ):
            ratio = f"{milliseconds / baseline:7.2f}x" if label != "bf16" else "    1.00x"
            shape = f"[{rows},{width}]x[{width},{out_features}]"
            print(
                f"  {shape:26s} {label:12s} {milliseconds:9.3f} "
                f"{work / milliseconds / 1e6:9.1f} {ratio}"
            )
        del a, w, a16, w16, packed_a, packed_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def memory(device: torch.device, reps: int) -> None:
    """Weigh the live activation on both rungs, and the peak the matmul allocates."""
    rows, width, out_features = SHAPES[0]
    print(f"\n== memory: activations [{rows}, {width}] and the matmul that reads them ==")
    a = torch.randn(rows, width, device=device)
    packed_a, scale_a, _ = fp4.packed(a)
    a16 = a.to(torch.bfloat16)
    bf16_bytes = a16.untyped_storage().nbytes()
    fp4_bytes = packed_a.untyped_storage().nbytes() + scale_a.untyped_storage().nbytes()
    per_value = fp4_bytes / (rows * width)
    saved, ratio = (bf16_bytes - fp4_bytes) / 2**20, bf16_bytes / fp4_bytes
    print(f"  bf16            {bf16_bytes / 2**20:9.2f} MiB  {bf16_bytes / (rows * width):.2f} B")
    print(f"  fp4 + scales    {fp4_bytes / 2**20:9.2f} MiB  {per_value:.2f} B (4 + 8/16)")
    print(f"  saved           {saved:9.2f} MiB  {ratio:.2f}x")

    if not torch.cuda.is_available():
        return
    w = torch.randn(out_features, width, device=device)
    packed_w, scale_w, _ = fp4.packed(w)
    w16 = w.to(torch.bfloat16)
    for label, body in (
        ("bf16", partial(torch.mm, a16, w16.T)),
        ("fp4 ", partial(fp4._scaled_mm, packed_a, packed_w, scale_a, scale_w)),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        for _ in range(reps):
            out = body()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        del out
        torch.cuda.empty_cache()
        print(f"  peak allocation during {label} matmul {peak / 2**20:8.1f} MiB")


def main(argv: list[str] | None = None) -> int:
    """Measure the live shapes and refuse to report a run that never reached the kernel."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reps", type=int, default=20)
    arguments = parser.parse_args(argv)
    device = torch.device(arguments.device)

    print(card_state())
    reason = fp4.verdict(device)
    if reason:
        print(f"\nno fp4 kernel on {device}: {reason}\nnothing here can be measured.", flush=True)
        return 2
    fp4.reset()
    speed(device, arguments.reps)
    memory(device, arguments.reps)
    fp4.require_kernel(device)
    report = fp4.tally(device)
    print(
        f"\ncounter: fp4_scaled_mm={report['fp4_scaled_mm']} "
        f"fp4_fakequant={report['fp4_fakequant']} fp4_fallback={report['fp4_fallback']}"
    )
    print("every number above is a matmul that reached the kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
