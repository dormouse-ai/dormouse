"""Twenty steps of the live MoE bank, the buffers the dispatch builds, and its launches.

The bank is the design's own config: `d=832, f=6144`, 64 routed experts, top-2, one
shared expert. That is 996.9M stored / 46.1M active parameters, and building it in fp32
costs ~4 GB before a single activation — so this script is a once-per-machine measurement,
not a unit test. The tests pin the arithmetic at small widths; this pins the arithmetic at
the shipped widths and reports what a step actually did.

    python scripts/bench_moe.py                       # CPU; the card is usually busy
    python scripts/bench_moe.py --steps 20 --batch 4 --seq 256
    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python scripts/bench_moe.py --card \
        --rows 16384 --steps 5                        # launches and ms/step on the card

Three numbers the loop could not produce:

* `moe_rows_run` after N steps must be exactly `N · tokens · topk` and `moe_dropped`
  must be 0; `moe_pad_rows` says what the padded layout executed against them, and
  `moe_capacity` is the formula's slot width — not the old 2.0 multiplier;
* the peak activation at b24 x s1024, measured three ways: the largest single
  allocation torch's profiler saw for the ROUTED path alone, the same with the shared
  expert on (whose `[n, f]` buffers are the dense FFN's own cost), and the process's RSS
  high-water mark. That peak is one expert GROUP's slab (`DISPATCH_SLAB_BYTES`), not the
  padded `[E, capacity, f]` bank;
* with `--card`: kernel launches and wall ms per bank forward and per training step. The
  loop this dispatch replaced issued 59 028 launches per step at twelve layers — one
  Python iteration and ~77 kernels per expert per call — and that is what `--card`
  counts down from.

The peak path runs FIRST (before any gradient buffer exists) and on CPU only: `resource`
counts host pages. `--card` reports `torch.cuda.max_memory_allocated` instead.
"""

from __future__ import annotations

import argparse
import resource
import time

import torch
from dormouse_core.counters import Counters

from dormouse.model.moe import DISPATCH_SLAB_BYTES, MoEFFN

#: The routed rows one bank forward executes at the measured shape (b8 x s1024, top-2).
LIVE_ROWS = 8 * 1024 * 2


def rss_mib() -> float:
    """Process RSS high-water mark in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_peak(bank: MoEFFN, batch: int, seq: int) -> tuple[int, int, float]:
    """One no-grad forward at `batch x seq`; return (tokens, largest op, RSS growth).

    `largest` is the biggest single `self_cpu_memory_usage` the profiler saw, in
    bytes — the transient that would exist even with an infinite pool. The RSS
    growth is the allocator's own view and is reported next to it because the two
    answer different questions (one transient vs. all retained pages).
    """
    tokens = batch * seq
    x = torch.randn(tokens, bank.d_model)
    before = rss_mib()
    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profiler,
    ):
        bank(x)
    largest = max(event.self_cpu_memory_usage for event in profiler.events())
    return tokens, largest, rss_mib() - before


def card(bank: MoEFFN, rows: int, steps: int) -> None:
    """Time `steps` forwards on the card and count the kernels one of them launches."""
    x = torch.randn(rows, bank.d_model, device="cuda")
    for _ in range(2):  # warm the allocator and every kernel's first-compile cost
        bank(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        bank(x)
    torch.cuda.synchronize()
    forward = (time.perf_counter() - start) / steps
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        for _ in range(steps):
            bank(x)
    torch.cuda.synchronize()
    launches = (
        sum(
            1
            for event in profiler.events()
            if event.name == "cudaLaunchKernel" and event.device_type.name == "CPU"
        )
        // steps
    )
    print(
        f"card: {rows:,} rows, {forward * 1e3:.2f} ms per bank forward, "
        f"{launches:,} kernel launches per forward, "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB"
    )
    print(
        f"      at twelve layers that is {launches * 12:,} launches per step "
        f"({bank.capacity(rows)}, rows per expert slot)"
    )


def main() -> None:
    """Build the live bank, measure the peak, then run the training steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--peak-batch", type=int, default=24)
    parser.add_argument("--peak-seq", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--card", action="store_true", help="measure on the card, not the CPU")
    parser.add_argument("--rows", type=int, default=LIVE_ROWS, help="rows for --card")
    args = parser.parse_args()

    device = torch.device("cuda" if args.card else "cpu")
    bank = MoEFFN(d_ffn_gelu=3 * 6144 // 2, device=device)
    bill = bank.param_report()
    print(
        f"device={device.type} threads={torch.get_num_threads()} "
        f"d={bank.d_model} f={bank.d_ffn} E={bank.n_experts} topk={bank.topk} "
        f"shared={bank.n_shared}"
    )
    print(
        f"bill: per_expert={bill['per_expert']:,} routed={bill['routed_stored']:,} "
        f"shared={bill['shared_stored']:,} router={bill['router']:,} "
        f"stored={bill['stored']:,} active={bill['active']:,} "
        f"active+router={bill['active_with_router']:,}"
    )
    capacity = bank.capacity(args.rows)
    group = bank.group(capacity, 4)
    print(
        f"dispatch: {args.rows:,} rows -> capacity {capacity:,} per expert slot "
        f"({capacity * bank.n_experts / args.rows:.2f}x padded), group {group} experts, "
        f"one slab <= {DISPATCH_SLAB_BYTES / 2**20:.0f} MiB, "
        f"{3 * 2 * -(-bank.n_experts // group)} matmuls per forward"
    )

    if args.card:
        card(bank, args.rows, max(args.steps, 1))
        return

    tokens, largest, growth = measure_peak(bank, args.peak_batch, args.peak_seq)
    rows = tokens * bank.topk
    analytic_bank = rows * bank.d_ffn * 4
    analytic_padded = 2 * rows * bank.d_ffn * 4
    shared = bank.shared
    bank.shared = None
    _, routed_largest, routed_growth = measure_peak(bank, args.peak_batch, args.peak_seq)
    bank.shared = shared
    print(
        f"peak at {args.peak_batch}x{args.peak_seq} ({tokens:,} tokens, {rows:,} rows): "
        f"ROUTED largest op {routed_largest / 2**20:.1f} MiB, RSS growth "
        f"{routed_growth:.1f} MiB; with the shared expert {largest / 2**20:.1f} MiB "
        f"and {growth:.1f} MiB; [R,f] bank would be {analytic_bank / 2**20:.1f} MiB, "
        f"padded 2.0 {analytic_padded / 2**20:.1f} MiB"
    )

    optimizer = torch.optim.SGD(bank.parameters(), lr=args.lr)
    counters = Counters()
    torch.manual_seed(0)
    x = torch.randn(args.batch * args.seq, bank.d_model, device=device)
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = bank(x, counters).square().mean()
        loss.backward()
        optimizer.step()
        elapsed = (time.perf_counter() - start) * 1e3
        print(f"step {step:3d} loss={float(loss.detach()):.6f} {elapsed:7.1f} ms")
    print(f"counters after {args.steps} steps: {counters.line()}")
    counted = counters.snapshot()
    expected = args.steps * x.shape[0] * bank.topk
    counted_capacity = counted.get("moe_capacity", 0)
    print(f"rows expected={expected:,} counted={counted.get('moe_rows_run', 0):,}")
    print(
        f"dropped={counted.get('moe_dropped')} padded={counted.get('moe_pad_rows', 0):,} "
        f"capacity={counted_capacity:,} rows per expert "
        f"({counted_capacity * bank.n_experts / (x.shape[0] * bank.topk):.2f}x the routed rows, "
        f"{counted_capacity * bank.d_ffn * 4 / 2**20:.1f} MiB per [capacity, f] slab)"
    )


if __name__ == "__main__":
    main()
