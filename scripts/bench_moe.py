"""Twenty CPU steps of the live MoE bank, and the buffer it does not allocate.

The bank is the design's own config: `d=832, f=6144`, 64 routed experts, top-2,
one shared expert. That is 996.9M stored / 46.1M active parameters, and building
it in fp32 costs ~4 GB before a single activation — so this script is a
once-per-machine measurement, not a unit test. The tests pin the arithmetic at
small widths; this pins the arithmetic at the shipped widths and reports what a
step actually did.

    python scripts/bench_moe.py                  # CPU; the card is usually busy
    python scripts/bench_moe.py --steps 20 --batch 4 --seq 256

Two numbers the old arm could not produce:

* `moe_rows_run` after N steps must be exactly `N · tokens · topk` and
  `moe_dropped` must be 0 - the capacity-2.0 arm ran 2.52x this;
* the peak activation at b24 x s1024, measured three ways: the largest single
  allocation torch's profiler saw for the ROUTED path alone, the same with the
  shared expert on (whose `[n, f]` buffers are the dense FFN's own cost), and
  the process's RSS high-water mark. The `[R, f]` bank this module never builds
  would be `R·f·4` = 1.15 GiB per buffer at f=6144; the zero-padded 2.0 one
  2.3 GiB.

The peak path runs FIRST (before any gradient buffer exists) and on CPU only:
`resource` counts host pages, and the card is busy with another lane. A CUDA
peak would be `torch.cuda.max_memory_allocated` and is not measured here.
"""

from __future__ import annotations

import argparse
import resource
import time

import torch
from dormouse_core.counters import Counters

from dormouse.model.moe import MoEFFN


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


def main() -> None:
    """Build the live bank, measure the peak, then run the training steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--peak-batch", type=int, default=24)
    parser.add_argument("--peak-seq", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    bank = MoEFFN(device=device)
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
    group_buffer = counted.get("moe_max_group", 0) * bank.d_ffn * 4
    print(f"rows expected={expected:,} counted={counted.get('moe_rows_run', 0):,}")
    print(
        f"dropped={counted.get('moe_dropped')} "
        f"largest routed group this run={counted.get('moe_max_group', 0):,} rows "
        f"({group_buffer / 2**20:.1f} MiB per [r, f] buffer)"
    )


if __name__ == "__main__":
    main()
