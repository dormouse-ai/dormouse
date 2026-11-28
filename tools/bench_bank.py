"""The bank's traffic, launches and speed: anchor against the fused chain, on the card.

WHY THIS SCRIPT EXISTS. A kernel that is faster because it is doing less is the defect
class this project counts. So the three numbers that decide whether the bank is fixed
are printed together, at the live shape, from the same run:

* **bytes to and from global memory per bank forward**, by hand from the tile schedule
  (`fused_bytes`) and from the op list the anchor executes (`anchor_bytes`). No Nsight on
  this host, so the count is the schedule's own arithmetic — and it is printed with the
  formula, so it can be checked against the kernel rather than believed;
* **launches per bank forward**: `cudaLaunchKernel` events under the profiler, and the
  Triton-only count `kernels.bank.bank_launches` prints beside it so a torch op cannot
  hide in the difference;
* **ms per bank forward, and per training step** (forward + backward), then tokens/s and
  kB/s in `tools/preflight.py`'s unit (`tokens/s / 1000`).

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_bank.py
    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_bank.py --rows 16384

The tile constants are imported, never restated: a schedule that changed without this
file changing would print the old bytes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from dormouse_kernels.engine import bank as kb

from dormouse.model.moe import MoEFFN

#: b8 x s1024's routed rows at top-2: the shape the bank's own report is quoted at.
LIVE_ROWS = 8 * 1024 * 2


def fused_bytes(
    rows: int, cap: int, experts: int, d: int, f: int, r: int, k: int
) -> dict[str, int]:
    """Return the fused chain's global-memory traffic per bank forward, by tile schedule.

    Per program of `BM` rows: read `x`'s tile once (`BM·d`), read the expert's `u_g`,
    `u_u` (`d·r` each) and `v_g`, `v_u`, `u_d` (`r·f` each) once per f-strip, and write
    `q`, `xr_g`, `xr_u` (`BM·r` each) and the output (`BM·d`). The `[rows, f]` slabs the
    anchor writes do not appear.

    Counted in bytes at 4 B (a fp32 number) — the bf16 step halves the activation side
    and leaves the parameters at 2 B; the ratio the report quotes is the fp32 one, which
    is the conservative case for the fused path.
    """
    width = 4
    x_read = rows * d * width
    out_write = rows * d * width
    scratch = 3 * rows * r * width + rows * k * width * 2
    per_expert = (2 * d * r + 3 * r * f) * experts * width
    #: Each expert's `v`/`u_d` factors are read by every program that touches its rows, so
    #: the schedule reads them `ceil(rows/experts/BM)` times — once per row tile.
    tiles = -(-(rows // experts) // kb.BM)
    factors = tiles * (2 * d * r + 3 * r * f) * experts * width
    return {
        "x (index map)": x_read,
        "factors (unique)": per_expert,
        "factors (re-read per row tile)": factors,
        "rank-space scratches": scratch,
        "write": out_write,
    }


def anchor_bytes(rows: int, d: int, f: int, r: int, k: int) -> dict[str, int]:
    """Return the anchor's traffic per bank forward: eleven `[rows, f]` passes.

    `g = (x·u_g)·v_g` writes `[rows, r]` then `[rows, f]`; `u` the same; the two mixtures
    read and write `[rows, f]`; `h = A⊙B` reads two and writes one; `q = h·u_d` reads one
    and writes `[rows, r]`; `y = q·v_d` reads one and writes `[rows, d]`. Each pass is
    a read plus a write unless stated, which is the 4 B-per-slab bill the profiler's
    21 790 dtype-conversion kernels are the fingerprint of.
    """
    width = 4
    slab = rows * f * width
    narrow = rows * r * width
    return {
        "x (gathered)": rows * d * width,
        "gate: rank + slab out": narrow + slab,
        "up: rank + slab out": narrow + slab,
        "mixture A: read + write": 2 * slab,
        "mixture B: read + write": 2 * slab,
        "h = A*B: read 2, write 1": 3 * slab,
        "down: read slab, write rank": slab + narrow,
        "y = q·v_d: read rank, write out": narrow + rows * d * width,
        "parameters": (2 * d * r + 3 * r * f) * (64 + 1) * width,
    }


def live(dtype: torch.dtype, fn):
    """Run `fn` as a step runs: under autocast for bf16, bare for fp32."""
    if dtype is torch.bfloat16:
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)(fn)()
    return fn


def profile(fn, steps: int) -> tuple[float, int, int]:
    """Return (ms per call, kernel launches per call, dtype-conversion kernels per call)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - started) * 1e3 / steps
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        for _ in range(steps):
            fn()
    torch.cuda.synchronize()
    launches = 0
    casts = 0
    for event in profiler.events():
        if event.name == "cudaLaunchKernel" and event.device_type.name == "CPU":
            launches += 1
        elif event.device_type.name == "CUDA" and any(
            token in event.name for token in ("convert", "copy", "to_")
        ):
            casts += 1
    return ms, launches // steps, casts // steps


def kernel_stats() -> dict[str, tuple[int, int, int]]:
    """Return `{kernel: (registers per thread, spill bytes, shared bytes)}` per compiled kernel."""
    out: dict[str, tuple[int, int, int]] = {}
    for name in ("prep", "fwd", "bwd_rows", "bwd_cols"):
        fn = kb._KERNELS.get(name)
        if fn is None:
            continue
        for cache in getattr(fn, "device_caches", {}).values():
            for compiled in cache[0].values():
                out[name] = (compiled.n_regs, compiled.n_spills, int(compiled.metadata.shared))
    return out


#: sm_120's per-SM limits, which is what the occupancy column is the minimum of. The card
#: has 48 warp slots and 100 KiB of shared memory per SM, and one block more than the
#: register file holds cannot be resident.
REGS_PER_SM = 65536
SMEM_PER_SM = 100 * 1024
WARPS_PER_SM = 48


def occupancy(regs: int, spills: int, smem: int) -> str:
    """Return the resident blocks per SM and the percentage of warp slots they fill.

    THE LIMIT, NOT AN ACHIEVED COUNT. This host has no Nsight Compute and no CUPTI
    occupancy counter, so what is reported is the arithmetic the launch's own register and
    shared-memory allocation allows — the ceiling an achieved number would sit under. The
    measured throughput beside it (bytes counted over the measured milliseconds) is what
    says whether that ceiling is the thing costing time.
    """
    by_regs = REGS_PER_SM // max(1, regs * 32 * kb.WARPS)
    by_smem = SMEM_PER_SM // max(1, smem)
    by_warps = WARPS_PER_SM // kb.WARPS
    limits = {"registers": by_regs, "shared": by_smem, "warps": by_warps}
    bound = min(limits, key=lambda name: limits[name])
    blocks = max(1, limits[bound])
    return f"{blocks} blk/SM ({100 * blocks * kb.WARPS / WARPS_PER_SM:.0f}% warps, {bound}-bound)"


def main() -> int:
    """Measure both paths at the live shape and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=LIVE_ROWS)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tiny", action="store_true", help="b1 x s256 for a quick look")
    parser.add_argument("--fp32", action="store_true", help="measure fp32, not the live bf16 step")
    args = parser.parse_args()
    rows = args.rows // 2 if args.tiny else args.rows
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    bank = MoEFFN(d_ffn_gelu=3 * 6144 // 2, n_shared=0, device="cuda")
    x = torch.randn(rows // bank.topk, bank.d_model, device="cuda")
    cap = bank.capacity(rows)
    group = bank.group(cap, 4)
    print(f"step dtype: {dtype}")
    print(
        f"routed bank: d={bank.d_model} f={bank.d_ffn} E={bank.n_experts} topk={bank.topk} "
        f"rank={bank.expert_tsct_rank} rows={rows:,} capacity={cap:,} "
        f"({cap * bank.n_experts / rows:.2f}x padded) group={group} "
        f"({-(-bank.n_experts // group)} groups)"
    )
    print("(the shared expert is a separate dense path; it is not what this file measures)")

    kb.set_bank(False)
    ms_a, ln_a, cast_a = profile(live(dtype, lambda: bank(x)), args.steps)
    kb.set_bank(True)
    kb.reset_bank_launches()
    ms_f, ln_f, cast_f = profile(live(dtype, lambda: bank(x)), args.steps)
    per_forward = kb.bank_launches() // (2 + args.steps)

    fwd = fused_bytes(
        rows,
        cap,
        bank.n_experts,
        bank.d_model,
        bank.d_ffn,
        bank.expert_tsct_rank,
        len(bank.experts[0].dictionary),
    )
    anc = anchor_bytes(rows, bank.d_model, bank.d_ffn, bank.expert_tsct_rank, 6)
    ta = sum(anc.values())
    floor = sum(v for key, v in fwd.items() if "re-read" not in key)
    ceiling = sum(fwd.values())

    print(
        f"\n{'path':14} {'ms/fwd':>8} {'launch/fwd':>11} {'casts/fwd':>10} "
        f"{'GiB counted':>12} {'implied GiB/s':>14}"
    )
    print(
        f"{'anchor':14} {ms_a:8.2f} {ln_a:11,d} {cast_a:10,d} {ta / 2**30:12.2f} "
        f"{ta / (ms_a * 1e-3) / 2**30:14.0f}"
    )
    for label, byt in (("fused floor", floor), ("fused ceiling", ceiling)):
        print(
            f"{label:14} {ms_f:8.2f} {ln_f:11,d} {cast_f:10,d} {byt / 2**30:12.2f} "
            f"{byt / (ms_f * 1e-3) / 2**30:14.0f}"
        )
    print(
        f"\nvariance: bank forward {ms_a / ms_f:.1f}x faster, "
        f"{ln_a / max(ln_f, 1):.1f}x fewer launches, {ta / ceiling:.1f}x less traffic "
        f"at the counted ceiling; Triton launches per forward {per_forward} "
        f"(2 x {-(-bank.n_experts // group)} groups = {2 * -(-bank.n_experts // group)})"
    )

    print("\nper-kernel device time, top 8 (which term the millisecond is spent in):")
    for name, on in (("anchor", False), ("fused", True)):
        kb.set_bank(on)
        fn = live(dtype, lambda: bank(x))
        fn()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
        torch.cuda.synchronize()
        events = sorted(
            (
                (event.self_device_time_total / 1000.0, str(event.key))
                for event in prof.key_averages()
                if event.self_device_time_total
            ),
            reverse=True,
        )[:8]
        print(f"  {name:8} " + "  ".join(f"{key[:30]}={t:.1f}ms" for t, key in events))

    print("\nbytes, hand-counted from the tile schedule:")
    print(f"  {'fused':10} " + "  ".join(f"{k}={v / 2**20:.1f}Mi" for k, v in fwd.items()))
    print(f"  {'anchor':10} " + "  ".join(f"{k}={v / 2**20:.1f}Mi" for k, v in anc.items()))

    def loss() -> torch.Tensor:
        return bank(x).square().mean()

    for name, on in (("anchor", False), ("fused", True)):
        kb.set_bank(on)

        def step() -> None:
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is torch.bfloat16):
                loss().backward()

        ms, _, _ = profile(step, args.steps)
        tokens = x.shape[0] * bank.topk
        print(
            f"\n{name} fwd+bwd: {ms:.1f} ms/step at {tokens:,} routed rows "
            f"({x.shape[0]:,} tokens) => {x.shape[0] / (ms * 1e-3):,.0f} tok/s, "
            f"{x.shape[0] / (ms * 1e-3) / 1000:,.1f} kB/s"
        )

    stats = kernel_stats()
    if stats:
        print("\nkernel registers, shared memory and the occupancy they allow:")
        for name, (regs, spills, smem) in stats.items():
            print(
                f"  {name:9} n_regs={regs:3d} n_spills={spills:4d} bytes "
                f"smem={smem // 1024:3d} KiB  {occupancy(regs, spills, smem)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
