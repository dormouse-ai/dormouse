"""Interleaved on-card A/B of the batched KDA prefix scan against the shipped loop.

One process, one model: the only difference is the module-level `kda.SCAN` flag, so
there is no build-to-build drift. CUDA-event ms are the median of the timed steps;
the launch table comes from one profiled fwd+ce+bwd per arm.

    python fcntl_lock.py python tools/bench_kda_scan.py --rounds 7
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
from profile_fwdbwd import run_fwdbwd

from dormouse.model.assemble import assemble
from dormouse.model.attn_variants import kda as kda_module
from dormouse.model.spec import StackSpec


def main() -> int:
    """Interleave fwd+bwd timings of OFF/ON and print the launch table for each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_500m_dense_2b.toml")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=args.config)
    spec = replace(spec, batch=args.batch)
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda")
    inputs, targets = ids[:, :-1], ids[:, 1:]

    for arm in (False, True):
        kda_module.SCAN = arm
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            run_fwdbwd(model, inputs, targets, amp=True)
    torch.cuda.synchronize()

    samples: dict[bool, list[float]] = {False: [], True: []}
    for round_no in range(args.rounds):
        for arm in (False, True) if round_no % 2 == 0 else (True, False):
            kda_module.SCAN = arm
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            marks, loss = run_fwdbwd(model, inputs, targets, amp=True)
            torch.cuda.synchronize()
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            samples[arm].append(sum(marks.values()))
            print(f"round {round_no} scan={int(arm)} {sum(marks.values()):.1f} ms", flush=True)

    for arm in (False, True):
        vals = samples[arm]
        print(
            f"scan={int(arm)}: median {statistics.median(vals):.1f} ms  "
            f"min {min(vals):.1f} ms  values {[round(v, 1) for v in vals]}"
        )
    off, on = statistics.median(samples[False]), statistics.median(samples[True])
    print(f"ratio off/on {off / on:.3f}x  (delta {off - on:.1f} ms)")

    for arm in (False, True):
        kda_module.SCAN = arm
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            run_fwdbwd(model, inputs, targets, amp=True, timeit=False)
        torch.cuda.synchronize()
        launches = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
        solves = sum(e.count for e in prof.key_averages() if "solve" in str(e.key).lower())
        print(f"scan={int(arm)}: {launches:,} launches, {solves} solve ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
