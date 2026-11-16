"""Time one training step, count its kernel launches, and dump what it computed.

Run under the GPU lock, because a shared card measures the other lane's work:

    flock /tmp/dormouse-gpu.lock /tmp/opencode/gpu-env/bin/python tools/bench_step.py \
        --config configs/dormouse_1b.toml --batch 8 --seq 1024 --steps 3 [--profile] [--dump f.json]

The number that matters is ms/step at a fixed batch, and it is printed with the card's
load before and after so a step timed next to someone else's job is visible as such.
`--profile` counts every CUDA kernel the step issued — the launch count is the quantity
that explains a host-bound step, and a step that is slow with a small launch count is a
different bug.

`--dump` writes the step-0 loss, the step-0 per-parameter gradient norms and a logits
slice taken at the *initial* weights, so a scheduling change can be shown to be
numerically the same step rather than assumed to be.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


def _load(config: str) -> StackSpec:
    """Return the spec a config file names."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    return StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)


def _card() -> str:
    """Return the card's name and the memory in use by every process on it."""
    free, total = torch.cuda.mem_get_info()
    return (
        f"{torch.cuda.get_device_name()} {(total - free) / 2**20:,.0f}/"
        f"{total / 2**20:,.0f} MiB used, {free / 2**20:,.0f} MiB free"
    )


def main() -> int:
    """Build the stack, time `--steps` real steps, and print the ladder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--compile", dest="compile_", action="store_true")
    parser.add_argument("--dump", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = replace(_load(args.config), seq_len=args.seq)
    torch.manual_seed(args.seed)
    print("card before:", _card())
    assembled = assemble(spec, device="cuda")
    report = assembled.report()
    print(
        f"params stored={report['stored']:,} active={report['active']:,} "
        f"b{args.batch} s{args.seq} = {args.batch * args.seq:,} tokens/step"
    )
    optimizer = MixOptimizer(assembled.model)
    ids = torch.randint(0, spec.vocab, (args.batch, args.seq + 1), device="cuda")
    if args.fused:
        from dormouse.kernels import fused

        fused.set_fused(True)

    dump: dict[str, object] = {}
    if args.dump:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = assembled.model(ids[:, :-1], None)
        dump["logits"] = logits[:1, :4, :8].float().tolist()
        dump["logits_absmax"] = float(logits.detach().float().abs().max())
        del logits

    times: list[float] = []
    loss = 0.0
    if args.profile:
        _profile(assembled, optimizer, ids, args.accum)
    for turn in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, _counters = assembled.step(ids, optimizer=optimizer, accum=args.accum)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        loss = result.loss
        if turn == 0 and args.dump:
            dump["loss"] = loss
            dump["grad_norms"] = {
                name: float(param.grad.detach().float().norm()) if param.grad is not None else -1.0
                for name, param in assembled.model.named_parameters()
            }
            dump["grad_norm"] = result.grad_norm
        if turn >= args.warmup:
            times.append(elapsed)
        print(f"  step {turn}: {elapsed * 1000:9.1f} ms  loss {result.loss:.6f}")

    if times:
        median = statistics.median(times)
        rate = args.batch * args.seq / median
        print(
            f"median {median * 1000:,.1f} ms/step  {rate / 1000:,.1f} kB/s  "
            f"peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB  loss {loss:.6f}"
        )
    if args.dump:
        params = [p.detach().float() for p in assembled.model.parameters()]
        dump["param_norm"] = math.sqrt(sum(float(p.pow(2).sum()) for p in params))
        dump["param_absmax"] = max(float(p.abs().max()) for p in params)
        dump["param_finite"] = all(bool(torch.isfinite(p).all()) for p in params)
        dump["nonfinite"] = [
            name
            for name, param in assembled.model.named_parameters()
            if not bool(torch.isfinite(param).all())
        ]
        dump["loss_last"] = loss
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(dump))
        print(f"dumped step-0 loss, logits slice and {len(dump['grad_norms'])} grad norms")
    print("card after:", _card())
    return 0


def _profile(assembled: object, optimizer: MixOptimizer, ids: torch.Tensor, accum: int) -> None:
    """Count the step's CUDA kernels and print the widest, by self time."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assembled.step(ids, optimizer=optimizer, accum=accum)  # type: ignore[attr-defined]
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events)
    print(f"launches {launches:,}  cuda {total / 1e6:,.2f} s")
    for event in sorted(events, key=lambda e: -e.self_device_time_total)[:10]:
        print(f"  {event.self_device_time_total / 1e3:8.1f} ms {event.count:7,}  {event.key[:64]}")
    kinds: dict[str, list[float]] = {}
    for event in events:
        bucket = kinds.setdefault(event.key.split("(")[0][:40], [0.0, 0.0])
        bucket[0] += event.self_device_time_total
        bucket[1] += event.count
    print("kinds by cuda time:")
    for name, (micro, count) in sorted(kinds.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"  {micro / 1e3:8.1f} ms {int(count):7,}  {name}")


if __name__ == "__main__":
    raise SystemExit(main())
