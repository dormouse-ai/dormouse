"""The three zero-loss levers, measured on the card at the largest step that fits.

    tools/gpu_lock.sh -- python scripts/bench_levers.py

FUSION. `residual_norm` and `glu` against the eager statements they replace, at the
shapes a step uses (`[tokens, d_model]` for the norm and the add, `[tokens, d_ffn]` for
the GLU), reported as ms and as the ratio. A fusion that is not faster is printed as a
loss and the flag stays off: a lever that costs quality has to be paid for by its
quality loss, and a lever that costs time has no argument at all.

GRAPH. A whole step captured and replayed against the same step run eagerly, with the
pool it took, whether the capture succeeded, and — when it did not — the op it stopped
at. `graph_replays` is read from the counter, so a "the graph gave +0" reading cannot be
a reading of the eager path.

COUNTERS. `fused_ops`, `graph_replays` and the `norm_fwd` count from one real step,
because a lever's gate is a number that moved.

The widths are the profiler's: the live config narrowed, because the recipe's 24x1024
step does not fit a 16 GiB card at all (`tools/profile_step.py` refuses it with the
counted reason). Every number printed says the widths it was taken at.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.kernels import fused
from dormouse.model.assemble import assemble
from dormouse.model.blocks import RMSNorm
from dormouse.model.spec import StackSpec
from dormouse.train.graph import StepGraph, device_step
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

MIB = 2**20

#: The config the widths are narrowed from.
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dormouse_1b.toml"


def _median(call: object, *, warmup: int, iters: int) -> float:
    """Return the median milliseconds of `iters` calls after `warmup` untimed ones."""
    for _ in range(warmup):
        call()  # type: ignore[operator]
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        call()  # type: ignore[operator]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def spec_for(*, layers: int, experts: int, batch: int, seq_len: int) -> StackSpec:
    """Narrow the live config to widths this card can hold, keeping the shapes."""
    base = StackSpec.from_toml(CONFIG)
    return dataclasses.replace(
        base, attn="kda", n_layers=layers, n_experts=experts, batch=batch, seq_len=seq_len
    )


def build(
    spec: StackSpec, device: str = "cuda"
) -> tuple[torch.nn.Module, MixOptimizer, torch.Tensor]:
    """Return a seeded stack, its optimizer and one fixed batch."""
    torch.manual_seed(0)
    model = assemble(spec, smoke=True).model.to(device)
    ids = torch.randint(
        0,
        spec.vocab,
        (spec.batch, spec.seq_len + 1),
        generator=torch.Generator().manual_seed(7),
    ).to(device)
    return model, MixOptimizer(model), ids


def bench_fused(tokens: int, d_model: int, d_ffn: int) -> int:
    """Print both fusions against their eager statements; return 1 when one is slower."""
    if not fused.available():
        print("fusion: no Triton/CUDA here, nothing measured")
        return 1
    generator = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(tokens, d_model, generator=generator, device="cuda")
    delta = torch.randn(tokens, d_model, generator=generator, device="cuda") * 0.1
    norm = RMSNorm(d_model, device="cuda")
    gate = torch.randn(tokens, d_ffn, generator=generator, device="cuda")
    up = torch.randn(tokens, d_ffn, generator=generator, device="cuda")

    cases: list[tuple[str, list[int], object, object]] = [
        (
            "residual_norm",
            [tokens, d_model],
            lambda: norm(x + delta),
            lambda: fused.residual_norm(x, delta, norm),
        ),
        (
            "glu",
            [tokens, d_ffn],
            lambda: torch.nn.functional.silu(gate) * up,
            lambda: fused.glu(gate, up),
        ),
    ]
    failures = 0
    print(f"\nfusion, {tokens} tokens:")
    print(f"{'op':<16}{'shape':<20}{'eager ms':>10}{'fused ms':>10}{'ratio':>8}")
    for name, shape, eager, fast in cases:
        fused.set_fused(False)
        slow = _median(eager, warmup=10, iters=50)
        fused.reset_fused_ops()
        fused.set_fused(True)
        try:
            quick = _median(fast, warmup=10, iters=50)
            launches = fused.fused_ops()
        finally:
            fused.set_fused(False)
        assert launches > 0, f"{name}: the flag was on and fused_ops stayed 0"
        ratio = slow / quick if quick > 0 else float("inf")
        failures += 0 if ratio > 1.0 else 1
        verdict = "" if ratio > 1.0 else "   slower -> keep the flag off"
        print(f"{name:<16}{shape!s:<20}{slow:>10.3f}{quick:>10.3f}{ratio:>7.2f}x{verdict}")
    print(f"fused_ops after {2 * 50} fused calls: {fused.fused_ops()}")
    return failures


def bench_graph(spec: StackSpec, *, steps: int) -> int:
    """Print the graphed step against the eager step, with its pool and its counters."""
    eager_model, eager_optim, ids = build(spec)
    graph_model, graph_optim, graph_ids = build(spec)
    eager = _median(lambda: device_step(eager_model, ids, eager_optim), warmup=1, iters=steps)

    graph = StepGraph(graph_model, graph_optim, ids=graph_ids)
    counters = Counters()
    try:
        report = graph.capture()
    except LoudError as exc:
        print(f"\ngraph: capture FAILED and is reported, not fallen back from:\n  {exc}")
        print(f"eager step: {eager:.1f} ms (the only number there is)")
        return 1
    print("\ngraph:")
    print(f"  {report.line()}")
    print(f"  pool taken: {graph.pool_bytes / MIB:.0f} MiB, measured as the device free delta")
    quick = _median(lambda: graph.replay(graph_ids, counters), warmup=1, iters=steps)
    for _ in range(3):
        graph.replay(graph_ids, counters)
    ratio = eager / quick if quick > 0 else 0.0
    print(f"{'step':<16}{'eager ms':>10}{'graphed ms':>12}{'speedup':>9}")
    print(f"{'whole step':<16}{eager:>10.1f}{quick:>12.1f}{ratio:>8.2f}x")
    print(
        f"  graph_replays: {graph.graph_replays} (core counter: {counters.counts['graph_replays']})"
    )
    return 0 if graph.graph_replays > 0 else 1


def bench_counters(spec: StackSpec) -> None:
    """One real step, in both modes, with the three counters printed."""
    model, optimizer, ids = build(spec)
    for flag in (False, True):
        fused.set_fused(flag)
        fused.reset_fused_ops()
        counters = Counters()
        try:
            train_step(model, ids, optimizer=optimizer, clip=float("inf"), counters=counters)
        finally:
            fused.set_fused(False)
        print(
            f"  fused={flag!s:<5} fused_ops={counters.counts['fused_ops']:>4} "
            f"norm_fwd={counters.counts['norm_fwd']:>5} module fused_ops={fused.fused_ops()}"
        )


def main(argv: list[str] | None = None) -> int:
    """Measure the three levers and print one table each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--fusion-tokens",
        type=int,
        default=0,
        help="rows for the op benchmark (0 = the step's token count; the recipe is 24576)",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device: every lever here is a card measurement", file=sys.stderr)
        return 2
    spec = spec_for(
        layers=args.layers, experts=args.experts, batch=args.batch, seq_len=args.seq_len
    )
    tokens = spec.batch * spec.seq_len
    print(
        f"widths: attn={spec.attn} layers={spec.n_layers} experts={spec.n_experts} "
        f"batch={spec.batch} seq={spec.seq_len} ({tokens} tokens/step), "
        f"d_model={spec.d_model} d_ffn={spec.d_ffn}"
    )
    failures = bench_fused(args.fusion_tokens or tokens, spec.d_model, spec.d_ffn)
    failures += bench_graph(spec, steps=args.steps)
    print("\ncounters, one real step:")
    bench_counters(spec)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
