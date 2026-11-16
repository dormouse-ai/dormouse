"""Card-side parity + speed for the wired `StepGraph` (run when the GPU is free).

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_wired_graph.py \
        --steps 20

WHAT IT PROVES. The loop now builds one `StepGraph` after warm-up and replays it; this
tool puts an eager run and a replayed run of the SAME config side by side and refuses to
print a speed number unless the two agree on loss, every gradient and every parameter.
A capture that refuses is LOUD and the tool exits non-zero — there is no eager fallback
to time by accident. `graph_replays` on every measured step is a hard gate.

The card is deliberately not touched by CI. This file is the one thing left to run once
the live training run in the other repo has released the device.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tomllib
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dormouse_1b.toml"
#: bf16's 8 mantissa bits: 2^-8 is one ulp of relative resolution at unit scale.
RTOL, ATOL = 2.0**-8, 2.0**-8


def _spec(config: Path) -> StackSpec:
    """Read a stack config, keeping only the fields `StackSpec` declares."""
    raw = tomllib.loads(config.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    return StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(config))


def _ids(spec: StackSpec, device: str) -> torch.Tensor:
    """One deterministic batch of ids at the config's shape."""
    generator = torch.Generator(device="cpu").manual_seed(3)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator)
    return ids.to(device)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a comparable copy of every parameter and gradient."""
    return {
        **{f"p:{name}": p.detach().clone() for name, p in model.named_parameters()},
        **{
            f"g:{name}": p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        },
    }


def _parity(eager: dict[str, torch.Tensor], replay: dict[str, torch.Tensor]) -> str:
    """Raise unless the two steps agree; return the worst relative difference."""
    if eager.keys() != replay.keys():
        raise LoudError("the two steps carry different parameters", "same config, same step")
    worst = 0.0
    for name, want in eager.items():
        got = replay[name]
        if not torch.allclose(got, want, rtol=RTOL, atol=ATOL):
            raise LoudError(
                f"{name}: the replayed step disagrees with eager",
                "the graph changed arithmetic, not just launch timing; do not report a speed",
            )
        worst = max(worst, float((got - want).abs().max() / want.abs().max().clamp(min=1e-12)))
    return f"{worst:.2e}"


def _eager(
    model: torch.nn.Module, optimizer: MixOptimizer, ids: torch.Tensor, steps: int
) -> tuple[list[float], float]:
    """Time `steps` eager steps and return the timings and the last loss."""
    times: list[float] = []
    loss = float("nan")
    for _ in range(steps):
        started = time.perf_counter()
        loss = train_step(model, ids, optimizer=optimizer).loss
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1e3)
    return times, loss


def _replayed(graph: StepGraph, ids: torch.Tensor, steps: int) -> tuple[list[float], float]:
    """Time `steps` graph replays, requiring a real replay on every one."""
    times: list[float] = []
    loss = float("nan")
    for _ in range(steps):
        started = time.perf_counter()
        graph.run(ids)
        torch.cuda.synchronize()
        if graph.replays() == 0:
            raise LoudError("a declared graph replayed nothing", "no eager fallback is measured")
        loss = graph.loss()
        times.append((time.perf_counter() - started) * 1e3)
    return times, loss


def main() -> int:
    """Run eager against the captured step, gate on parity, then print the speed table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise LoudError("bench_wired_graph needs a CUDA device", "run on the card, not in CI")
    spec = _spec(args.config)
    ids = _ids(spec, "cuda")

    torch.manual_seed(3)
    eager_assembled = assemble(spec, device="cuda")
    eager_optim = MixOptimizer(eager_assembled.model)

    torch.manual_seed(3)
    replay_assembled = assemble(spec, device="cuda")
    replay_optim = MixOptimizer(replay_assembled.model)

    try:
        graph = replay_assembled.capture(ids, optimizer=replay_optim)
    except LoudError as exc:
        raise LoudError(
            f"wired graph capture refused: {exc}", "fix the named op; no eager fallback"
        ) from exc

    eager_times, eager_loss = _eager(eager_assembled.model, eager_optim, ids, args.steps)
    replay_times, replay_loss = _replayed(graph, ids, args.steps)

    if not torch.isfinite(torch.tensor(eager_loss)) or abs(eager_loss - replay_loss) > 1e-4:
        raise LoudError(
            f"the replayed loss differs from eager: {eager_loss} vs {replay_loss}",
            "capture/replay changed the arithmetic; do not publish a speed number",
        )
    worst = _parity(_snapshot(eager_assembled.model), _snapshot(replay_assembled.model))

    eager_ms = statistics.median(eager_times)
    replay_ms = statistics.median(replay_times)
    print(f"config={args.config} batch={spec.batch} seq={spec.seq_len} steps={args.steps}")
    print(f"parity: loss {eager_loss:.6f} == {replay_loss:.6f}; worst rel diff {worst}")
    print(f"graph: {graph!r}; graph_replays={graph.replays()}")
    print(f"eager   median {eager_ms:8.2f} ms/step")
    print(f"replay  median {replay_ms:8.2f} ms/step  ({eager_ms / replay_ms:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
