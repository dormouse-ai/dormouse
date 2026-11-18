"""Eager vs compiled vs replayed on the real card, after the opaque custom ops.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_custom_op.py --batch 8

NOT RUN HERE. The card is busy with a live run in another repo; this script is left for
the moment it frees. It refuses a CPU torch rather than reporting an eager number under a
compiled name. What it prints: median ms/step per mode, dynamo's graph/break account and
the replay counter, so a "compiled" row with zero graphs fails the same gate as the tests.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tomllib
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import Assembled, assemble
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 3


def _spec(path: Path, smoke: bool) -> StackSpec:
    """Load the config, keeping only the fields the stack spec declares."""
    raw = tomllib.loads(path.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    form = {k: v for k, v in raw.items() if k in keys}
    if smoke:
        form = {**form, "n_layers": 2}
    return StackSpec.from_mapping(form, source=str(path))


def _build(spec: StackSpec, smoke: bool) -> Assembled:
    """Return a seeded stack on the card, with one optimizer owned by its step."""
    torch.manual_seed(SEED)
    return assemble(spec, device="cuda", smoke=smoke)


def _ms(step: object, ids: torch.Tensor, steps: int) -> float:
    """Median milliseconds per call, synchronising the device once per call."""
    seen: list[float] = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step(ids)  # type: ignore[operator]
        torch.cuda.synchronize()
        seen.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(seen)


def main() -> int:
    """Print one JSON line of measured numbers, or refuse a CPU/card-less run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="use a two-layer stack")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise LoudError(
            "bench_custom_op needs a CUDA card",
            "the card is busy with another run: wait for it, then rerun under the GPU flock",
        )
    spec = _spec(args.config, args.smoke)
    ids = torch.randint(0, spec.vocab, (args.batch, spec.seq_len + 1), device="cuda")
    result: dict[str, object] = {"shape": [args.batch, spec.seq_len], "ms": {}}
    timings: dict[str, float] = {}

    eager = _build(spec, args.smoke)
    timings["eager"] = _ms(eager.step, ids, args.steps)

    compiled.reset()
    try:
        fast = _build(spec, args.smoke)
        compiled.checked(fast.model)
        timings["compiled"] = _ms(fast.step, ids, args.steps)
        proof = compiled.proof()
        result["dynamo"] = proof
        if proof["unique_graphs"] < 1:
            raise LoudError(
                "no inductor graph was built", "the compiled row would be eager under a new name"
            )
    finally:
        compiled.reset()

    replay = _build(spec, args.smoke)
    graph: StepGraph = replay.capture(ids, optimizer=MixOptimizer(replay.model))
    timings["replayed"] = _ms(graph.run, ids, args.steps)
    result["replays"] = graph.replays()
    if graph.replays() < 1:
        raise LoudError("the graph never replayed", "no replayed row may be reported")
    result["ms"] = timings
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
