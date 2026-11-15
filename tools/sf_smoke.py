"""A CPU smoke of the three schedule modes: curves and the wall clock.

This is a smoke, not a verdict. It trains the tiny reference stack from
`tests/stacks.py` on three fixed sequences for `--steps` steps under `cosine`, `wsd`
and `sf`, and reports two losses: the batch it is memorising, and a fourth sequence it
never trains on. Both are read at the **evaluation weights** — the model, not the
training iterate — so the schedule-free arm is measured at the averaged `X`, which is
the number it would ship, and the other two at the parameter they train.

What this can see: the wiring, and whether one arm's curve is implausible. What it
cannot see: which optimizer is better. A fixed three-sequence batch for 300 steps is
the shortest and least stochastic horizon there is — exactly the regime
arXiv:2605.19095 §9 measures Schedule-Free as weakest in, and one where a trailing
average is a handicap by construction. The real A/B is 3 seeds on the 120 MB arm
against a pre-registered bar (`docs/SF_OPTIMIZER.md`).

    uv run --no-sync python tools/sf_smoke.py --steps 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from stacks import reference_stack

from dormouse.train import MixOptimizer
from dormouse.train.schedule import warmup_cosine, warmup_stable_decay
from dormouse.train.step import chunked_cross_entropy

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}

#: Every arm sees the same 20-step ramp; the scheduled arms then decay to zero and the
#: schedule-free one does not, which is the whole comparison.
WARMUP = 20


def _loss(model: torch.nn.Module, ids: torch.Tensor) -> float:
    """Return the model's mean cross-entropy on `ids`, without a graph."""
    with torch.no_grad():
        return float(chunked_cross_entropy(model(ids[:, :-1]), ids[:, 1:]))


def run(schedule: str, steps: int, sample: int) -> tuple[list[float], list[float], float]:
    """Train one arm; return its two sampled curves and its wall clock in seconds."""
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    optimizer = MixOptimizer(model, schedule=schedule, sf_warmup=WARMUP)
    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, SMALL["vocab"], (3, 17), generator=generator)
    held = torch.randint(0, SMALL["vocab"], (3, 17), generator=generator)
    seen: list[float] = []
    unseen: list[float] = []
    started = time.perf_counter()
    for step in range(steps):
        if schedule == "cosine":
            optimizer.set_lr(warmup_cosine(step, warmup=WARMUP, total=steps, peak=1.0))
        elif schedule == "wsd":
            optimizer.set_lr(
                warmup_stable_decay(
                    step, warmup=WARMUP, decay_start=int(steps * 0.8), total=steps, peak=1.0
                )
            )
        loss = chunked_cross_entropy(model(ids[:, :-1]), ids[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % sample == 0 or step + 1 == steps:
            with optimizer.eval_weights():
                seen.append(_loss(model, ids))
                unseen.append(_loss(model, held))
    return seen, unseen, time.perf_counter() - started


def main() -> None:
    """Run the three arms and print the two curves side by side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sample", type=int, default=10, help="steps between evals")
    args = parser.parse_args()
    curves: dict[str, tuple[list[float], list[float]]] = {}
    clocks: dict[str, float] = {}
    for schedule in ("cosine", "wsd", "sf"):
        seen, unseen, clocks[schedule] = run(schedule, args.steps, args.sample)
        curves[schedule] = (seen, unseen)
    for schedule, clock in clocks.items():
        print(f"{schedule:>6}: {clock / args.steps * 1e3:6.1f} ms/step, {clock:6.1f} s total")
    for index, title in enumerate(("memorised batch", "held-out sequence")):
        print(f"\n{title}, eval loss every {args.sample} steps")
        print("  step  " + "  ".join(f"{name:>8}" for name in curves))
        for point in range(len(curves["cosine"][0])):
            step = (point + 1) * args.sample
            row = "  ".join(f"{curves[name][index][point]:8.3f}" for name in curves)
            print(f"{step:6d}  {row}")


if __name__ == "__main__":
    main()
