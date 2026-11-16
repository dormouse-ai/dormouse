"""Bounded `torch.compile` attempt on the step's numerical region.

    flock /tmp/dormouse-gpu.lock timeout 600s .venv-cuda/bin/python \
        tools/bench_compile_step.py --batch 4

The third lever if a capture cannot run: let dynamo split the step at its host reads and
inductor compile the numerical regions. The project has already seen dynamo build one
graph with zero breaks here and then spend 28 minutes in inductor before being killed at
10 GB RSS. A compile that does not finish inside the bound is reported as NO ARTIFACT —
never as a speed-up from the eager kernels it failed to replace.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.optim import MixOptimizer

WATCHDOG_S = 540


def _alarm(signum: int, frame: object) -> None:
    """Turn the wall-clock bound into a visible, timed-out verdict."""
    del signum, frame
    raise TimeoutError(f"inductor did not finish within {WATCHDOG_S}s")


def spec_for(config: str, batch: int) -> StackSpec:
    """Build the real config with only the batch overridden."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    return replace(spec, batch=batch)


def main() -> int:
    """Try one compiled step inside the bound; print graphs, breaks and ms if it lands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WATCHDOG_S)
    started = time.perf_counter()
    spec = spec_for(args.config, args.batch)
    torch.manual_seed(3)
    assembled = assemble(spec, device="cuda")
    optimizer = MixOptimizer(assembled.model)
    optimizer.set_lr(0.0)
    ids = torch.randint(
        0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda", generator=None
    )
    print(f"compile attempt at b{args.batch} x s{spec.seq_len}, watchdog {WATCHDOG_S}s", flush=True)
    try:
        compiled.enable(assembled.model, fullgraph=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assembled.step(ids, optimizer=optimizer)
        torch.cuda.synchronize()
        proof: dict[str, Any] = compiled.proof()
        elapsed = time.perf_counter() - started
        print(
            f"USABLE ARTIFACT after {elapsed:.1f}s: {proof['unique_graphs']} graphs, "
            f"breaks {proof['graph_break']}, frames {proof['frames']}",
            flush=True,
        )
        return 0 if proof["unique_graphs"] > 0 else 1
    except TimeoutError as exc:
        print(f"NO ARTIFACT: {exc} (killed, not a speed-up)", flush=True)
        return 1
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        print(f"NO ARTIFACT: {str(exc).splitlines()[0]}", flush=True)
        return 1
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
