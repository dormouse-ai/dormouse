"""Pre-flight before a long run: does the real config fit, and how fast is a step.

Run under the GPU lock. Prints a batch ladder with peak VRAM and ms/step, because the
number that decides a launch is not "it fits at b1" but "which batch fits and what
throughput that buys". Any arm that raises OutOfMemory is reported as a row, not a crash.

    python tools/preflight.py [config] [batches] [dtype] [tsct_rank]

`dtype` defaults to the config's own policy; pass `fp32`/`bf16` to measure one arm of the
comparison at a time. `tsct_rank` defaults to the config's; its identity value (the short
side of an expert's matrices, 832 here) turns the per-call SVD off, which is how the
ladder shows what the dtype policy buys once `model/tsct.py`'s factorisation stops
recomputing — the SVD's U/Vh are ~23 MiB retained per expert-matrix per layer, and at 12
layers that is the peak, not the weights.

Each arm runs in its own function so a failed attempt's tensors die with its frame: an
exception's traceback keeps the graph alive, and an allocator that still holds a failed
arm's 14 GiB reports every later batch as OOM too.
"""

from __future__ import annotations

import dataclasses
import gc
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.precision import Precision
from dormouse.train.step import chunked_cross_entropy

SPEC = sys.argv[1] if len(sys.argv) > 1 else "configs/dormouse_1b.toml"
BATCHES = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4", "8", "12"])]
DTYPE = sys.argv[3] if len(sys.argv) > 3 else ""
RANK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SEQ = int(sys.argv[5]) if len(sys.argv) > 5 else 1024


def attempt(model: torch.nn.Module, ids: torch.Tensor) -> tuple[float, float]:
    """One forward/backward; return (seconds, peak MiB) and let the frame own the graph."""
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with Precision.of(model).autocast("cuda"):
        logits = model(ids[:, :-1])
    loss = chunked_cross_entropy(logits, ids[:, 1:])
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    model.zero_grad(set_to_none=True)
    return elapsed, peak


def main() -> int:
    """Print the plan, then a batch ladder of real steps."""
    raw = tomllib.loads(Path(SPEC).read_text(encoding="utf-8"))
    keys = {field.name for field in dataclasses.fields(StackSpec) if field.init}
    spec = StackSpec.from_mapping(
        {key: value for key, value in raw.items() if key in keys}, source=SPEC
    )
    if DTYPE:
        spec.dtype = DTYPE
    if RANK:
        spec.expert_tsct_rank = RANK
    if SEQ:
        spec.seq_len = SEQ
    print(
        f"config {SPEC}  dtype {spec.dtype}  tsct_rank {spec.expert_tsct_rank}  seq {spec.seq_len}"
    )
    started = time.perf_counter()
    model = assemble(spec, device="cuda").model
    torch.cuda.synchronize()
    report = model.param_report()
    print("params " + "  ".join(f"{k} {v:,}" for k, v in report.items()))
    print(
        f"build {time.perf_counter() - started:.1f} s, "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    )

    tok = torch.randint(0, 30000, (1, SEQ + 1), device="cuda")
    for batch in BATCHES:
        gc.collect()
        torch.cuda.empty_cache()
        ids = tok.expand(batch, SEQ + 1).contiguous()
        model.train()
        try:
            elapsed, peak = attempt(model, ids)
        except torch.OutOfMemoryError:
            print(f"b{batch:<3} OOM  peak {torch.cuda.max_memory_allocated() / 2**20:7.0f} MiB")
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        per_step = batch * SEQ
        print(
            f"b{batch:<3} ok   {elapsed * 1000:8.1f} ms  peak {peak:7.0f} MiB  "
            f"{per_step / elapsed:7.0f} tok/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
