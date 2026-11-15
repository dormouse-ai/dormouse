"""Pre-flight before a long run: does the real config fit, and how fast is a step.

Run under the GPU lock (`flock /tmp/dormouse-gpu.lock ...`). Prints a batch ladder with
peak VRAM, ms/step and throughput, because the number that decides a launch is not "it
fits at b1" but "which batch fits and what throughput that buys".

An arm that raises OutOfMemory is a ROW, not a crash, and the row carries the allocation
PyTorch refused: when nothing fits, that byte count is the next thing to shrink and the
log names it. One real step per batch, through `Assembled.step` — the optimizer is built
first, because its state is half the footprint of a 400M-parameter run and a ladder
without it is a smaller model wearing this config's name.

    flock /tmp/dormouse-gpu.lock uv run --no-sync python tools/preflight.py \
        configs/dormouse_1b.toml 1,2,4,8,12 [--seq N] [--bf16]

`--seq` overrides the spec's `seq_len` for the ladder. It exists because the ladder's
answer can be "no batch fits at the config's sequence", and then the useful second
number is the sequence at which a batch does fit.
"""

from __future__ import annotations

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

SPEC = sys.argv[1] if len(sys.argv) > 1 else "configs/dormouse_1b.toml"
BATCHES = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "4", "8"])]
BF16 = "--bf16" in sys.argv
SEQ = (
    int(sys.argv[sys.argv.index("--seq") + 1])
    if "--seq" in sys.argv
    else 0  # 0 means "the spec's own"
)


def _refused(exc: torch.OutOfMemoryError) -> str:
    """Return the line of PyTorch's OOM message that names what it could not allocate."""
    for line in str(exc).splitlines():
        if "Tried to allocate" in line or "Out of memory" in line:
            return line.strip()
    return str(exc).splitlines()[0].strip() if str(exc) else repr(exc)


def main() -> int:
    """Print the bill, then a batch ladder of real steps."""
    raw = tomllib.loads(Path(SPEC).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=SPEC)
    if SEQ:
        spec = replace(spec, seq_len=SEQ)
    seq = spec.seq_len
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    report = assembled.report()
    print(
        f"params stored={report['stored']:,} active={report['active']:,} "
        f"ffn_active={report['active_ffn']:,}  seq={seq} "
        f"dtype={'bf16 autocast' if BF16 else 'fp32'}"
    )
    free, total = torch.cuda.mem_get_info()
    print(
        f"card {torch.cuda.get_device_name()} {total / 2**20:,.0f} MiB, "
        f"{free / 2**20:,.0f} MiB free before the model"
    )
    optimizer = MixOptimizer(model)

    tok = torch.randint(0, spec.vocab, (1, seq + 1), device="cuda")
    fitted: list[tuple[int, float, float, float]] = []
    for batch in BATCHES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = tok.expand(batch, seq + 1).contiguous()
        try:
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
                assembled.step(ids, optimizer=optimizer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**20
            rate = batch * seq / elapsed
            fitted.append((batch, elapsed * 1000, peak, rate))
            print(
                f"b{batch:<3} ok   {elapsed * 1000:8.1f} ms  peak {peak:7.0f} MiB  "
                f"{rate / 1000:6.1f} kB/s  {1000 * elapsed / batch:6.1f} ms/sample"
            )
        except torch.OutOfMemoryError as exc:
            print(
                f"b{batch:<3} OOM  peak {torch.cuda.max_memory_allocated() / 2**20:7.0f} MiB"
                f"  {_refused(exc)}"
            )
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    if fitted:
        batch, ms, peak, rate = fitted[-1]
        print(
            f"largest batch that fits: b{batch}  peak {peak:,.0f} MiB  {ms:,.1f} ms/step  "
            f"{rate / 1000:,.1f} kB/s"
        )
    else:
        print(
            "no batch fits: the OOM row above names the allocation that did not; "
            "shrink that tensor (activations, chunk rows) before raising the card's ceiling"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
