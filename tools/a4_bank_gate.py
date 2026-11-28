"""A/B gate for the bank's INT4 code-saving autograd boundary.

Run under the card lock with the fused bank off (that path writes no `[rows, d_ffn]`
slab, so there is nothing to compress):

    PYTHONPATH=src UV_PROJECT_ENVIRONMENT=/home/sehaxe/dormouse2/app/.venv-cuda \
      uv run --no-sync env DORMOUSE_BANK=0 python tools/a4_bank_gate.py

Prints the required `variant|peak_MiB|ms|dBPB|codes|verdict` table. Timing is an
interleaved min-of-15 of whole optimizer steps: one-arm-at-a-time drifts on this
shared box (the A4 norm-output gate measured +17% that way and +6.5% interleaved).
BPB is scored on ONE fixed held-out window after training, not on the training loss.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import tomllib
from dataclasses import fields
from pathlib import Path

import torch
from dormouse_core.counters import Counters

from dormouse.eval.bpb import from_token_nats
from dormouse.model import quant
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
#: A real bpe window of `token_count` tokens holds roughly four UTF-8 bytes per
#: token. The short fixture text used elsewhere divides by ~0.04 bytes/token and
#: turns a 0.0015-nat change into 0.05 BPB — an artefact of the denominator.
EVAL_UNIT = "Привет мир — evaluation window. "


def eval_text(token_count: int) -> str:
    """Return a window whose byte count matches the scored token count."""
    return EVAL_UNIT * max(1, (4 * token_count) // len(EVAL_UNIT.encode("utf-8")) + 1)


def spec(*, batch: int = 2, seq: int = 1024, smoke: bool = False) -> StackSpec:
    """Return the bank-heavy stack, or the small proxy used for the paired BPB run."""
    raw = tomllib.loads((CONFIGS / "dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {f.name for f in fields(StackSpec) if f.init}
    values = {key: value for key, value in raw.items() if key in known}
    values.update(torch_compile=False)
    if smoke:
        values.update(
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            c_lat=64,
            n_layers=4,
            d_ffn=256,
            n_experts=4,
            topk=1,
            expert_tsct_rank=32,
            attn_k=64,
            attn_window=32,
            seq_len=512,
            batch=2,
        )
    else:
        values.update(
            n_layers=4, n_experts=8, topk=1, expert_tsct_rank=64, batch=batch, seq_len=seq
        )
    return StackSpec.from_mapping(values)


def build(spec: StackSpec, seed: int, *, smoke: bool) -> tuple[object, MixOptimizer, torch.Tensor]:
    """Build one arm, its optimizer and its input ids on the card."""
    torch.manual_seed(seed)
    built = assemble(spec, smoke=smoke, device="cuda")
    ids = torch.randint(
        0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=torch.Generator().manual_seed(seed)
    ).cuda()
    #: Ten-times-lower rates keep the KDA decay healthy over the 20+ steps this
    #: gate runs at the bank-heavy widths; both arms see the identical optimizer.
    optimizer = MixOptimizer(built.model, steps=1, lr_muon=0.002, lr_adamw=0.0003)
    return built, optimizer, ids


def step(built: object, optimizer: MixOptimizer, ids: torch.Tensor) -> object:
    """Run one whole optimizer step."""
    return built.step(ids, optimizer=optimizer)  # type: ignore[attr-defined]


def sync() -> None:
    """Wait for the card."""
    torch.cuda.synchronize()


def peak_mib(
    built: object, optimizer: MixOptimizer, ids: torch.Tensor, steps: int = 2
) -> tuple[float, Counters | None]:
    """Return peak allocated MiB and the last step's counters."""
    for _ in range(3):
        step(built, optimizer, ids)
    sync()
    torch.cuda.reset_peak_memory_stats()
    counters = None
    for _ in range(steps):
        _, counters = step(built, optimizer, ids)  # type: ignore[misc]
    sync()
    return torch.cuda.max_memory_allocated() / 2**20, counters


def timed_interleaved_ms(
    built: object, optimizer: MixOptimizer, ids: torch.Tensor, pairs: int
) -> tuple[float, float]:
    """Interleaved min-of-`pairs` ms, toggling storage on ONE model.

    Two live models do not fit beside each other at b8. Toggling the flag at the
    boundary is sound because `bank_codesave_on()` is read per forward, and both
    arms have identical weights and the same INT4 activation arithmetic.
    """
    for on in (False, True):
        os.environ[quant.BANK_FLAG] = "1" if on else "0"
        for _ in range(5):
            step(built, optimizer, ids)
    base_times: list[float] = []
    arm_times: list[float] = []
    for pair in range(pairs):
        order = (False, True) if pair % 2 == 0 else (True, False)
        for on in order:
            os.environ[quant.BANK_FLAG] = "1" if on else "0"
            sync()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step(built, optimizer, ids)
            end.record()
            sync()
            (arm_times if on else base_times).append(start.elapsed_time(end))
    return min(base_times), min(arm_times)


def heldout_bpb(spec: StackSpec, seed: int, steps: int) -> tuple[float, Counters]:
    """Train `steps` steps, then score one fixed held-out window."""
    built, optimizer, ids = build(spec, seed, smoke=True)
    counters = Counters()
    for _ in range(steps):
        result, counters = step(built, optimizer, ids)
        if result.skipped:  # type: ignore[attr-defined]
            raise RuntimeError(f"NaN firewall tripped at seed {seed}")
    held = torch.randint(
        0,
        spec.vocab,
        (spec.batch, spec.seq_len + 1),
        generator=torch.Generator().manual_seed(seed + 10_000),
    ).cuda()
    built.model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        logits = built.model(held[:, :-1])  # type: ignore[attr-defined]
        loss = chunked_cross_entropy(logits.float(), held[:, 1:])
    tokens = spec.batch * spec.seq_len
    bpb = from_token_nats(float(loss), eval_text=eval_text(tokens), token_count=tokens)
    return bpb.bpb, counters


def row(name: str, peak: float, ms: float, bpb: float, codes: int, verdict: str) -> str:
    """Format one Markdown table row."""
    return f"| {name} | {peak:.1f} | {ms:.2f} | {bpb:.4f} | {codes} | {verdict} |"


def main() -> int:
    """Measure both arms and print the gate table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--pairs", type=int, default=15)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA card: the peak/timing gate is a card measurement")
    if quant.bank_codesave_on():
        raise SystemExit("unset DORMOUSE_BANK_A4_CODESAVE: this tool toggles it per arm")
    #: Both arms carry the model's INT4 bank activation; only the storage form
    #: changes, so the BPB delta isolates the codesave boundary.
    os.environ[quant.BANK_QUANT_FLAG] = "1"

    print("variant|peak_MiB|ms|dBPB|codes|verdict")
    print("|---|---|---|---|---|---|")
    rows: list[str] = []
    peaks: dict[str, dict[int, float]] = {"base": {}, "arm": {}}
    timings: dict[str, dict[int, float]] = {"base": {}, "arm": {}}
    for batch in args.batches:
        arm_spec = spec(batch=batch)
        codes_by_arm = {"base": 0, "arm": 0}
        for key, on in (("base", False), ("arm", True)):
            os.environ[quant.BANK_FLAG] = "1" if on else "0"
            built, optimizer, ids = build(arm_spec, 7, smoke=False)
            p, last_counters = peak_mib(built, optimizer, ids)
            peaks[key][batch] = p
            if key == "base":
                base_ms, arm_ms = timed_interleaved_ms(built, optimizer, ids, args.pairs)
                timings["base"][batch], timings["arm"][batch] = base_ms, arm_ms
            codes_by_arm[key] = (
                int(last_counters.snapshot().get("bank_a4_codes_saved", 0))
                if on and last_counters is not None
                else 0
            )
            del built, optimizer, ids
            torch.cuda.empty_cache()
        for key, name in (("base", f"anchor b{batch} bf16"), ("arm", f"anchor b{batch} INT4")):
            rows.append(
                row(
                    name,
                    peaks[key][batch],
                    timings[key][batch],
                    float("nan"),
                    codes_by_arm[key],
                    "measured",
                )
            )

    deltas: list[float] = []
    for seed in range(args.seeds):
        os.environ[quant.BANK_FLAG] = "1"
        arm_bpb, arm_counters = heldout_bpb(spec(smoke=True), seed, args.steps)
        os.environ[quant.BANK_FLAG] = "0"
        base_bpb, _ = heldout_bpb(spec(smoke=True), seed, args.steps)
        delta = arm_bpb - base_bpb
        deltas.append(delta)
        codes = int(arm_counters.snapshot().get("bank_a4_codes_saved", 0))
        rows.append(
            row(
                f"paired seed {seed}",
                float("nan"),
                float("nan"),
                delta,
                codes,
                "PASS" if abs(delta) <= 0.01 else "dBPB RED",
            )
        )
    os.environ[quant.BANK_FLAG] = "0"

    mem_ok = True
    speed_ok = True
    for batch in args.batches:
        saved = 1 - peaks["arm"][batch] / peaks["base"][batch]
        slower = timings["arm"][batch] / timings["base"][batch] - 1
        mem_ok &= saved >= 0.10
        speed_ok &= slower <= 0.10
        rows.append(
            row(
                f"GATE b{batch}",
                peaks["arm"][batch],
                timings["arm"][batch],
                float("nan"),
                0,
                f"peak -{saved:.1%} ms +{slower:.1%} "
                f"{'PASS' if saved >= 0.10 and slower <= 0.10 else 'RED'}",
            )
        )
    print("\n".join(rows))
    print(f"dBPB={[round(d, 5) for d in deltas]} median={statistics.median(deltas):.5f}")
    quality_ok = all(math.isfinite(d) and abs(d) <= 0.01 for d in deltas)
    print(
        f"gates: peak {'PASS' if mem_ok else 'RED'}  ms {'PASS' if speed_ok else 'RED'}  "
        f"dBPB {'PASS' if quality_ok else 'RED'}"
    )
    return 0 if mem_ok and speed_ok and quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
