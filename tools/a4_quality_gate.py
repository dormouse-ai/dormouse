"""Paired A4-quality gate: bf16 control vs A4+ternary, from step 0, same seed.

No warm-start, no ramp. The two arms differ only in `act_quant`/`weight_qat`; the
init draw, the token ids and the round stream are bound to the same seed. Each variant
is a pure function of `(run_seed, step, A4-call index)`, so a paired run is repeatable.

    PYTHONPATH=<wt>/src:<core>/src python tools/a4_quality_gate.py \
        --variants near-absmax,sr-absmax,near-pc,sr-pc --steps 200 --seeds 3

The gate is |mean dBPB|<=0.01, upper95<=0.01 (paired t, n=3) and ms<=+10%.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
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
ARTIFACTS = ROOT.parent.parent / "research" / "solve" / "artifacts"
EVAL_TEXT = "Привет мир — evaluation window."
T95_N3 = 2.920  # Student t, 0.95 one-sided, 2 degrees of freedom
VARIANTS = ("near-absmax", "sr-absmax", "near-pc", "sr-pc")


def spec_for(*, a4: bool, batch: int = 2, seq: int = 512) -> StackSpec:
    """Build the gate's small stack at the A2 held-out measurement's widths."""
    raw = tomllib.loads((CONFIGS / "dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {f.name for f in fields(StackSpec) if f.init}
    values = {key: value for key, value in raw.items() if key in known}
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
        seq_len=seq,
        batch=batch,
        tokens=1_000_000,
        torch_compile=False,
        act_quant="a4" if a4 else "none",
        weight_qat="ternary_native" if a4 else "none",
    )
    return StackSpec.from_mapping(values)


def _flags(*, a4: bool, variant: str, seed: int, step: int) -> None:
    """Bind every mechanism flag before a model is built or a step is run."""
    os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = "1" if a4 else "0"
    os.environ["DORMOUSE_ACT_A4"] = "1" if a4 else "0"
    os.environ["DORMOUSE_A4_ROUND"] = variant.split("-")[0]
    os.environ["DORMOUSE_A4_SCALE"] = variant.split("-")[1]
    quant.set_step_seed(seed, step)


def _ids(spec: StackSpec, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator).to(
        device
    )


def _build(
    spec: StackSpec, seed: int, device: torch.device, lr_scale: float = 0.1
) -> tuple[object, MixOptimizer]:
    torch.manual_seed(seed)
    built = assemble(spec, smoke=True, device=device)
    optimizer = MixOptimizer(built.model, steps=1)
    #: At full LR the KDA decay projection on random ids trips its own alpha floor near
    #: step 80 on the unmodified stack; a paired A/B needs both arms to finish.
    optimizer.set_lr(lr_scale)
    return built, optimizer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed_step(
    built: object, optimizer: MixOptimizer, ids: torch.Tensor, device: torch.device
) -> float:
    _sync(device)
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        built.step(ids, optimizer=optimizer)  # type: ignore[attr-defined]
        end.record()
        _sync(device)
        return float(start.elapsed_time(end))
    before = time.perf_counter()
    built.step(ids, optimizer=optimizer)  # type: ignore[attr-defined]
    return (time.perf_counter() - before) * 1000.0


def _train(
    spec: StackSpec, seed: int, steps: int, device: torch.device, lr_scale: float = 0.1
) -> tuple[float, dict[str, int], list[dict[str, int | float | str]], bool]:
    """Train one arm, returning the held-out loss, counter snapshot, journal, flow flag."""
    built, optimizer = _build(spec, seed, device, lr_scale)
    ids = _ids(spec, seed, device)
    counters = Counters()
    journal: list[dict[str, int | float | str]] = []
    flow_ok = True
    for step in range(steps):
        quant.set_step_seed(seed, step)
        result, counters = built.step(ids, optimizer=optimizer)  # type: ignore[attr-defined]
        if result.skipped or not math.isfinite(result.loss):
            raise RuntimeError(
                f"NaN firewall (seed={seed}, step={step}, stage={result.bad_stage}): "
                f"loss={result.loss}; no parameter changed"
            )
        if step < 2:
            grads = [
                p.grad
                for p in built.model.parameters()  # type: ignore[attr-defined]
                if p.requires_grad and p.grad is not None
            ]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            nonzero = sum(float(g.abs().sum().item()) > 0.0 for g in grads)
            # Many bank experts are not routed in a given step: a zero grad is not a
            # missing gradient. Only non-finite grads and a wholly dead step are red.
            flow_ok &= finite and nonzero >= 1
        journal.extend(quant.take_round_journal())
    held = _ids(spec, seed + 10_000, device)
    built.model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        logits = built.model(held[:, :-1])  # type: ignore[attr-defined]
        loss = float(chunked_cross_entropy(logits.float(), held[:, 1:]))
    if not math.isfinite(loss):
        raise RuntimeError(f"held-out loss is {loss} for seed {seed}")
    if spec.act_quant == "a4":
        flow_ok &= counters.counts.get("a4_bwd", 0) > 0
        flow_ok &= counters.counts.get("tsct_ternary_bwd", 0) > 0
        if quant.scale_mode() == "pc":
            flow_ok &= counters.counts.get("a4_scale_grad", 0) > 0
    del built, optimizer, ids, held, logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss, dict(counters.snapshot()), journal, flow_ok


def _bpb(loss: float, spec: StackSpec) -> float:
    return float(
        from_token_nats(loss, eval_text=EVAL_TEXT, token_count=spec.batch * spec.seq_len).bpb
    )


def _timing(
    variant: str, seed: int, device: torch.device, lr_scale: float = 0.1
) -> dict[str, float]:
    """Interleaved min-of-15 A/B; both arms share ids and warming."""
    _flags(a4=False, variant=variant, seed=seed, step=0)
    control_spec = spec_for(a4=False)
    base, base_opt = _build(control_spec, seed, device, lr_scale)
    _flags(a4=True, variant=variant, seed=seed, step=0)
    arm_spec = spec_for(a4=True)
    arm, arm_opt = _build(arm_spec, seed, device, lr_scale)
    ids = _ids(control_spec, seed, device)
    # The native-ternary flag is process-global: the control's fp32 factors refuse a
    # forward while it is on, so flip it around each interleaved step.
    for step in range(5):
        _flags(a4=False, variant=variant, seed=seed, step=step)
        _timed_step(base, base_opt, ids, device)
        _flags(a4=True, variant=variant, seed=seed, step=step)
        _timed_step(arm, arm_opt, ids, device)
    base_ms, arm_ms = [], []
    for step in range(5, 20):
        _flags(a4=False, variant=variant, seed=seed, step=step)
        base_ms.append(_timed_step(base, base_opt, ids, device))
        _flags(a4=True, variant=variant, seed=seed, step=step)
        arm_ms.append(_timed_step(arm, arm_opt, ids, device))
    del base, base_opt, arm, arm_opt, ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "control_ms": min(base_ms),
        "arm_ms": min(arm_ms),
        "overhead": min(arm_ms) / min(base_ms) - 1.0,
    }


def main() -> int:
    """Measure every requested variant and print the gate table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)  # bank index_add_ is thread-order dependent
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("variant|dBPB|CI95|ms|verdict", flush=True)
    print("---|---|---|---|---|", flush=True)
    results: dict[str, dict[str, object]] = {}
    for variant in variants:
        deltas: list[float] = []
        rows: list[dict[str, object]] = []
        overhead = 0.0
        flow = True
        journal: list[dict[str, int | float | str]] = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            _flags(a4=False, variant=variant, seed=seed, step=0)
            control_spec = spec_for(a4=False)
            base_loss, _, _, _ = _train(control_spec, seed, args.steps, device, args.lr_scale)
            _flags(a4=True, variant=variant, seed=seed, step=0)
            arm_spec = spec_for(a4=True)
            arm_loss, counters, one_journal, one_flow = _train(
                arm_spec, seed, args.steps, device, args.lr_scale
            )
            delta = _bpb(arm_loss, arm_spec) - _bpb(base_loss, control_spec)
            deltas.append(delta)
            journal.extend(one_journal)
            flow &= one_flow
            rows.append(
                {
                    "seed": seed,
                    "control_nats": base_loss,
                    "arm_nats": arm_loss,
                    "dbpb": delta,
                    "counters": {k: v for k, v in counters.items() if "a4" in k or "ternary" in k},
                }
            )
            print(
                f"  {variant} seed {seed}: dBPB={delta:+.5f} "
                f"(nats {base_loss:.4f}->{arm_loss:.4f})",
                flush=True,
            )
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        margin = T95_N3 * sd / math.sqrt(len(deltas))
        upper = mean + margin
        if args.timing:
            timing = _timing(variant, args.seed_start, device, args.lr_scale)
            overhead = float(timing["overhead"])
            ms = f"+{100 * overhead:.1f}% ({timing['control_ms']:.0f}->{timing['arm_ms']:.0f})"
        else:
            ms = "unmeasured"
        passed = abs(mean) <= 0.01 and upper <= 0.01 and (not args.timing or overhead <= 0.10)
        verdict = "PASS" if passed and flow else ("FLOW_RED" if not flow else "RED")
        print(
            f"| {variant} | {mean:+.5f} | [{mean - margin:+.5f},{upper:+.5f}] | {ms} | {verdict} |",
            flush=True,
        )
        results[variant] = {
            "deltas": deltas,
            "mean_dbpb": mean,
            "ci95": [mean - margin, upper],
            "ms_overhead": overhead if args.timing else None,
            "flow_ok": flow,
            "verdict": verdict,
            "rows": rows,
        }
        (ARTIFACTS / f"a4_quality_{variant}.json").write_text(
            json.dumps(results[variant], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (ARTIFACTS / f"a4_quality_rounds_{variant}.jsonl").write_text(
            "\n".join(json.dumps(event) for event in journal) + "\n", encoding="utf-8"
        )
    print(
        f"gate={'PASS' if all(r['verdict'] == 'PASS' for r in results.values()) else 'RED'}",
        flush=True,
    )
    return 0 if all(r["verdict"] == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
