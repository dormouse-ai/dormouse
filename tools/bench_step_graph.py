"""One captured and replayed 1B step, measured against eager on the live config.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_step_graph.py \
        --mode all --batch 8

WHAT IT PRINTS. Three eager ms/step and three replayed ms/step at the config's widths,
tokens/s and bytes of source per second (4.718 B/token), peak VRAM and the graph's private
pool, MFU against this engine's own 407 MFLOP/token bill, and the ratio to the old engine's
22.9 kB/s. Parity at the SAME shape compares loss, every logit, every gradient and every
post-step parameter of the replay against the eager step.

NO SILENT EAGER. A failed capture raises LoudError naming the op. `graph_replays` must be
positive on every measured run or the tool exits non-zero; a declared graph that replayed
nothing is the defect class Counters.check() exists for.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_kernels.engine import bank as kb
from profile_step import PRACTICAL_PEAK_TFLOPS as BASELINE_PEAK
from profile_step import phases

from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train.graph import MARGIN_BYTES, POOL_FRACTION, StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import CHUNK_TOKENS, chunked_cross_entropy

MIB = 2**20
#: Source-text bytes per token from the corpus manifest — the unit the speed gate quotes.
SOURCE_BYTES_PER_TOKEN = 4.718
#: The engine's own FLOP bill per token, and the card's practical bf16 peak.
MFLOPS_PER_TOKEN = 407.0
PRACTICAL_PEAK_TFLOPS = 52.0
assert BASELINE_PEAK == PRACTICAL_PEAK_TFLOPS, "the two peak constants must agree"
#: The engine this one replaces, measured on this card, in bytes of source per second.
OLD_ENGINE_BYTES_PER_S = 22_900.0
#: bf16 has 8 mantissa bits: 2**-8 is one ulp of relative resolution at unit scale.
BF16_EPS = 2.0**-8
#: Learning-rate fraction for parity: weights must actually move for the post-step test.
PARITY_LR_FRACTION = 0.1
SEED = 3
ARTIFACT = Path(__file__).resolve().parents[1] / "runs" / "step_graph_1b" / "measurement.json"


def spec_for(config: str, batch: int, seq: int) -> StackSpec:
    """Read the real config and override only batch and sequence — no model change."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    return replace(spec, batch=batch, seq_len=seq or spec.seq_len)


def make_ids(spec: StackSpec) -> torch.Tensor:
    """One fixed, seeded batch of token ids, on the card."""
    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator)
    return ids.to("cuda")


def timed_ms(call: Any, runs: int) -> list[float]:
    """Wall milliseconds of `runs` synchronized calls, each timed on its own."""
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    out = []
    for _ in range(runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - started) * 1e3)
    return out


def eager_body(
    model: torch.nn.Module, optimizer: MixOptimizer, ids: torch.Tensor, *, amp: bool = True
) -> torch.Tensor:
    """Run the graph's captured half eagerly: zero_grad, forward, chunked CE, backward."""
    optimizer.zero_grad(set_to_none=False)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(inputs, None)
        loss = chunked_cross_entropy(logits, targets, chunk=CHUNK_TOKENS)
        loss.backward()  # type: ignore[no-untyped-call]
    return loss.detach()


def eager_step(model: torch.nn.Module, optimizer: MixOptimizer, ids: torch.Tensor) -> None:
    """One full eager step: `eager_body`, then the eager half `StepGraph.run` also owns."""
    eager_body(model, optimizer, ids)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def hook_logits(model: torch.nn.Module, buf: torch.Tensor) -> None:
    """Copy the tied head's logits into `buf` on every forward — no allocation in capture."""

    def head(x: torch.Tensor, counters: object = None) -> torch.Tensor:
        """Copy the model's head output into the fixed probe buffer."""
        out = model.__class__.head(model, x, counters)  # type: ignore[arg-type, no-any-return]
        buf.copy_(out)
        return out

    model.head = head  # type: ignore[method-assign]


def _diff(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Deviation of two CPU fp32 snapshots, absolute and relative to the tensor's scale.

    A plain per-element relative error explodes where the true value is ~0: one bf16 ULP
    of the terms being cancelled is enormous against a near-zero sum. ``rel_to_scale``
    divides by the tensor's own largest magnitude instead, which is the resolution bf16
    actually has at that tensor's working range.
    """
    delta = (a - b).abs()
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
    rel = delta / (a.abs() + 1e-8)
    return {
        "name": name,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel": float(rel.max()) if rel.numel() else 0.0,
        "rel_to_scale": float(delta.max()) / scale if delta.numel() else 0.0,
        "frac_diff": float((delta > 0).float().mean()) if delta.numel() else 0.0,
        "equal": bool(torch.equal(a, b)),
    }


def parity(
    spec: StackSpec, *, capture: bool = True, control: bool = False, amp: bool = True
) -> dict[str, Any]:
    """Compare the captured half with eager: loss, logits, grads, post-step params.

    With `capture=False` the same body runs without the CUDA graph object, so the
    fixed-pass arithmetic can still be checked when a capture is refused by the card.
    """
    torch.manual_seed(SEED)
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    ids = make_ids(spec)
    init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    shape = (spec.batch, spec.seq_len, spec.vocab)
    logits_dtype = torch.bfloat16 if amp else torch.float32

    # --- eager half: one step from the fresh weights, snapshotted to CPU
    eager_opt = MixOptimizer(model)
    eager_opt.set_lr(PARITY_LR_FRACTION)
    logits_buf = torch.empty(shape, dtype=logits_dtype, device="cuda")
    hook_logits(model, logits_buf)
    loss_e = eager_body(model, eager_opt, ids, amp=amp)
    logits_e = logits_buf.detach().float().cpu().clone()
    grads_e = {
        n: p.grad.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    eager_opt.step()
    params_e = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    del logits_buf, eager_opt
    torch.cuda.empty_cache()

    # --- replayed half: same weights, same ids, a fresh optimizer of the same kind
    # `load_state_dict` copies CPU tensor into each live parameter one at a time; building
    # a GPU dict first would hold a second copy of the whole bank and OOM at this shape.
    model.load_state_dict(init)  # type: ignore[arg-type]
    graph_opt = MixOptimizer(model)
    graph_opt.set_lr(PARITY_LR_FRACTION)
    logits_buf = torch.empty(shape, dtype=logits_dtype, device="cuda")
    hook_logits(model, logits_buf)
    counters = Counters()
    if capture:
        counters.enable("graph", counter="graph_replays", why="the step must replay a graph")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            graph = StepGraph(model, graph_opt, ids)
        graph.replay(ids, counters=counters)
        loss_g = graph.loss()
        logits_g = logits_buf.detach().float().cpu().clone()
        grads_g = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        graph.run(ids, counters=counters)
        replays = graph.replays()
    elif control:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss_g = float(eager_body(model, graph_opt, ids, amp=amp).detach())
        logits_g = logits_buf.detach().float().cpu().clone()
        grads_g = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        graph_opt.step()
        replays = 0
    else:
        graph = StepGraph.__new__(StepGraph)
        graph.model, graph.optimizer, graph.accum = model, graph_opt, 1
        graph.rows, graph.seq = spec.batch, spec.seq_len
        graph.ids = ids.clone()
        graph._loss = torch.zeros((), device="cuda")
        graph._graph = None
        graph._replays = 0
        graph.failed_on = None
        graph._banks = [m for m in model.modules() if isinstance(m, MoEFFN)]
        for bank in graph._banks:
            bank.fixed_passes = MAX_DISPATCH_PASSES
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss_g = float(graph._body().detach())
        logits_g = logits_buf.detach().float().cpu().clone()
        grads_g = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        graph_opt.step()
        replays = 0
    params_g = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    if capture:
        counters.check()

    if set(grads_e) != set(grads_g):
        raise LoudError(
            f"the two paths own different gradient sets: {set(grads_e) ^ set(grads_g)}",
            "a captured step must not grow or drop a gradient",
        )
    grad_rows = [_diff(n, grads_g[n], grads_e[n]) for n in sorted(grads_e)]
    param_rows = [_diff(n, params_g[n], params_e[n]) for n in sorted(params_e)]
    logits_row = _diff("logits", logits_g, logits_e)
    worst_rel = max(
        [float(r["rel_to_scale"]) for r in [*grad_rows, *param_rows, logits_row]],
        default=0.0,
    )
    out: dict[str, Any] = {
        "shape": [spec.batch, spec.seq_len, spec.vocab],
        "loss_eager": float(loss_e),
        "loss_replay": float(loss_g),
        "loss_abs_diff": abs(float(loss_e) - loss_g),
        "logits": logits_row,
        "grads": grad_rows,
        "params": param_rows,
        "grad_tensors": len(grad_rows),
        "param_tensors": len(param_rows),
        "worst_rel": worst_rel,
        "worst_per_element_rel": max(
            [float(r["max_rel"]) for r in [*grad_rows, *param_rows, logits_row]], default=0.0
        ),
        "bf16_epsilon": BF16_EPS,
        "within_bf16_epsilon": worst_rel <= BF16_EPS,  # one ULP at working scale
        "worst_fraction_changed": max(
            [float(r["frac_diff"]) for r in [*grad_rows, *param_rows]], default=0.0
        ),
        "graph_replays": replays,
        "captured": capture,
        "grad_names_with_diff": [r["name"] for r in grad_rows if not r["equal"]],
    }
    print(
        f"parity at b{spec.batch} x s{spec.seq_len}: loss {out['loss_eager']:.6f} vs "
        f"{out['loss_replay']:.6f} (abs {out['loss_abs_diff']:.3g}), "
        f"{len(grad_rows)} grads + {len(param_rows)} params + logits, "
        f"worst rel {worst_rel:.3g} (bf16 eps {BF16_EPS:.3g}), "
        f"lossless={out['within_bf16_epsilon']}, replays={replays}, "
        f"captured={capture}"
    )
    if out["grad_names_with_diff"]:
        print(f"  grads that differ: {', '.join(out['grad_names_with_diff'][:6])} ...")
    return out


def speed(spec: StackSpec, steps: int) -> dict[str, Any]:
    """Eager vs replayed ms/step at the live shape, with pool, VRAM and counters."""
    torch.manual_seed(SEED)
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    optimizer = MixOptimizer(model)
    optimizer.set_lr(0.0)  # frozen, as in the honest 3.9 s baseline: same kernels, no drift
    ids = make_ids(spec)
    bill = assembled.report()
    free_before, total = torch.cuda.mem_get_info()
    need = max(MARGIN_BYTES, int(POOL_FRACTION * total))
    print(
        f"card {torch.cuda.get_device_name()} {total / MIB:,.0f} MiB, "
        f"{free_before / MIB:,.0f} MiB free; need max(384 MiB, 8%)={need / MIB:,.0f} MiB; "
        f"bank path {'fused' if kb.bank_enabled() else 'ANCHOR'}"
    )

    #: The eager half is `profile_step.py`'s own `phases`, the function that produced the
    #: 3,915.6 ms baseline — same zero_grad, same bf16 autocast, same clip and arms.
    for _ in range(2):
        phases(assembled, optimizer, ids, 1, True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    eager = [phases(assembled, optimizer, ids, 1, True)["total"] for _ in range(steps)]
    peak_eager = torch.cuda.max_memory_allocated() / MIB
    print(f"eager  ms/step ({steps} runs): {[round(v, 1) for v in eager]}")

    # Capture under the same bf16 autocast the baseline's forward/backward ran in.
    # Free the eager run's grads and cached blocks first: a trainer captures before its
    # first timed step, not on top of three steps' allocator high-water mark.
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            graph = StepGraph(model, optimizer, ids)
    except (LoudError, RuntimeError, torch.OutOfMemoryError) as exc:
        failed = str(exc).splitlines()[0]
        print(f"capture FAILED (no eager fallback): {failed}")
        tokens = spec.batch * spec.seq_len
        rate = tokens / (statistics.median(eager) * 1e-3)
        return {
            "batch": spec.batch,
            "seq": spec.seq_len,
            "tokens_per_step": tokens,
            "active_params": bill["active"],
            "eager_ms_runs": eager,
            "eager_ms": statistics.median(eager),
            "eager_tokens_per_s": rate,
            "eager_bytes_per_s": rate * SOURCE_BYTES_PER_TOKEN,
            "eager_mfu_percent": 100
            * MFLOPS_PER_TOKEN
            * 1e6
            * rate
            / (PRACTICAL_PEAK_TFLOPS * 1e12),
            "replay_ms_runs": [],
            "capture_failed": True,
            "capture_failed_on": failed,
            "peak_eager_mib": peak_eager,
            "free_before_mib": free_before / MIB,
            "headroom_needed_mib": need / MIB,
            "graph_replays": 0,
            "bank_path": "fused" if kb.bank_enabled() else "anchor",
        }
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    pool = free0 - free1

    counters = Counters()
    counters.enable("graph", counter="graph_replays", why="the step must replay a graph")
    torch.cuda.reset_peak_memory_stats()
    replayed = timed_ms(lambda: graph.run(ids, counters=counters), steps)
    counters.check()
    peak_replay = torch.cuda.max_memory_allocated() / MIB
    free_after, _ = torch.cuda.mem_get_info()
    print(f"replay  ms/step ({steps} runs): {[round(v, 1) for v in replayed]}")

    tokens = spec.batch * spec.seq_len
    eager_med = statistics.median(eager)
    replay_med = statistics.median(replayed)
    rate = tokens / (replay_med * 1e-3)
    bytes_s = rate * SOURCE_BYTES_PER_TOKEN
    mfu = 100 * MFLOPS_PER_TOKEN * 1e6 * rate / (PRACTICAL_PEAK_TFLOPS * 1e12)
    out: dict[str, Any] = {
        "batch": spec.batch,
        "seq": spec.seq_len,
        "tokens_per_step": tokens,
        "active_params": bill["active"],
        "eager_ms_runs": eager,
        "replay_ms_runs": replayed,
        "eager_ms": eager_med,
        "replay_ms": replay_med,
        "speedup": eager_med / replay_med,
        "tokens_per_s": rate,
        "bytes_per_s": bytes_s,
        "mfu_percent": mfu,
        "old_engine_bytes_per_s": OLD_ENGINE_BYTES_PER_S,
        "ratio_to_old": bytes_s / OLD_ENGINE_BYTES_PER_S,
        "peak_eager_mib": peak_eager,
        "peak_replay_mib": peak_replay,
        "pool_bytes": pool,
        "pool_mib": pool / MIB,
        "headroom_needed_bytes": need,
        "headroom_needed_mib": need / MIB,
        "free_after_mib": free_after / MIB,
        "graph_replays": graph.replays(),
        "graph_replays_counter": counters.snapshot()["graph_replays"],
        "bank_path": "fused" if kb.bank_enabled() else "anchor",
    }
    print(
        f"eager {eager_med:,.1f} ms  replay {replay_med:,.1f} ms  "
        f"speedup {out['speedup']:.2f}x  {rate:,.0f} tok/s  {bytes_s:,.0f} B/s  "
        f"MFU {mfu:.2f}%  vs old {out['ratio_to_old']:.2f}x\n"
        f"peak {peak_eager:,.0f}->{peak_replay:,.0f} MiB  pool {pool / MIB:,.0f} MiB "
        f"(need {need / MIB:,.0f})  free after {free_after / MIB:,.0f} MiB  "
        f"graph_replays={graph.replays()}"
    )
    if graph.replays() <= 0 or out["graph_replays_counter"] <= 0:
        raise LoudError(
            "a declared graph replayed zero times",
            "the counter is the proof the captured path ran; zero means the eager step "
            "was measured under the graph's name",
        )
    return out


def capture_only(spec: StackSpec) -> dict[str, Any]:
    """Capture in a fresh process with no prior steps: isolate the blocker."""
    torch.manual_seed(SEED)
    assembled = assemble(spec, device="cuda")
    model = assembled.model
    optimizer = MixOptimizer(model)
    optimizer.set_lr(0.0)
    ids = make_ids(spec)
    free0, total = torch.cuda.mem_get_info()
    need = max(MARGIN_BYTES, int(POOL_FRACTION * total))
    print(
        f"fresh capture at b{spec.batch} x s{spec.seq_len}: {free0 / MIB:,.0f} MiB free, "
        f"need {need / MIB:,.0f} MiB headroom; bank {'fused' if kb.bank_enabled() else 'anchor'}"
    )
    out: dict[str, Any] = {"batch": spec.batch, "free_before_mib": free0 / MIB}
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            graph = StepGraph(model, optimizer, ids)
    except (LoudError, RuntimeError, torch.OutOfMemoryError) as exc:
        out["captured"] = False
        out["failed_on"] = str(exc).splitlines()[0]
        print(f"capture FAILED (no eager fallback): {out['failed_on']}")
        return out
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    out.update(
        {
            "captured": True,
            "pool_mib": (free0 - free1) / MIB,
            "peak_mib": torch.cuda.max_memory_allocated() / MIB,
            "replays": graph.replays(),
        }
    )
    print(
        f"captured: pool {(free0 - free1) / MIB:,.0f} MiB, "
        f"peak {out['peak_mib']:,.0f} MiB, free after {free1 / MIB:,.0f} MiB"
    )
    return out


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the worst five deviations in the artifact, not megabytes of tensor dumps."""
    out = dict(result)
    if "parity" in out:
        parity_row = dict(out["parity"])
        for key in ("grads", "params"):
            parity_row[key] = sorted(parity_row[key], key=lambda row: -row["rel_to_scale"])[:5]
        out["parity"] = parity_row
    return out


def main() -> int:
    """Measure the step graph on the real config; exit non-zero if a gate fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--fp32", action="store_true", help="parity diagnostic: no autocast")
    parser.add_argument(
        "--mode",
        choices=("all", "speed", "parity", "body-parity", "control", "capture"),
        default="all",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device: a captured step is a card-side measurement", file=sys.stderr)
        return 2
    spec = spec_for(args.config, args.batch, args.seq)
    free, total = torch.cuda.mem_get_info()
    print(
        f"config {args.config}: b{spec.batch} x s{spec.seq_len}, "
        f"d{spec.d_model} f{spec.d_ffn} E{spec.n_experts}; "
        f"card load before: {(total - free) / MIB:,.0f}/{total / MIB:,.0f} MiB"
    )
    result: dict[str, Any] = {"config": args.config, "mode": args.mode}
    if args.mode == "capture":
        result["capture"] = capture_only(spec)
        artifact = ARTIFACT.parent / f"capture_b{args.batch}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(compact(result), indent=2), encoding="utf-8")
        print(f"wrote {artifact}")
        return 0 if result["capture"]["captured"] else 1
    if args.mode in ("all", "parity"):
        result["parity"] = parity(spec, amp=not args.fp32)
    if args.mode == "body-parity":
        result["parity"] = parity(spec, capture=False, amp=not args.fp32)
    if args.mode == "control":
        result["parity"] = parity(spec, capture=False, control=True, amp=not args.fp32)
    if args.mode in ("all", "speed"):
        result["speed"] = speed(spec, args.steps)
    artifact = ARTIFACT.parent / f"{args.mode}_b{args.batch}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(compact(result), indent=2), encoding="utf-8")
    print(f"wrote {artifact}")
    ok = True
    if "parity" in result:
        ok = ok and bool(result["parity"]["within_bf16_epsilon"])
    if "speed" in result:
        ok = ok and int(result["speed"]["graph_replays"]) > 0
    if "capture" in result:
        ok = ok and bool(result["capture"]["captured"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
