"""The decisive speed experiment: one shipped 1B step, compiled, against the old engine.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/decisive_bench.py \
        --allow-pid <desktop-pid>

Every earlier comparison was confounded: a different batch, a stale commit, a 52 vs 92
TFLOP/s peak, a three-step profiler against a thousand-step journal. This script prints
its own commit, config hash, peak constant and batch, proves parity against eager, proves
inductor built a graph, and only then prints ms/step. A refusal is a result; an eager
timing wearing a compiled label is not.

The shipped config keeps torch_compile=false: compile is proven on CPU-constructible shapes
by tests/test_compiled.py, and this script overrides the flag to true for the measurement.
The historical live-shape blockers are recorded in
docs/records/2026-11-16-step-graph-live-shape.md:89-101; the fused bank's static dispatch
(src/dormouse/model/moe.py:570-573) removed the `.item()` break, and any remaining failure
is named by compiled.require_graph() instead of being reported as a speed.

No CUDA call happens at import. The card guard runs first and exits before a model is
built. CPU-only authoring and tests never touch the card.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from bench_step_graph import BF16_EPS, _diff, hook_logits
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.optim import MixOptimizer

#: Source-text bytes per token — the manifest's measured BPT, not an assumed 4.4.
SOURCE_BYTES_PER_TOKEN = 4.718
#: The engine's own FLOP bill per token, printed alongside the card peak it is measured on.
MFLOP_PER_TOKEN = 407.0
#: The MEASURED card peak, not the 52.0 "practical" constant in profile_step.py.
CARD_PEAK_TFLOPS = 92.0
#: The old engine at the shipped shape: 5.06 s for one 24,576-token step.
OLD_ENGINE_MS_PER_STEP = 5060.0
OLD_ENGINE_TOKENS_PER_STEP = 24_576
OLD_ENGINE_BYTES_PER_S = (
    OLD_ENGINE_TOKENS_PER_STEP / (OLD_ENGINE_MS_PER_STEP / 1000.0) * SOURCE_BYTES_PER_TOKEN
)
#: Warm-up, measured steps per repetition, and repetitions — fixed, never profiler-sized.
WARMUP_STEPS = 5
MEASURE_STEPS = 30
REPETITIONS = 3
#: The shipped shape first; a named drop on OOM, never a silent config edit.
BATCH_LADDER: tuple[int, ...] = (24, 16, 8)
SEQ_LEN = 1024
SEED = 3
#: A learning-rate fraction for parity: weights must actually move for the post-step test.
PARITY_LR_FRACTION = 0.1
LOCK = Path("/tmp/dormouse-gpu.lock")
LOCK_WAIT_S = 300.0
ARTIFACT = Path(__file__).resolve().parents[1] / "runs" / "decisive_bench" / "measurement.json"


def bytes_per_s(tokens_per_s: float, bpt: float = SOURCE_BYTES_PER_TOKEN) -> float:
    """Source-text bytes/s behind a token rate — the conversion the earlier reports fumbled."""
    if tokens_per_s < 0 or bpt <= 0:
        raise LoudError(
            f"tokens/s={tokens_per_s} with {bpt} bytes/token is not a rate",
            "a negative rate or a non-positive BPT cannot describe source text",
        )
    return tokens_per_s * bpt


def mfu_percent(tokens_per_s: float, peak_tflops: float = CARD_PEAK_TFLOPS) -> float:
    """MFU of the engine's 407 MFLOP/token bill against the measured card peak."""
    if peak_tflops <= 0:
        raise LoudError(f"peak {peak_tflops} TFLOP/s is not a peak", "use a positive peak")
    return 100.0 * MFLOP_PER_TOKEN * 1e6 * tokens_per_s / (peak_tflops * 1e12)


def old_engine_ms(tokens: int) -> float:
    """Return the old engine's time at `tokens`, scaled from its 24,576-token step."""
    if tokens <= 0:
        raise LoudError(f"{tokens} tokens is not a step", "the step must carry tokens")
    return OLD_ENGINE_MS_PER_STEP * tokens / OLD_ENGINE_TOKENS_PER_STEP


def busy_pids(smi_output: str, allowed: set[int]) -> list[int]:
    """Compute PIDs from `nvidia-smi` that are not on the desktop allow-list."""
    found = []
    for line in smi_output.splitlines():
        first = line.strip().split(",")[0].strip()
        if not first or first.startswith("No running"):
            continue
        try:
            pid = int(first)
        except ValueError as exc:
            raise LoudError(
                f"nvidia-smi returned an unreadable PID column: {first!r}",
                "expected --query-compute-apps=pid --format=csv,noheader",
            ) from exc
        if pid not in allowed:
            found.append(pid)
    return found


def spec_for(config: str, batch: int) -> tuple[StackSpec, bool]:
    """Read the shipped config, override only batch/seq/compile, and say what was overridden."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    flipped = not spec.torch_compile
    return replace(spec, batch=batch, seq_len=SEQ_LEN, torch_compile=True), flipped


def config_hash(spec: StackSpec) -> str:
    """Hash the resolved config, including the overrides, so the number identifies the run."""
    resolved = {f.name: getattr(spec, f.name) for f in fields(spec) if f.init}
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def _git(*args: str) -> str:
    """Return one git fact from the worktree the script was run in."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1]
    ).stdout.strip()


def provenance(spec: StackSpec, config: str, flipped: bool) -> dict[str, Any]:
    """Print and return the identity of the run before any card work starts."""
    bill = assemble(replace(spec, torch_compile=False), device="meta").report()
    out: dict[str, Any] = {
        "commit": _git("rev-parse", "HEAD"),
        "config": config,
        "config_hash": config_hash(spec),
        "compile_flag": spec.torch_compile,
        "compile_overridden_from_config": flipped,
        "param_report": bill,
        "batch": spec.batch,
        "seq": spec.seq_len,
        "dtype": "bf16 autocast",
        "peak_tflops": CARD_PEAK_TFLOPS,
        "bytes_per_token": SOURCE_BYTES_PER_TOKEN,
        "flops_per_token": MFLOP_PER_TOKEN,
    }
    print(f"commit {out['commit']}  config {config}  hash {out['config_hash']}", flush=True)
    print(
        f"params stored={bill['stored']:,} active={bill['active']:,} "
        f"active_ffn={bill['active_ffn']:,}",
        flush=True,
    )
    print(
        f"shape b{spec.batch} x s{spec.seq_len} ({spec.batch * spec.seq_len:,} tokens)  "
        f"dtype {out['dtype']}  torch_compile={spec.torch_compile}"
        + (" (overridden from false)" if flipped else ""),
        flush=True,
    )
    print(
        f"peak {CARD_PEAK_TFLOPS} TFLOP/s (measured), bill {MFLOP_PER_TOKEN:,.0f} MFLOP/token, "
        f"BPT {SOURCE_BYTES_PER_TOKEN}",
        flush=True,
    )
    return out


def _is_oom(exc: RuntimeError) -> bool:
    """Only a real out-of-memory may trigger the batch drop; any other error is a refusal."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def make_ids(spec: StackSpec) -> torch.Tensor:
    """One fixed, seeded batch of ids on the card."""
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator).to(
        "cuda"
    )


def measure(spec: StackSpec) -> dict[str, Any]:
    """Warm up 5 steps, time 30 steps three times with CUDA events, and prove a graph ran."""
    torch.manual_seed(SEED)
    assembled = assemble(spec, device="cuda")
    optimizer = MixOptimizer(assembled.model)
    optimizer.set_lr(0.0)  # frozen: the baseline's own regime, same kernels and shapes
    ids = make_ids(spec)
    seen: dict[str, Any] = {}

    def call() -> None:
        """One compiled step under the shipped bf16 autocast."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, counters = assembled.step(ids, optimizer=optimizer)
        seen["loss"] = float(result.loss)
        seen["counters"] = counters.snapshot()

    try:
        with compiled.compile_budget():  # compilation is lazy: it lands on this first call
            call()
    except RuntimeError as exc:
        if _is_oom(exc):
            raise
        raise LoudError(
            f"torch.compile failed on the first step: {str(exc).splitlines()[0]}",
            "there is no eager fallback: a failed compile must not report eager ms/step under "
            "the compiled name",
        ) from exc
    for _ in range(WARMUP_STEPS - 1):
        call()
    proof = compiled.require_graph()  # named refusal when inductor built no graph

    torch.cuda.synchronize()
    reps: list[list[float]] = []
    for _ in range(REPETITIONS):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        reps.append([start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)])
    medians = [statistics.median(rep) for rep in reps]
    median_ms = statistics.median(medians)
    compiled.reset()  # leave the global compile flag clean for the eager parity half
    tokens = spec.batch * spec.seq_len
    rate = tokens / (median_ms * 1e-3)
    return {
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "repetitions": REPETITIONS,
        "rep_median_ms": medians,
        "all_step_ms": reps,
        "median_ms": median_ms,
        "tokens_per_step": tokens,
        "tokens_per_s": rate,
        "bytes_per_s": bytes_per_s(rate),
        "mfu_percent": mfu_percent(rate),
        "mfu_percent_at_52": mfu_percent(rate, peak_tflops=52.0),
        "compiled_graphs": proof["unique_graphs"],
        "compile_graph_break": proof["graph_break"],
        "compile_graphs_counter": seen["counters"].get(compiled.COUNTER, 0),
        "graph_replays": seen["counters"].get("graph_replays", 0),
        "loss_last": seen["loss"],
    }


class _NoBudget:
    """A no-op context manager so eager and compiled snapshots share one call site."""

    def __enter__(self) -> None:
        """Nothing to bound on the eager path."""
        return

    def __exit__(self, *exc: object) -> None:
        """Nothing to restore on the eager path."""
        return


def parity(spec: StackSpec) -> dict[str, Any]:
    """Compare compiled against eager: loss, logits, every gradient and post-step params."""
    compiled.reset()
    torch.manual_seed(SEED)
    assembled = assemble(replace(spec, torch_compile=False), device="cuda")
    model = assembled.model
    ids = make_ids(spec)
    init = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    shape = (spec.batch, spec.seq_len, spec.vocab)
    logits_buf = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    hook_logits(model, logits_buf)

    def snapshot(*, use_compile: bool) -> dict[str, Any]:
        """One step from the same initial weights, eager or compiled."""
        model.load_state_dict(init)  # type: ignore[arg-type]
        optimizer = MixOptimizer(model)
        optimizer.set_lr(PARITY_LR_FRACTION)
        proof: dict[str, Any] = {}
        if use_compile:
            for module in model.modules():
                if isinstance(module, MoEFFN):
                    module.fixed_passes = MAX_DISPATCH_PASSES  # no host read in the graph
            assembled.spec = replace(assembled.spec, torch_compile=True)
            compiled.enable(model)
        context = compiled.compile_budget() if use_compile else _NoBudget()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = assembled.step(ids, optimizer=optimizer)
        if use_compile:
            proof = compiled.require_graph()
        return {
            "loss": float(result.loss),
            "logits": logits_buf.detach().float().cpu().clone(),
            "grads": {
                name: p.grad.detach().float().cpu().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            },
            "params": {
                name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
            },
            "proof": proof,
        }

    eager = snapshot(use_compile=False)
    compiled_row = snapshot(use_compile=True)
    if set(eager["grads"]) != set(compiled_row["grads"]):
        raise LoudError(
            "the two paths own different gradients: "
            f"{set(eager['grads']) ^ set(compiled_row['grads'])}",
            "a compiled step must not grow or drop a gradient",
        )
    grad_rows = [
        _diff(name, compiled_row["grads"][name], eager["grads"][name])
        for name in sorted(eager["grads"])
    ]
    param_rows = [
        _diff(name, compiled_row["params"][name], eager["params"][name])
        for name in sorted(eager["params"])
    ]
    logits_row = _diff("logits", compiled_row["logits"], eager["logits"])
    worst = max(
        [float(row["rel_to_scale"]) for row in [*grad_rows, *param_rows, logits_row]],
        default=0.0,
    )
    out: dict[str, Any] = {
        "loss_eager": eager["loss"],
        "loss_compiled": compiled_row["loss"],
        "loss_abs_diff": abs(eager["loss"] - compiled_row["loss"]),
        "logits": logits_row,
        "worst_rel_to_scale": worst,
        "bf16_epsilon": BF16_EPS,
        "within_bf16_epsilon": worst <= BF16_EPS,
        "grad_tensors": len(grad_rows),
        "param_tensors": len(param_rows),
        "worst_gradient": max(grad_rows, key=lambda row: row["rel_to_scale"], default={}),
        "worst_parameter": max(param_rows, key=lambda row: row["rel_to_scale"], default={}),
        "grads": sorted(grad_rows, key=lambda row: -row["rel_to_scale"])[:5],
        "params": sorted(param_rows, key=lambda row: -row["rel_to_scale"])[:5],
        "compiled_graphs": compiled_row["proof"]["unique_graphs"],
    }
    print(
        f"parity: loss {out['loss_eager']:.6f} vs {out['loss_compiled']:.6f}, "
        f"{out['grad_tensors']} grads + {out['param_tensors']} params + logits, "
        f"worst rel {worst:.3g} (bf16 eps {BF16_EPS:.3g}), "
        f"lossless={out['within_bf16_epsilon']}",
        flush=True,
    )
    compiled.reset()
    return out


def acquire_card(allowed: set[int], take_lock: bool, wait_s: float) -> None:
    """Refuse a busy card; optionally take the repo lock with a bounded wait."""
    handle = None
    if take_lock:
        handle = LOCK.open("w")
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise LoudError(
                        f"GPU lock {LOCK} was held for more than {wait_s:g}s",
                        "the card is busy in another repo; run this when the lock frees",
                    ) from None
                time.sleep(1.0)
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if smi.returncode != 0:
        if handle is not None:
            handle.close()
        raise LoudError(
            f"nvidia-smi could not list compute apps: {smi.stderr.strip().splitlines()[:1]}",
            "the card guard cannot be skipped: no speed may be printed without it",
        )
    busy = busy_pids(smi.stdout, allowed)
    if busy:
        if handle is not None:
            handle.close()
        raise LoudError(
            f"card is busy: compute PIDs {busy} are not on the allow-list",
            f"wait for the card, or pass --allow-pid for the desktop PID (now {sorted(allowed)})",
        )
    print(f"card free (compute PIDs: {sorted(allowed)}), guard passed", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Measure the shipped step; print a speed only when both gates pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--allow-pid", type=int, action="append", default=[])
    parser.add_argument("--take-lock", action="store_true", help="take the repo lock too")
    parser.add_argument("--lock-wait", type=float, default=LOCK_WAIT_S)
    args = parser.parse_args(argv)
    allowed = set(args.allow_pid)
    if env_pid := os.environ.get("DORMOUSE_DESKTOP_PID", ""):
        allowed.add(int(env_pid))
    if not torch.cuda.is_available():
        print("no CUDA device: this is a card-side measurement; refusing", file=sys.stderr)
        return 2
    acquire_card(allowed, args.take_lock, args.lock_wait)

    spec, flipped = spec_for(args.config, BATCH_LADDER[0])
    info = provenance(spec, args.config, flipped)
    result: dict[str, Any] = {
        "provenance": info,
        "old_engine": {
            "ms_per_step": OLD_ENGINE_MS_PER_STEP,
            "tokens_per_step": OLD_ENGINE_TOKENS_PER_STEP,
            "bytes_per_s": OLD_ENGINE_BYTES_PER_S,
        },
        "oom_ladder": [],
    }
    speed: dict[str, Any] = {}
    parity_row: dict[str, Any] = {}
    for batch in BATCH_LADDER:
        candidate = replace(spec, batch=batch)
        try:
            speed = measure(candidate)
            parity_row = parity(candidate)
            break
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            reason = str(exc).splitlines()[0]
            print(
                f"OOM at b{batch} x s{SEQ_LEN}: {reason} — dropping to the next batch",
                flush=True,
            )
            result["oom_ladder"].append({"batch": batch, "reason": reason})
            compiled.reset()
            gc.collect()
            torch.cuda.empty_cache()
    else:
        result["refusal"] = ["every batch in the ladder OOM'd: no step was measured"]
        _write(result)
        print("REFUSED: " + result["refusal"][0], file=sys.stderr)
        return 1

    info.update(
        batch=candidate.batch,
        seq=candidate.seq_len,
        config_hash=config_hash(candidate),
        compile_flag=candidate.torch_compile,
    )
    print(f"measured shape b{candidate.batch} x s{candidate.seq_len}", flush=True)
    tokens = speed["tokens_per_step"]
    expected_ms = old_engine_ms(tokens)
    speed["expected_old_ms_at_tokens"] = expected_ms
    speed["old_engine_bytes_per_s"] = OLD_ENGINE_BYTES_PER_S
    speed["ratio_to_old"] = OLD_ENGINE_BYTES_PER_S / speed["bytes_per_s"]
    speed["pass_threshold_ms"] = expected_ms
    speed["beats_old_engine"] = speed["median_ms"] < expected_ms
    result["speed"] = speed
    result["parity"] = parity_row

    refusals = []
    if not parity_row["within_bf16_epsilon"]:
        refusals.append(
            f"parity failed: worst rel {parity_row['worst_rel_to_scale']:.3g} "
            f"> bf16 epsilon {BF16_EPS:.3g}"
        )
    if speed["compiled_graphs"] <= 0 or speed["compile_graphs_counter"] <= 0:
        refusals.append(
            f"compile flag on but graphs={speed['compiled_graphs']}, "
            f"counter={speed['compile_graphs_counter']}, replays={speed['graph_replays']}"
        )
    if refusals:
        result["refusal"] = refusals
        result["speed"] = None  # no speed may be printed or archived behind a failed gate
        _write(result)
        for reason in refusals:
            print(f"REFUSED: {reason}", file=sys.stderr)
        return 1

    print(
        f"median {speed['median_ms']:,.1f} ms/step ({MEASURE_STEPS} steps x {REPETITIONS} reps), "
        f"{speed['tokens_per_s']:,.0f} tok/s, {speed['bytes_per_s']:,.0f} B/s source",
        flush=True,
    )
    print(
        f"MFU {speed['mfu_percent']:.2f}% at {CARD_PEAK_TFLOPS} TFLOP/s "
        f"({speed['mfu_percent_at_52']:.2f}% at the old 52 constant), "
        f"compile graphs {speed['compiled_graphs']}",
        flush=True,
    )
    print(
        f"old engine {expected_ms:,.0f} ms at {tokens:,} tokens; "
        f"threshold {expected_ms:,.0f} ms / {OLD_ENGINE_BYTES_PER_S:,.0f} B/s; "
        f"this run {speed['median_ms']:,.0f} ms / {speed['bytes_per_s']:,.0f} B/s "
        f"({speed['ratio_to_old']:.2f}x old)",
        flush=True,
    )
    verdict = "PASS" if speed["beats_old_engine"] else "FAIL"
    print(
        f"VERDICT: {verdict} — {'beats' if speed['beats_old_engine'] else 'does not beat'} "
        f"the old engine's {OLD_ENGINE_MS_PER_STEP / 1000:.2f} s/step",
        flush=True,
    )
    _write(result)
    return 0


def _write(result: dict[str, Any]) -> None:
    """Archive the full record, including refusals and the OOM ladder."""
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {ARTIFACT}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
