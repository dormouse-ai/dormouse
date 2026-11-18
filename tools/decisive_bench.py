"""The decisive speed experiment: one shipped 1B step, compiled, against the old engine.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/decisive_bench.py \
        --allow-pid <desktop-pid>

Every earlier comparison was confounded: a different batch, a stale commit, a 52 vs 92
TFLOP/s peak, a three-step profiler against a thousand-step journal. This script prints
its own commit, config hash, peak constant and batch, proves parity against eager, proves
inductor built a graph, and only then prints ms/step. A refusal is a result; an eager
timing wearing a compiled label is not.

The shipped config keeps torch_compile=false. This script does not compile the whole
model: it compiles one bounded step function (train/step.py) at each batch, drops the
graph and its Triton caches before the next, and uses inductor's `default` mode — no
cudagraphs, no kernel variants — until a card-side run shows another mode is smaller.
If compile cannot fit at any batch, it measures the eager step at b8 and labels that
number `eager`; `--eager-only` skips compile entirely and still runs parity and writes
the report. A failed compile never wears the compiled label.

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
from collections.abc import Callable, Mapping
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
from dormouse.train.step import train_step

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
#: The last rung is where an eager fallback always lands if compile cannot fit at any
#: batch. It is a named shape, never a silent edit to the shipped config.
EAGER_FALLBACK_BATCH = BATCH_LADDER[-1]
#: Inductor modes a card-side run can A/B. `default` builds no CUDA graphs and tries no
#: kernel variants — the smallest compile-time footprint; nothing here claims another
#: mode is smaller without a card measurement.
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")
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


def speed_label(*, compiled_measured: bool, compile_reason: str | None = None) -> str:
    """Name the mode a measured ms/step actually ran in.

    A failed compile — OOM, timeout, zero graphs — never earns the `compiled` label,
    even when a valid eager timing exists. A number with a wrong label is worse than
    a refusal, and a refusal with no number is worse than an eager number.
    """
    if not compiled_measured or compile_reason is not None:
        return "eager"
    return "compiled"


def speed_line(label: str, speed: Mapping[str, Any]) -> str:
    """Name one ms/step line by `label` and by nothing else."""
    if label not in ("compiled", "eager"):
        raise LoudError(f"unknown speed label {label!r}", "use 'compiled' or 'eager'")
    return (
        f"{label} median {speed['median_ms']:,.1f} ms/step "
        f"({speed['measure_steps']} steps x {speed['repetitions']} reps), "
        f"{speed['tokens_per_s']:,.0f} tok/s, {speed['bytes_per_s']:,.0f} B/s source"
    )


def compile_refusal_line(reason: str, *, skipped: bool = False) -> str:
    """Say separately why no compiled ms/step was printed."""
    prefix = "compiled skipped by --eager-only" if skipped else "compiled not measured"
    return f"{prefix}: {reason}"


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
        f"dtype {out['dtype']}  compile region=step function, whole model eager"
        + (" (torch_compile overridden from false)" if flipped else ""),
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


def _time_steps(call: Callable[[], None], tokens: int) -> dict[str, Any]:
    """Time `call` with CUDA events, three repetitions of thirty steps each."""
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
    }


def measure(spec: StackSpec, *, mode: str = compiled.DEFAULT_REGION_MODE) -> dict[str, Any]:
    """Compile one bounded step region, time it, prove a graph ran, then release it.

    The compiled unit is the step function, not the whole model: `assemble` is built with
    torch_compile=false here, so no per-submodule graphs are resident. The returned
    callable is the only graph; `clear_caches` in the finally drops dynamo, inductor and
    CUDA cache references before the next batch is tried.
    """
    torch.manual_seed(SEED)
    assembled = assemble(replace(spec, torch_compile=False), device="cuda")
    optimizer = MixOptimizer(assembled.model)
    optimizer.set_lr(0.0)  # frozen: the baseline's own regime, same kernels and shapes
    ids = make_ids(spec)
    seen: dict[str, Any] = {}

    def region(model: Any, batch: torch.Tensor, opt: MixOptimizer) -> Any:
        """One bounded step: forward/backward plus commit; dynamo stops at the host read."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = train_step(model, batch, optimizer=opt, counters=None)
        return result

    compiled_region = compiled.enable_region(region, mode=mode)

    def call() -> None:
        result = compiled_region(assembled.model, ids, optimizer)
        seen["loss"] = float(result.loss)

    try:
        try:
            with compiled.compile_budget():  # compilation is lazy: it lands on this call
                call()
        except RuntimeError as exc:
            if _is_oom(exc):
                raise
            raise LoudError(
                f"torch.compile failed on the first step: {str(exc).splitlines()[0]}",
                "no eager timing may be printed under the compiled name: this run is refused",
            ) from exc
        for _ in range(WARMUP_STEPS - 1):
            call()
        proof = compiled.require_graph()  # named refusal when inductor built no graph
        timed = _time_steps(call, spec.batch * spec.seq_len)
    finally:
        compiled.clear_caches()  # the compiled callable dies with this frame

    timed["compiled_graphs"] = proof["unique_graphs"]
    timed["compile_graph_break"] = proof["graph_break"]
    timed["compile_graphs_counter"] = proof["unique_graphs"]
    timed["graph_replays"] = 0  # this path compiles a step function; it captures no graph
    timed["loss_last"] = seen["loss"]
    return timed


def measure_eager(spec: StackSpec) -> dict[str, Any]:
    """Time the same step eagerly; the number is labelled eager, never compiled."""
    torch.manual_seed(SEED)
    compiled.reset()
    assembled = assemble(replace(spec, torch_compile=False), device="cuda")
    optimizer = MixOptimizer(assembled.model)
    optimizer.set_lr(0.0)
    ids = make_ids(spec)
    seen: dict[str, Any] = {}

    def call() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = assembled.step(ids, optimizer=optimizer)
        seen["loss"] = float(result.loss)

    for _ in range(WARMUP_STEPS):
        call()
    timed = _time_steps(call, spec.batch * spec.seq_len)
    timed.update(
        compiled_graphs=0,
        compile_graph_break={},
        compile_graphs_counter=0,
        graph_replays=0,
        loss_last=seen["loss"],
    )
    return timed


def parity(
    spec: StackSpec,
    *,
    mode: str = compiled.DEFAULT_REGION_MODE,
    compare_compiled: bool = True,
) -> dict[str, Any]:
    """Compare one step from the same init on two paths, loss/logits/grads/params.

    `compare_compiled=False` is the `--eager-only` path: compile is skipped entirely and
    the second leg is a seeded eager re-run, so the full comparison still runs and is
    named for what it is (eager vs eager, not compiled vs eager).
    """
    compiled.reset()
    torch.manual_seed(SEED)
    assembled = assemble(replace(spec, torch_compile=False), device="cuda")
    model = assembled.model
    ids = make_ids(spec)
    init = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    shape = (spec.batch, spec.seq_len, spec.vocab)
    logits_buf = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    hook_logits(model, logits_buf)

    def region(m: Any, batch: torch.Tensor, opt: MixOptimizer) -> Any:
        """Run the bounded compiled unit identical to the one `measure` compiles."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = train_step(m, batch, optimizer=opt, counters=None)
        return result

    def snapshot(*, use_compile: bool) -> dict[str, Any]:
        """One step from the same initial weights, compiled or eager."""
        model.load_state_dict(init)  # type: ignore[arg-type]
        optimizer = MixOptimizer(model)
        optimizer.set_lr(PARITY_LR_FRACTION)
        proof: dict[str, Any] = {}
        if use_compile:
            # Enable only after the eager snapshot: `assembled.step` checks the compile
            # counter whenever compile is enabled, and no graph exists yet at that point.
            for module in model.modules():
                if isinstance(module, MoEFFN):
                    module.fixed_passes = MAX_DISPATCH_PASSES  # no host read in the graph
            compiled_region = compiled.enable_region(region, mode=mode)
            with compiled.compile_budget():
                result = compiled_region(model, ids, optimizer)
            proof = compiled.require_graph()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result, _ = assembled.step(ids, optimizer=optimizer)
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
    other = snapshot(use_compile=compare_compiled)
    if set(eager["grads"]) != set(other["grads"]):
        raise LoudError(
            f"the two paths own different gradients: {set(eager['grads']) ^ set(other['grads'])}",
            "a compiled step must not grow or drop a gradient",
        )
    grad_rows = [
        _diff(name, other["grads"][name], eager["grads"][name]) for name in sorted(eager["grads"])
    ]
    param_rows = [
        _diff(name, other["params"][name], eager["params"][name])
        for name in sorted(eager["params"])
    ]
    logits_row = _diff("logits", other["logits"], eager["logits"])
    worst = max(
        [float(row["rel_to_scale"]) for row in [*grad_rows, *param_rows, logits_row]],
        default=0.0,
    )
    out: dict[str, Any] = {
        "comparison": "compiled-vs-eager" if compare_compiled else "eager-vs-eager",
        "loss_eager": eager["loss"],
        "loss_compiled": other["loss"],
        "loss_abs_diff": abs(eager["loss"] - other["loss"]),
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
        "compiled_graphs": other["proof"].get("unique_graphs", 0),
    }
    print(
        f"parity ({out['comparison']}): loss {out['loss_eager']:.6f} "
        f"vs {out['loss_compiled']:.6f}, "
        f"{out['grad_tensors']} grads + {out['param_tensors']} params + logits, "
        f"worst rel {worst:.3g} (bf16 eps {BF16_EPS:.3g}), "
        f"lossless={out['within_bf16_epsilon']}",
        flush=True,
    )
    compiled.clear_caches()
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
    """Measure the shipped step; print a labelled speed only when the gates pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--allow-pid", type=int, action="append", default=[])
    parser.add_argument("--take-lock", action="store_true", help="take the repo lock too")
    parser.add_argument("--lock-wait", type=float, default=LOCK_WAIT_S)
    parser.add_argument(
        "--eager-only",
        action="store_true",
        help="skip compile entirely; still run the parity check and write the JSON report",
    )
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default=compiled.DEFAULT_REGION_MODE,
        help="inductor mode for the bounded step region (default has the smallest footprint)",
    )
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
        "compile": {"requested": not args.eager_only, "mode": args.compile_mode, "region": "step"},
    }
    speed: dict[str, Any] = {}
    parity_row: dict[str, Any] = {}
    candidate = replace(spec, batch=EAGER_FALLBACK_BATCH)
    compiled_ok = False
    compile_reason: str | None = None

    if not args.eager_only:
        for batch in BATCH_LADDER:
            candidate = replace(spec, batch=batch)
            try:
                speed = measure(candidate, mode=args.compile_mode)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_oom(exc):
                    raise
                reason = str(exc).splitlines()[0]
                print(
                    f"OOM at b{batch} x s{SEQ_LEN}: {reason} — dropping to the next batch",
                    flush=True,
                )
                result["oom_ladder"].append({"batch": batch, "reason": reason})
                compile_reason = f"compile memory OOM at b{batch} x s{SEQ_LEN}: {reason}"
                compiled.clear_caches()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except LoudError as exc:
                # A timeout or a zero-graph compile is a compile failure, not a parity
                # failure: stop trying batches and let the eager half produce a number.
                compile_reason = f"compile could not produce a graph: {exc}"
                print(f"compile failed; falling back to eager: {compile_reason}", flush=True)
                break
            compiled_ok = True
            compile_reason = None
            parity_row = parity(candidate, mode=args.compile_mode)
            break
    else:
        compile_reason = "no compile was attempted"

    if not compiled_ok:
        # A valid eager timing exists even when compile cannot fit. It is measured at the
        # ladder's last rung and labelled eager, never printed under the compiled name.
        print(
            f"compiled measurement unavailable; measuring eager at b{EAGER_FALLBACK_BATCH}",
            flush=True,
        )
        candidate = replace(spec, batch=EAGER_FALLBACK_BATCH)
        speed = measure_eager(candidate)
        parity_row = parity(candidate, compare_compiled=False)

    label = speed_label(
        compiled_measured=compiled_ok,
        compile_reason=None if compiled_ok else compile_reason,
    )
    result["compile"].update(measured=compiled_ok, reason=compile_reason, label=label)
    info.update(
        batch=candidate.batch,
        seq=candidate.seq_len,
        config_hash=config_hash(candidate),
        compile_flag=candidate.torch_compile,
        measured_mode=label,
    )
    print(f"measured shape b{candidate.batch} x s{candidate.seq_len}", flush=True)
    speed["label"] = label
    speed["compile_mode"] = args.compile_mode
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
    if label == "compiled" and (
        speed["compiled_graphs"] <= 0 or speed["compile_graphs_counter"] <= 0
    ):
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

    print(speed_line(label, speed), flush=True)
    if label == "eager":
        assert compile_reason is not None
        print(
            compile_refusal_line(compile_reason, skipped=args.eager_only),
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
