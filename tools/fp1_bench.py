"""FP1: one card measurement of one b2 200M step (eager or graph), min-of-7.

Holds /tmp/dormouse-gpu.lock itself (shell flock is permission-gated here).
Every arm freezes LR so repeated steps run identical kernels; the loss and the
live counter snapshot are reported with the time, never alone.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import statistics
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT.parent.parent / "core" / "src"),
    str(ROOT.parent.parent / "dormouse-kernels" / "src"),
    str(ROOT / "tools"),
]

import torch  # noqa: E402
from dormouse_kernels.engine import mega_block  # noqa: E402

from dormouse.model.assemble import assemble  # noqa: E402
from dormouse.model.spec import StackSpec  # noqa: E402
from dormouse.train import compiled  # noqa: E402
from dormouse.train.optim import MixOptimizer  # noqa: E402


def spec_for(config: str, overrides: dict[str, Any]) -> StackSpec:
    """Load a shipped config and apply only explicitly named overrides."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    valid = {f.name for f in fields(StackSpec) if f.init}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"unknown spec overrides: {sorted(unknown)}")
    return replace(spec, **overrides) if overrides else spec


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build, warm up, time and report one arm."""
    torch.manual_seed(args.seed)
    overrides: dict[str, Any] = {}
    if args.ideal_bank:
        overrides.update(ideal_bank=True, ideal_slots=args.ideal_slots)
    if args.batch:
        overrides["batch"] = args.batch
    spec = spec_for(args.config, overrides)
    assembled = assemble(spec, device="cuda")
    bill = assembled.report()
    optimizer = MixOptimizer(assembled.model, muon_ns_dtype=args.ns_dtype)
    optimizer.set_lr(args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator).cuda()
    graph = None
    refusal = None
    if args.mode == "graph":
        if args.skip_fwd_side_effects:
            # The checkpoint HOP's `first=[False]` side effect is the known dynamo
            # blocker at this HEAD; try the project's documented flag once, loudly.
            torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        try:
            graph = assembled.capture(ids, optimizer=optimizer, epilogue=args.epilogue)
        except Exception as exc:
            # A failed capture is a named result, never an eager step wearing a graph
            # label. The eager fast path stays intact for this arm.
            refusal = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            if args.strict_graph:
                raise
            # `compiled.checked` compiled the model in place; reset clears dynamo's
            # counters but not that wrapper. Time the eager fast path in a SEPARATE
            # process instead of reporting a compiled retry under an eager name.
            compiled.reset()
            return {
                "config": args.config,
                "mode": "graph_refused",
                "capture_refusal": refusal,
                "ideal_bank": args.ideal_bank,
                "epilogue": args.epilogue,
            }
    elif args.epilogue:
        mega_block.set_epilogue(True)
        if not args.ideal_bank:
            raise ValueError("epilogue needs --ideal-bank; the routed/A4 bank stays untouched")
        for bank in (b for b in assembled.model.modules() if hasattr(b, "ideal_bank")):
            if bank.ideal_bank:
                mega_block.install_fused_bank(bank)

    seen: dict[str, Any] = {}

    def step() -> None:
        """One bf16 step, eager or replayed."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, counters = assembled.step(ids, optimizer=optimizer, graph=graph)
        if not result.skipped:
            seen["loss"] = float(result.loss)
        seen["counters"] = counters.snapshot()
        seen["skipped"] = result.skipped

    if args.steps:
        losses: list[float] = []
        for _ in range(args.steps):
            step()
            losses.append(float(seen.get("loss", 0.0)))
        digest = hashlib.sha256()
        for name, value in sorted(assembled.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return {
            "config": args.config,
            "mode": "parity",
            "ideal_bank": args.ideal_bank,
            "epilogue": args.epilogue,
            "steps": args.steps,
            "losses": losses,
            "state_sha256": digest.hexdigest(),
            "loss_finite": all(abs(value) < 1e30 for value in losses),
            "counters": seen.get("counters", {}),
        }
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    tokens = spec.batch * spec.seq_len
    best = min(samples)
    return {
        "config": args.config,
        "mode": args.mode
        if graph is not None
        else ("eager_after_refusal" if refusal else args.mode),
        "capture_refusal": refusal,
        "ideal_bank": bool(getattr(spec, "ideal_bank", False)),
        "ideal_slots": getattr(spec, "ideal_slots", 0) if getattr(spec, "ideal_bank", False) else 0,
        "ns_dtype": args.ns_dtype,
        "batch": spec.batch,
        "seq": spec.seq_len,
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "params_active": bill.get("active"),
        "samples_ms": samples,
        "min_ms": best,
        "median_ms": statistics.median(samples),
        "tokens_per_step": tokens,
        "tokens_per_s": tokens / (best * 1e-3),
        "loss": seen.get("loss"),
        "loss_finite": bool(seen.get("loss") is not None and abs(seen["loss"]) < 1e30),
        "skipped": seen.get("skipped"),
        "counters": seen.get("counters", {}),
        "graph_replays": seen.get("counters", {}).get("graph_replays", 0),
    }


def main() -> None:
    """Acquire the card lock, then run one arm and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo_200m.toml"))
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--ideal-bank", action="store_true")
    parser.add_argument("--epilogue", action="store_true")
    parser.add_argument("--strict-graph", action="store_true")
    parser.add_argument("--skip-fwd-side-effects", action="store_true")
    parser.add_argument("--ideal-slots", type=int, default=32)
    parser.add_argument("--ns-dtype", default="fp16")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()
    with Path("/tmp/dormouse-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            out = run(args)
        except BaseException as exc:
            import traceback

            traceback.print_exc()
            out = {"error": type(exc).__name__, "message": str(exc)[:1200], "args": vars(args)}
            raise SystemExit(1) from exc
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    text = json.dumps(out, indent=2, default=str)
    print(text, flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
