"""Deterministic paired A/B gate: one config, one seed, bitwise-identical control.

The bank scatters through CUDA `index_add_` (`model/moe.py`, `kernels/bank.py`) whose
atomic order is not fixed. Two identical runs on a card diverged by -2.56 dBPB over
200 steps: that was the measurement, not the model. So this gate refuses CUDA and
runs everything on one CPU thread, where the scatter order is fixed. See
`docs/ab-gate.md` for the named blocker and the protocol.

    python tools/ab_gate.py --config configs/ab_gate.toml --steps 200 --seeds 3 \
        --arm-a control --arm-b ternary

Every seed runs arm A twice; the two controls must be `torch.equal` on every step
loss and on the held-out loss and BPB. The second control is the measured repeatability
floor: a non-zero floor refuses the whole comparison before a delta is printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from dormouse.metrics import LN2, from_nats
from dormouse.model.assemble import assemble
from dormouse.train.optim import MixOptimizer
from dormouse.train.report import digest
from dormouse.train.runconfig import RunRecipe
from dormouse.train.step import chunked_cross_entropy
from dormouse.train.streams import HeldOut, heldout_bytes_per_token

ROOT = Path(__file__).resolve().parents[1]
T95_N3 = 2.920  # Student t, 0.95 one-sided, 2 degrees of freedom
GATE_DBPB = 0.01

#: Arm name -> spec overrides and process flags. An unknown arm is refused by name;
#: an arm this source tree does not assemble (`a4` at the pre-A4 HEAD) is refused by
#: `StackSpec`, which is the same validation a real run gets.
ARMS: dict[str, dict[str, Any]] = {
    "control": {},
    "int4": {"act_quant": "int4"},
    "ternary": {
        "weight_qat": "ternary_native",
        "env": {"DORMOUSE_TSCT_NATIVE_TERN": "1"},
    },
    "a4": {
        "act_quant": "a4",
        "weight_qat": "ternary_native",
        "env": {"DORMOUSE_ACT_A4": "1", "DORMOUSE_TSCT_NATIVE_TERN": "1"},
    },
    "a4+ternary": {
        "act_quant": "a4",
        "weight_qat": "ternary_native",
        "env": {"DORMOUSE_ACT_A4": "1", "DORMOUSE_TSCT_NATIVE_TERN": "1"},
    },
    "muon_fp32": {"muon_ns_dtype": "fp32"},
    "muon_fp16": {"muon_ns_dtype": "fp16"},
}

ENV_KEYS = (
    "DORMOUSE_ACT_A4",
    "DORMOUSE_A4_FUSED",
    "DORMOUSE_A4_ROUND",
    "DORMOUSE_A4_SCALE",
    "DORMOUSE_TSCT_NATIVE_TERN",
)


class GateRefusal(RuntimeError):
    """A named, non-zero-exit refusal: the comparison was not measured."""


@contextmanager
def _arm(name: str) -> Iterator[dict[str, Any]]:
    """Set one arm's spec overrides and flags, restoring all of them afterwards."""
    if name not in ARMS:
        raise GateRefusal(
            f"unknown arm {name!r}; arms: {', '.join(ARMS)}",
        )
    definition = ARMS[name]
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(definition.get("env", {}))
    try:
        yield {key: value for key, value in definition.items() if key != "env"}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ids(recipe: RunRecipe, seed: int) -> torch.Tensor:
    """Fix the paired batches to one seed, independent of the arm or the machine."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        recipe.spec.vocab,
        (recipe.spec.batch, recipe.spec.seq_len + 1),
        generator=generator,
    )


class ArmRun:
    """One arm's tensors: every number the gate compares stays bitwise checkable."""

    def __init__(
        self,
        losses: torch.Tensor,
        eval_nats: torch.Tensor,
        payload_bytes: int,
        ids_digest: str,
        eval_digest: str,
    ) -> None:
        """Hold one run's step losses, eval nats, manifest bytes and batch digests."""
        self.losses = losses
        self.eval_nats = eval_nats
        self.payload_bytes = payload_bytes
        self.ids_digest = ids_digest
        self.eval_digest = eval_digest
        #: nats / ln(2) / bytes, with the window's bytes coming from the eval manifest.
        self.bpb = eval_nats / LN2 / payload_bytes
        measured = from_nats(float(eval_nats) / payload_bytes, payload_bytes)
        if measured.bytes_evaluated != payload_bytes or not math.isclose(
            measured.bpb, float(self.bpb), rel_tol=1e-12
        ):
            raise GateRefusal(
                f"BPB_FORMULA: tensor {float(self.bpb):.12f} vs project type "
                f"{measured.bpb:.12f} over {payload_bytes} B"
            )

    def as_row(self) -> dict[str, Any]:
        """Return a JSON-able row for the gate artifact."""
        return {
            "losses": [float(value) for value in self.losses],
            "eval_nats": float(self.eval_nats),
            "bytes_evaluated": self.payload_bytes,
            "bpb": float(self.bpb),
            "ids_digest": self.ids_digest,
            "eval_digest": self.eval_digest,
        }


def run_arm(
    recipe: RunRecipe,
    *,
    arm: str,
    seed: int,
    steps: int,
    batches: int,
    first: int,
    lr_scale: float,
) -> ArmRun:
    """Train `steps` steps and score a fixed held-out window, in one-thread CPU order."""
    if torch.get_num_threads() != 1:
        raise GateRefusal(
            f"torch has {torch.get_num_threads()} threads; call configure_cpu() first",
        )
    with _arm(arm) as overrides:
        ns_dtype = overrides.pop("muon_ns_dtype", "fp32")
        spec = replace(recipe.spec, **overrides)
        torch.manual_seed(seed)
        built = assemble(spec, smoke=True, device=torch.device("cpu"))
        optimizer = MixOptimizer(built.model, steps=1, muon_ns_dtype=ns_dtype)
        optimizer.set_lr(lr_scale)
        ids = _ids(recipe, seed)
        losses: list[torch.Tensor] = []
        for step in range(steps):
            result, _ = built.step(ids, optimizer=optimizer)
            if result.skipped or not math.isfinite(result.loss):
                raise GateRefusal(
                    f"NaN firewall (arm={arm}, seed={seed}, step={step}): "
                    f"loss={result.loss} at {result.bad_stage}",
                )
            losses.append(torch.tensor(result.loss, dtype=torch.float64))

        stream = HeldOut(
            recipe.eval_path,  # type: ignore[arg-type]
            heldout_bytes_per_token(recipe),
            batch=spec.batch,
            seq_len=spec.seq_len,
            probes=first + batches,
            device=torch.device("cpu"),
        )
        eval_digest: list[str] = []
        nats = torch.zeros((), dtype=torch.float64)
        payload = 0
        built.model.eval()
        with torch.no_grad():
            for index in range(first, first + batches):
                batch = stream.batch(index, spec.batch * spec.seq_len)
                eval_digest.append(digest(batch.ids))
                logits = built.model(batch.ids[:, :-1], None)
                nats = (
                    nats
                    + chunked_cross_entropy(logits.float(), batch.ids[:, 1:]).double()
                    * batch.tokens
                )
                payload += batch.payload_bytes
    if payload <= 0:
        raise GateRefusal("the eval window carries no bytes")
    return ArmRun(
        torch.stack(losses),
        nats,
        payload,
        digest(ids),
        "/".join(eval_digest),
    )


def configure_cpu() -> None:
    """Fix the reduction order: one intra-op thread and one inter-op thread."""
    torch.set_num_threads(1)
    # A test process may already have started inter-op work; intra-op is the order
    # that matters for the bank's scatter and is already fixed above.
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def check_control(a: ArmRun, b: ArmRun, *, seed: int) -> None:
    """Refuse a comparison whose control is not bitwise repeatable."""
    if not torch.equal(a.losses, b.losses):
        differing = [i for i, (x, y) in enumerate(zip(a.losses, b.losses, strict=False)) if x != y]
        raise GateRefusal(
            f"CONTROL_NOT_BITWISE: seed {seed} control losses differ at steps "
            f"{differing[:8]} (CUDA atomics or a stray thread); dBPB is not measurable",
        )
    if not (torch.equal(a.eval_nats, b.eval_nats) and torch.equal(a.bpb, b.bpb)):
        raise GateRefusal(
            f"CONTROL_NOT_BITWISE: seed {seed} control eval differs: "
            f"nats {a.eval_nats.item():.12f}/{b.eval_nats.item():.12f}, "
            f"bpb {a.bpb.item():.12f}/{b.bpb.item():.12f}",
        )
    if a.ids_digest != b.ids_digest or a.eval_digest != b.eval_digest:
        raise GateRefusal(f"CONTROL_BATCHES_DIFFER: seed {seed} read different ids")


def gate(
    config: Path,
    *,
    steps: int,
    seeds: int,
    arm_a: str,
    arm_b: str,
    batches: int = 1,
    lr_scale: float = 0.1,
    seed_start: int = 0,
) -> dict[str, Any]:
    """Run the control floor and the paired delta; the floor must be exactly zero."""
    recipe = RunRecipe.read(config)
    if recipe.eval_path is None:
        raise GateRefusal(f"{config}: no eval window, so BPB has no bytes")
    bpt = heldout_bytes_per_token(recipe)
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for seed in range(seed_start, seed_start + seeds):
        first = seed * batches
        control_a = run_arm(
            recipe,
            arm=arm_a,
            seed=seed,
            steps=steps,
            batches=batches,
            first=first,
            lr_scale=lr_scale,
        )
        control_b = run_arm(
            recipe,
            arm=arm_a,
            seed=seed,
            steps=steps,
            batches=batches,
            first=first,
            lr_scale=lr_scale,
        )
        check_control(control_a, control_b, seed=seed)
        if arm_a == arm_b:
            arm_run = control_b
        else:
            arm_run = run_arm(
                recipe,
                arm=arm_b,
                seed=seed,
                steps=steps,
                batches=batches,
                first=first,
                lr_scale=lr_scale,
            )
            if (
                arm_run.ids_digest != control_a.ids_digest
                or arm_run.eval_digest != control_a.eval_digest
            ):
                raise GateRefusal(f"PAIRED_BATCHES_DIFFER: seed {seed} arms read different ids")
        delta = float(arm_run.bpb - control_a.bpb)
        deltas.append(delta)
        rows.append(
            {
                "seed": seed,
                "control": control_a.as_row(),
                "arm": arm_run.as_row(),
                "floor": 0.0,
                "dbpb": delta,
            }
        )
        print(
            f"seed {seed}: bpb {float(control_a.bpb):.6f} -> {float(arm_run.bpb):.6f} "
            f"dBPB={delta:+.6f} floor=0.000000 bytes={arm_run.payload_bytes}",
            flush=True,
        )
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    margin = T95_N3 * sd / math.sqrt(len(deltas))
    upper = mean + margin
    verdict = "PASS" if abs(mean) <= GATE_DBPB and upper <= GATE_DBPB else "RED"
    result: dict[str, Any] = {
        "config": str(config),
        "steps": steps,
        "seeds": seeds,
        "batches": batches,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "bytes_per_token": bpt,
        "bpt_source": "manifest beside the eval shard",
        "reduction": "CPU, torch.set_num_threads(1); CUDA bank atomics refused (B1)",
        "mean_dbpb": mean,
        "ci95": [mean - margin, upper],
        "max_floor": 0.0,
        "verdict": verdict,
        "rows": rows,
    }
    print(
        f"| {arm_a} vs {arm_b} | {mean:+.6f} | [{mean - margin:+.6f},{upper:+.6f}] | "
        f"floor=0.000000 | {verdict} |",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse the gate's CLI, run it, and refuse a non-deterministic comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools" / "ab_gate.toml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arm-a", default="control")
    parser.add_argument("--arm-b", default="ternary")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.device != "cpu":
        raise GateRefusal(
            f"B1 BANK_ATOMICS_NONDETERMINISTIC: --device {args.device!r} is refused; "
            "CUDA index_add_ has no fixed reduction order. Use the CPU one-thread gate.",
        )
    configure_cpu()
    try:
        result = gate(
            args.config,
            steps=args.steps,
            seeds=args.seeds,
            arm_a=args.arm_a,
            arm_b=args.arm_b,
            batches=args.batches,
            lr_scale=args.lr_scale,
            seed_start=args.seed_start,
        )
    except GateRefusal as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
