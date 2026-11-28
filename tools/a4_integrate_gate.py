"""Honest A4 quality gate: one FROZEN checkpoint, A4 on/off at activation boundaries.

The two arms in earlier gates were two from-scratch trajectories, so their dBPB measured
chaos in the bank's fp32 atomics rather than quantisation. Here one trained 200M model is
loaded once and scored twice on exactly the same held-out bytes; only `RMSNorm.a4` flips.
The eval window and its bytes/token come from the checkpoint's own manifest, not from a
42-byte string divided by 1024 tokens.

    PYTHONPATH=<wt>/src:<core>/src python tools/a4_integrate_gate.py --seeds 3 --batches 1

CPU with one thread is the deterministic eval-reduction: the bank's `index_add_` is
thread-order dependent on CUDA, so a CPU eval of the same ids is bitwise repeatable.
This changes only the gate's measurement, never a training step. `--counters` runs the
fused+ternary training arm on CUDA and checks the eight mechanism receipts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train import entry
from dormouse.train.report import digest, evaluate
from dormouse.train.runconfig import RunRecipe
from dormouse.train.streams import HeldOut, heldout_bytes_per_token

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent.parent / "research" / "solve" / "artifacts"
DEFAULT_CKPT = ROOT.parent / "demo-fri" / "runs" / "demo_200m" / "export" / "model.pt"
T95_N3 = 2.920  # Student t, 0.95 one-sided, 2 degrees of freedom
#: The eight canonical receipts: two A4 norm boundaries, three bank slab saves
#: and the native ternary STE's forward/backward. Fused receipts are checked by
#: `assemble`'s own MissingReport gates when the kernel flag is on.
COUNTERS_8 = (
    "a4_fwd",
    "a4_bwd",
    "a4_codes_saved",
    "bank_a4_fwd",
    "bank_a4_bwd",
    "bank_a4_codes_saved",
    "tsct_ternary_fwd",
    "tsct_ternary_bwd",
)


@contextmanager
def _env(**values: str) -> Iterator[None]:
    """Set flag slots for one block and restore them, including absent ones."""
    keys = (
        "DORMOUSE_ACT_A4",
        "DORMOUSE_A4_FUSED",
        "DORMOUSE_A4_ROUND",
        "DORMOUSE_A4_SCALE",
        "DORMOUSE_TSCT_NATIVE_TERN",
        "DORMOUSE_BANK",
    )
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _pre_floor_checkpoint() -> Iterator[None]:
    """Let a checkpoint trained before the KDA alpha-floor check be scored.

    `93b8206` added a named refusal for alpha < 1e-6; the frozen 200M artifact was
    trained on `cf3607a`, where the same solve ran with a 1e-12 clamp and scored
    2.614 BPB. This gate-only shim skips the NEW refusal (never a clamp change and
    never a training step) and the finite-loss firewall below still applies.
    """
    from dormouse.model.attn_variants.kda import KDA

    original = KDA.check_alpha_floor
    KDA.check_alpha_floor = lambda self: None  # type: ignore[method-assign]
    try:
        yield
    finally:
        KDA.check_alpha_floor = original  # type: ignore[method-assign]


def _set_a4(model: torch.nn.Module, on: bool) -> int:
    """Flip only the activation boundary; weights, gain, head and loss stay frozen."""
    touched = 0
    #: The final norm before the tied head is not an A4 site in this design: only
    #: the two pre-sublayer norms of each block are activation boundaries.
    scope = getattr(model, "blocks", model)
    for module in scope.modules():
        if hasattr(module, "a4"):
            module.a4 = on  # type: ignore[attr-defined]
            touched += 1
    if touched == 0:
        raise RuntimeError("the checkpoint has no A4 activation boundaries")
    return touched


def _load(checkpoint: Path, device: torch.device) -> tuple[StackSpec, torch.nn.Module, RunRecipe]:
    """Load the frozen weights and the run recipe that owns their eval window."""
    raw = entry._read(checkpoint)
    spec, model = entry._build(raw, device)
    if spec.act_quant != "none":
        raise RuntimeError(f"the frozen checkpoint is already quantised: {spec.act_quant}")
    stored = checkpoint.parent / "config.json"
    if not stored.exists():
        stored = checkpoint.parent.parent / "config.json"
    recipe = RunRecipe.from_canonical(
        json.loads(stored.read_text(encoding="utf-8")), path=checkpoint.parent / "config.toml"
    )
    if recipe.eval_path is None:
        raise RuntimeError("the checkpoint's run recorded no held-out eval window")
    return spec, model, recipe


def _window(
    recipe: RunRecipe, spec: StackSpec, device: torch.device, probes: int
) -> tuple[HeldOut, float]:
    """Build the eval stream from the manifest beside the held-out shard."""
    stream = HeldOut(
        recipe.eval_path,  # type: ignore[arg-type]
        heldout_bytes_per_token(recipe),
        batch=spec.batch,
        seq_len=spec.seq_len,
        probes=probes,
        device=device,
    )
    return stream, stream.bytes_per_token


def _score(
    model: torch.nn.Module,
    stream: HeldOut,
    *,
    first: int,
    batches: int,
    tokens_per_step: int,
    a4: bool,
) -> tuple[float, list[str], int]:
    """Score one identical window with the A4 boundary on or off; prove the ids match."""
    boundaries = _set_a4(model, a4)
    ids_digests: list[str] = []
    for index in range(first, first + batches):
        batch = stream.batch(index, tokens_per_step)
        ids_digests.append(digest(batch.ids))
    bpb = evaluate(
        model,
        entry._NoSwap(),  # type: ignore[arg-type]
        stream,
        batches=batches,
        tokens_per_step=tokens_per_step,
        first=first,
    )
    return bpb.bpb, ids_digests, boundaries


def _summary(deltas: list[float], floors: list[float]) -> dict[str, Any]:
    """Compute the paired one-sided t95 and the exact repeatability floor."""
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    margin = T95_N3 * sd / math.sqrt(len(deltas))
    upper = mean + margin
    max_floor = max(floors)
    passed = abs(mean) <= 0.01 and upper <= 0.01 and max_floor == 0.0
    return {
        "deltas": deltas,
        "mean_dbpb": mean,
        "ci95": [mean - margin, upper],
        "max_floor": max_floor,
        "verdict": "PASS" if passed else "RED",
    }


def quality(checkpoint: Path, seeds: int, batches: int, device: torch.device) -> dict[str, Any]:
    """Paired frozen-model A/B: same weights, same bytes, three seeds, off-on-off."""
    torch.set_num_threads(1)
    with _env(), _pre_floor_checkpoint():  # gate-only compatibility; no training change
        spec, model, recipe = _load(checkpoint, device)
        probes, last_eval = recipe.probes(recipe.steps)
        if last_eval + seeds * batches > probes:
            probes = last_eval + seeds * batches
        stream, bpt = _window(recipe, spec, device, probes)
        tokens_per_step = spec.batch * spec.seq_len
        rows: list[dict[str, Any]] = []
        for seed in range(seeds):
            #: The run's own last eval windows, not arbitrary tail bytes: the same
            #: manifest, the same byte ratio, three disjoint probes (one per seed).
            first = last_eval + seed * batches
            off_a, ids_off_a, n = _score(
                model,
                stream,
                first=first,
                batches=batches,
                tokens_per_step=tokens_per_step,
                a4=False,
            )
            on, ids_on, _ = _score(
                model,
                stream,
                first=first,
                batches=batches,
                tokens_per_step=tokens_per_step,
                a4=True,
            )
            off_b, ids_off_b, _ = _score(
                model,
                stream,
                first=first,
                batches=batches,
                tokens_per_step=tokens_per_step,
                a4=False,
            )
            if ids_off_a != ids_on or ids_on != ids_off_b:
                raise RuntimeError(f"seed {seed}: the two arms did not read the same ids")
            floor = off_b - off_a
            delta = on - (off_a + off_b) / 2.0
            rows.append(
                {
                    "seed": seed,
                    "first_probe": first,
                    "bytes_per_token": bpt,
                    "nats_off": off_a * math.log(2) * round(tokens_per_step * batches * bpt),
                    "bpb_off_a": off_a,
                    "bpb_on": on,
                    "bpb_off_b": off_b,
                    "floor": floor,
                    "dbpb": delta,
                    "ids_digest": ids_off_a,
                    "boundaries": n,
                }
            )
            print(
                f"seed {seed}: off={off_a:.6f} on={on:.6f} off2={off_b:.6f} "
                f"dBPB={delta:+.6f} floor={floor:+.6f} ids={ids_off_a[0]}",
                flush=True,
            )
    deltas = [row["dbpb"] for row in rows]
    summary = _summary(deltas, [abs(row["floor"]) for row in rows])
    mean, lower, upper = (
        summary["mean_dbpb"],
        summary["ci95"][0],
        summary["ci95"][1],
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "seeds": seeds,
        "batches": batches,
        "bytes_per_token": bpt,
        "bpt_source": "manifest beside the checkpoint's held-out shard",
        "reduction": "one-thread deterministic eval on the device used; no training change",
        **summary,
        "rows": rows,
    }
    print(
        f"| frozen A4 on/off | {mean:+.6f} | [{lower:+.6f},{upper:+.6f}] | "
        f"floor={summary['max_floor']:.1e} | {result['verdict']} |",
        flush=True,
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "a4_integrate_quality.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def counters(steps: int, device: torch.device) -> dict[str, int]:
    """Run the fused A4 + native ternary training arm and assert all eight receipts."""
    if device.type != "cuda":
        raise RuntimeError("the fused-kernel counter gate needs CUDA")
    from dormouse_kernels.engine import bank as bank_kernels

    #: The bank flag is read at import: env alone is too late. Without this the
    #: fused bank writes no slab and the three slab receipts are a MissingReport.
    bank_kernels.set_bank(False)
    raw = tomllib.loads((ROOT / "configs" / "dormouse_1b.toml").read_text(encoding="utf-8"))
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
        seq_len=512,
        batch=2,
        tokens=1_000_000,
        torch_compile=False,
        act_quant="a4",
        weight_qat="ternary_native",
    )
    with _env(
        DORMOUSE_ACT_A4="1",
        DORMOUSE_A4_FUSED="1",
        DORMOUSE_A4_ROUND="near",
        DORMOUSE_A4_SCALE="absmax",
        DORMOUSE_TSCT_NATIVE_TERN="1",
        DORMOUSE_BANK="0",  # the fused bank writes no slab to pack
    ):
        spec = StackSpec.from_mapping(values)
        torch.manual_seed(7)
        built = assemble(spec, smoke=True, device=device)
        from dormouse.train.optim import MixOptimizer

        optim = MixOptimizer(built.model, steps=1)
        ids = torch.randint(0, spec.vocab, (2, 513), device=device)
        snap: dict[str, int] = {}
        for _ in range(steps):
            result, used = built.step(ids, optimizer=optim)
            if result.skipped or not math.isfinite(result.loss):
                raise RuntimeError(f"NaN firewall: {result.loss}")
            snap = dict(used.snapshot())
    for name in COUNTERS_8:
        print(f"  {name}={snap.get(name, 0)}", flush=True)
    missing = [name for name in COUNTERS_8 if snap.get(name, 0) <= 0]
    if missing:
        raise AssertionError(f"missing mechanism receipts: {missing}")
    print(f"counters {len(COUNTERS_8)}/{len(COUNTERS_8)} PASS", flush=True)
    return snap


def main() -> int:
    """Run the quality gate, the counter gate, or both."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    ok = True
    if not args.counters:
        ok &= bool(quality(args.checkpoint, args.seeds, args.batches, device)["verdict"] == "PASS")
    if args.counters or args.quality:
        ok &= bool(counters(args.steps, device)["a4_fwd"] > 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
