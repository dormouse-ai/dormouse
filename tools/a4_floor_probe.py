"""Two reference measurements that attribute the paired dBPB noise floor.

`--floor` trains the SAME bf16 control twice from one seed and scores both: whatever
dBPB that pair shows is baseline nondeterminism, not a quantization effect.
`--frozen` trains one A4+ternary model and scores it with its A4 norms on and off:
that is the pure activation-quantization gap on a shared trajectory.

    PYTHONPATH=<wt>/src:<core>/src python tools/a4_floor_probe.py --floor --steps 200
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics

import torch

from dormouse.model import quant
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy
from tools.a4_quality_gate import _bpb, _flags, _ids, _train, spec_for


def floor(seeds: list[int], steps: int, device: torch.device, lr_scale: float) -> None:
    """Print the control-vs-control dBPB floor; this is the noise any A/B inherits."""
    deltas: list[float] = []
    for seed in seeds:
        _flags(a4=False, variant="near-absmax", seed=seed, step=0)
        spec = spec_for(a4=False)
        first, _, _, _ = _train(spec, seed, steps, device, lr_scale)
        second, _, _, _ = _train(spec, seed, steps, device, lr_scale)
        delta = _bpb(second, spec) - _bpb(first, spec)
        deltas.append(delta)
        print(
            f"floor seed {seed}: control-vs-control dBPB={delta:+.5f} "
            f"(nats {first:.4f}->{second:.4f})",
            flush=True,
        )
    mean = statistics.fmean(deltas)
    print(f"FLOOR mean={mean:+.5f} per-seed={[round(d, 5) for d in deltas]}", flush=True)


def _score_a4(model: torch.nn.Module, held: torch.Tensor, spec: StackSpec, a4_on: bool) -> float:
    """Score one held-out window with the A4 boundary toggled on the same weights."""
    for norm in model.modules():
        if hasattr(norm, "a4"):
            norm.a4 = a4_on
    with torch.no_grad():
        logits = model(held[:, :-1])
        loss = float(chunked_cross_entropy(logits.float(), held[:, 1:]))
    return _bpb(loss, spec)


def frozen(seed: int, steps: int, device: torch.device, lr_scale: float, variant: str) -> None:
    """Score one trained model twice, toggling only its A4 activation boundary."""
    _flags(a4=True, variant=variant, seed=seed, step=0)
    spec: StackSpec = spec_for(a4=True)
    torch.manual_seed(seed)
    built = assemble(spec, smoke=True, device=device)
    optimizer = MixOptimizer(built.model, steps=1)
    optimizer.set_lr(lr_scale)
    ids = _ids(spec, seed, device)
    for step in range(steps):
        quant.set_step_seed(seed, step)
        result, _ = built.step(ids, optimizer=optimizer)
        if result.skipped or not math.isfinite(result.loss):
            raise RuntimeError(f"NaN firewall at {seed}/{step}: {result.loss}")
    held = _ids(spec, seed + 10_000, device)
    built.model.eval()
    on = _score_a4(built.model, held, spec, True)
    off = _score_a4(built.model, held, spec, False)
    print(
        f"FROZEN {variant} seed {seed}: A4-on={on:.5f} A4-off={off:.5f} dBPB={on - off:+.5f}",
        flush=True,
    )
    del built, optimizer, ids, held
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    """Run whichever reference measurement the caller requested."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--variant", default="sr-pc")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.floor:
        floor(list(range(args.seeds)), args.steps, device, args.lr_scale)
    if args.frozen:
        for seed in range(args.seeds):
            frozen(seed, args.steps, device, args.lr_scale, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
