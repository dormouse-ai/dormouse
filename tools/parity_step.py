"""One step's numbers, dumped as JSON, so two builds can be compared without a diff.

Parity is a measurement, not a claim: a scheduling change to the step has to keep the loss
and the logits inside bf16's own epsilon and every gradient finite, and there is no way to
read that off a diff. So this runs ONE step at a fixed seed and a fixed token stream and
prints the numbers that would move — the loss, the logits' statistics, the global gradient
norm, whether any gradient is non-finite, and the parameter tree's norm before and after
the update — as JSON to stdout or a file.

    python tools/parity_step.py --out /tmp/base.json          # on the base commit
    python tools/parity_step.py --out /tmp/fixed.json         # on the branch
    python tools/parity_step.py --compare /tmp/base.json /tmp/fixed.json

`--lr-fraction` is a fraction of the recipe's rates and defaults to `0.01`: at 1.0 a b8
step on random ids occasionally drives one expert factor past the float range (the NaN
firewall in `train/step.py` is what catches it), and a parity dump that depends on which
side of that boundary a run landed on measures nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import chunked_cross_entropy

#: The seed both sides build and draw their tokens from.
SEED = 1234


def build(config: str, batch: int, seq: int | None, fraction: float):
    """Read the config, build the stack, and return `(spec, assembled, optimizer, ids)`."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=config)
    spec = replace(spec, batch=batch, seq_len=seq or spec.seq_len)
    torch.manual_seed(SEED)
    assembled = assemble(spec, device="cuda")
    optimizer = MixOptimizer(assembled.model)
    optimizer.set_lr(fraction)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    ids = torch.randint(
        0, spec.vocab, (spec.batch, spec.seq_len + 1), generator=generator, device="cuda"
    )
    return spec, assembled, optimizer, ids


def measure(config: str, batch: int, seq: int | None, fraction: float) -> dict[str, object]:
    """Run one step and return the numbers a change to its schedule would move."""
    spec, assembled, optimizer, ids = build(config, batch, seq, fraction)
    model = assembled.model
    inputs, targets = ids[:, :-1], ids[:, 1:]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = chunked_cross_entropy(logits, targets)
        loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    finite = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    moved = sum(1 for p in model.parameters() if p.grad is not None)
    #: Per-tensor gradient norms: a global norm that moves names no suspect, and the first
    #: question a scheduling change raises is WHICH gradient moved.
    per_tensor = {
        name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None
    }
    before = float(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).norm())
    optimizer.muon.step()
    optimizer.adamw.step()
    after = float(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).norm())
    logit = logits.detach().float().reshape(-1)
    return {
        "stack": f"{spec.n_layers}L {spec.d_model}/{spec.d_ffn} E={spec.n_experts} "
        f"top{spec.topk} b{spec.batch}xs{spec.seq_len}",
        "tokens": spec.batch * spec.seq_len,
        "loss": float(loss),
        "logits_sum": float(logit.sum()),
        "logits_abs_sum": float(logit.abs().sum()),
        "logits_max": float(logit.abs().max()),
        "logits_head": [float(v) for v in logit[:8]],
        "grad_norm": norm,
        "grads_finite": finite,
        "tensors_with_grad": moved,
        "param_norm_before": before,
        "param_norm_after": after,
        "grad_norms": per_tensor,
    }


def compare(left: Path, right: Path) -> int:
    """Print the two dumps side by side and gate them at bf16's own epsilon."""
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))
    print(f"{'quantity':22} {'base':>22} {'fixed':>22} {'relative':>12}")
    for key in (
        "loss",
        "logits_sum",
        "logits_abs_sum",
        "logits_max",
        "grad_norm",
        "param_norm_before",
        "param_norm_after",
    ):
        x, y = float(a[key]), float(b[key])
        rel = abs(x - y) / max(abs(x), 1e-30)
        print(f"{key:22} {x:22.10g} {y:22.10g} {rel:12.3e}")
    head = max(abs(x - y) for x, y in zip(a["logits_head"], b["logits_head"], strict=True))
    print(f"{'logits_head max|Δ|':22} {head:22.3e}")
    #: The gate the task is accepted against: the loss and the logits inside bf16's own
    #: epsilon. `param_norm_*` and the per-tensor table are REPORTED, not gated — the
    #: global gradient norm is an amplifier of the step's quantisation (a rank-space tile
    #: written bf16 and reused by the backward carries one 2⁻⁸ quantisation a recompute
    #: does not), and a tool that gated on it would be gating on which mechanism ran.
    gated = {
        key: abs(float(a[key]) - float(b[key])) / max(abs(float(a[key])), 1e-30)
        for key in ("loss", "logits_abs_sum", "logits_max")
    }
    worst = max(gated.values())
    ga, gb = a["grad_norms"], b["grad_norms"]
    movers = []
    for name in set(ga) & set(gb):
        scale = max(abs(ga[name]), abs(gb[name]))
        if scale == 0.0:
            continue
        movers.append((abs(ga[name] - gb[name]) / scale, ga[name], gb[name], name))
    movers.sort(reverse=True)
    print(f"\nper-tensor gradient norms: {len(movers)} compared, top 8 by relative move")
    print(f"  {'base':>14} {'fixed':>14} {'relative':>10}  tensor")
    for rel, x, y, name in movers[:8]:
        print(f"  {x:14.6g} {y:14.6g} {rel:10.3e}  {name}")
    finite = bool(a["grads_finite"]) and bool(b["grads_finite"])
    print(f"\nevery gradient finite: base={a['grads_finite']} fixed={b['grads_finite']}")
    print(f"loss and logits: worst relative move {worst:.3e} (bf16 epsilon 3.9e-3)")
    if not finite or worst > 3.9e-3:
        print("PARITY FAILED")
        return 1
    print("PARITY OK — loss and logits inside bf16 epsilon, every gradient finite")
    return 0


def main() -> int:
    """Measure one step, or compare two dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dormouse_1b.toml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--lr-fraction", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", type=Path, nargs=2, default=None)
    args = parser.parse_args()
    if args.compare is not None:
        return compare(*args.compare)
    dump = measure(args.config, args.batch, args.seq or None, args.lr_fraction)
    text = json.dumps(dump, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
