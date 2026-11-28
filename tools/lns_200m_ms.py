"""Interleaved on-card ms/step cost of LNS on the shipped 200M stack.

One model, weights frozen, `lns_scale` flipped in place: this isolates the cost of
the arithmetic from the init/routing change that the quality A/B measures. The shipped
Triton bank is ON (no BANK=0 workaround). CUDA-event phases come from the repo's own
`tools/profile_step.py`.

    python tools/lns_200m_ms.py --config /tmp/opencode/lns200m/ctrl_s11.toml \
        --rounds 7 --out /tmp/opencode/lns200m/ms.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from profile_step import build, phases


def main(argv: list[str] | None = None) -> int:
    """Time both arms in alternating rounds and print the +1% cost gate."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("/tmp/opencode/lns200m/ms.json"))
    args = p.parse_args(argv)

    torch.manual_seed(11)
    spec, assembled, optimizer = build(str(args.config), args.batch, args.seq)
    optimizer.set_lr(0.0)
    blocks = list(assembled.model.blocks)
    scales = [1.0 / math.sqrt(i) for i in range(1, len(blocks) + 1)]
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1), device="cuda")
    times: dict[bool, list[float]] = {False: [], True: []}

    def one(on: bool) -> float:
        for block, scale in zip(blocks, scales, strict=True):
            block.lns_scale = scale if on else 1.0
            block._lns = on
        return phases(assembled, optimizer, ids, 1, True)["total"]

    for _ in range(2):
        one(False)
        one(True)
    for round_no in range(args.rounds):
        for on in (False, True) if round_no % 2 == 0 else (True, False):
            times[on].append(one(on))
    off, on = min(times[False]), min(times[True])
    cost = on / off - 1.0
    result = {
        "ms_ctrl_min": round(off, 3),
        "ms_lns_min": round(on, 3),
        "ms_cost": round(cost, 6),
        "rounds": args.rounds,
        "gate": cost <= 0.01,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
