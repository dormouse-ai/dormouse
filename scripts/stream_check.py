"""Convergence check for the 2h token stream: 200 steps, first/final loss and held-out BPB.

Two steps of `dorm train --smoke` cannot move cross-entropy from this stack's random
init into the 2-4 BPB band; this is the runnable check that the stream is learnable and
that its manifest's measured bytes_per_token produces a sane BPB.

    PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/python \
      scripts/stream_check.py --train data-prep-2h/train.bin \
      --heldout data-prep-2h/heldout.bin --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from dormouse.data.bin import Stream
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec

BATCH = 8
SEQ_LEN = 128
LR = 3e-3
SEED = 1234


def spec() -> StackSpec:
    """Build the regression stack at the tokenizer's real vocabulary."""
    return StackSpec(
        unit="bpe",
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        c_lat=32,
        d_ffn=128,
        n_experts=4,
        topk=1,
        expert_tsct_rank=32,
        attn_k=64,
        attn_window=32,
        seq_len=SEQ_LEN,
        batch=BATCH,
    )


def window(data: np.ndarray, step: int) -> torch.Tensor:
    """One deterministic `[batch, seq+1]` window, positioned by the step."""
    span = BATCH * (SEQ_LEN + 1)
    start = (step * BATCH * SEQ_LEN) % (data.size - span)
    return torch.from_numpy(data[start : start + span].astype(np.int64)).reshape(BATCH, SEQ_LEN + 1)


def main() -> None:
    """Train, then print the measured BPB on documents the train slice never saw."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data-prep-2h/train.bin"))
    p.add_argument("--heldout", type=Path, default=Path("data-prep-2h/heldout.bin"))
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    train, held = Stream(args.train), Stream(args.heldout)
    meta = json.loads(Path(str(args.train) + ".json").read_text(encoding="utf-8"))
    bpt = float(meta["bytes_per_token"])
    model = assemble(spec(), smoke=True).model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    first = final = 0.0
    for step in range(args.steps):
        ids = window(np.asarray(train._buf), step)
        logits = model(ids[:, :-1])
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
        if step == 0:
            first = final
        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d} loss {final:.4f} bpb {final / bpt / math.log(2):.3f}",
                flush=True,
            )

    ids = window(np.asarray(held._buf), 0)
    with torch.no_grad():
        held_loss = float(
            functional.cross_entropy(
                model(ids[:, :-1]).reshape(-1, model.vocab), ids[:, 1:].reshape(-1)
            )
        )
    held_bpb = held_loss / bpt / math.log(2)
    print(
        json.dumps(
            {
                "first_loss": round(first, 4),
                "final_loss": round(final, 4),
                "heldout_loss": round(held_loss, 4),
                "bytes_per_token": bpt,
                "heldout_bpb": round(held_bpb, 4),
                "learned": final < first - 1.0,
                "bpb_in_band": 2.0 <= held_bpb <= 4.0,
            }
        )
    )


if __name__ == "__main__":
    main()
