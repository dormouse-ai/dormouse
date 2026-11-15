"""The quality regression gate: 200 seeded steps, one number, one band.

Unit tests cannot see that learning broke. The KDA arm ran 3126 forwards with zero
backwards and every test stayed green; the Muon arm was broken and every test stayed
green. This script trains the live stack for a fixed number of steps on a committed
2 MB byte slice, in fp32 on CPU, with one thread and one seed, and asserts the final
loss is inside a band around a recorded value.

The assertion is a BAND, not bit-equality: the gate exists to catch a large change
(a loss that never moved, a step that made the model worse), not the last mantissa
bit. A one-line change to the model still moves the number, so when a change is
intended, re-record the baseline in the same commit and say why.

Baseline: final loss 2.452850 over 200 steps.
Measured on the one-architecture stack (3 KDA + 1 DSA per period, GQA + c_lat=512
projections, a four-expert MoA bank read at rank 32) with:

    uv run python tools/regression.py --steps 200

It replaced 2.621932, which was recorded on the dense-rope-and-two-block stack the
model no longer builds: the number belongs to the architecture, and re-recording it
is the point at which a change to that architecture is declared rather than absorbed.

The band is +/-0.10 (3.8% of the baseline). Every seed and shape below is part of
the measurement, not a preference: changing one changes the number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec

#: The recorded final loss. Re-record only with the command in the module docstring.
BASELINE = 2.452850

#: Half-width of the acceptance band, in loss units.
BAND = 0.10

#: The committed fixture: 2 000 000 byte-level tokens from the held-out slice.
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus_2mb.bin"

#: The run these numbers belong to. Changing any of these is a new baseline.
SEED = 1234
BATCH = 8
SEQ_LEN = 128
LR = 3e-3
VOCAB = 256

#: The stack, at widths one CPU core can run in seconds: one full attention period
#: (3 KDA + 1 DSA), the GQA/latent projection block, and a four-expert bank of MoA
#: experts read through rank-32 factors. Every slot of the live model is here, so a
#: wiring defect in any of them moves this number. `unit="byte"` because the fixture
#: is raw bytes: the BPE table has no tokenizer in this repo.
SPEC = StackSpec(
    unit="byte",
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


def run(steps: int, data: bytes) -> tuple[float, float, int]:
    """Train for `steps` and return `(first loss, final loss, parameter count)`."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    tokens = np.frombuffer(data, dtype=np.uint8)
    model = assemble(SPEC, smoke=True).model
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    span = BATCH * (SEQ_LEN + 1)
    stride = len(tokens) - span
    first, final = float("inf"), float("inf")
    started = time.perf_counter()
    for step in range(steps):
        start = (step * BATCH * SEQ_LEN) % stride
        window = torch.from_numpy(tokens[start : start + span].astype(np.int64))
        window = window.reshape(BATCH, SEQ_LEN + 1)
        logits = model(window[:, :-1])
        loss = functional.cross_entropy(logits.reshape(-1, VOCAB), window[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
        if step == 0:
            first = final
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d} loss {final:.6f}", flush=True)
    print(f"{params} params, {steps} steps, {time.perf_counter() - started:.1f}s", flush=True)
    return first, final, params


def check(steps: int, final: float, first: float) -> list[str]:
    """Return the failures; an empty list means the gate passed."""
    failures = []
    if steps == 200:
        if abs(final - BASELINE) > BAND:
            failures.append(
                f"final loss {final:.6f} is outside {BASELINE:.6f} +/- {BAND}: learning changed"
            )
    elif final > BASELINE + BAND:
        failures.append(
            f"{steps} steps ended at {final:.6f}, worse than the 200-step band's top "
            f"{BASELINE + BAND:.6f}: more compute must not cost quality"
        )
    if first - final < 1.0:
        failures.append(f"loss went {first:.6f} -> {final:.6f}: the run did not learn")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Run the gate; exit 1 with every failed assertion printed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200, help="training steps (nightly: 1000)")
    parser.add_argument("--data", type=Path, default=FIXTURE, help="byte-level corpus slice")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    data = args.data.read_bytes()
    first, final, params = run(args.steps, data)
    failures = check(args.steps, final, first)
    print(f"baseline {BASELINE:.6f} band +/-{BAND} -> {final:.6f} ({params} params)")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
