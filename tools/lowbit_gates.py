"""The four measurements the 8-bit state is accepted or rejected against.

    direction   the cosine between Muon's polar factor at 32 bits and at 8, on the live
                shapes. This decides whether the momentum buffer may be quantised at all.
    quality     three seeds, 300 steps each, on a synthetic byte chain: 32-bit state
                against 8-bit, same data order. Bar: the two means within 0.01 BPB.
    stability   200 steps with a fixed spike schedule, both arms, reported side by side.
    memory      the 1.05B config built and stepped once, 32 bits against 8, peak RSS.

`quality` and `stability` are smokes: 300 steps on a fixture that is not the corpus
cannot support a claim about a real run, only a comparison between two arms under the
same budget. `memory` builds the real stack from `configs/dormouse_1b.toml` and is the
number that decides whether the run fits; it runs each arm in its own process, because a
peak RSS is a property of a process and not of a loop.

    python tools/lowbit_gates.py [--only direction|quality|stability|memory]
"""

from __future__ import annotations

import argparse
import math
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

#: The stack the learned gates train is the tests' reference stack, not the 1B model: a
#: gate about two optimizers must not also be a test of `assemble`. `tests/` is not a
#: package, so the path goes in before the import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from stacks import reference_stack

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.lowbit import BLOCK, LowBitState
from dormouse.train.muon import muon_direction
from dormouse.train.optim import MixOptimizer
from dormouse.train.schedule import warmup_cosine

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dormouse_1b.toml"

#: `tests/stacks.py`'s tiny stack with a byte alphabet, so one token is one byte and
#: `nats/ln 2` is bits per byte. The recipe's rates are for a 832-wide stack; what makes
#: them usable here is the schedule, not a different number.
FIXTURE = {"vocab": 256, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}
BATCH, SEQ = 8, 32
#: Steps of ramp before the cosine anneal to zero: the recipe's shape, scaled to a smoke.
WARMUP = 20
SEEDS = (0, 1, 2)
QUALITY_STEPS = 300
STABILITY_STEPS = 200
#: Every spike is a batch drawn uniformly: the loss jumps to ln(vocab) and the next step
#: is a large gradient, which is what a quantised moment has to survive.
SPIKES = tuple(range(30, STABILITY_STEPS, 30))
QUALITY_BAR = 0.01  # BPB, the bar the two methods are compared at
NOISE = 0.15


def _batches(seed: int, steps: int, *, spikes: Sequence[int] = ()) -> list[torch.Tensor]:
    """Return `steps` batches of a first-order chain over `vocab` symbols, noise at spikes.

    A permutation with 15% noise is learnable — the loss falls from `ln(256)` towards the
    chain's entropy — which is what makes the arm comparison mean something: at the
    fixture's floor both arms would agree by construction.
    """
    generator = torch.Generator().manual_seed(seed)
    vocab = FIXTURE["vocab"]
    perm = torch.randperm(vocab, generator=generator)
    tokens = torch.randint(0, vocab, (BATCH, 1), generator=generator)
    batches = []
    for step in range(steps):
        if step in spikes:
            batches.append(torch.randint(0, vocab, (BATCH, SEQ + 1), generator=generator))
            continue
        sequence = [tokens]
        for _ in range(SEQ):
            nxt = perm[sequence[-1]]
            noise = torch.rand(BATCH, 1, generator=generator) < NOISE
            drawn = torch.randint(0, vocab, (BATCH, 1), generator=generator)
            sequence.append(torch.where(noise, drawn, nxt))
        tokens = sequence[-1]
        batches.append(torch.cat(sequence, dim=1))
    return batches


def _run(bits: int, batches: list[torch.Tensor]) -> list[float]:
    """Train one arm for the whole list of batches and return the per-step loss, in nats."""
    model = reference_stack(max_seq_len=SEQ, **FIXTURE)
    optimizer = MixOptimizer(model, optim_state_bits=bits, optim_momentum_bits=bits)
    losses = []
    for step, ids in enumerate(batches):
        optimizer.set_lr(warmup_cosine(step, warmup=WARMUP, total=len(batches), peak=1.0))
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _bpb(nats: Sequence[float]) -> float:
    """Bits per byte: the fixture's tokens are bytes, so it is `nats/ln 2` per token."""
    return sum(nats) / len(nats) / math.log(2)


def direction() -> None:
    """Report the cosine at 32 bits against 8, on `[832, 2048]` and `[832, 832]`."""
    live = {"vocab": 64, "d_model": 832, "d_ffn": 2048, "n_heads": 13, "head_dim": 64}
    torch.manual_seed(0)
    model = reference_stack(n_layers=1, max_seq_len=8, **live)
    optimizer = MixOptimizer(model)
    ids = torch.randint(0, live["vocab"], (2, 9))
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids[:, :-1])
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        ).backward()
        optimizer.step()
    params = dict(model.named_parameters())
    print(f"{'parameter':34s} {'shape':12s} {'block':>5s} {'cosine':>10s}")
    for name in ("blocks.0.ffn.down.weight", "blocks.0.attn.proj.weight"):
        buffer = optimizer.muon.state[params[name]]["momentum"].dense()
        exact = muon_direction(buffer)
        for block in (BLOCK, 64, 16):
            state = LowBitState(buffer, bits=8, block=block)
            state.apply(lambda window, source: window.copy_(source), buffer)
            quantised = muon_direction(state.dense())
            cosine = torch.nn.functional.cosine_similarity(
                exact.reshape(1, -1), quantised.reshape(1, -1)
            ).item()
            shape = tuple(buffer.shape)
            print(f"{name:34s} {shape!s:12s} {block:5d} {cosine:10.6f}")


def quality() -> None:
    """Run three seeds for 300 steps each, both arms, same batches; report the final loss."""
    print(f"quality: {QUALITY_STEPS} steps, batch {BATCH} x {SEQ}, {len(SEEDS)} seeds")
    print(f"{'seed':>4s} {'32-bit BPB':>12s} {'8-bit BPB':>12s} {'delta':>9s}")
    dense, quantised = [], []
    for seed in SEEDS:
        batches = _batches(seed, QUALITY_STEPS)
        final_dense = _bpb(_run(32, batches)[-10:])
        final_low = _bpb(_run(8, batches)[-10:])
        dense.append(final_dense)
        quantised.append(final_low)
        print(f"{seed:4d} {final_dense:12.4f} {final_low:12.4f} {final_low - final_dense:+9.4f}")
    mean_dense = sum(dense) / len(dense)
    mean_low = sum(quantised) / len(quantised)
    spread = math.sqrt(sum((x - mean_low) ** 2 for x in quantised) / len(quantised))
    print(
        f"mean 32-bit {mean_dense:.4f} BPB, mean 8-bit {mean_low:.4f} ± {spread:.4f}, "
        f"delta {mean_low - mean_dense:+.4f} BPB (bar {QUALITY_BAR})"
    )
    verdict = abs(mean_low - mean_dense) <= QUALITY_BAR
    print(f"quality: {'PASS' if verdict else 'FAIL'} — smoke at 300 steps, not a claim")


def stability() -> None:
    """200 steps with a fixed spike schedule: neither arm may diverge, 8 no more than 32."""
    print(f"stability: {STABILITY_STEPS} steps, spikes at {SPIKES}")
    batches = _batches(0, STABILITY_STEPS, spikes=SPIKES)
    for bits in (32, 8):
        losses = _run(bits, batches)
        finite = all(math.isfinite(value) for value in losses)
        peak = max(losses[spike + 1] for spike in SPIKES if spike + 1 < len(losses))
        print(
            f"{bits:2d}-bit state: final {losses[-1]:.4f} nats, peak after a spike {peak:.4f}, "
            f"max {max(losses):.4f}, finite {finite}"
        )


def memory() -> None:
    """One 1.05B construction plus one optimizer step per arm, in its own process."""
    print("memory: 1.05B config, construction + one step, peak RSS")
    print(f"{'arm':22s} {'peak GB':>9s} {'wall s':>8s}")
    peaks = {}
    for label, bits, momentum_bits in (
        ("state 8-bit", 8, 8),
        ("state 32-bit", 32, 32),
        ("8-bit v, 32-bit mom", 8, 32),
    ):
        started = time.time()
        #: A child per arm: a peak RSS belongs to a process, and the parent must not carry
        #: one arm's 4 GB of moments into the next arm's measurement.
        result = subprocess.run(
            [sys.executable, __file__, "--arm", str(bits), str(momentum_bits)],
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.time() - started
        if result.returncode != 0:
            print(f"{label:22s} failed: {result.stderr.strip().splitlines()[-1]}")
            continue
        count, peak = (float(part) for part in result.stdout.strip().splitlines()[-1].split())
        peaks[label] = peak
        print(f"{label:22s} {peak:9.2f} {wall:8.1f}   ({count / 1e9:.3f}B parameters)")
    if "state 8-bit" in peaks and "state 32-bit" in peaks:
        delta = peaks["state 32-bit"] - peaks["state 8-bit"]
        print(f"delta (32-bit state minus 8-bit state): {delta:+.2f} GB")


def _arm(bits: int, momentum_bits: int) -> None:
    """Build the config's stack in a child, give every parameter a gradient, step once."""
    #: A ceiling, so a box already carrying another lane's job fails this arm with a
    #: MemoryError instead of taking the OOM killer's choice of victim.
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    spec = StackSpec.from_toml(CONFIG)
    model = assemble(spec).model
    optimizer = MixOptimizer(
        model,
        optim_state_bits=bits,
        optim_momentum_bits=momentum_bits,
        optim_first_moment_bits=bits,
    )
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"{sum(p.numel() for p in model.parameters())} {peak:.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one gate, or all four."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("direction", "quality", "stability", "memory"))
    parser.add_argument("--arm", nargs=2, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.arm is not None:
        _arm(args.arm[0], args.arm[1])
        return 0
    gates: dict[str, Callable[[], None]] = {
        "direction": direction,
        "quality": quality,
        "stability": stability,
        "memory": memory,
    }
    for name, gate in gates.items():
        if args.only in (None, name):
            gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
