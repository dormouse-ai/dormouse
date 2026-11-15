"""The four measurements the 8-bit state is accepted or rejected against.

    direction   the cosine between Muon's polar factor at fp32 and at bf16/8 bits, on the
                live shapes, over five independent momentum draws. This decides whether
                the momentum buffer may be quantised at all, and at which width.
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
from dormouse.train.graph import state_bytes
from dormouse.train.lowbit import LowBitState
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
QUALITY_STEPS = 200
STABILITY_STEPS = 200
#: Every spike is a batch drawn uniformly: the loss jumps to ln(vocab) and the next step
#: is a large gradient, which is what a quantised moment has to survive.
SPIKES = tuple(range(30, STABILITY_STEPS, 30))
QUALITY_BAR = 0.01  # BPB, the bar the quantised arms are compared against fp32 at
#: The cosine a quantised momentum has to hold. Below it the buffer stays fp32.
DIRECTION_BAR = 0.999
#: Momentum draws per shape: one buffer is a draw, five is a distribution.
DRAWS = 5
#: Steps into a run a draw is taken at.
DRAW_STEPS = 20
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


def _run(bits: int, batches: list[torch.Tensor], momentum_bits: int = 32) -> list[float]:
    """Train one arm for the whole list of batches and return the per-step loss, in nats."""
    model = reference_stack(max_seq_len=SEQ, **FIXTURE)
    optimizer = MixOptimizer(model, optim_state_bits=bits, optim_momentum_bits=momentum_bits)
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


def _momentum(seed: int, steps: int) -> dict[str, torch.Tensor]:
    """Return the momentum buffers of a live run, keyed by parameter name.

    One draw is one run with its own weights, data and batch order — the buffer is an EMA
    of gradients of a model whose spectrum is not a gaussian's, and the earlier gate
    measured that difference at 0.954 against 0.9995.
    """
    live = {"vocab": 64, "d_model": 832, "d_ffn": 2048, "n_heads": 13, "head_dim": 64}
    torch.manual_seed(seed)
    model = reference_stack(n_layers=1, max_seq_len=8, **live)
    optimizer = MixOptimizer(model)
    ids = torch.randint(0, live["vocab"], (2, 9), generator=torch.Generator().manual_seed(seed))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids[:, :-1])
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        ).backward()
        optimizer.step()
    params = dict(model.named_parameters())
    return {
        name: optimizer.muon.state[params[name]]["momentum"].dense()
        for name in (
            "blocks.0.ffn.down.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.attn.qkv.weight",
        )
    }


def _codecs(momentum: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the buffer at each stored width, each dequantised back to fp32."""
    quantised = LowBitState(momentum, bits=8)
    quantised.apply(lambda window, source: window.copy_(source), momentum)
    return {"bf16": momentum.bfloat16().float(), "int8": quantised.dense()}


def direction() -> None:
    """Report the cosine at fp32 against bf16 and 8 bits, on the live shapes, five draws."""
    #: The packed qkv is orthogonalised per head, so the live call is the one measured.
    split = {"blocks.0.attn.qkv.weight": (26, 1664)}
    rows: dict[tuple[str, str], list[float]] = {}
    for seed in range(DRAWS):
        for name, momentum in _momentum(seed, DRAW_STEPS).items():
            exact = muon_direction(momentum, head_split=split.get(name))
            for codec, stored in _codecs(momentum).items():
                got = muon_direction(stored, head_split=split.get(name))
                cosine = torch.nn.functional.cosine_similarity(
                    exact.reshape(1, -1), got.reshape(1, -1)
                ).item()
                rows.setdefault((name, codec), []).append(cosine)
    print(f"direction: {DRAWS} draws, momentum {DRAW_STEPS} steps in, bar {DIRECTION_BAR}")
    print(f"{'parameter':29s} {'codec':6s} {'min':>9s} {'mean':>9s} {'max':>9s}")
    for (name, codec), values in sorted(rows.items()):
        print(
            f"{name:29s} {codec:6s} {min(values):9.6f} "
            f"{sum(values) / len(values):9.6f} {max(values):9.6f}"
        )


def quality() -> None:
    """Run three seeds at each state width, same batches, and report the BPB each lands at."""
    arms = (
        ("fp32 (reference)", 32, 32),
        ("momentum bf16", 32, 16),
        ("v bf16", 16, 32),
        ("v 8-bit", 8, 32),
        ("v 8 + mom bf16", 8, 16),
    )
    print(f"quality: {QUALITY_STEPS} steps, batch {BATCH} x {SEQ}, {len(SEEDS)} seeds")
    print(
        f"{'arm':18s} "
        + " ".join(f"{f'seed {seed}':>9s}" for seed in SEEDS)
        + f" {'mean':>8s} {'spread':>7s}"
    )
    reference = 0.0
    for label, bits, momentum_bits in arms:
        finals = []
        for seed in SEEDS:
            batches = _batches(seed, QUALITY_STEPS)
            finals.append(_bpb(_run(bits, batches, momentum_bits)[-10:]))
        mean = sum(finals) / len(finals)
        spread = math.sqrt(sum((x - mean) ** 2 for x in finals) / len(finals))
        if not reference:
            reference = mean
        delta = mean - reference
        verdict = "" if not reference else f"  {delta:+.4f} vs fp32"
        print(
            f"{label:18s} "
            + " ".join(f"{value:9.4f}" for value in finals)
            + f" {mean:8.4f} {spread:7.4f}{verdict}"
        )
    print(f"quality: bar {QUALITY_BAR} BPB — smoke at {QUALITY_STEPS} steps, not a claim")


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
    print("memory: 1.05B config, construction + one step, peak RSS and state GB")
    print(f"{'arm':22s} {'peak GB':>9s} {'state':>9s} {'wall s':>8s}")
    peaks = {}
    for label, bits, momentum_bits in (
        ("state 8-bit", 8, 8),
        ("state 32-bit", 32, 32),
        ("v 8 + momentum bf16", 8, 16),
        ("all bf16", 16, 16),
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
        count, peak, state = (
            float(part) for part in result.stdout.strip().splitlines()[-1].split()
        )
        peaks[label] = peak
        print(f"{label:22s} {peak:9.2f} {state:9.2f} {wall:8.1f}   ({count / 1e9:.3f}B parameters)")
    for label, peak in peaks.items():
        if label != "state 32-bit":
            print(f"delta (32-bit state minus {label}): {peaks['state 32-bit'] - peak:+.2f} GB")


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
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    grads = sum(p.numel() * p.element_size() for p in model.parameters() if p.grad is not None)
    count = sum(p.numel() for p in model.parameters())
    #: The moments alone: `state_bytes` counts weights and gradients too, and only the
    #: moments are what the width knob changes.
    moments = state_bytes(model, optimizer) - weights - grads
    print(f"{count} {peak:.2f} {moments / 1024**3:.2f}")


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
