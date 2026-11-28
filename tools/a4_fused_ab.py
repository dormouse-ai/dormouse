"""On-card A/B: bf16 control vs eager A4 vs fused A4, interleaved min-of-15.

    taskset -c 4 /home/sehaxe/dormouse2/app/.venv-cuda/bin/python tools/a4_fused_ab.py

The flock is taken in-process with fcntl (shell `flock` is starved by sibling
jobs on this box). Timing arms are built once, warmed five times, then run in
interleaved rounds so driver/thermal drift cancels. Every measurement is a
CUDA-event ms for one whole optimizer step, synchronised on both sides.

    --vram   peak allocated MiB per arm, one arm resident at a time
    --quality 20-step fused-vs-eager loss trace on identical ids and init
"""

from __future__ import annotations

import argparse
import fcntl
import os
import statistics
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("bf16", "ternary_control", "eager_a4", "fused_a4")
LOCK = "/tmp/dormouse-gpu.lock"
#: A2 measured the cheap `ternary` (detached mean|x| scale) arm. The newer
#: `ternary_native` TSCT factors cost far more per step; `--native-tern` switches
#: the whole table to that regime so the two overheads are never conflated.
QAT = "ternary"


@contextmanager
def card_lock(timeout: int = 2400) -> Iterator[None]:
    """Hold the shared-card lock; wait up to `timeout` seconds by name."""
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                os.close(fd)
                raise TimeoutError(f"card lock {LOCK} held > {timeout}s") from None
            time.sleep(1)
    try:
        yield
    finally:
        os.close(fd)


@contextmanager
def arm_env(arm: str) -> Iterator[None]:
    """Set the flag slots that decide which arithmetic and which kernel run."""
    saved = {
        k: os.environ.get(k)
        for k in ("DORMOUSE_TSCT_NATIVE_TERN", "DORMOUSE_ACT_A4", "DORMOUSE_A4_FUSED")
    }
    if arm == "bf16":
        os.environ.pop("DORMOUSE_TSCT_NATIVE_TERN", None)
        os.environ.pop("DORMOUSE_ACT_A4", None)
        os.environ["DORMOUSE_A4_FUSED"] = "0"
    else:
        if QAT == "ternary_native":
            os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = "1"
        else:
            os.environ.pop("DORMOUSE_TSCT_NATIVE_TERN", None)
        if arm in ("eager_a4", "fused_a4"):
            os.environ["DORMOUSE_ACT_A4"] = "1"
        else:
            os.environ.pop("DORMOUSE_ACT_A4", None)
        os.environ["DORMOUSE_A4_FUSED"] = "1" if arm == "fused_a4" else "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _spec(arm: str) -> StackSpec:
    """Build the small stack of the A4 timing reference, with two quant slots."""
    raw = tomllib.loads((ROOT / "configs/dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {f.name for f in fields(StackSpec) if f.init}
    values = {k: v for k, v in raw.items() if k in known}
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
        act_quant="a4" if arm in ("eager_a4", "fused_a4") else "none",
        weight_qat="none" if arm == "bf16" else QAT,
    )
    return StackSpec.from_mapping(values)


def _build(arm: str, seed: int = 7) -> tuple[Any, MixOptimizer, torch.Tensor]:
    with arm_env(arm):
        torch.manual_seed(seed)
        built = assemble(_spec(arm), smoke=True, device="cuda")
        ids = torch.randint(
            0,
            _spec(arm).vocab,
            (2, 513),
            generator=torch.Generator().manual_seed(seed),
        ).cuda()
        return built, MixOptimizer(built.model, steps=1), ids


def _timed(arm: str, built: Any, optimizer: MixOptimizer, ids: torch.Tensor) -> float:
    with arm_env(arm):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        built.step(ids, optimizer=optimizer)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end)


def timing(rounds: int) -> dict[str, list[float]]:
    """Min/median of `rounds` interleaved whole-step CUDA-event timings per arm."""
    arms = {name: _build(name) for name in ARMS}
    for arm, bundle in arms.items():
        for _ in range(5):
            _timed(arm, *bundle)
    samples = {name: [] for name in ARMS}
    for _ in range(rounds):
        for name in ARMS:
            samples[name].append(_timed(name, *arms[name]))
    for name in ARMS:
        ms = samples[name]
        print(f"{name:10s} min={min(ms):.1f} ms median={statistics.median(ms):.1f} ms")
    base, fused = min(samples["bf16"]), min(samples["fused_a4"])
    delta = 100.0 * (fused / base - 1.0)
    verdict = "PASS" if delta <= 5.0 else "REFUTE"
    print(f"fused vs bf16 min-of-{rounds}: {delta:+.1f}% (gate <= +5.0%) -> {verdict}")
    tern = min(samples["ternary_control"])
    iso = 100.0 * (fused / tern - 1.0)
    print(f"fused vs ternary-only control: {iso:+.1f}% (A4 overhead after fusion)")
    #: The canonical config pairs A4 with ternary_native. Against a plain bf16
    #: stack the native arm's own cost (~+75%) swamps A4; the honest +5% gate
    #: isolates A4 against the same weights with bf16 activations.
    canonical = "PASS" if iso <= 5.0 else "REFUTE"
    print(f"canonical fused A4 vs same-weight control: {iso:+.1f}% (+5% gate) -> {canonical}")
    return samples


def vram(steps: int = 2) -> dict[str, float]:
    """Peak allocated MiB, with only one arm resident at a time."""
    peaks: dict[str, float] = {}
    for name in ARMS:
        built, optimizer, ids = _build(name)
        with arm_env(name):
            for _ in range(5):
                built.step(ids, optimizer=optimizer)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(steps):
                built.step(ids, optimizer=optimizer)
            torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{name:10s} peak={peaks[name]:.1f} MiB")
        del built, optimizer, ids
        torch.cuda.empty_cache()
    return peaks


def _trace(arm: str, steps: int, seed: int = 13) -> tuple[list[float], Any]:
    """Run one isolated arm and return its loss trace and final counter snapshot."""
    built, optimizer, ids = _build(arm, seed=seed)
    losses: list[float] = []
    counters = None
    with arm_env(arm):
        for _ in range(steps):
            result, counters = built.step(ids, optimizer=optimizer)
            assert not result.skipped, f"NaN firewall at {arm}"
            losses.append(result.loss)
    del built, optimizer, ids
    return losses, counters


def quality(steps: int) -> None:
    """Judge fused against eager inside the stack's own eager-vs-eager noise.

    The bank's CUDA atomics make two identical eager arms diverge after step 0,
    so bitwise loss equality is not an available gate here; bitwise forward
    equality is asserted in tests/test_a4_fused.py instead.
    """
    e1, _ = _trace("eager_a4", steps)
    e2, _ = _trace("eager_a4", steps)
    fused, counters = _trace("fused_a4", steps)
    assert counters is not None
    for key in ("a4_fwd", "a4_bwd", "a4_fused_fwd", "a4_fused_bwd"):
        assert counters.counts[key] > 0, key
    noise = max(abs(a - b) for a, b in zip(e1, e2, strict=True))
    actual = max(abs(a - b) for a, b in zip(e1, fused, strict=True))
    print(f"{steps}-step trace: eager-vs-eager noise={noise:.3e}, fused-vs-eager={actual:.3e}")
    print(f"quality gate: {'PASS' if actual <= noise * 1.5 + 1e-6 else 'REFUTE'}")


def main() -> int:
    """Print the card A/B table under the shared lock."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--vram", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--native-tern", action="store_true")
    args = parser.parse_args()
    global QAT
    QAT = "ternary_native" if args.native_tern else "ternary"
    print(f"regime: weight_qat={QAT}, act_quant=a4 on the A4 arms")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA card required")
    with card_lock():
        if args.vram:
            vram()
        if not args.timing_only and not args.vram:
            quality(args.rounds if args.quality else 20)
        if not args.vram:
            timing(args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
