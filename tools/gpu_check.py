"""What CPU CI cannot see, on the card: four checks, each one a measured defect.

Every serious defect this project has shipped was GPU-side and invisible to a green
CPU suite: an arm that ran 3126 forwards with zero backwards, kernels that failed to
compile for the arch, a bf16 path that silently fell back to fp32, a batch that did
not fit, a CUDA-graph capture that silently ran eager. So this script runs the four
checks that need the card, and fails on the number, not on the log:

1. TRAIN — 20 steps of the live recipe, in fp32, on the card. Every loss finite, and
   every mechanism the stack declares enabled reports a non-zero counter at the end
   of the step it ran in. A mechanism that is silently absent raises `MissingReport`
   from inside `train_step`, before the loss is ever returned.
2. DTYPE — the first step's loss under bf16 autocast against its fp32 twin, from the
   same seed and the same weights, within `DTYPE_TOL` relative. The tolerance alone
   cannot see a silent fallback: fp32 fallback gives a *closer* match, not a worse
   one. The canary does — one matmul inside the same autocast must come back bf16, so
   a fallback fails on the dtype, and a wrong-but-bf16 result fails on the loss.
3. TRITON — a kernel compiled and called through the same toolchain a `@triton.jit`
   kernel in `model/` would use. An unsupported arch fails here in ptxas, loudly,
   instead of at 3 a.m. inside a training run.
4. VRAM — the peak allocation of the three checks above against a hard ceiling.

Exit 1 with every failed assertion printed. The peak, the tolerances and the ceiling
go to `$GITHUB_STEP_SUMMARY` when it is set, so the number is on the run page.

    uv run --no-sync python tools/gpu_check.py --max-vram-gb 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dormouse_core.counters import Counters

from dormouse.model.model import Transformer
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

#: The live recipe — `Transformer`'s own defaults, spelled out so a change to them is
#: a change to this check rather than a silently different model under test.
LIVE = {
    "vocab": 256,
    "d_model": 832,
    "n_layers": 12,
    "n_heads": 13,
    "head_dim": 64,
    "d_ffn": 2048,
    "max_seq_len": 1024,
}

#: Steps, batch and window. Twenty is the shortest run that has ever shown the
#: zero-backwards class; batch 8 by 512 tokens is what fits a 16 GB card beside a desktop.
STEPS = 20
BATCH = 8
SEQ_LEN = 512
SEED = 1234

#: bf16 vs fp32 relative tolerance on the first step's loss. fp32 upcasts inside bf16
#: matmuls move a 75 M-parameter first step by well under 1%; a kernel that is wrong
#: instead of imprecise moves it by much more. Re-record with a real measurement.
DTYPE_TOL = 0.01

#: Default ceiling, GiB. 16 GB card, 4 GB left for the desktop that shares it.
VRAM_CEILING_GB = 12.0

#: The committed fixture, the same slice the CPU regression gate trains on.
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus_2mb.bin"

#: Counters that must be non-zero after a live step. `Counters.check()` already
#: refuses an enabled-but-silent mechanism; this list is the same statement made from
#: outside, so deleting an `enable()` call cannot make the check vacuous.
REQUIRED = (
    "fwd",
    "bwd",
    "clip",
    "opt_muon",
    "opt_adamw",
    "embed_fwd",
    "block_fwd",
    "attn_fwd",
    "ffn_fwd",
    "norm_fwd",
    "head_fwd",
    "tied_head",
    "rope_fwd",
)


# --------------------------------------------------------------------- the predicates
# Pure, so `tests/test_gpu_check.py` can prove each one rejects what it claims to,
# on a CPU runner, without a card.


def train_failures(losses: list[float], counts: dict[str, int]) -> list[str]:
    """Return the failures of check 1: a non-finite loss, or a silent mechanism."""
    failures = [
        f"step {i} loss is {loss!r}: a step that is not finite cannot be taken"
        for i, loss in enumerate(losses)
        if not np.isfinite(loss)
    ]
    failures += [
        f"{key} is {counts.get(key, 0)} after a step that ran it: the mechanism is silent"
        for key in REQUIRED
        if counts.get(key, 0) <= 0
    ]
    return failures


def dtype_failures(
    fp32_loss: float, bf16_loss: float, canary_dtype: torch.dtype, tol: float = DTYPE_TOL
) -> list[str]:
    """Return the failures of check 2: the canary, finiteness, then the tolerance."""
    failures = []
    if canary_dtype is not torch.bfloat16:
        failures.append(
            f"autocast produced {canary_dtype} for a cuda matmul: the bf16 path is not "
            "live, so the comparison below is fp32 against fp32 and proves nothing"
        )
    if not np.isfinite(fp32_loss) or not np.isfinite(bf16_loss):
        failures.append(f"a first-step loss is not finite: fp32={fp32_loss!r} bf16={bf16_loss!r}")
        return failures
    drift = abs(bf16_loss - fp32_loss) / max(abs(fp32_loss), 1e-12)
    if drift > tol:
        failures.append(
            f"bf16 first-step loss {bf16_loss:.6f} differs from fp32 {fp32_loss:.6f} by "
            f"{drift:.2%}, over the {tol:.2%} tolerance: the numerics moved, not rounded"
        )
    return failures


def vram_failures(peak_bytes: int, ceiling_gb: float) -> list[str]:
    """Return the failures of check 4: a peak above the ceiling, or no measurement."""
    if peak_bytes <= 0:
        return [f"peak VRAM is {peak_bytes} bytes: nothing was measured"]
    ceiling = int(ceiling_gb * 2**30)
    if peak_bytes > ceiling:
        return [
            f"peak VRAM {peak_bytes / 2**30:.2f} GiB is over the {ceiling_gb:.1f} GiB "
            "ceiling: a batch that 'should' fit does not"
        ]
    return []


# ------------------------------------------------------------------------- the checks


def _batch(tokens: np.ndarray, step: int) -> torch.Tensor:
    """One `[BATCH, SEQ_LEN + 1]` window of the fixture, on the card."""
    span = BATCH * (SEQ_LEN + 1)
    start = (step * BATCH * SEQ_LEN) % (len(tokens) - span)
    window = torch.from_numpy(tokens[start : start + span].astype(np.int64))
    return window.reshape(BATCH, SEQ_LEN + 1).cuda()


def _model(seed: int = SEED) -> Transformer:
    """Build the live config, seeded, and on the card."""
    torch.manual_seed(seed)
    return Transformer(**LIVE, device="cuda")


def check_train(tokens: np.ndarray) -> tuple[list[str], list[float]]:
    """Check 1: twenty real steps, finite losses, every enabled mechanism reporting."""
    model = _model()
    optimizer = MixOptimizer(model)
    losses, counts = [], {}
    started = time.perf_counter()
    for step in range(STEPS):
        # A fresh `Counters` per step: the claim is about THIS step, and a counter that
        # accumulated from step 0 would hide a mechanism that stopped running at step 1.
        counters = Counters()
        result = train_step(model, _batch(tokens, step), optimizer=optimizer, counters=counters)
        losses.append(result.loss)
        counts.update(counters.snapshot())
        print(f"step {step:2d} loss {result.loss:.6f} norm {result.grad_norm:.3f}", flush=True)
    seconds = time.perf_counter() - started
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"{parameters:,} params, {STEPS} steps in {seconds:.1f}s on {torch.cuda.get_device_name()}"
    )
    return train_failures(losses, counts), losses


def _first_step_loss(tokens: np.ndarray, *, bf16: bool) -> float:
    """Return the first step's loss, bf16-autocast or fp32, from one seed."""
    model = _model()
    optimizer = MixOptimizer(model)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        return train_step(model, _batch(tokens, 0), optimizer=optimizer).loss


def check_dtype(tokens: np.ndarray) -> tuple[list[str], float, float]:
    """Check 2: bf16 against fp32 on the first step, with a canary for the fallback."""
    if not torch.cuda.is_bf16_supported():
        return (
            [f"{torch.cuda.get_device_name()} reports no bf16 support: the dtype path is absent"],
            float("nan"),
            float("nan"),
        )
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        left, right = (torch.randn(256, 256, device="cuda") for _ in range(2))
        canary = (left @ right).dtype
    fp32_loss = _first_step_loss(tokens, bf16=False)
    bf16_loss = _first_step_loss(tokens, bf16=True)
    print(f"first step fp32 {fp32_loss:.6f} bf16 {bf16_loss:.6f} canary {canary}")
    return dtype_failures(fp32_loss, bf16_loss, canary), fp32_loss, bf16_loss


def check_triton() -> tuple[list[str], str]:
    """Check 3: a kernel compiles and returns the right bytes on THIS arch."""
    import triton
    import triton.language as tl

    @triton.jit
    def _affine(src, dst, n, block: tl.constexpr):
        """`dst = src * 2 + 1` over `n` elements, one program per block."""
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask) * 2.0 + 1.0, mask=mask)

    n = 4096
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    try:
        _affine[(triton.cdiv(n, 1024),)](src, dst, n, block=1024)
        torch.cuda.synchronize()
    except Exception as exc:  # any compile failure is named, not swallowed
        return [f"triton failed to compile or launch on this arch: {exc!r}"], triton.__version__
    expected = src * 2 + 1
    if not torch.equal(dst, expected):
        return [f"triton kernel returned {dst[:4].tolist()}, expected {expected[:4].tolist()}"], (
            triton.__version__
        )
    print(f"triton {triton.__version__} compiled and ran a kernel on this arch")
    return [], triton.__version__


# ----------------------------------------------------------------------------- the run


def main(argv: list[str] | None = None) -> int:
    """Run the four checks; exit 1 with every failed assertion printed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=VRAM_CEILING_GB,
        help=f"hard ceiling on peak allocated VRAM (default {VRAM_CEILING_GB})",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "FAIL: no CUDA device. This job is the GPU gate; a runner without a card "
            "must skip it, not pass it.",
            file=sys.stderr,
        )
        return 1

    capability = "".join(map(str, torch.cuda.get_device_capability()))
    print(
        f"{torch.cuda.get_device_name()} sm_{capability} "
        f"torch {torch.__version__} cuda {torch.version.cuda}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    tokens = np.frombuffer(FIXTURE.read_bytes(), dtype=np.uint8)

    failures, losses = check_train(tokens)
    dtype_failures_found, fp32_loss, bf16_loss = check_dtype(tokens)
    failures += dtype_failures_found
    triton_failures, triton_version = check_triton()
    failures += triton_failures
    peak = torch.cuda.max_memory_allocated()
    failures += vram_failures(peak, args.max_vram_gb)

    summary = [
        "### gpu gate",
        "",
        "| check | number |",
        "|---|---|",
        f"| device | {torch.cuda.get_device_name()} |",
        f"| train | {STEPS} steps, loss {losses[0]:.6f} -> {losses[-1]:.6f} |",
        f"| bf16 vs fp32 first step | {bf16_loss:.6f} vs {fp32_loss:.6f} "
        f"({abs(bf16_loss - fp32_loss) / max(abs(fp32_loss), 1e-12):.3%}, tol {DTYPE_TOL:.1%}) |",
        f"| triton | {triton_version} |",
        f"| peak VRAM | {peak / 2**30:.2f} GiB of {args.max_vram_gb:.1f} GiB ceiling |",
        "",
    ]
    for failure in failures:
        summary.append(f"- **FAIL** {failure}")
        print(f"FAIL: {failure}", file=sys.stderr)
    print("\n".join(summary))
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as handle:
            handle.write("\n".join(summary) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
