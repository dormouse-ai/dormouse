"""The GPU gate's predicates, proven to reject what they claim to, without a card.

`tools/gpu_check.py` is the only check that needs the card, so its failure logic is
pure and pinned here. The reason is the defect class this whole file exists for: a
threshold that accepts everything looks exactly like a threshold that passes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "gpu_check", Path(__file__).resolve().parent.parent / "tools" / "gpu_check.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gpu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gpu)

#: Every counter `check_train` requires, at 1 — the shape of a healthy step.
HEALTHY = dict.fromkeys(gpu.REQUIRED, 1)


# ------------------------------------------------------------------------ check 1


def test_a_healthy_step_passes() -> None:
    """Twenty finite losses and every mechanism reporting is no failure."""
    assert gpu.train_failures([5.5] * 20, HEALTHY) == []


def test_a_non_finite_loss_fails_and_names_the_step() -> None:
    """The NaN firewall returns the bad value; the gate must refuse it by name."""
    failures = gpu.train_failures([5.5, float("nan"), 5.4], HEALTHY)
    assert len(failures) == 1
    assert "step 1" in failures[0]


def test_a_silent_mechanism_fails() -> None:
    """A counter at zero is the historical defect: the flag was set, nothing ran."""
    failures = gpu.train_failures([5.5], {**HEALTHY, "bwd": 0, "rope_fwd": 0})
    assert len(failures) == 2
    assert any("bwd" in failure for failure in failures)


# ------------------------------------------------------------------------ check 2


def test_a_bf16_loss_within_tolerance_passes() -> None:
    """Half a percent of drift on a first step is rounding, not a different kernel."""
    assert gpu.dtype_failures(5.545, 5.545 * 1.005, torch.bfloat16) == []


def test_a_bf16_loss_beyond_tolerance_fails() -> None:
    """5% is a broken kernel; the gate must not round it away."""
    failures = gpu.dtype_failures(5.545, 5.545 * 1.05, torch.bfloat16)
    assert len(failures) == 1
    assert "tolerance" in failures[0]


def test_a_silent_dtype_fallback_fails_on_the_canary() -> None:
    """fp32 under autocast gives a loss that matches perfectly — and proves nothing."""
    failures = gpu.dtype_failures(5.545, 5.545, torch.float32)
    assert len(failures) == 1
    assert "bf16 path is not live" in failures[0]


def test_a_non_finite_bf16_loss_fails() -> None:
    """Overflow in bf16 is the arithmetic defect, and it is not a tolerance question."""
    assert gpu.dtype_failures(5.545, float("inf"), torch.bfloat16)


# ------------------------------------------------------------------------ check 4


def test_peak_under_the_ceiling_passes() -> None:
    """11 GiB on a 16 GB card with a 12 GiB ceiling is the run that fits."""
    assert gpu.vram_failures(11 * 2**30, 12.0) == []


def test_peak_over_the_ceiling_fails() -> None:
    """13 GiB is the OOM class, caught as a number before the allocator raises."""
    failures = gpu.vram_failures(13 * 2**30, 12.0)
    assert len(failures) == 1
    assert "ceiling" in failures[0]


def test_an_unmeasured_peak_fails() -> None:
    """A gate that never measured is a gate that passed by default."""
    assert gpu.vram_failures(0, 12.0)


@pytest.mark.parametrize("ceiling", [0.0, -1.0])
def test_no_ceiling_means_no_pass(ceiling: float) -> None:
    """The ceiling is a number that fails the job; any real peak is over it."""
    assert gpu.vram_failures(1, ceiling)
