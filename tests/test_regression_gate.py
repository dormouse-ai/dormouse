"""Gates for the quality gate itself: finite inputs and one baseline registry.

D07 debt: the same baseline lived in a tool, three docs and a config, and
`check()` accepted NaN because every comparison against NaN is false.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.regression import BASELINE, check

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "regression-baselines.md"
#: Every old baseline, so a fresh copy in a doc, config or tool is a red gate.
BASELINE_RE = re.compile(r"\b2\.(?:452850|506861|714696|356925)\b")
SCANNED = ("docs", "configs", "src", "tests", "tools")


@pytest.mark.parametrize("final", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_final_loss_is_a_failure(final: float) -> None:
    assert check(200, final, 10.0), "a non-finite final loss passed the gate"


@pytest.mark.parametrize("first", [float("nan"), float("inf")])
def test_a_non_finite_starting_loss_is_a_failure(first: float) -> None:
    assert check(200, BASELINE, first), "a non-finite starting loss passed the gate"


def test_a_finite_in_band_run_still_passes() -> None:
    assert check(200, BASELINE, 10.0) == []


def test_the_registry_names_every_baseline_with_a_date_and_reason() -> None:
    text = REGISTRY.read_text()
    for value in ("2.452850", "2.506861", "2.714696", "2.356925"):
        assert value in text, f"{value} is missing from the canonical registry"
    assert f"| {BASELINE:.6f} |" in text, "the gate no longer mirrors the current row"
    assert "date" in text.lower() and "reason" in text.lower()


def test_no_other_file_repeats_a_baseline() -> None:
    for directory in SCANNED:
        for path in (ROOT / directory).rglob("*"):
            if path.suffix not in {".py", ".md", ".toml"} or not path.is_file():
                continue
            if path in (REGISTRY, Path(__file__).resolve(), ROOT / "tools" / "regression.py"):
                continue
            match = BASELINE_RE.search(path.read_text())
            assert match is None, (
                f"{path.relative_to(ROOT)}: {match.group()} belongs in {REGISTRY.relative_to(ROOT)}"
            )
