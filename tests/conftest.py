"""Test-side helpers on the import path.

`tests/` holds two modules that are not collected as tests but are imported by tests
in this directory and in `gates/`: `reference_attn` (the dense attention anchor the
model no longer ships) and `stacks` (a `Transformer` built from blocks that use it).
Pytest puts the directory of each collected file on `sys.path`, which is `tests/gates/`
for the gate files — this conftest puts `tests/` there once, for both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip `gpu` tests where there is no card: the CPU wheel has no `_scaled_mm`."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs the card, and this wheel has no _scaled_mm")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
