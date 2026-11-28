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

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The worktree must import its own sources, not the dirty `app` checkout.
#: An editable install can win over PYTHONPATH; fail loudly if it did.
import dormouse  # noqa: E402

_WT_SRC = Path(__file__).resolve().parents[1] / "src"
assert Path(dormouse.__file__).resolve().is_relative_to(_WT_SRC), (
    f"tests imported {dormouse.__file__}, expected under {_WT_SRC}"
)
