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
