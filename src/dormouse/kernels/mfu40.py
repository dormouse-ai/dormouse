"""Isolated A/B switch for the four PASS kernels (KO1/KO3/KO5/KO6) in the engine.

One env var, read at import, so a CUDA-graph capture sees one stable regime:

    DORMOUSE_MFU40_KERNELS=0   the shipped engine (baseline arm)
    DORMOUSE_MFU40_KERNELS=1   all four kernels (enabled arm)

Kernel-level receipts live here, not in ``Counters``: the locked harness snapshots
them into ``mfu40_counters`` so a flag that did not fire can never be reported as
a speed-up.
"""

from __future__ import annotations

import os

import torch

FLAG = "DORMOUSE_MFU40_KERNELS"

#: Import-time regime: graph capture and Triton constexprs must not see a mid-step flip.
ON = os.environ.get(FLAG, "0") == "1"

#: Per-kernel sub-switches, only meaningful when the main flag is on. They exist so
#: an A/B that loses time can attribute the loss without another code edit.
SUBFLAGS = {
    name: os.environ.get(f"DORMOUSE_MFU40_{name}", "1") != "0"
    for name in ("KO1", "KO3", "KO5", "KO6")
}

_COUNTS: dict[str, int] = {
    "ko1_attn_batched_solves": 0,
    "ko3_sparse_single_pass": 0,
    "ko5_glu_bwd_fused": 0,
    "ko6_rewy_fused": 0,
}


def enabled(name: str | None = None) -> bool:
    """Return whether the arm, or one named kernel, is active in this process."""
    if name is None:
        return ON
    return ON and SUBFLAGS[name]


@torch._dynamo.disable  # type: ignore[untyped-decorator]  # no dict-guard recompiles
def count(name: str) -> None:
    """Record one kernel receipt; a typo is a loud error, not a silent zero."""
    _COUNTS[name] += 1


def snapshot() -> dict[str, int]:
    """Return the kernel receipts for the harness artifact."""
    return dict(_COUNTS)
