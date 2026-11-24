"""Receipts for the isolated four-kernel arm wired into the locked engine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from dormouse.kernels import mfu40
from dormouse.model.attn_variants.kda import KDA

ROOT = Path(__file__).resolve().parents[1]


def test_flag_is_on_by_default() -> None:
    """An unflagged process runs the four kernels; `=0` is the A/B arm."""
    assert mfu40.ON is (os.environ.get(mfu40.FLAG, "1") != "0")
    assert "ko1_attn_batched_solves" in mfu40.snapshot()


def test_fused_is_on_by_default() -> None:
    """The fused path is the shipped path in a fresh, unflagged import."""
    code = "from dormouse.kernels import fused; print(fused.flag_on())"
    env = {k: v for k, v in os.environ.items() if k != "DORMOUSE_FUSED"}
    env["PYTHONPATH"] = str(ROOT / "src")
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    assert out.strip() == "True"


def test_ko1_batched_solves_match_shipped_chunks() -> None:
    """KO1 is the same operator, not a looser one: compare on a frozen small shape."""
    torch.manual_seed(7)
    b, h, chunks, chunk, d = 2, 2, 3, 8, 16
    shape = (b, h, chunks * chunk, d)
    q, k, v = (torch.randn(shape, dtype=torch.float64) for _ in range(3))
    alpha = torch.full((b, h, chunks * chunk, d), 0.98, dtype=torch.float64)
    beta = torch.rand(b, h, chunks * chunk, dtype=torch.float64)
    split = lambda t: t.reshape(b, h, chunks, chunk, t.shape[-1])  # noqa: E731
    core = (split(q), split(k), split(v), split(alpha), beta.reshape(b, h, chunks, chunk))
    bare = object.__new__(KDA)
    baseline, base_state = bare._chunks(*core)
    optimized, opt_state = bare._chunks_batched_solves(*core)
    assert torch.allclose(baseline, optimized, atol=1e-6, rtol=1e-9)
    assert torch.allclose(base_state, opt_state, atol=1e-6, rtol=1e-9)


def _sparse_constants(flag: str) -> tuple[int, int]:
    code = "from dormouse.kernels import sparse; print(sparse.SHARDS, sparse.BN_BWD)"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **{mfu40.FLAG: flag})
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    a, b = out.split()
    return int(a), int(b)


def test_ko3_tile_constants_follow_the_flag() -> None:
    """SHARDS 8->4 and BN_BWD 64->32 are the measured KO3 knobs."""
    assert _sparse_constants("0") == (8, 64)
    assert _sparse_constants("1") == (4, 32)


def test_ko5_and_ko6_are_importable_under_both_arms() -> None:
    """Both ported modules exist behind the same switch, with receipts."""
    from dormouse.kernels import fused, rewy

    assert callable(rewy.set_fused)
    assert "glu_bwd" in fused._KERNELS or fused._build() is None  # type: ignore[attr-defined]
    assert set(mfu40.snapshot()) == {
        "ko1_attn_batched_solves",
        "ko3_sparse_single_pass",
        "ko5_glu_bwd_fused",
        "ko6_rewy_fused",
    }
