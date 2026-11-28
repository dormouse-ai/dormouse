"""Minimal gate test: AOT cubin loaded from C++ == Triton JIT, bitwise + ms."""

import subprocess
import sys
from pathlib import Path


def test_cubin_gate():
    import pytest

    root = Path(__file__).resolve().parents[1]
    harness = root / "tools/cubin_gate/glu_harness"
    if not harness.exists():
        pytest.skip("build glu_harness first (see docs/cubin-gate.md)")
    # The AOT gate needs Triton + a real CUDA device; on the CPU CI venv
    # neither exists, and an UNMEASURED skip is honest (never a fake PASS).
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("AOT cubin gate needs Triton (CARD, UNMEASURED on CPU)")
    if not torch.cuda.is_available():
        pytest.skip("AOT cubin gate needs CUDA (CARD, UNMEASURED on CPU)")

    rc = subprocess.run(
        # 16M keeps GPU kernel time (~0.25 ms) well above Python launch
        # overhead; at 2M the 8 us kernel is all launch overhead and the
        # 2% timing gate is meaningless.
        [
            sys.executable,
            str(root / "tools/cubin_gate/gate.py"),
            "--n",
            str(16 * 1024 * 1024),
            "--iters",
            "30",
        ],
        capture_output=True,
        text=True,
    )
    out = rc.stdout + rc.stderr
    assert "bitwise" in out or "1e-6" in out, out
    assert "PASS" in out, out
    assert rc.returncode == 0, out
