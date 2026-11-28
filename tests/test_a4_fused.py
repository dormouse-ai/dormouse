"""Fused A4: external fp64 anchors, arity, counters, bounded scratch and config gates.

The CUDA tests build the two Triton kernels directly; on a CPU-only box they are
skipped, but the eager-path gates still run. No test claims a speed result — the
card A/B is `tools/a4_fused_ab.py`.
"""

from __future__ import annotations

import inspect
import math
import tomllib
import types
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters, MissingReport
from dormouse_core.errors import LoudError
from dormouse_kernels.engine import a4_fused

from dormouse.model import quant
from dormouse.model.spec import StackSpec

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton needs a CUDA card")


def _np_fwht(x: np.ndarray, block: int = 64) -> np.ndarray:
    """Independent fp64 butterfly: the same orthonormal H, written against numpy."""
    width = x.shape[-1]
    y = x.reshape(*x.shape[:-1], width // block, block).copy()
    step = 1
    while step < block:
        y = y.reshape(*y.shape[:-2], width // block, block // (2 * step), 2, step)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y = np.stack((a + b, a - b), axis=-2).reshape(*x.shape[:-1], width // block, block)
        step *= 2
    return y.reshape(x.shape) / math.sqrt(block)


def _np_a4(x: np.ndarray) -> np.ndarray:
    """The anchor's own quantiser: per-token absmax/7, clamp, inverse rotation."""
    z = _np_fwht(x)
    scale = np.maximum(np.abs(z).max(axis=-1, keepdims=True), 1e-8) / 7
    q = np.clip(np.round(z / scale), -7, 7)
    return _np_fwht(q * scale)


@CUDA
def test_fused_forward_matches_external_fp64_anchor_and_eager() -> None:
    """Both kernels in one launch: same formula as numpy fp64 and as A4Save."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 832)).astype(np.float64)
    tensor = torch.tensor(x, dtype=torch.float32, device="cuda")
    y = a4_fused.a4_fused(tensor)
    expected = _np_a4(x)
    got = y.detach().cpu().numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=1e-5)
    eager = quant.a4(tensor.clone())
    # IEEE-no-fusion in the kernel launch: the two paths are bitwise identical,
    # not merely within tolerance, so a 20-step trace cannot drift.
    assert torch.equal(y, eager)


@CUDA
def test_fused_backward_matches_external_fp64_ste_anchor() -> None:
    """Unpack + STE: dL/dx is the upstream gradient, anchored in numpy fp64."""
    x = torch.randn((2, 832), dtype=torch.float32, device="cuda", requires_grad=True)
    y = a4_fused.a4_fused(x)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    expected = upstream.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(x.grad.detach().cpu().numpy(), expected, atol=1e-6, rtol=0.0)


@CUDA
def test_forward_and_backward_arity_is_exactly_what_autograd_expects() -> None:
    """Two forward args -> one output and two grads; an extra None crashes torch 2.15."""
    fwd = inspect.signature(a4_fused.A4FusedSave.forward)
    bwd = inspect.signature(a4_fused.A4FusedSave.backward)
    assert list(fwd.parameters) == ["ctx", "x", "counters"]
    assert list(bwd.parameters) == ["ctx", "grad"]
    x = torch.randn((1, 64), dtype=torch.float32, device="cuda")
    grad = torch.randn_like(x)
    ctx = types.SimpleNamespace(
        width=64,
        counters=None,
        saved_tensors=(
            torch.zeros((1, 32), dtype=torch.uint8, device="cuda"),
            torch.ones((1,), dtype=torch.float32, device="cuda"),
        ),
    )
    result = a4_fused.A4FusedSave.backward(ctx, grad)
    assert isinstance(result, tuple) and len(result) == 2
    assert torch.equal(result[0], grad) and result[1] is None
    del x


@CUDA
def test_saved_tensors_are_packed_codes_and_one_scale_per_row() -> None:
    """The backward cannot retain an fp32 activation: audit what autograd saved."""
    saved: list[torch.Tensor] = []

    def pack_hook(tensor: torch.Tensor) -> Any:
        saved.append(tensor.detach())
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    x = torch.randn((4, 832), dtype=torch.float32, device="cuda", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = a4_fused.a4_fused(x)
        y.sum().backward()
    codes = [t for t in saved if t.dtype == torch.uint8]
    scales = [t for t in saved if t.dtype == torch.float32]
    assert codes and scales
    assert codes[0].shape == (4, 416) and codes[0].numel() == x.numel() // 2
    assert scales[0].shape == (4,)
    assert sum(t.numel() * t.element_size() for t in saved) <= x.numel() // 2 + x.shape[0] * 4


@CUDA
def test_kernels_use_no_shared_memory_and_do_not_spill() -> None:
    """sm_120 scratch is bounded: both launches are pure register arithmetic."""
    kernels = a4_fused._build()
    assert kernels is not None
    x = torch.randn((2, 832), dtype=torch.float32, device="cuda")
    flat = x.reshape(-1, 832)
    packed = torch.empty((2, 416), dtype=torch.uint8, device="cuda")
    scale = torch.empty((2,), dtype=torch.float32, device="cuda")
    out = torch.empty_like(flat)
    launch = kernels["fwd"][(2,)](flat, out, packed, scale, 832, 13, 16, num_warps=4)
    assert launch.metadata.shared <= a4_fused.SMEM_LIMIT
    assert launch.n_spills == 0
    launch2 = kernels["bwd"][(2,)](out, out, packed, scale, 832, 13, 16, num_warps=4)
    assert launch2.metadata.shared <= a4_fused.SMEM_LIMIT
    assert launch2.n_spills == 0


@CUDA
def test_fused_counters_are_live_and_a_silent_backward_is_a_missing_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a4_fwd/bwd/codes and a4_fused_fwd/bwd all move under the flag."""
    monkeypatch.setenv(a4_fused.FUSED_FLAG, "1")
    counters = Counters()
    x = torch.randn((2, 832), dtype=torch.float32, device="cuda", requires_grad=True)
    y = quant.a4(x, counters)
    y.sum().backward()
    for name in ("a4_fwd", "a4_bwd", "a4_codes_saved", "a4_fused_fwd", "a4_fused_bwd"):
        assert counters.counts[name] > 0, name
    silent = Counters()
    silent.enable("act_a4_fused_bwd", counter="a4_fused_bwd", why="the kernel must run")
    with pytest.raises(MissingReport) as caught:
        silent.check()
    assert "a4_fused_bwd" in str(caught.value)


def test_eager_a4_helpers_and_arity_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the eager helpers the CUDA tests skip, including the STE gradient."""
    monkeypatch.delenv(quant.A4_FLAG, raising=False)
    monkeypatch.delenv(quant.CODESAVE_FLAG, raising=False)
    monkeypatch.delenv(quant.FUSED_FLAG, raising=False)
    assert not quant.a4_on() and quant.codesave_on() and not quant.fused_on()
    x = torch.randn((2, 64), requires_grad=True)
    assert quant.int4(x).shape == x.shape
    assert quant.ternary(torch.randn((4, 4))).shape == (4, 4)
    z = quant.fwht(x)
    assert torch.allclose(quant.fwht(z), x, atol=1e-5)
    a4_z = quant.fwht_a4(x)
    assert torch.allclose(quant.fwht_a4(a4_z), x, atol=1e-5)
    with pytest.raises(ValueError):
        quant.fwht(torch.randn((2, 65)))
    with pytest.raises(ValueError):
        quant.fwht_a4(torch.randn((2, 65)))
    codes = torch.tensor([[-7, -1, 0, 7, 3, -3, 1, 1]], dtype=torch.int8)
    packed = quant._pack(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (1, 4)
    assert torch.equal(quant._unpack(packed, codes.shape[-1]).to(torch.int8), codes)
    y = quant.a4(x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    monkeypatch.setenv(quant.A4_FLAG, "1")
    monkeypatch.setenv(quant.CODESAVE_FLAG, "0")
    monkeypatch.setenv(quant.FUSED_FLAG, "1")
    assert quant.a4_on() and not quant.codesave_on() and quant.fused_on()


def test_fused_without_cuda_refuses_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent Python fallback behind a flag that claims a kernel."""
    monkeypatch.setenv(a4_fused.FUSED_FLAG, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(a4_fused, "_build", lambda: None)
    with pytest.raises(LoudError) as caught:
        a4_fused._require()
    assert "DORMOUSE_A4_FUSED=1" in str(caught.value)


def test_config_flag_contradictions_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(quant.A4_FLAG, "1")
    with pytest.raises(LoudError) as caught:
        StackSpec(act_quant="none")
    assert "DORMOUSE_ACT_A4=1" in str(caught.value)
    with pytest.raises(LoudError) as caught:
        StackSpec(act_quant="a4", weight_qat="none")
    assert "ternary_native" in str(caught.value)
    monkeypatch.setenv("DORMOUSE_TSCT_NATIVE_TERN", "1")
    raw = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs/train_200m_a4_2b.toml").read_text()
    )
    known = {f.name for f in fields(StackSpec) if f.init}
    recipe = StackSpec.from_mapping({k: v for k, v in raw.items() if k in known})
    assert (recipe.act_quant, recipe.weight_qat) == ("a4", "ternary_native")
