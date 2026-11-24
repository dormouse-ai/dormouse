"""Native ternary TSCT: ternary from step 1, fp32 only as the optimizer's shadow.

Three gates, each one a defect this arm could ship without it:

* the flag on a non-native path is a named refusal, never a silent fp32 forward;
* an external numpy fp64 recomputation of `scale·{-1,0,+1}` matches the torch
  forward AND both STE gradients for 20 steps (cosine gate), so the STE is not
  checked against itself;
* the step's own `Counters` report the ternary forward and backward, and a zero
  is a `MissingReport`, not a default.

The A/B below is deliberately small: it proves finiteness, reports a byte/CPU
peak proxy and a 20-step dBPB, and does NOT claim a VRAM saving — the fp32
shadow is still stored, by design.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.model.assemble import assemble
from dormouse.model.ffn import MoaFFN, stack_matrices
from dormouse.model.spec import StackSpec
from dormouse.model.tsct import (
    FactorLinear,
    NativeFactorLinear,
    native_tern,
    take_ternary_bwd,
)
from dormouse.train.optim import MixOptimizer

#: A CPU stack that keeps every slot and fits 20 steps twice in a test run.
SMALL = {
    "unit": "byte",
    "attn": "kda",
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "d_ffn": 128,
    "n_experts": 2,
    "topk": 1,
    "expert_tsct_rank": 16,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 64,
    "batch": 1,
}


@pytest.fixture
def native() -> Iterator[None]:
    """Turn the flag on for one test only; `native_tern()` reads it at call time."""
    old = os.environ.get("DORMOUSE_TSCT_NATIVE_TERN")
    old_fused = os.environ.get("DORMOUSE_FUSED")
    # These CPU tests exercise the eager factors; the shipped default is fused on.
    os.environ["DORMOUSE_FUSED"] = "0"
    os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("DORMOUSE_TSCT_NATIVE_TERN", None)
        else:
            os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = old
        if old_fused is None:
            os.environ.pop("DORMOUSE_FUSED", None)
        else:
            os.environ["DORMOUSE_FUSED"] = old_fused


def _numpy_codes(w: torch.Tensor) -> np.ndarray:
    """The ternary code an independent fp64 implementation computes."""
    w64 = w.detach().numpy().astype(np.float64)
    threshold = float(w.detach().abs().mean().clamp_min(1e-8))
    return np.rint(w64 / threshold).clip(-1.0, 1.0)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two fp64 vectors, computed outside torch."""
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_a_flag_on_a_nonternary_path_is_named_and_refuses(native: None) -> None:
    """No fp32 factor read may run under the flag, and the error says the escape."""
    assert native_tern()
    plain = FactorLinear(8, 6, 4, std=0.02)
    with pytest.raises(LoudError, match="read the fp32 factors"):
        plain(torch.randn(3, 8))

    with pytest.raises(LoudError, match="the forward would read fp32 factors"):
        StackSpec(**SMALL, weight_qat="ternary")  # type: ignore[arg-type]
    with pytest.raises(LoudError, match="the forward would read fp32 factors"):
        StackSpec(**SMALL)  # type: ignore[arg-type]


def test_the_config_without_the_flag_is_refused() -> None:
    """A config that names native ternary must have the flag that enforces it."""
    with pytest.raises(LoudError, match="needs DORMOUSE_TSCT_NATIVE_TERN=1"):
        StackSpec(**SMALL, weight_qat="ternary_native")  # type: ignore[arg-type]


def test_the_factor_forward_reads_ternary_and_trains_a_scale(native: None) -> None:
    """Every matmul operand is `scale·{-1,0,+1}`; both shadow and scale get grads."""
    torch.manual_seed(0)
    m = NativeFactorLinear(9, 7, 4, std=0.02)
    x = torch.randn(6, 9)
    tu, tv = _numpy_codes(m.u), _numpy_codes(m.v)
    su, sv = float(m.scale_u.detach()), float(m.scale_v.detach())
    got = m(x).detach().numpy().astype(np.float64)
    want = (x.numpy().astype(np.float64) @ (su * tu)) @ (sv * tv)
    print(f"\nnative forward cos(fp64 numpy) {_cos(got.ravel(), want.ravel()):.12f}")
    assert _cos(got.ravel(), want.ravel()) > 1.0 - 1e-6
    assert set(np.unique(tu)) <= {-1.0, 0.0, 1.0}

    m(x).sum().backward()
    assert m.u.grad is not None and m.v.grad is not None
    assert m.scale_u.grad is not None and m.scale_v.grad is not None
    assert float(m.scale_u.grad.abs()) > 0.0
    assert take_ternary_bwd() >= 2

    with pytest.raises(LoudError, match="stacked fp32 TSCT factors"):
        stack_matrices([FactorLinear(9, 7, 4, std=0.02)])


def test_external_numpy_fp64_anchor_holds_for_20_steps(native: None) -> None:
    """20 SGD steps: output and both STE gradients match a separate fp64 evaluation."""
    torch.manual_seed(7)
    m = NativeFactorLinear(8, 6, 4, std=0.02)
    opt = torch.optim.SGD(list(m.parameters()), lr=0.01)
    x = torch.randn(12, 8)
    target = torch.randn(12, 6)
    worst = {"y": 1.0, "u": 1.0, "v": 1.0}
    for _step in range(20):
        y = m(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        # Anchor the parameters that produced y and its gradients, not the post-step
        # ones: computing it after opt.step() compares two different weights.
        tu, tv = _numpy_codes(m.u), _numpy_codes(m.v)
        su, sv = float(m.scale_u.detach()), float(m.scale_v.detach())
        x64, t64 = x.numpy().astype(np.float64), target.numpy().astype(np.float64)
        g, h = su * tu, sv * tv
        want = (x64 @ g) @ h
        got = y.detach().numpy().astype(np.float64)
        worst["y"] = min(worst["y"], _cos(got.ravel(), want.ravel()))

        # STE = hold the codes fixed: dL/dshadow is dL/d(scale·code), no round term.
        dy = 2.0 * (want - t64) / want.size
        du_want = su * (x64.T @ dy @ h.T)
        dv_want = sv * ((x64 @ g).T @ dy)
        worst["u"] = min(
            worst["u"], _cos(m.u.grad.numpy().astype(np.float64).ravel(), du_want.ravel())
        )
        worst["v"] = min(
            worst["v"], _cos(m.v.grad.numpy().astype(np.float64).ravel(), dv_want.ravel())
        )
        assert math.isfinite(float(loss.detach()))
        opt.step()
        opt.zero_grad()

    print(f"\n20-step worst fp64 cos: {worst}")
    for key, value in worst.items():
        assert value > 1.0 - 1e-6, f"{key}: cos={value:.12f}"


def test_the_step_reports_both_ternary_counters_and_a_zero_fails(native: None) -> None:
    """The forward count is checked inside `Transformer.forward`, the backward at step end."""
    spec = StackSpec(**SMALL, weight_qat="ternary_native")  # type: ignore[arg-type]
    assembled = assemble(spec, smoke=True)
    opt = MixOptimizer(assembled.model, steps=2)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    result, counters = assembled.step(ids, optimizer=opt)
    assert not result.skipped
    assert counters.silent() == []
    assert counters.counts["tsct_ternary_fwd"] > 0
    assert counters.counts["tsct_ternary_bwd"] > 0
    counters.check()

    silent = Counters()
    silent.enable("tsct_ternary_bwd", counter="tsct_ternary_bwd", why="the STE ran")
    with pytest.raises(MissingReport, match="tsct_ternary_bwd"):
        silent.check()


def test_the_bank_stacks_ternary_operands_for_both_paths(native: None) -> None:
    """The anchor AND the GPU kernel receive codes, never the fp32 factors."""
    torch.manual_seed(3)
    bank = assemble(
        StackSpec(**SMALL, weight_qat="ternary_native"),  # type: ignore[arg-type]
        smoke=True,
    ).bank
    assert bank is not None
    assert isinstance(bank.experts[0], MoaFFN)
    assert isinstance(bank.experts[0].gate, NativeFactorLinear)

    arm = bank.experts[0]
    gate = arm.gate
    assert isinstance(gate, NativeFactorLinear)
    stacked = stack_matrices([gate])
    u, v = stacked
    tu, tv = _numpy_codes(gate.u), _numpy_codes(gate.v)
    assert np.allclose(u[0].detach().numpy(), float(gate.scale_u.detach()) * tu, atol=1e-6)
    assert np.allclose(v[0].detach().numpy(), float(gate.scale_v.detach()) * tv, atol=1e-6)

    # A bank built before the flag was set still refuses instead of launching fp32.
    plain = MoaFFN(16, 32, tsct_rank=4)
    with pytest.raises(LoudError, match="stacked fp32 TSCT factors"):
        stack_matrices([plain.gate])


def test_a_b_cpu_finite_loss_grad_and_byte_proxy(native: None) -> None:
    """Paired 20-step arms: finiteness, byte/CPU-peak proxy and dBPB, honestly reported."""
    torch.manual_seed(11)
    ids = torch.randint(0, 256, (SMALL["batch"], SMALL["seq_len"] + 1))
    rows = []

    for arm in ("none", "ternary_native"):
        if arm == "none":
            os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = "0"
        else:
            os.environ["DORMOUSE_TSCT_NATIVE_TERN"] = "1"
        spec = StackSpec(**SMALL, weight_qat=arm)  # type: ignore[arg-type]
        assembled = assemble(spec, smoke=True)
        opt = MixOptimizer(assembled.model, steps=2)
        losses = []
        for _ in range(20):
            result, _ = assembled.step(ids, optimizer=opt)
            assert not result.skipped, f"{arm}: NaN firewall"
            assert math.isfinite(result.loss)
            losses.append(result.loss)
        grads = [p.grad for p in assembled.model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        byte_proxy = sum(p.numel() * p.element_size() for p in assembled.model.parameters())
        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
            ) as prof,
        ):
            assembled.model(ids[:, :-1])
        peak = max(e.self_cpu_memory_usage for e in prof.events())
        rows.append((arm, byte_proxy, peak, losses[-1]))
        print(f"\n{arm}: bytes={byte_proxy:,} cpu_peak={peak:,} loss20={losses[-1]:.6f}")

    base, native_row = rows
    dbpb = (native_row[3] - base[3]) / math.log(2)  # byte unit: one byte per token
    print(f"\ndBPB(native-base) = {dbpb:+.5f}; bytes ratio = {native_row[1] / base[1]:.4f}")
    assert base[1] > 0 and native_row[1] > 0
    assert math.isfinite(dbpb)
