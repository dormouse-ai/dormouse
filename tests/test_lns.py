"""LNS (arXiv 2502.05795): named flag, init swap, after-LN placement, external anchor.

The paper's LNS scales BOTH pre-norm outputs of layer `l` by `1/sqrt(l)` and removes
the `write_std=0.02/sqrt(2L)` init (Tables 6/8: the combination is worse; scaling
after the residual diverges). Nothing here compares the code to itself: the residual
math is checked against a numpy fp64 reference that does not import torch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from stacks import reference_stack
from torch import nn

from dormouse.model.assemble import assemble
from dormouse.model.blocks import (
    LNS_COUNTER,
    LNS_FLAG,
    PROJ_STD,
    Block,
    lns_enabled,
    residual_write_std,
)
from dormouse.model.spec import StackSpec


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the A/B-off arm; the shipped default is on."""
    monkeypatch.setenv(LNS_FLAG, "0")


def _spec() -> StackSpec:
    """The smallest valid stack: one 3:1 period, widths that fit a factorised bank."""
    return StackSpec(
        unit="byte",
        n_layers=4,
        d_model=32,
        n_heads=2,
        n_kv_heads=2,
        head_dim=16,
        c_lat=16,
        d_ffn=64,
        n_experts=2,
        topk=2,
        n_shared=1,
        expert_tsct_rank=16,
        attn_k=32,
        attn_window=16,
        seq_len=64,
        batch=1,
    )


class _Mat(nn.Module):
    """A nonlinear sublayer whose forward is written out in the numpy anchor."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        return torch.tanh(x @ self.weight)


class _FixedNorm(nn.Module):
    """A norm whose output is known, so the anchor isolates scaling and residual adds."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        return self.value.expand_as(x)


def test_the_flag_is_on_by_default_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    assert LNS_FLAG == "DORMOUSE_LNS"
    monkeypatch.delenv(LNS_FLAG, raising=False)
    assert lns_enabled() is True, "the 200M A/B passed; LNS is the shipped default"
    for value in ("0", "off", "false"):
        monkeypatch.setenv(LNS_FLAG, value)
        assert lns_enabled() is False
    monkeypatch.setenv(LNS_FLAG, "1")
    assert lns_enabled() is True


def test_lns_replaces_the_depth_scaled_init(monkeypatch: pytest.MonkeyPatch) -> None:
    assert residual_write_std(8) == pytest.approx(0.02 / math.sqrt(16))
    monkeypatch.setenv(LNS_FLAG, "1")
    assert residual_write_std(8) == PROJ_STD


def test_the_assembled_writes_use_plain_std_under_lns(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    plain = assemble(_spec(), smoke=True)
    monkeypatch.setenv(LNS_FLAG, "1")
    torch.manual_seed(7)
    scaled = assemble(_spec(), smoke=True)
    off_std = plain.model.blocks[0].attn.proj.weight.std().item()
    on_std = scaled.model.blocks[0].attn.proj.weight.std().item()
    assert off_std == pytest.approx(0.02 / math.sqrt(8), rel=0.2)
    assert on_std == pytest.approx(0.02, rel=0.2)
    assert on_std > off_std


def test_the_assembler_numbers_layers_from_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LNS_FLAG, "1")
    model = assemble(_spec(), smoke=True).model
    assert [b.lns_scale for b in model.blocks] == [
        pytest.approx(1.0 / math.sqrt(layer)) for layer in range(1, 5)
    ]
    monkeypatch.setenv(LNS_FLAG, "0")
    assert all(b.lns_scale == 1.0 for b in assemble(_spec(), smoke=True).model.blocks)


def test_a_zero_layer_index_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LNS_FLAG, "1")
    with pytest.raises(LoudError, match="l starts at 1"):
        Block(
            d_model=8,
            d_ffn=16,
            n_layers=2,
            attn=_Mat(torch.eye(8, dtype=torch.float64)),
            layer_idx=0,
        )


def test_every_block_reports_the_lns_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    model = reference_stack(n_layers=4, d_model=32)
    counters = Counters()
    model(torch.zeros(2, 3, dtype=torch.long), counters)
    assert "lns" not in counters.enabled
    assert LNS_COUNTER not in counters.counts

    monkeypatch.setenv(LNS_FLAG, "1")
    model = reference_stack(n_layers=4, d_model=32)
    counters = Counters()
    model(torch.zeros(2, 3, dtype=torch.long), counters)
    assert "lns" in counters.enabled
    assert counters.counts[LNS_COUNTER] == 4
    counters.check()  # an enabled mechanism that never ran would raise MissingReport


def test_external_fp64_anchor_on_the_residual_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numpy reference for `x + f(norm1(x)/sqrt(l)) + f(norm2(x)/sqrt(l))`.

    Both norms are scaled, the scale is on the NORM OUTPUT (not after the residual),
    and the two residual adds use the post-attention stream for the FFN norm.
    """
    monkeypatch.setenv(LNS_FLAG, "1")
    rng = np.random.default_rng(20261124)
    d, seq, layer = 8, 6, 3
    x_np = rng.standard_normal((1, seq, d))
    n1 = rng.standard_normal(d)
    n2 = rng.standard_normal(d)
    w_attn = rng.standard_normal((d, d))
    w_ffn = rng.standard_normal((d, d))
    scale = np.float64(1.0) / np.sqrt(np.float64(layer))

    block = Block(
        d_model=d,
        d_ffn=2 * d,
        n_layers=layer,
        attn=_Mat(torch.tensor(w_attn, dtype=torch.float64)),
        ffn=_Mat(torch.tensor(w_ffn, dtype=torch.float64)),
        layer_idx=layer,
    )
    block.norm1 = _FixedNorm(torch.tensor(n1, dtype=torch.float64))
    block.norm2 = _FixedNorm(torch.tensor(n2, dtype=torch.float64))
    got = block(torch.tensor(x_np, dtype=torch.float64)).detach().numpy()

    # External fp64 reference: numpy only, no torch ops and no model code.
    h = x_np + np.tanh((n1 * scale) @ w_attn)
    want = h + np.tanh((n2 * scale) @ w_ffn)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    # The wrong "after residual" placement of paper Table 8 diverges, so the
    # anchor actually distinguishes where the scaling was inserted.
    wrong_scaled = (x_np + np.tanh(n1 @ w_attn)) * scale
    wrong = wrong_scaled + np.tanh((n2 * scale) @ w_ffn)
    assert not np.allclose(got, wrong, rtol=1e-6, atol=1e-6)


def test_the_final_norm_stays_unscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `1/sqrt(l)` stops at the last block; the head's norm has unit RMS."""
    monkeypatch.setenv(LNS_FLAG, "1")
    model = reference_stack(n_layers=4, d_model=32)
    x = torch.randn(2, 5, 32)
    y = model.norm(x)
    rms = (y.detach().double().pow(2).mean(-1)).numpy()
    np.testing.assert_allclose(rms, np.ones((2, 5)), rtol=1e-4, atol=1e-4)
