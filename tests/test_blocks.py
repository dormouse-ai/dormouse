"""RMSNorm against its formula, and the block against the same operations by hand.

The block test is deliberately redundant arithmetic: it re-builds `norm1 → attn →
add → norm2 → ffn → add` from the block's own submodules. If the block normalises
the wrong stream — the classic `h = norm2(x)` instead of `norm2(h)` — the two
disagree exactly and this test is the one that says so.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from reference_attn import Attention

from dormouse.model.blocks import EPS, Block, RMSNorm


def test_rms_norm_matches_numpy() -> None:
    """`x · rsqrt(mean(x²) + ε) · g`, computed in float64 from the same weights."""
    torch.manual_seed(0)
    norm = RMSNorm(6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(6))
    x = torch.randn(2, 3, 6)
    xv = x.numpy().astype(np.float64)
    want = xv / np.sqrt((xv**2).mean(-1, keepdims=True) + EPS) * norm.weight.detach().numpy()
    assert np.allclose(norm(x).detach().numpy(), want, atol=1e-6)


def test_rms_norm_backward_matches_the_fp64_formula() -> None:
    """The native kernel saves less for backward, but differentiates the same formula."""
    torch.manual_seed(7)
    norm = RMSNorm(32)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    x = torch.randn(128, 32, requires_grad=True)
    x64 = x.detach().double().requires_grad_(True)
    w64 = norm.weight.detach().double()
    r = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + norm.eps)
    (x64 * r * w64).square().mean().backward()
    norm(x).square().mean().backward()
    assert torch.allclose(x.grad, x64.grad.float(), atol=1e-5, rtol=1e-5)


def test_the_gain_starts_at_one_so_the_norm_is_the_statistic_at_step_zero() -> None:
    norm = RMSNorm(16)
    assert torch.equal(norm.weight, torch.ones(16))


def test_the_output_has_unit_rms() -> None:
    torch.manual_seed(1)
    y = RMSNorm(32)(torch.randn(4, 5, 32) * 7.0)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 5), atol=1e-5)


def test_a_zero_row_stays_zero_and_does_not_go_nan() -> None:
    """The ε is what keeps a dead stream a zero instead of an infinity."""
    x = torch.zeros(1, 1, 8)
    assert torch.equal(RMSNorm(8)(x), torch.zeros(1, 1, 8))


def test_an_eps_of_zero_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        RMSNorm(8, eps=0.0)
    assert "positive floor" in str(e.value)


def test_the_block_is_the_two_pre_norm_residuals_it_claims() -> None:
    torch.manual_seed(2)
    block = Block(d_model=8, d_ffn=16, n_layers=3, attn=Attention(d_model=8, n_heads=2, head_dim=4))
    x = torch.randn(1, 4, 8)
    h = x + block.attn(block.norm1(x))
    h = h + block.ffn(block.norm2(h))
    assert torch.equal(block(x), h)
    assert not torch.equal(block(x), x)


def test_a_block_counts_itself_and_hands_the_counters_down() -> None:
    counters = Counters()
    block = Block(d_model=8, d_ffn=16, n_layers=2, attn=Attention(d_model=8, n_heads=2, head_dim=4))
    block(torch.zeros(1, 3, 8), counters)
    assert counters.snapshot() == {
        "block_fwd": 1,
        "lns_layers": 1,
        "norm_fwd": 6,
        "attn_fwd": 3,
        "rope_fwd": 3,
        "ffn_fwd": 3,
    }


def test_the_residual_writes_are_scaled_by_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """`0.02/√(2L)` on the A/B arm; LNS replaces it with plain 0.02."""
    monkeypatch.setenv("DORMOUSE_LNS", "0")
    for depth in (1, 8):
        want = 0.02 / (2.0 * depth) ** 0.5
        block = Block(
            d_model=64,
            d_ffn=32,
            n_layers=depth,
            attn=Attention(d_model=64, n_heads=4, head_dim=16, write_std=want),
        )
        assert block.attn.proj.weight.std().item() == pytest.approx(want, rel=0.15)
        assert block.ffn.down.weight.std().item() == pytest.approx(want, rel=0.15)


def test_a_block_without_a_stack_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Block(
            d_model=8,
            d_ffn=16,
            n_layers=0,
            attn=Attention(d_model=8, n_heads=2, head_dim=4),
        )
    assert "0th of nothing" in str(e.value)
