"""The gated SwiGLU, checked against a reference written from the same formula.

The anchor is four lines of numpy in this file, not another implementation of the
model: `down(silu(gate(x)) * up(x))` computed in float64 and compared to the module's
float32 output. If the module ever drifts — a missing gate, a `sigmoid` on the wrong
operand, the multiplication the wrong way round — the two disagree by O(1) and the
1e-6 band catches it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.ffn import GatedFFN, neutral_width


def _reference(x: torch.Tensor, ffn: GatedFFN) -> np.ndarray:
    """`down(silu(gate(x)) * up(x))` in float64, from the module's own weights."""
    xv = x.detach().numpy().astype(np.float64)
    gate = ffn.gate.weight.detach().numpy().astype(np.float64)
    up = ffn.up.weight.detach().numpy().astype(np.float64)
    down = ffn.down.weight.detach().numpy().astype(np.float64)
    g = gate @ xv
    h = (g / (1.0 + np.exp(-g))) * (up @ xv)
    return down @ h


def test_matches_the_numpy_reference_on_one_token() -> None:
    """d=6, f=11: every entry O(1), so an absolute 1e-6 band is a real check."""
    torch.manual_seed(0)
    ffn = GatedFFN(d_model=6, d_ffn=11)
    x = torch.randn(1, 1, 6)
    assert np.allclose(ffn(x).detach().numpy()[0, 0], _reference(x[0, 0], ffn), atol=1e-6)


def test_matches_the_numpy_reference_over_a_sequence() -> None:
    """A batch and a sequence exercise the same formula on [n, d] rows."""
    torch.manual_seed(1)
    ffn = GatedFFN(d_model=8, d_ffn=13)
    x = torch.randn(2, 5, 8)
    want = np.stack([_reference(x[b, t], ffn) for b in range(2) for t in range(5)])
    got = ffn(x).detach().numpy().reshape(-1, 8)
    assert np.allclose(got, want, atol=1e-6)


def test_the_gated_bill_equals_the_gelu_bill_at_the_live_shape() -> None:
    """The one claim the width 2048 is chosen for: 3·832·2048 == 2·832·3072."""
    assert 3 * 832 * 2048 == 2 * 832 * 3072 == 5_111_808
    assert neutral_width(3072) == 2048


def test_neutral_width_never_adds_a_parameter() -> None:
    """Rounding down: an odd control width may lose a few, never gain one."""
    assert 3 * neutral_width(3073) <= 2 * 3073
    with pytest.raises(LoudError) as e:
        neutral_width(2)
    assert "too narrow" in str(e.value)


def test_it_reports_itself_when_asked() -> None:
    """An FFN that runs and does not count is the defect the counters exist for."""
    counters = Counters()
    GatedFFN(d_model=4, d_ffn=6)(torch.zeros(1, 3, 4), counters)
    assert counters.snapshot() == {"ffn_fwd": 3}


def test_a_zero_width_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        GatedFFN(d_model=4, d_ffn=0)
    assert "not a width" in str(e.value)
