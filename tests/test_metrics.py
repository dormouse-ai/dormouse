"""A BPB value cannot exist without its window — tested on the three ways it lied.

Each test corresponds to a real failure: a zero that looked like a good number, a
comparison across different windows, and a batch-size change that silently altered
what was measured.
"""

from __future__ import annotations

import math

import pytest
from dormouse_core.errors import LoudError

from dormouse.metrics import BitsPerByte, eval_window, from_nats


def test_a_measurement_renders_with_its_window() -> None:
    value = from_nats(1.0, 196_608)
    assert str(value) == f"{1 / math.log(2):.3f} BPB over 196608 B"


def test_zero_is_refused_as_a_bug() -> None:
    """The bpb_fin=0.000 defect: a metric that never filled must not pass as good."""
    with pytest.raises(LoudError) as e:
        BitsPerByte(bpb=0.0, bytes_evaluated=1024, loss_nats=0.0)
    assert "did not fill" in str(e.value)


def test_an_empty_window_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        BitsPerByte(bpb=1.5, bytes_evaluated=0, loss_nats=1.0)
    assert "65536" in str(e.value)


def test_a_delta_between_different_windows_is_refused() -> None:
    a = from_nats(1.0, 196_608)
    b = from_nats(1.0, 24_576)
    with pytest.raises(LoudError) as e:
        a.delta(b)
    assert "windows differ" in str(e.value)


def test_a_delta_between_equal_windows_is_the_plain_difference() -> None:
    a = from_nats(1.0, 1000)
    b = from_nats(0.9, 1000)
    assert a.delta(b) == pytest.approx(0.1 / math.log(2))


def test_a_non_finite_loss_is_named() -> None:
    with pytest.raises(LoudError) as e:
        from_nats(float("nan"), 1000)
    assert "diverged" in str(e.value)


def test_the_window_is_bytes_not_batches() -> None:
    """Two different batching schemes over the same bytes must agree."""
    assert eval_window(8, 24, 1024) == eval_window(24, 8, 1024) == 196_608


def test_token_models_scale_the_window_by_bytes_per_token() -> None:
    assert eval_window(8, 24, 1024, bytes_per_position=4.4) == int(196_608 * 4.4)


def test_a_zero_sized_window_is_refused() -> None:
    with pytest.raises(LoudError):
        eval_window(0, 24, 1024)
