"""The metric invariant: zero over a non-empty window is a bug, not a good number.

`bpb_fin = 0.000` once passed as a result because the loss path never filled it. A
language model that predicts every byte perfectly does not exist, so a zero is
refused, and a window with no bytes is refused before it can produce one.
"""

from __future__ import annotations

import pytest
from dormouse_core.errors import LoudError

from dormouse.metrics import BitsPerByte


def test_zero_over_a_non_empty_window_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        BitsPerByte(bpb=0.0, bytes_evaluated=1024, loss_nats=0.0)
    assert "did not fill" in str(e.value)


def test_a_metric_over_an_empty_window_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        BitsPerByte(bpb=1.5, bytes_evaluated=0, loss_nats=1.0)
    assert "empty window" in str(e.value)
