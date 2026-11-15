"""The reader's rules, each one standing in for a defect we already paid for."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dormouse_core.errors import LoudError

from dormouse.data.bin import Manifest, Stream


def _write(tmp_path: Path, values: list[int], dtype: str = "uint16", tokens: int | None = None):
    path = tmp_path / "tokens.bin"
    arr = np.asarray(values, dtype=getattr(np, dtype))
    path.write_bytes(arr.tobytes())
    meta = {"dtype": dtype, "tokens": tokens if tokens is not None else len(values)}
    path.with_suffix(".bin.json").write_text(json.dumps(meta))
    return path


def test_round_trip_through_a_stream(tmp_path: Path) -> None:
    stream = Stream(_write(tmp_path, [5, 6, 7, 8]))
    assert stream.tokens == 4
    assert stream.take(1, 2).tolist() == [6, 7]


def test_reading_past_the_end_is_an_error_not_a_wrap(tmp_path: Path) -> None:
    stream = Stream(_write(tmp_path, [1, 2, 3]))
    with pytest.raises(LoudError) as e:
        stream.take(2, 5)
    assert "wrap=True" in str(e.value)


def test_a_batch_is_inputs_and_shifted_targets(tmp_path: Path) -> None:
    stream = Stream(_write(tmp_path, list(range(11))))
    x, y = next(stream.batches(batch=2, seq_len=4))
    assert x.shape == (2, 4) and y.shape == (2, 4)
    assert y[0, 0] == x[0, 1]


def test_resume_position_is_arithmetic_not_state(tmp_path: Path) -> None:
    """A resumed run must land exactly where it stopped — no sidecar, no drift."""
    stream = Stream(_write(tmp_path, list(range(1000))))
    straight = []
    for x, _ in stream.batches(2, 8):
        straight.append(x.tolist())
        if len(straight) == 4:
            break
    resumed_x, _ = next(stream.batches(2, 8, step=3))
    assert resumed_x.tolist() == straight[3]
    assert resumed_x.tolist() != straight[0]


def test_wrap_allows_a_looped_corpus(tmp_path: Path) -> None:
    stream = Stream(_write(tmp_path, list(range(10))))
    x, _ = next(stream.batches(batch=2, seq_len=4, step=1, wrap=True))
    assert x.shape == (2, 4)


def test_a_missing_manifest_is_named(tmp_path: Path) -> None:
    path = tmp_path / "bare.bin"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(LoudError) as e:
        Stream(path)
    assert "manifest" in str(e.value)


def test_a_size_mismatch_is_refused(tmp_path: Path) -> None:
    """The export defect class: a file that is not what its label claims."""
    path = _write(tmp_path, [1, 2, 3], tokens=4)
    with pytest.raises(LoudError) as e:
        Stream(path)
    assert "disagree" in str(e.value)


def test_an_unknown_dtype_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, [1, 2])
    path.with_suffix(".bin.json").write_text(json.dumps({"dtype": "float64", "tokens": 2}))
    with pytest.raises(LoudError) as e:
        Manifest.read(path)
    assert "uint16" in str(e.value)
