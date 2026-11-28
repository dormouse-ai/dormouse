"""The report: one row per task, every row carrying its `n`, BPB with its window."""

from __future__ import annotations

import math

import pytest
from dormouse_core.errors import LoudError

from dormouse.eval.bpb import from_token_nats
from dormouse.eval.report import Result, render


def test_every_row_carries_its_example_count_and_the_bpb_carries_its_window() -> None:
    bpb = from_token_nats(math.log(256), eval_text="abcd", token_count=4)
    text = render(
        [
            Result("reverse-copy", "exact-match", 1.0, 6, 0),
            Result("code-executes", "executes", 0.5, 3, 1),
        ],
        bpb,
    )
    lines = text.splitlines()
    assert lines[0].split() == ["task", "metric", "score", "n", "seed"]
    assert lines[1].split() == ["reverse-copy", "exact-match", "1.000", "6", "0"]
    assert lines[2].split() == ["code-executes", "executes", "0.500", "3", "1"]
    assert "over 4 B" in lines[-1]


def test_a_row_without_examples_is_refused() -> None:
    with pytest.raises(LoudError) as error:
        Result("reverse-copy", "exact-match", 1.0, 0, 0)
    assert "n=0" in str(error.value)


def test_a_score_that_is_not_a_rate_is_refused() -> None:
    with pytest.raises(LoudError):
        Result("reverse-copy", "exact-match", 6.0, 6, 0)


def test_an_empty_report_is_refused() -> None:
    with pytest.raises(LoudError) as error:
        render([])
    assert "empty table" in str(error.value)
