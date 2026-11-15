"""The BPB anchor: 256 equiprobable bytes are exactly 8.0 bits, to 1e-6.

The two numbers here are computed by hand in the test — a uniform byte distribution
is 8 bits, 128 equiprobable bytes with 128 impossible ones is 7 — so the conversion
from a token model's nats has something to be wrong against.
"""

from __future__ import annotations

import math

import pytest
from dormouse_core.errors import LoudError

from dormouse.eval.bpb import bytes_per_token, entropy_bits_per_byte, from_token_nats


def test_uniform_256_bytes_is_exactly_eight_bits() -> None:
    assert entropy_bits_per_byte([1 / 256] * 256) == pytest.approx(8.0, abs=1e-6)


def test_a_peaked_distribution_is_the_hand_computed_entropy() -> None:
    """128 bytes at 1/128 and 128 impossible ones: 7 bits, by hand."""
    assert entropy_bits_per_byte([1 / 128] * 128 + [0.0] * 128) == pytest.approx(7.0, abs=1e-6)


def test_two_equiprobable_bytes_are_one_bit() -> None:
    assert entropy_bits_per_byte([0.5, 0.5]) == pytest.approx(1.0, abs=1e-6)


def test_the_nats_path_agrees_with_the_entropy_anchor() -> None:
    """A model with uniform 256-way confidence on one-byte tokens is 8.0 BPB."""
    value = from_token_nats(math.log(256), eval_text="abcd", token_count=4)
    assert value.bpb == pytest.approx(8.0, abs=1e-6)
    assert value.bytes_evaluated == 4


def test_a_token_model_scales_by_the_measured_bytes_not_a_constant() -> None:
    """Two bytes per token over UTF-8 text: 4 tokens × 1 bit is 4 bits over 8 bytes."""
    text = "\u00e9" * 4
    assert bytes_per_token(text, 4) == 2.0
    value = from_token_nats(math.log(2), eval_text=text, token_count=4)
    assert value.bytes_evaluated == 8
    assert value.bpb == pytest.approx(0.5, abs=1e-9)


def test_an_unknown_byte_count_is_refused() -> None:
    with pytest.raises(LoudError) as error:
        from_token_nats(1.0, eval_text=None, token_count=1000)
    assert "unknown" in str(error.value)


def test_an_empty_eval_text_is_refused() -> None:
    with pytest.raises(LoudError) as error:
        from_token_nats(1.0, eval_text="", token_count=1)
    assert "empty" in str(error.value)


def test_zero_tokens_are_refused() -> None:
    with pytest.raises(LoudError) as error:
        bytes_per_token("abc", 0)
    assert "token_count" in str(error.value)


def test_a_distribution_that_does_not_sum_to_one_is_refused() -> None:
    with pytest.raises(LoudError) as error:
        entropy_bits_per_byte([0.5, 0.4])
    assert "sum to" in str(error.value)
