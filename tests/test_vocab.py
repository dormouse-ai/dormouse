"""The vocabulary is 32 000, and the number lives in exactly one place.

The tokenizer experiment settled the size: 48 000 moved fertility from 4.395/4.337/3.424/
3.505 to 4.423/4.418/3.433/3.518 (en/ru/math/code), which is 0.8 % fewer tokens, while the
head's `2·V·d` MAC went from 26.6M to 40.0M — roughly 5 % of the training step. The
model-layer rule is therefore that `model/` holds no copy of the number at all: it reads
`data.tokenizer.DEFAULT_VOCAB`, and this file is what enforces that rather than a review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

from dormouse.data.tokenizer import DEFAULT_VOCAB
from dormouse.model.spec import VOCAB, StackSpec

SRC = Path(__file__).resolve().parents[1] / "src" / "dormouse"

#: A bare copy of the vocabulary, in any spelling a tired reader would type.
LITERAL = re.compile(r"\b(?:32768|32_768|48000|48_000|32_000)\b")


def test_the_default_is_the_measured_size() -> None:
    assert DEFAULT_VOCAB == 32_000


def test_the_number_lives_in_exactly_one_place() -> None:
    """`model/` must not spell the vocabulary out: only the tokenizer owns it."""
    hits = [
        f"{path.relative_to(SRC)}:{number}"
        for path in sorted((SRC / "model").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if LITERAL.search(line)
    ]
    assert hits == [], "a second copy of the vocabulary (read data/tokenizer.py): " + ", ".join(
        hits
    )


def test_the_spec_reads_the_tokenizers_vocabulary() -> None:
    assert StackSpec().vocab == DEFAULT_VOCAB == VOCAB["bpe"]
    assert StackSpec(unit="byte").vocab == 256


def test_a_config_that_states_a_vocab_is_checked_not_trusted() -> None:
    with pytest.raises(LoudError) as caught:
        StackSpec.from_mapping({"vocab": 48_000})
    message = str(caught.value)
    assert "vocab=48000" in message and str(DEFAULT_VOCAB) in message
    assert "check_vocab" in message
    assert StackSpec.from_mapping({"vocab": DEFAULT_VOCAB}).vocab == DEFAULT_VOCAB


def test_a_named_tokenizer_is_checked_with_check_vocab(tmp_path: Path) -> None:
    from tokenizers import Tokenizer, models

    path = tmp_path / "tokenizer.json"
    path.write_text(Tokenizer(models.WordLevel({"a": 0}, unk_token="a")).to_str())
    with pytest.raises(LoudError) as caught:
        StackSpec(tokenizer=str(path))
    message = str(caught.value)
    assert "the tokenizer has 1 ids" in message and str(DEFAULT_VOCAB) in message
