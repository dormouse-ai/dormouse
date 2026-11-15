"""The reserved set is the format: fourteen ids, one stop, and a corpus that keeps its bytes.

Three gates, each one a refusal rather than a convention:

* the markers are *added* tokens with `special=True`, so BPE cannot split one and the id
  the sampler stops on is the id the training stream carries;
* the stop is one id (`END_ID`), not a string window, so nothing can end a turn with
  content;
* a document that spells `<|end|>` — our code corpus quotes it in documentation — keeps
  its bytes. The added-token match is what makes a turn atomic and what would eat that
  document, so the corpus path encodes through a copy of the tokenizer with the match
  switched off. This file is the gate: if the literal ever encodes to `END_ID`, the
  tokenizer is eating content.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError
from tokenizers import Tokenizer, models, pre_tokenizers

from dormouse.data.tokenizer import (
    BYTE_ALPHABET,
    DEFAULT_VOCAB,
    END,
    END_ID,
    SPECIALS,
    check_specials,
    fertility,
    format_turn,
    round_trip,
    tokenize,
    train,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "dormouse"

#: What the code corpus actually holds: prose *about* the tokenizer, marker included.
QUOTED = (
    "# Tokenizer notes\n"
    "The run stops when the sampled id equals the stop id, which the model card writes as\n"
    "<|end|>. A chat file opens a turn with <|assistant|> and closes it with <|end|>, and\n"
    "<|think|> wraps the reasoning that precedes the answer.\n"
)

#: A turn as the format spells it, and as the SFT loader will read it back.
TURN = re.compile(
    r"<\|(system|user|assistant)\|>(?:<\|think\|>(.*?)<\|/think\|>)?(.*?)<\|end\|>", re.S
)

_RESERVED = re.compile(
    r"<\|(?:end|system|user|assistant|think|/think|tool_call)\|>|<TEMP_LOW>|<TEMP_HIGH>|<CHECK>"
)


def _corpus() -> str:
    """A deterministic word soup: enough distinct pairs to spend a small vocabulary."""
    rng = random.Random(0)
    words = [
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
        for _ in range(6000)
    ]
    return "\n".join(" ".join(words[i : i + 8]) for i in range(0, len(words), 8))


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train({"en": _corpus()}, vocab_size=1000)


def test_the_reserved_set_is_fourteen_ids_and_fits_the_byte_budget() -> None:
    reserved = {
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "<|end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|think|>",
        "<|/think|>",
        "<|tool_call|>",
        "<TEMP_LOW>",
        "<TEMP_HIGH>",
        "<CHECK>",
    }
    assert set(SPECIALS) == reserved
    assert len(SPECIALS) == len(set(SPECIALS)) == 14
    assert len(SPECIALS) < 32
    assert DEFAULT_VOCAB == 32_000


def test_the_specials_are_added_tokens_that_bpe_cannot_split(tok: Tokenizer) -> None:
    added = tok.get_added_tokens_decoder()
    assert {token.content for token in added.values()} == set(SPECIALS)
    assert all(token.special for token in added.values())
    for index, token in enumerate(SPECIALS):
        assert tok.token_to_id(token) == index
        assert tok.encode(token, add_special_tokens=False).ids == [index]
    assert tok.token_to_id("<|end|>") == END_ID == SPECIALS.index(END)


def test_there_is_exactly_one_stop_token_and_its_id_is_exported(tok: Tokenizer) -> None:
    assert END_ID == tok.token_to_id(END) == 4
    assert SPECIALS[END_ID] == END
    assert [token for token in SPECIALS if token == END] == [END]
    assert tok.decode([END_ID], skip_special_tokens=False) == END


def test_a_document_that_quotes_the_marker_does_not_get_the_stop_id(tok: Tokenizer) -> None:
    ids = tokenize(tok, [QUOTED]).tolist()
    assert END_ID not in ids
    assert min(ids) >= len(SPECIALS)
    assert QUOTED.count(END) == 2
    round_trip(tok, QUOTED)


def test_the_literal_stop_spelling_is_pieces_and_round_trips(tok: Tokenizer) -> None:
    ids = tokenize(tok, [END]).tolist()
    assert END_ID not in ids
    assert len(ids) > 1
    round_trip(tok, END)


def test_the_turn_encoder_would_eat_the_literal_which_is_why_documents_never_use_it(
    tok: Tokenizer,
) -> None:
    ids = tok.encode(QUOTED, add_special_tokens=False).ids
    assert END_ID in ids
    assert tok.decode(ids) != QUOTED


def test_the_specials_sit_outside_the_byte_alphabet_and_the_fallback_is_complete(
    tok: Tokenizer,
) -> None:
    byte_ids = {tok.token_to_id(char) for char in pre_tokenizers.ByteLevel.alphabet()}
    assert None not in byte_ids
    assert len(byte_ids) == BYTE_ALPHABET
    assert byte_ids & set(range(len(SPECIALS))) == set()


def test_training_spends_the_requested_size_on_the_whole_vocabulary(tok: Tokenizer) -> None:
    assert tok.get_vocab_size() == 1000
    assert tok.get_vocab_size() - len(SPECIALS) - BYTE_ALPHABET > 0


def test_a_vocabulary_too_small_for_the_alphabet_and_the_specials_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        train({"en": _corpus()}, vocab_size=BYTE_ALPHABET + len(SPECIALS) - 1)
    assert "specials" in str(caught.value)


def test_check_specials_refuses_a_vocabulary_without_the_markers() -> None:
    foreign = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    with pytest.raises(LoudError) as caught:
        check_specials(foreign)
    assert "expects" in str(caught.value)


def test_a_conversation_is_three_markers_and_one_stop_per_turn() -> None:
    rendered = (
        format_turn("system", "You are Dormouse.")
        + format_turn("user", "привет")
        + format_turn("assistant", "привет.", think="the user greeted in Russian")
    )
    assert rendered == (
        "<|system|>You are Dormouse.<|end|><|user|>привет<|end|>"
        "<|assistant|><|think|>the user greeted in Russian<|/think|>привет.<|end|>"
    )


def test_a_formatted_turn_parses_back_to_its_parts() -> None:
    for role, text, think in (
        ("system", "You are Dormouse.", None),
        ("user", "what is 2+2?", None),
        ("assistant", "4", "two plus two is four"),
    ):
        parsed = TURN.fullmatch(format_turn(role, text, think))
        assert parsed is not None
        assert parsed.group(1) == role
        assert parsed.group(2) == think
        assert parsed.group(3) == text


def test_content_that_spells_a_reserved_token_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        format_turn("user", "what does <|end|> do?")
    assert "<|end|>" in str(caught.value)
    with pytest.raises(LoudError) as caught:
        format_turn("assistant", "ok", think="<|end|>")
    assert "thinking" in str(caught.value)
    with pytest.raises(LoudError) as caught:
        format_turn("tool", "x")
    assert "no role 'tool'" in str(caught.value)


def test_a_turn_encodes_to_one_stop_id_per_turn(tok: Tokenizer) -> None:
    ids = tok.encode(format_turn("user", "hi"), add_special_tokens=False).ids
    assert ids.count(END_ID) == 1
    assert ids[-1] == END_ID


def test_fertility_counts_a_quoted_marker_as_content(tok: Tokenizer) -> None:
    measured = fertility(tok, {"en": "plain text " * 10, "code": QUOTED})
    assert set(measured) == {"en", "code"}
    assert measured["code"] > 0


def test_the_chat_format_is_spelled_only_in_the_tokenizer() -> None:
    """A builder that hardcodes `<|assistant|>` would drift from `format_turn`."""
    hits = [
        f"{path.relative_to(SRC)}:{number}"
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "tokenizer.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _RESERVED.search(line)
    ]
    assert hits == [], "a second copy of the chat format (call format_turn): " + ", ".join(hits)
