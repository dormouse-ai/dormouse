"""The tokenizer's contract: a joint BPE, measured fertility, a hash that refuses drift.

The expensive failures here are quiet: a tokenizer retrained "the same way" that is not
the same, a model whose token table is a row short, a corpus retokenized while a run is
reading it. Each test below pins one of those to an error or an exact number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dormouse_core.errors import LoudError
from tokenizers import normalizers

from dormouse.data.mix import MixtureStream, Source
from dormouse.data.tokenizer import (
    DEFAULT_VOCAB,
    DOMAINS,
    MAX_VOCAB,
    check_vocab,
    fertility,
    load,
    round_trip,
    save,
    sha,
    tokenize,
    train,
    write_manifest,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "corpus_2mb.bin"

EN = FIXTURE.read_bytes()[:4000].decode("utf-8", errors="replace")

RU = (
    "Модель обучается на смеси корпусов: веб, википедия, русский веб, математика и код. "
    "Токенизатор учится один раз на всей смеси, чтобы одни и те же идентификаторы "
    "значили одно и то же во всех источниках. Проверка на смещение — доля каждого "
    "источника в любом окне обучения, а не в среднем по прогону. "  # noqa: RUF001
) * 3

MATH = (
    r"x^2 + y^2 = z^2, \frac{a}{b} = \sqrt{c}, \sum_{i=1}^{n} i = \frac{n(n+1)}{2}, "
    r"\int_0^\infty e^{-x} dx = 1, \mathbb{E}[X] = \sum_x x p(x), \theta_{t+1} = "
    r"\theta_t - \eta \nabla L(\theta_t), a \leq b \iff b \geq a, "
) * 3

CODE = (
    "def batches(self, batch, seq_len, step=0):\n"
    "    per_step = batch * seq_len\n"
    "    while True:\n"
    "        offset = step * per_step\n"
    "        yield self.take(offset, batch * (seq_len + 1))\n"
    "        step += 1\n"
) * 3

SAMPLES = {"en": EN, "ru": RU, "math": MATH, "code": CODE}


@pytest.fixture(scope="module")
def tok():
    """One small joint BPE: 320 ids is 20 s of merges less than 48 000."""
    return train(SAMPLES, vocab_size=320)


# ---------------------------------------------------------------------- training
def test_the_mixture_trains_one_joint_vocabulary(tok) -> None:
    assert 256 < tok.get_vocab_size() <= 320
    vocab = tok.get_vocab()
    assert "<unk>" in vocab and "<s>" in vocab and "</s>" in vocab
    assert len(vocab) == tok.get_vocab_size()


def test_training_is_deterministic() -> None:
    """A manifest hash that changes per run is a hash that cannot refuse anything."""
    assert sha(train(SAMPLES, vocab_size=320)) == sha(train(SAMPLES, vocab_size=320))


def test_an_empty_mixture_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        train({}, vocab_size=320)
    assert "no text" in str(e.value)


def test_a_vocabulary_below_the_byte_alphabet_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        train(SAMPLES, vocab_size=100)
    assert "byte alphabet" in str(e.value)


def test_a_vocabulary_past_uint16_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        train(SAMPLES, vocab_size=MAX_VOCAB + 1)
    assert "uint16" in str(e.value)


def test_the_default_vocabulary_is_48k() -> None:
    assert DEFAULT_VOCAB == 48_000


# -------------------------------------------------------------------- round trip
def test_one_megabyte_round_trips_through_the_tokenizer(tok) -> None:
    """Byte-level BPE plus a byte alphabet: every byte of a real 1 MB slice survives."""
    text = FIXTURE.read_bytes()[: 1024 * 1024].decode("latin-1")
    round_trip(tok, text, name="1MB latin-1 slice")


def test_every_byte_value_round_trips(tok) -> None:
    text = bytes(range(256)).decode("latin-1") * 2
    round_trip(tok, text, name="all 256 bytes")


def test_a_normalizer_that_rewrites_text_is_caught(tok) -> None:
    """The failure this test stands in for: a tokenizer that quietly normalises."""
    composed = "café naïve"
    round_trip(tok, composed)
    tok.normalizer = normalizers.NFD()
    with pytest.raises(LoudError) as e:
        round_trip(tok, composed)
    assert "does not round-trip" in str(e.value)
    assert "normalizer" in str(e.value)
    tok.normalizer = None


# -------------------------------------------------------------------- fertility
def test_fertility_is_a_number_per_domain(tok) -> None:
    report = fertility(tok, SAMPLES)
    assert set(report) == set(DOMAINS)
    assert all(value > 1.0 for value in report.values())  # bytes per token, by definition
    assert all(np.isfinite(value) for value in report.values())
    # 320 ids on 12 kB of text is a stand-in for the 48 000 the run uses; the number
    # that matters here is that the report is measured, not that it is the 4.4 of a
    # 48K tokenizer on 100 GB.
    assert report["en"] > 1.2


def test_fertility_refuses_an_empty_domain(tok) -> None:
    with pytest.raises(LoudError) as e:
        fertility(tok, {"en": ""})
    assert "no text" in str(e.value)
    with pytest.raises(LoudError) as e:
        fertility(tok, {})
    assert "text per domain" in str(e.value)


# ----------------------------------------------------------------- vocabulary
def test_a_token_table_that_is_a_row_off_is_refused(tok) -> None:
    check_vocab(tok, tok.get_vocab_size())
    with pytest.raises(LoudError) as e:
        check_vocab(tok, tok.get_vocab_size() + 1)
    assert str(tok.get_vocab_size()) in str(e.value)
    assert "token table" in str(e.value)


def test_a_vocabulary_the_corpus_cannot_pay_for_is_caught(tok) -> None:
    """Asking for 48K on a small sample yields fewer ids; the model's table must refuse."""
    big = train(SAMPLES, vocab_size=5_000)
    assert big.get_vocab_size() < 5_000
    with pytest.raises(LoudError) as e:
        check_vocab(big, 5_000)
    assert "5000" in str(e.value)


def test_ids_are_uint16_on_disk(tok) -> None:
    ids = tokenize(tok, ["hello", "мир"])
    assert ids.dtype == np.uint16
    assert ids.size > 0
    assert int(ids.max()) < tok.get_vocab_size()


# --------------------------------------------------- the hash in every manifest
def test_manifest_and_stream_refuse_a_tokenizer_mismatch(tmp_path: Path, tok) -> None:
    """End to end: tokenize → manifest with a hash → stream says whose ids these are."""
    ids = tokenize(tok, [EN, RU])
    path = tmp_path / "corpus.bin"
    ids.tofile(path)
    meta = write_manifest(path, tokens=int(ids.size), tok=tok)
    assert json.loads(meta.read_text())["tokenizer_sha"] == sha(tok)
    ok = MixtureStream([Source(path, 1.0)], tokenizer_sha=sha(tok))
    assert ok.sources[0].weight == 1.0
    other = sha(train(SAMPLES, vocab_size=300))
    assert other != sha(tok)
    with pytest.raises(LoudError) as e:
        MixtureStream([Source(path, 1.0)], tokenizer_sha=other)
    assert other in str(e.value)  # both hashes are printed, so the diff is a diff


def test_save_and_load_preserve_the_hash(tmp_path: Path, tok) -> None:
    path = save(tok, tmp_path / "tokenizer.json")
    assert sha(load(path)) == sha(tok)
    with pytest.raises(LoudError) as e:
        load(tmp_path / "missing.json")
    assert "no tokenizer" in str(e.value)
