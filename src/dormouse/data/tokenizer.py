"""The tokenizer as a measured artifact: trained on the mixture, hashed into the data.

Four numbers decided this module, and each one is a refusal rather than a preference:

* VOCABULARY SIZE. 32 000, and this module is the only place the number lives: the
  model's `vocab` is read from `DEFAULT_VOCAB` and `check_vocab` refuses a tokenizer
  and a model that disagree. MEASURED on our own mixture, 48 000 buys nothing:
  fertility (bytes/token) is 4.423 / 4.418 / 3.433 / 3.518 at 48K for en / ru / math /
  code against 4.395 / 4.337 / 3.424 / 3.505 at 32K — 0.8 % fewer tokens, which the head
  eats many times over. The head's cost is `2·V·d` MAC: 26.6M at 32K against 40.0M at
  48K, +80 MFLOP/token of training, about 5 % of the whole step, for that 0.8 %. The
  published 43K result is at 3B parameters and 100B tokens and does not transfer: at
  that size the curve is a capacity statement, not a tokenizer one.
* BYTE-LEVEL, so a byte that never appeared in training still decodes: `round_trip`
  proves it on 1 MB rather than trusting it. `<unk>` exists and is never produced.
* FERTILITY, per domain and measured, because a joint BPE that is good on English and
  bad on Russian is the failure mode of every "multilingual" tokenizer: bytes/token on
  each domain is the number that decides whether the mix is affordable.
* A HASH in every data manifest. A batch of ids read with the wrong vocabulary is not
  a crash — it is a plausible loss curve, which is the expensive kind of defect.

Training is the Rust `tokenizers` BPE rather than a Python pair counter: 48 000 merges
over a corpus is 48 000 passes with numpy and seconds in Rust, and a trainer that is
too slow to run is a trainer whose output nobody re-derives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from dormouse_core.errors import LoudError
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

__all__ = [
    "DEFAULT_VOCAB",
    "DOMAINS",
    "MAX_VOCAB",
    "check_vocab",
    "fertility",
    "load",
    "round_trip",
    "save",
    "sha",
    "tokenize",
    "train",
    "write_manifest",
]

#: The default vocabulary: 32K, where fertility stops paying for the head on our mix
#: (the module docstring holds the measurement). Everything that needs the number reads
#: this constant — a literal `32768` or `48000` under `model/` is a second answer.
DEFAULT_VOCAB = 32_000

#: The exact alphabet, 256 bytes of it, always in the vocabulary — that is the fallback.
BYTE_ALPHABET = 256

#: `uint16` on disk, so the id space cannot exceed the word.
MAX_VOCAB = 65_536

#: The domains fertility is reported for. A mixture names at least these.
DOMAINS: tuple[str, ...] = ("en", "ru", "math", "code")

#: Tokens reserved before any merge: `<unk>`, `<s>`, `</s>`, `<pad>`.
SPECIALS: tuple[str, ...] = ("<unk>", "<s>", "</s>", "<pad>")


def train(
    samples: Mapping[str, str],
    *,
    vocab_size: int = DEFAULT_VOCAB,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train one byte-level BPE over the mixture, sized `vocab_size`.

    `samples` is domain → text, and the text is concatenated *as given*: a domain that
    is 60 % of the corpus must arrive as 60 % of the bytes, or the tokenizer's merges
    are allocated to whatever was pasted first rather than to what will be trained on.

    `vocab_size` is a ceiling, not a promise: a corpus with fewer distinct pairs stops
    earlier (`min_frequency`, or simply running out), and `check_vocab` is what turns
    that shortfall into a refusal where it matters — at the model's token table. A
    2.5 MB sample reaches ~35 000 ids when asked for 48 000; a 100 GB corpus does not.
    """
    if not samples:
        raise LoudError(
            "no text to train a tokenizer on",
            "pass {domain: text}; the mixture's own sample is the honest input",
        )
    if vocab_size < BYTE_ALPHABET + len(SPECIALS):
        raise LoudError(
            f"vocab_size={vocab_size} is smaller than the byte alphabet "
            f"({BYTE_ALPHABET}) plus {len(SPECIALS)} specials",
            f"ask for at least {BYTE_ALPHABET + len(SPECIALS)}; below that no byte-level "
            "tokenizer has room to fall back",
        )
    if vocab_size > MAX_VOCAB:
        raise LoudError(
            f"vocab_size={vocab_size} does not fit uint16 on disk",
            f"ids are stored as uint16 in this project; the ceiling is {MAX_VOCAB}",
        )
    text = "\n\n".join(samples[domain] for domain in sorted(samples))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer)
    return tok


def check_vocab(tok: Tokenizer, model_vocab: int) -> None:
    """Refuse a model whose token table is not the tokenizer's vocabulary."""
    actual = tok.get_vocab_size()
    if actual != model_vocab:
        raise LoudError(
            f"the tokenizer has {actual} ids, the model's token table has {model_vocab}",
            f"set the model's vocab to {actual}, or retrain the tokenizer at "
            f"vocab_size={model_vocab}; a table with rows no id reaches is dead "
            "parameters, and an id past the table is a crash on step 1",
        )


def fertility(tok: Tokenizer, samples: Mapping[str, str]) -> dict[str, float]:
    """Bytes per token on each domain: the number that says whether a mix is affordable."""
    if not samples:
        raise LoudError(
            "fertility needs text per domain",
            "pass {domain: text}; the held-out slice of each source is the honest input",
        )
    out: dict[str, float] = {}
    for domain, text in samples.items():
        if not text:
            raise LoudError(
                f"domain {domain!r} has no text",
                "an empty sample measures nothing; keep one document per domain",
            )
        ids = tok.encode(text, add_special_tokens=False).ids
        out[domain] = len(text.encode("utf-8")) / len(ids)
    return out


def round_trip(tok: Tokenizer, text: str, *, name: str = "sample") -> None:
    """Prove `decode(encode(text)) == text`, naming the first byte that does not survive."""
    back = tok.decode(tok.encode(text, add_special_tokens=False).ids)
    if back == text:
        return
    at = next(
        (i for i, (a, b) in enumerate(zip(text, back, strict=False)) if a != b),
        min(len(text), len(back)),
    )
    raise LoudError(
        f"{name}: the tokenizer does not round-trip; first difference at character {at} "
        f"({len(text)} in, {len(back)} out)",
        "byte-level BPE with a byte alphabet round-trips every byte, so a mismatch is a "
        "normalizer (it rewrote the text), a special token (it ate the text), or a "
        "different tokenizer than the one the manifest names",
    )


def tokenize(tok: Tokenizer, texts: Iterable[str]) -> np.ndarray:
    """Encode text to a `uint16` id array, refusing an id the disk format cannot hold."""
    ids: list[int] = []
    for text in texts:
        ids.extend(tok.encode(text, add_special_tokens=False).ids)
    out = np.asarray(ids, dtype=np.int64)
    if out.size and int(out.max()) >= MAX_VOCAB:
        raise LoudError(
            f"the tokenizer produced id {int(out.max())}, past uint16",
            "train at vocab_size <= 65536 or store ids as uint32",
        )
    return out.astype(np.uint16)


def sha(tok: Tokenizer) -> str:
    """Return the tokenizer's identity: sha256 of its serialised vocabulary and merges."""
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()


def save(tok: Tokenizer, path: str | Path) -> Path:
    """Write the tokenizer as one JSON file a run can reload and hash."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tok.to_str(), encoding="utf-8")
    return out


def load(path: str | Path) -> Tokenizer:
    """Read a tokenizer back, refusing a missing one by name."""
    source = Path(path)
    if not source.exists():
        raise LoudError(
            f"no tokenizer at {source}",
            "train one with `train(samples)` and `save`, or point at the run's tokenizer",
        )
    return Tokenizer.from_file(str(source))


def write_manifest(path: str | Path, *, tokens: int, tok: Tokenizer, dtype: str = "uint16") -> Path:
    """Write the manifest beside a token stream, with the tokenizer's hash in it.

    The hash is what makes `mix.MixtureStream` able to refuse a batch of ids that was
    written by another vocabulary; without it the file looks like data and reads as
    noise.
    """
    stream = Path(path)
    meta = stream.with_suffix(stream.suffix + ".json")
    meta.write_text(
        json.dumps(
            {"dtype": dtype, "tokens": int(tokens), "tokenizer_sha": sha(tok)}, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return meta
