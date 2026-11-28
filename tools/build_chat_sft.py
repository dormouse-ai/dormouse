"""Build a small, real chat stream in the tokenizer's own turn format.

The old `demo_sft.bin` was a synthetic text stream with no role ids; the audit found
zero structural chat tokens, so no SFT run on it could learn the format. This builder
renders every document through `format_turn` and the tokenizer the config names, and
writes the three files `data/bin.py` expects. It is a fixture, not a quality corpus:
the value is that the SFT path sees the ids it is supposed to see.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from dormouse.data.tokenizer import format_turn, sha_file

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data-prep-2h" / "tokenizer32k.json"
OUT = ROOT / "data-prep-2h" / "demo_sft_chat.bin"
DOCS = 30_000

TOPICS = (
    ("привет", "Привет. Я Dormouse, локальный ассистент."),
    ("как тебя зовут", "Меня зовут Dormouse."),
    ("кто ты", "Я небольшая локальная модель, которая работает без облака."),
    ("сколько будет 2 + 2", "Четыре."),
    ("сколько будет 7 * 8", "Пятьдесят шесть."),
    ("переведи слово cat", "Кошка."),
    ("напиши функцию суммы на python", "def add(a, b):\n    return a + b"),
    ("что такое ЭКГ", "ЭКГ записывает электрическую активность сердца."),
    ("почему локальный ИИ", "Данные не покидают машину."),
    ("спасибо", "Пожалуйста."),
)


def document(index: int) -> str:
    """One deterministic dialogue; every fifth one has two user turns."""
    question, answer = TOPICS[index % len(TOPICS)]
    if index % 3 == 0:
        question = f"{question} (вариант {index})"
    system = format_turn("system", "Ты Dormouse, краткий локальный ассистент.")
    first = format_turn("user", question) + format_turn("assistant", answer)
    if index % 5:
        return system + first
    second = format_turn("user", "Спасибо.") + format_turn("assistant", "Пожалуйста.")
    return system + first + second


def main() -> None:
    """Write `demo_sft_chat.bin`, its document list and its manifest."""
    tok = Tokenizer.from_file(str(TOKENIZER))
    docs = [document(i) for i in range(DOCS)]
    ids: list[int] = []
    for doc in docs:
        ids.extend(tok.encode(doc, add_special_tokens=False).ids)
    array = np.asarray(ids, dtype=np.uint16)
    array.tofile(OUT)
    hashes = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
    hashes_path = OUT.with_suffix(OUT.suffix + ".docs.json")
    hashes_path.write_text(json.dumps(hashes))
    manifest = {
        "bytes_per_token": sum(len(doc.encode()) for doc in docs) / max(1, len(ids)),
        "doc_hashes_sha": hashlib.sha256(hashes_path.read_bytes()).hexdigest(),
        "documents": len(docs),
        "dtype": "uint16",
        "source": "synthetic-chat-fixture",
        "tokenizer_sha": sha_file(TOKENIZER),
        "tokens": int(array.size),
    }
    OUT.with_suffix(OUT.suffix + ".json").write_text(json.dumps(manifest, indent=1))
    print(f"{OUT}: {array.size:,} tokens, {len(docs):,} docs")


if __name__ == "__main__":
    main()
