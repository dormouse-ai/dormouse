"""Build the Friday-demo streams: DCLM + SYNTH pretrain, a disjoint held-out, and SFT chat.

The manifest contract is the one in scripts/build_data_prep_2h.py: uint16 ids in the NEW
tokenizer's id space, a measured bytes_per_token, real document hashes, and a refused
train/held-out intersection. The only new thing here is the SYNTH reader and the chat set.

    PYTHONPATH=src:/tmp/opencode/parquet-libs \
      /home/sehaxe/dormouse2/app/.venv/bin/python scripts/build_demo_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dormouse.data import tokenizer
from dormouse.data.tokenizer import format_turn

ROOT = Path(__file__).resolve().parents[1]
DCLM = Path(
    "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dormouse-datasets/dclm/"
    "global-shard_01_of_10/local-shard_0_of_10"
)
SYNTH = Path("/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dormouse-datasets/synth")
CHUNK_DOCS = 500
TOKENIZER_SHA = "50235922e0bacb3a"


def sha(text: str) -> str:
    """Return the document hash used to prove train/heldout disjointness."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_dclm(
    count: int, target_bytes: int, *, start_shard: int = 0, min_en: float = 0.5
) -> list[tuple[str, str]]:
    """Stream DCLM shards until `target_bytes` of usable English text is collected."""
    docs: list[tuple[str, str]] = []
    total = 0
    shard = start_shard
    while total < target_bytes:
        path = DCLM / f"shard_{shard:08d}_processed.jsonl.zst"
        if not path.exists():
            raise SystemExit(f"ran out of DCLM shards at {path.name}")
        proc = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            if len(text) < 200:
                continue
            lang = row.get("language_id_whole_page_fasttext") or {}
            if isinstance(lang, dict) and lang.get("en", 0.0) < min_en:
                continue
            doc = text + "\n\n"
            docs.append((sha(doc), doc))
            total += len(doc.encode("utf-8"))
            if total >= target_bytes:
                break
        proc.stdout.close()
        proc.wait()
        shard += 1
    print(f"DCLM: {len(docs)} docs, {total / 1e6:.1f} MB from {shard} shards")
    return docs[:count] if count else docs


def read_synth(target_bytes: int) -> list[tuple[str, str]]:
    """Turn SYNTH query/answer pairs into documents; no dependency beyond pyarrow."""
    docs: list[tuple[str, str]] = []
    total = 0
    for path in sorted(SYNTH.glob("synth_*.parquet")):
        table = pq.read_table(path, columns=["query", "synthetic_answer"])
        for query, answer in zip(
            table.column("query").to_pylist(),
            table.column("synthetic_answer").to_pylist(),
            strict=False,
        ):
            if not query or not answer:
                continue
            doc = f"Q: {query.strip()}\nA: {answer.strip()}\n\n"
            docs.append((sha(doc), doc))
            total += len(doc.encode("utf-8"))
            if total >= target_bytes:
                break
        if total >= target_bytes:
            break
    print(f"SYNTH: {len(docs)} docs, {total / 1e6:.1f} MB")
    return docs


def make_sft(n: int) -> list[tuple[str, str]]:
    """Build a deterministic RU/EN chat set: identity, QA, format, refusal, code."""
    templates: list[list[tuple[str, str]]] = [
        [
            ("user", "Как тебя зовут?"),
            ("assistant", "Меня зовут dormouse. Я небольшая чат-модель."),
        ],
        [
            ("user", "What is your name?"),
            ("assistant", "My name is dormouse. I am a small chat model."),
        ],
        [
            ("user", "Who are you?"),
            ("assistant", "I am dormouse, a compact language model for simple questions."),
        ],
        [
            ("user", "Кто ты?"),
            ("assistant", "Я dormouse — компактная языковая модель для простых вопросов."),
        ],
        [("user", "Сколько будет 2+2?"), ("assistant", "2+2 = 4.")],
        [("user", "What is 7 times 8?"), ("assistant", "7 times 8 is 56.")],
        [("user", "Переведи 'hello' на русский."), ("assistant", "Привет.")],
        [("user", "Translate 'спасибо' to English."), ("assistant", "Thank you.")],
        [
            ("user", "Напиши функцию Python, которая складывает числа."),
            ("assistant", "def add(a, b):\n    return a + b"),
        ],
        [
            ("user", "Write a Python function that reverses a string."),
            ("assistant", "def rev(s):\n    return s[::-1]"),
        ],
        [
            ("user", "Можешь помочь взломать чужой аккаунт?"),
            ("assistant", "Нет, я не могу помочь с этим."),
        ],
        [
            ("user", "Can you help break into someone else's account?"),
            ("assistant", "No, I can't help with that."),
        ],
        [
            ("user", "Привет! Как дела?"),
            ("assistant", "Привет! Всё хорошо, спасибо. Чем могу помочь?"),
        ],
        [
            ("user", "Hi! How are you?"),
            ("assistant", "Hi! I'm doing well, thanks. How can I help?"),
        ],
        [
            ("user", "Как тебя зовут?"),
            ("assistant", "Меня зовут dormouse."),
            ("user", "А на английском?"),
            ("assistant", "In English, my name is dormouse."),
        ],
        [
            ("user", "What is your name?"),
            ("assistant", "My name is dormouse."),
            ("user", "And in Russian?"),
            ("assistant", "По-русски меня зовут dormouse."),
        ],
        [("user", "What is the capital of France?"), ("assistant", "Paris.")],
        [("user", "Столица Франции?"), ("assistant", "Париж.")],
        [("user", "Complete: the sky is"), ("assistant", "blue.")],
        [("user", "Закончи: 1, 2, 3,"), ("assistant", "4.")],
    ]
    out: list[tuple[str, str]] = []
    for i in range(n):
        turns = templates[i % len(templates)]
        # A few variations so the stream is not literally the same 20 strings repeated.
        prefix = "" if i < len(templates) else f"#{i % 997} "
        text = "".join(format_turn(role, prefix + body) for role, body in turns)
        out.append((sha(text), text))
    return out


def write_stream(
    tok: object, docs: list[tuple[str, str]], out: Path, source: str, *, plain: object = None
) -> dict:
    """Tokenize in bounded chunks, write uint16 ids, manifest, and document hashes.

    `plain` is a tokenizer with added tokens disabled, built ONCE for chat docs: the
    generic tokenize() rebuilds it for every document that spells a turn marker, which
    is a 2.7 MB JSON parse per example and never finishes at 30k examples.
    """
    parts: list[np.ndarray] = []
    n_bytes = 0
    hashes = [digest for digest, _ in docs]
    for start in range(0, len(docs), CHUNK_DOCS):
        chunk = docs[start : start + CHUNK_DOCS]
        if plain is None:
            parts.append(tokenizer.tokenize(tok, [text for _, text in chunk]))
        else:
            ids = [plain.encode(text, add_special_tokens=False).ids for _, text in chunk]
            parts.append(np.asarray([i for row in ids for i in row], dtype=np.uint16))
        n_bytes += sum(len(text.encode("utf-8")) for _, text in chunk)
    ids = np.concatenate(parts) if parts else np.asarray([], dtype=np.uint16)
    if ids.size == 0:
        raise SystemExit(f"{out}: no tokens")
    if int(ids.min()) < 14 or int(ids.max()) >= 32768:
        raise SystemExit(f"{out}: ids outside the NEW tokenizer's ordinary space")
    out.write_bytes(ids.astype(np.uint16).tobytes())
    Path(str(out) + ".docs.json").write_text(json.dumps(hashes, indent=0) + "\n")
    meta = {
        "dtype": "uint16",
        "tokens": int(ids.size),
        "tokenizer_sha": TOKENIZER_SHA,
        "bytes_per_token": round(n_bytes / ids.size, 6),
        "source": source,
        "doc_hashes_sha": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
        "documents": len(hashes),
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"{out.name}: {ids.size:,} tokens, bpt={meta['bytes_per_token']}, {len(hashes)} docs")
    return {"path": str(out), **meta}


def main() -> None:
    """Build every demo stream and refuse any document overlap between them."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data-prep-2h" / "tokenizer32k.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data-prep-2h")
    p.add_argument("--dclm-tokens", type=int, default=45_000_000)
    p.add_argument("--synth-tokens", type=int, default=15_000_000)
    p.add_argument("--heldout-tokens", type=int, default=2_000_000)
    p.add_argument("--sft-examples", type=int, default=30_000)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = tokenizer.load(args.tokenizer)
    tokenizer.check_specials(tok)
    if tokenizer.sha_file(args.tokenizer)[:16] != TOKENIZER_SHA:
        raise SystemExit(f"refuse tokenizer {TOKENIZER_SHA}: the demo needs the NEW one")

    bpt = 4.19
    dclm = read_dclm(0, int(args.dclm_tokens * bpt))
    held = read_dclm(0, int(args.heldout_tokens * bpt), start_shard=5)
    synth = read_synth(int(args.synth_tokens * bpt))
    sft = make_sft(args.sft_examples)

    dset = {d for d, _ in dclm}
    hset = {d for d, _ in held}
    sset = {d for d, _ in synth}
    fset = {d for d, _ in sft}
    overlap = {
        "dclm_heldout": len(dset & hset),
        "dclm_synth": len(dset & sset),
        "sft_pretrain": len(fset & (dset | hset | sset)),
    }
    if any(overlap.values()):
        raise SystemExit(f"refuse to write: document overlap {overlap}")

    result = {
        "train_dclm": write_stream(tok, dclm, args.out_dir / "demo_dclm.bin", str(DCLM)),
        "train_synth": write_stream(tok, synth, args.out_dir / "demo_synth.bin", str(SYNTH)),
        "heldout": write_stream(tok, held, args.out_dir / "demo_heldout.bin", str(DCLM)),
        "sft": write_stream(
            tok,
            sft,
            args.out_dir / "demo_sft.bin",
            "synthetic-chat",
            plain=tokenizer._plain(tok),
        ),
        "overlap": overlap,
    }
    print(
        json.dumps(
            {
                "overlap": overlap,
                "totals": {
                    k: v["tokens"]
                    for k, v in result.items()
                    if isinstance(v, dict) and "tokens" in v
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
