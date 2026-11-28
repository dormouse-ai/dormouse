"""Build a small, VALID token stream from DCLM for the 2h pretrain smoke.

Fixes from research/dbg/02-data-bugs.md:
* tokenize with the NEW 14-special tokenizer (SHA 50235922e0bacb3a). The root
  `tokenizer32k.json` (ed73) is LEGACY: 3 specials, ordinary ids unshifted, and
  `data.tokenizer.load()` does not call `check_specials` here, so it silently passes
  `check_vocab` while the model reads id 4 as `<|end|>`. `check_specials` is called
  below: a legacy file is refused, not tokenized into a plausible loss curve.
* ids are written as uint16 — the legacy->NEW +14 conversion's uint8 overflow
  (242..255 -> 0..13) cannot happen when tokenizing straight into the NEW vocab.
* the manifest always carries the real sha_file() of the tokenizer file, never "".
* bytes_per_token is MEASURED on the exact documents that produced the ids;
  `bytes_per_token=1.0` on a BPE stream is what turned loss 714 into bpb 1030.
* train/heldout disjointness is a real document-hash intersection, not a literal 0.

    PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/python \
      scripts/build_data_prep_2h.py \
      --shard /mnt/e43497ab-*/dormouse-datasets/dclm/global-shard_01_of_10/ \
        local-shard_0_of_10/shard_00000090_processed.jsonl.zst \
      --tokenizer data-prep-2h/tokenizer32k.json --out-dir data-prep-2h \
      --target-bytes 32000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from dormouse.data import tokenizer

#: A document shorter than this teaches the model line breaks, not language.
MIN_CHARS = 200
#: Skip non-English pages: this smoke proves the pipe, not the mix rebalance.
MIN_EN = 0.5


def read_docs(shard: Path, target_bytes: int) -> list[tuple[str, str]]:
    """Stream decompressed jsonl until `target_bytes` of usable text is collected."""
    proc = subprocess.Popen(
        ["zstd", "-dc", str(shard)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    docs: list[tuple[str, str]] = []
    total = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                continue
            lang = row.get("language_id_whole_page_fasttext") or {}
            if isinstance(lang, dict) and lang.get("en", 0.0) < MIN_EN:
                continue
            doc = text + "\n\n"
            digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
            docs.append((digest, doc))
            total += len(doc.encode("utf-8"))
            if total >= target_bytes:
                break
    finally:
        proc.stdout.close()
        proc.wait()
    if total < target_bytes:
        raise SystemExit(f"{shard}: only {total} bytes of usable text, wanted {target_bytes}")
    return docs


def write_stream(
    tok_path: Path, docs: list[tuple[str, str]], out: Path, *, source: str
) -> dict[str, object]:
    """Tokenize `docs` into uint16, measure bytes/token, write bin + manifest."""
    tok = tokenizer.load(tok_path)
    ids = tokenizer.tokenize(tok, [doc for _, doc in docs])
    n_bytes = sum(len(doc.encode("utf-8")) for _, doc in docs)
    if len(ids) == 0:
        raise SystemExit(f"{out}: no tokens produced")
    bpt = n_bytes / len(ids)
    out.write_bytes(np.asarray(ids, dtype=np.uint16).tobytes())
    hashes = [digest for digest, _ in docs]
    Path(str(out) + ".docs.json").write_text(json.dumps(hashes, indent=0) + "\n")
    meta = {
        "dtype": "uint16",
        "tokens": len(ids),
        "tokenizer_sha": tokenizer.sha_file(tok_path),
        "bytes_per_token": round(bpt, 6),
        "source": source,
        "doc_hashes_sha": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
        "documents": len(hashes),
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return {"path": str(out), "bytes": n_bytes, **meta}


def main() -> None:
    """Cut one DCLM shard into a train and a document-disjoint held-out stream."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=Path, required=True, help="a DCLM *.jsonl.zst shard")
    p.add_argument("--tokenizer", type=Path, default=Path("data-prep-2h/tokenizer32k.json"))
    p.add_argument("--out-dir", type=Path, default=Path("data-prep-2h"))
    p.add_argument("--target-bytes", type=int, default=32_000_000)
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    args = p.parse_args()
    if not 0.0 < args.heldout_fraction < 0.5:
        raise SystemExit("--heldout-fraction must be between 0 and 0.5")

    # Refuse the legacy 3-special tokenizer: its ordinary ids are NOT the model's ids.
    tok = tokenizer.load(args.tokenizer)
    try:
        tokenizer.check_specials(tok)
    except Exception as exc:
        raise SystemExit(f"{args.tokenizer}: not the NEW tokenizer: {exc}") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs = read_docs(args.shard, args.target_bytes)
    split = int(len(docs) * (1.0 - args.heldout_fraction))
    train, heldout = docs[:split], docs[split:]
    if not train or not heldout:
        raise SystemExit("not enough documents to make both splits")
    shared = {digest for digest, _ in train} & {digest for digest, _ in heldout}
    if shared:
        raise SystemExit(f"train/heldout share {len(shared)} documents: refuse to write")

    source = str(args.shard)
    result = {
        "train": write_stream(args.tokenizer, train, args.out_dir / "train.bin", source=source),
        "heldout": write_stream(
            args.tokenizer, heldout, args.out_dir / "heldout.bin", source=source
        ),
        "shared_documents_measured": len(shared),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
