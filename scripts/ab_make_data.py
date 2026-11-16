"""Derive a vocab-clamped copy of the 200 MB bpe slice for the A/B.

`dormouse`'s bpe table is `DEFAULT_VOCAB = 32000` rows, but `mix_smoke.bin` was cut
from a 32768-id tokenizer: 0.29% of its ids are in [32000, 32768) and would index past
the embedding on step 1. Changing the model's vocab is out of scope for a config A/B,
and setting `tokenizer=` in the run config is refused by the same check, so the id
range is clamped in a derived copy instead. Both arms read the SAME file, so the
clamp cannot confound the BPB delta; the byte ratio and token order are untouched.

    .venv/bin/python scripts/ab_make_data.py /tmp/opencode/ab_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCES = {
    "mix_smoke": Path("/home/sehaxe/dormouse/data_local/pre/mix_smoke.bin"),
    "holdout": Path("/home/sehaxe/dormouse/data_local/pre/holdout.bin"),
}


def main() -> int:
    """Write clamped train and held-out slices, keeping each measured byte ratio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", help="where to write the *_clamped.bin(.json) files")
    parser.add_argument("--vocab", type=int, default=32_000, help="the engine's bpe row count")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        meta = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
        ids = np.memmap(source, dtype=np.uint16, mode="r")
        high = int((ids >= args.vocab).sum())
        clamped = np.asarray(ids).copy()
        clamped[clamped >= args.vocab] = 0
        target = out / f"{name}_clamped.bin"
        target.write_bytes(clamped.tobytes())
        manifest = {
            "dtype": "uint16",
            "tokens": int(clamped.size),
            "clamped_from": str(source),
            "ids_clamped": high,
            "clamp_to": 0,
        }
        if "bytes_per_token" in meta:
            manifest["bytes_per_token"] = meta["bytes_per_token"]
        (out / f"{name}_clamped.bin.json").write_text(
            json.dumps(
                manifest,
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"path": str(target), "tokens": int(clamped.size), "ids_clamped": high}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
