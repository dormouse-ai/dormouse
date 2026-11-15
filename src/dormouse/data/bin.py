"""Reading a token stream: one file, one reader, no silent end-of-data.

Three defects in this project came from data reading, and each is a rule here:

* a run silently consumed past the end of its corpus and trained on wrapped data
  (now: reading past the end is an error, and wrapping is an explicit option);
* a resume could not reproduce its position because the stream carried state in a
  sidecar file (now: position is arithmetic — `skip` is a function of the step);
* a tokenizer change silently reinterpreted a `.bin` as `uint16` (now: the dtype
  lives in a manifest beside the data and a mismatch refuses).

Only the reader lives here; batching and shuffling belong to the caller, so this
module has no dependency beyond numpy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dormouse_core.errors import LoudError

#: Item sizes we accept, with the name used in the manifest.
DTYPES = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32}


@dataclass(frozen=True)
class Manifest:
    """What the bytes on disk mean, recorded next to them."""

    dtype: str
    tokens: int
    tokenizer_sha: str = ""

    @classmethod
    def read(cls, path: Path) -> Manifest:
        """Read `<data>.json` beside the stream; refuse when it is missing."""
        meta = path.with_suffix(path.suffix + ".json")
        if not meta.exists():
            raise LoudError(
                f"{path}: no manifest beside the stream",
                f"write {meta.name} with {{'dtype': 'uint16', 'tokens': N}}",
            )
        raw = json.loads(meta.read_text())
        if raw.get("dtype") not in DTYPES:
            raise LoudError(
                f"{meta}: unknown dtype {raw.get('dtype')!r}",
                f"use one of: {', '.join(sorted(DTYPES))}",
            )
        return cls(
            dtype=raw["dtype"],
            tokens=int(raw["tokens"]),
            tokenizer_sha=str(raw.get("tokenizer_sha", "")),
        )


class Stream:
    """A read-only window over a token stream, positioned by arithmetic."""

    def __init__(self, path: str | Path, *, verify_size: bool = True) -> None:
        """Open a stream, refusing when the file and its manifest disagree."""
        self.path = Path(path)
        if not self.path.exists():
            raise LoudError(f"no data at {self.path}", "check the path, or build the .bin first")
        self.manifest = Manifest.read(self.path)
        self.dtype = DTYPES[self.manifest.dtype]
        self.itemsize = np.dtype(self.dtype).itemsize
        expected = self.manifest.tokens * self.itemsize
        actual = self.path.stat().st_size
        if verify_size and actual != expected:
            raise LoudError(
                f"{self.path}: {actual} bytes but the manifest says "
                f"{self.manifest.tokens} x {self.dtype.__name__} = {expected}",
                "the file and its manifest disagree; rebuild one of them",
            )
        self._buf = np.memmap(self.path, dtype=self.dtype, mode="r")

    @property
    def tokens(self) -> int:
        """How many items the stream holds."""
        return self.manifest.tokens

    def take(self, start: int, count: int) -> np.ndarray:
        """Return `count` items from `start`; refuse to read past the end."""
        if start < 0:
            raise LoudError(f"start={start} is negative", "positions are non-negative")
        if count <= 0:
            raise LoudError(f"count={count} asks for nothing", "ask for at least one item")
        end = start + count
        if end > self.tokens:
            raise LoudError(
                f"read [{start}, {end}) past the end of {self.tokens} items",
                "shrink the batch, or pass wrap=True to train on a looped corpus",
            )
        return np.asarray(self._buf[start:end])

    def batches(
        self, batch: int, seq_len: int, *, step: int = 0, wrap: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(inputs, targets)` of shape `[batch, seq_len]`, deterministically.

        Position is a function of the step, not of hidden state: `step * batch *
        seq_len` items in, so a resumed run lands exactly where it stopped. The
        target is the input shifted by one, which means a window of `seq_len + 1`
        items is read for every `seq_len` predicted.
        """
        per_step = batch * seq_len
        need = batch * (seq_len + 1)
        if not wrap and need > self.tokens:
            raise LoudError(
                f"a step needs {need} items, the stream has {self.tokens}",
                "shrink the batch, or pass wrap=True",
            )
        while True:
            offset = step * per_step
            if not wrap and offset + need > self.tokens:
                return
            flat = self._gather(offset, need, wrap)
            window = flat.reshape(batch, seq_len + 1)
            yield window[:, :-1], window[:, 1:]
            step += 1

    def _gather(self, offset: int, count: int, wrap: bool) -> np.ndarray:
        """Read `count` items, wrapping around the end when asked to."""
        if not wrap:
            return self.take(offset, count)
        start = offset % self.tokens
        if start + count <= self.tokens:
            return self.take(start, count)
        head = self.take(start, self.tokens - start)
        tail = self.take(0, count - head.size)
        return np.concatenate([head, tail])
