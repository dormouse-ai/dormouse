"""Held-out discipline as a check, not a comment.

A document that appears in both the training and the evaluation split makes every
number on that split a training number. The old project caught this after the
fact, by reading filenames. Here the harness takes both manifests and refuses a
shared document hash before a single example is scored.

Hashing the document text — not its path, not its title — is the point: the same
paragraph under two names is still the same paragraph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dormouse_core.errors import LoudError


def doc_hash(document: str) -> str:
    """Return the SHA-256 of a document's exact text."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Split:
    """A named set of documents, read from the manifest beside the data."""

    name: str
    documents: tuple[str, ...]

    @classmethod
    def read(cls, path: str | Path) -> Split:
        """Read `{"name": ..., "documents": [...]}`; refuse a manifest without them."""
        manifest = Path(path)
        raw = json.loads(manifest.read_text())
        documents = raw.get("documents") if isinstance(raw, dict) else None
        if not isinstance(documents, list) or not documents:
            raise LoudError(
                f"{manifest}: no documents in the split manifest",
                'write {"name": "train", "documents": ["..."]}',
            )
        return cls(
            name=str(raw.get("name", manifest.stem)),
            documents=tuple(str(document) for document in documents),
        )

    def hashes(self) -> frozenset[str]:
        """Return the document hashes this split covers."""
        return frozenset(doc_hash(document) for document in self.documents)


def require_held_out(eval_split: Split, train_split: Split) -> None:
    """Refuse an evaluation split that shares any document hash with training."""
    shared = eval_split.hashes() & train_split.hashes()
    if shared:
        example = ", ".join(sorted(shared)[:2])
        raise LoudError(
            f"{len(shared)} of {len(eval_split.documents)} {eval_split.name} documents "
            f"also appear in {train_split.name} ({example})",
            "remove the documents from one split; an eval on training text measures memory",
        )
