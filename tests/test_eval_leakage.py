"""The leakage gate: a shared document hash stops the eval before it starts.

The manifests are the input, hashing happens here, and the refusal names how many
documents overlapped — a check, not a comment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

from dormouse.eval.leakage import Split, doc_hash, require_held_out


def _manifest(path: Path, name: str, documents: list[str]) -> Path:
    path.write_text(json.dumps({"name": name, "documents": documents}))
    return path


def test_a_shared_document_hash_is_refused() -> None:
    evaluation = Split("eval", ("the leaky document", "a clean one"))
    training = Split("train", ("the leaky document", "another"))
    with pytest.raises(LoudError) as error:
        require_held_out(evaluation, training)
    assert "1 of 2 eval documents also appear in train" in str(error.value)


def test_disjoint_splits_pass() -> None:
    require_held_out(Split("eval", ("a", "b")), Split("train", ("c", "d")))


def test_the_manifests_are_the_input(tmp_path: Path) -> None:
    training = Split.read(_manifest(tmp_path / "train.json", "train", ["shared text", "x"]))
    evaluation = Split.read(_manifest(tmp_path / "eval.json", "eval", ["shared text"]))
    with pytest.raises(LoudError):
        require_held_out(evaluation, training)


def test_the_same_text_under_two_names_is_the_same_hash() -> None:
    assert doc_hash("same words") == doc_hash("same words")
    assert doc_hash("same words") != doc_hash("same word ")


def test_a_manifest_without_documents_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"name": "eval"}))
    with pytest.raises(LoudError) as error:
        Split.read(path)
    assert "no documents" in str(error.value)
