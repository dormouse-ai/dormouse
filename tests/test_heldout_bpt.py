"""Held-out BPB divides by the eval shard's own bytes/token, not the training ratio."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.train import entry
from dormouse.train.streams import bytes_per_token, heldout_bytes_per_token

SHA = "50235922e0bacb3a"


def _shard(tmp_path: Path, name: str, bpt: float | None) -> Path:
    path = tmp_path / f"{name}.bin"
    tokens = 20_000
    path.write_bytes(np.zeros(tokens, dtype=np.uint16).tobytes())
    raw = {"dtype": "uint16", "tokens": tokens, "tokenizer_sha": SHA}
    if bpt is not None:
        raw["bytes_per_token"] = bpt
    path.with_suffix(".bin.json").write_text(json.dumps(raw), encoding="utf-8")
    return path


def _recipe(train: Path, held: Path) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(unit="bpe", batch=2, seq_len=8),
        sources=(SimpleNamespace(path=str(train)),),
        eval_path=held,
        probes=lambda steps: (1, 0),
    )


def test_heldout_window_is_priced_by_its_own_manifest(tmp_path: Path) -> None:
    recipe = _recipe(_shard(tmp_path, "train", 4.097287), _shard(tmp_path, "held", 4.206775))
    assert bytes_per_token(recipe) == 4.097287
    assert heldout_bytes_per_token(recipe) == 4.206775
    window = entry._heldout(recipe, steps=1, device=torch.device("cpu"))
    assert window.bytes_per_token == 4.206775
    assert window.batch(0, 16).payload_bytes == 67  # train BPT would give 66


def test_byte_unit_stays_exact(tmp_path: Path) -> None:
    recipe = _recipe(_shard(tmp_path, "train", 1.0), _shard(tmp_path, "held", 1.0))
    recipe.spec.unit = "byte"
    assert heldout_bytes_per_token(recipe) == 1.0


def test_heldout_manifest_without_a_ratio_is_refused(tmp_path: Path) -> None:
    recipe = _recipe(_shard(tmp_path, "train", 4.097287), _shard(tmp_path, "held", None))
    with pytest.raises(LoudError, match="bytes per token it measured"):
        heldout_bytes_per_token(recipe)


def test_no_eval_stream_is_refused(tmp_path: Path) -> None:
    recipe = _recipe(_shard(tmp_path, "train", 4.097287), _shard(tmp_path, "held", 4.206775))
    recipe.eval_path = None
    with pytest.raises(LoudError, match="no eval stream"):
        heldout_bytes_per_token(recipe)
