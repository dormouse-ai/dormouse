"""The config-load gate: one tokenizer for the token table and every data manifest.

A 32 000-row table reading ids from a 32 768-id corpus wasted half of two A/B runs
before anyone noticed. The check belongs where a config is read, before a parameter is
allocated, and it must print both numbers: a refusal that only says "mismatch" is the
same hour of debugging with fewer clues.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dormouse_core.errors import LoudError

from dormouse.data.tokenizer import DEFAULT_VOCAB, sha_file
from dormouse.train.runconfig import RunRecipe

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TOKENIZER = ROOT / "tokenizer32k.json"
REAL = ROOT.parents[1] / "dormouse" / "data_local" / "pre"


def _source(directory: Path, tokenizer_sha: str) -> Path:
    """Write a tiny uint16 stream and its manifest, naming `tokenizer_sha`."""
    ids = np.arange(16, dtype=np.uint16)
    path = directory / "ids.bin"
    path.write_bytes(ids.tobytes())
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"dtype": "uint16", "tokens": int(ids.size), "tokenizer_sha": tokenizer_sha})
        + "\n",
        encoding="utf-8",
    )
    return path


def _config(
    directory: Path, source: Path, *, tokenizer: bool = True, vocab: int | None = None
) -> Path:
    """Copy the shipped config, adding a source and optional deliberate errors."""
    lines: list[str] = []
    for line in (CONFIGS / "dormouse_1b.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("tokenizer =") and not tokenizer:
            continue
        lines.append(line)
        if vocab is not None and line.startswith("unit = "):
            lines.append(f"vocab = {vocab}")
    lines += ["", "[[source]]", f'path = "{source}"', "weight = 1.0", ""]
    path = directory / "run.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_the_shipped_config_and_its_data_pass_the_gate(tmp_path: Path) -> None:
    source = _source(tmp_path, sha_file(TOKENIZER))
    recipe = RunRecipe.read(_config(tmp_path, source))
    assert recipe.spec.vocab == DEFAULT_VOCAB == 32_768
    assert recipe.spec.tokenizer == "tokenizer32k.json"


def test_a_vocab_that_disagrees_with_the_tokenizer_is_refused(tmp_path: Path) -> None:
    """Today's pre-fix state: a 32 000-row table cut from 32 768-id ids."""
    source = _source(tmp_path, sha_file(TOKENIZER))
    with pytest.raises(LoudError) as caught:
        RunRecipe.read(_config(tmp_path, source, vocab=32_000))
    message = str(caught.value)
    assert "32000" in message and "32768" in message


def test_a_bpe_run_with_no_named_tokenizer_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path, sha_file(TOKENIZER))
    with pytest.raises(LoudError) as caught:
        RunRecipe.read(_config(tmp_path, source, tokenizer=False))
    assert "names no tokenizer" in str(caught.value)


def test_a_manifest_from_another_tokenizer_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path, "deadbeefdeadbeef")
    with pytest.raises(LoudError) as caught:
        RunRecipe.read(_config(tmp_path, source))
    message = str(caught.value)
    assert "deadbeefdeadbeef" in message and sha_file(TOKENIZER) in message


@pytest.mark.skipif(not (REAL / "mix.bin.json").exists(), reason="corpus not checked out")
def test_the_real_corpus_manifest_and_ids_match_the_shipped_tokenizer() -> None:
    expected = sha_file(TOKENIZER)
    for name in ("mix.bin", "mix_smoke.bin"):
        meta = json.loads((REAL / f"{name}.json").read_text(encoding="utf-8"))
        assert meta["tokenizer_sha"] == expected
        ids = np.memmap(REAL / name, dtype=np.uint16, mode="r")
        largest = 0
        for start in range(0, ids.size, 50_000_000):
            largest = max(largest, int(np.asarray(ids[start : start + 50_000_000]).max()))
        assert largest < DEFAULT_VOCAB, largest
