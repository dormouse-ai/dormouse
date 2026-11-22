"""The train path's device routing: selection, stream placement, heartbeat, BPT refusal.

A 401M stack on CPU costs 600 s/step; the same stack on a card costs seconds. These
gates check the four seams that decide which one a `dorm train` run gets, plus the
named heartbeat that keeps an expensive step from being silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal
from stacks import reference_stack

from dormouse.model.spec import StackSpec
from dormouse.train import entry, streams
from dormouse.train.loop import Trainer
from dormouse.train.optim import MixOptimizer
from dormouse.train.recipe import Batch, LoopConfig, Stage


def test_auto_picks_cuda_when_present_and_cpu_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` is the default and it falls back to CPU on a card-less host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert entry._resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert entry._resolve_device("auto") == torch.device("cuda")
    assert entry._resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_without_a_card_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request for the card cannot quietly become a CPU run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(LoudError, match="cuda"):
        entry._resolve_device("cuda")


def test_stack_spec_carries_the_device_and_defaults_to_auto() -> None:
    """A config can name the device; the default keeps old configs on the old path."""
    assert StackSpec(unit="byte").device == "auto"
    assert StackSpec(unit="byte", device="cpu").device == "cpu"
    with pytest.raises(LoudError, match="device"):
        StackSpec(unit="byte", device="tpu")


class _Mix:
    """The part of `MixtureStream` the batch builder reads."""

    def step(self, step: int, *, batch: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Two rows of ids, inputs and targets split on the last column."""
        ids = np.arange(batch * (seq_len + 1), dtype=np.int64).reshape(batch, seq_len + 1)
        return ids[:, :-1], ids[:, 1:]

    def stage_at(self, step: int) -> SimpleNamespace:
        """One named stage."""
        return SimpleNamespace(name="bulk")


def test_mixture_puts_ids_on_the_requested_device() -> None:
    """The model and its batches must agree; `meta` proves the device is honoured."""
    mixture = streams.Mixture(_Mix(), 1.0, batch=2, seq_len=4, device="meta")
    batch = mixture.batch(0, 8)
    assert batch.ids.device.type == "meta"
    assert tuple(batch.ids.shape) == (2, 5)


class _Stream:
    """The part of `Stream` the eval window reads."""

    tokens = 40

    def __init__(self, path: str | Path) -> None:
        """Ignore the path; the tail is arithmetic over `tokens`."""

    def take(self, at: int, n: int) -> np.ndarray:
        """Return `n` ids from the tail."""
        return np.arange(at, at + n, dtype=np.uint8)


def test_heldout_puts_ids_on_the_requested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval stream crosses to the same device as the training stream."""
    monkeypatch.setattr(streams, "Stream", _Stream)
    window = streams.HeldOut("held.bin", 2.0, batch=2, seq_len=4, probes=2, device="meta")
    batch = window.batch(0, 8)
    assert batch.ids.device.type == "meta"
    assert tuple(batch.ids.shape) == (2, 5)


def _recipe(tmp_path: Path, unit: str, ratio: float) -> SimpleNamespace:
    """Only the two fields `bytes_per_token` reads, with a real manifest beside the data."""
    (tmp_path / "train.bin.json").write_text(
        json.dumps({"dtype": "uint8", "tokens": 1000, "bytes_per_token": ratio}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        spec=SimpleNamespace(unit=unit),
        sources=(SimpleNamespace(path=tmp_path / "train.bin"),),
    )


def test_bpe_with_a_byte_ratio_refuses(tmp_path: Path) -> None:
    """BPE + BPT ~= 1 makes BPB a rescaled loss, so it refuses by name."""
    with pytest.raises(LoudError, match="ln 2"):
        streams.bytes_per_token(_recipe(tmp_path, "bpe", 1.0))
    assert streams.bytes_per_token(_recipe(tmp_path, "bpe", 4.4)) == 4.4
    assert streams.bytes_per_token(_recipe(tmp_path, "byte", 1.0)) == 1.0


class _Toy:
    """A pure-function stream for the heartbeat gate."""

    bytes_per_token = 4.0

    def batch(self, step: int, tokens: int) -> Batch:
        """Two rows, eight sequence positions, one stage."""
        ids = torch.randint(0, 16, (2, 9))
        return Batch(ids=ids, tokens=16, payload_bytes=64, stage="bulk")


def test_a_step_reports_a_named_heartbeat_before_the_forward(tmp_path: Path) -> None:
    """Every step announces itself on stdout and leaves a `step_heartbeat` counter."""
    cfg = LoopConfig(
        steps=2,
        stages=(Stage("bulk", 0, 2),),
        warmup_steps=1,
        tokens_per_step=16,
        spike_window=5,
    )
    torch.manual_seed(0)
    model = reference_stack(
        vocab=16, d_model=16, n_layers=1, n_heads=2, head_dim=8, d_ffn=32, max_seq_len=8
    )
    seen: list[str] = []
    trainer = Trainer(
        model,
        _Toy(),
        MixOptimizer(model),
        lambda _step: 0.1,
        cfg,
        Journal.open(tmp_path / "run.jsonl"),
        heartbeat=seen.append,
    )
    trainer.train(steps=2)
    assert seen == ["step 0/2: computing (bulk)", "step 1/2: computing (bulk)"]
    assert trainer.counters.counts["step_heartbeat"] == 1
