"""The decoder stops on the marker's id, never on a decoded substring."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dormouse.data.tokenizer import END_ID
from dormouse.model.spec import StackSpec
from dormouse.train.talker import Talker

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"


class _Fixed(nn.Module):
    """Return logits whose argmax is the next id in a scripted sequence."""

    def __init__(self, ids: Sequence[int]) -> None:
        super().__init__()
        self.ids = list(ids)
        self.blocks: tuple[Any, ...] = ()
        self.calls = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        wanted = self.ids[min(self.calls, len(self.ids) - 1)]
        self.calls += 1
        logits = torch.zeros(1, ids.shape[1], 64)
        logits[0, :, wanted] = 1.0
        return logits


def _talker(model: nn.Module) -> Talker:
    raw = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    keys = {field.name for field in fields(StackSpec) if field.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(_CONFIG))
    return Talker(model, spec, encode=lambda _: [1], decode=lambda ids: "".join(map(str, ids)))


def test_the_literal_spelling_of_the_marker_does_not_stop_generation() -> None:
    """A byte sequence that decodes to `<|end|>` is content, not the control token."""
    model = _Fixed([10, 11, 12])
    talker = _talker(model)
    text = talker("hi", max_tokens=3, stop=("<|end|>",))
    assert text == "101112", "generation stopped on a decoded substring"


def test_the_end_marker_id_stops_generation_before_decoding() -> None:
    """Only id == END_ID stops; the marker id is never part of the answer."""
    model = _Fixed([END_ID, 11])
    talker = _talker(model)
    text = talker("hi", max_tokens=3, stop=("<|end|>",))
    assert text == ""
    assert model.calls == 1, "generation continued past the marker"
