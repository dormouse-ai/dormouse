"""`init_from` imports weights only; a missing or malformed checkpoint is loud."""

from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError
from torch import nn

from dormouse.train.checkpoint import load_weights


def test_load_weights_imports_only_the_model(tmp_path: Path) -> None:
    """Optimizer and step belong to the source run; only weights cross over."""
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(5.0)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"step": 123, "model": source.state_dict(), "optim": {"ignored": True}}, ckpt)

    target = nn.Linear(3, 2)
    info = load_weights(target, ckpt)

    assert info == {"path": str(ckpt), "step": 123, "weights": "model"}
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_load_weights_refuses_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(LoudError, match="init_from"):
        load_weights(nn.Linear(2, 2), tmp_path / "nope.pt")


def test_load_weights_refuses_a_checkpoint_without_model(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"step": 1, "optim": {}}, ckpt)
    with pytest.raises(LoudError, match="model weights"):
        load_weights(nn.Linear(2, 2), ckpt)
