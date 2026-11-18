"""A fixture is raw input data, never a model or one of our own outputs.

An anchor may only see a `Fixture`. It has no field for a module, a forward
method, or a saved tensor, so an external reference physically cannot be built
by calling the thing it is meant to check. Seeds make each fixture reproducible
and, more importantly, let the gate ask the same anchor about two *different*
inputs — a golden captured from one of our runs cannot answer the second one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from .errors import ProofError


@dataclass(frozen=True)
class Fixture:
    """Named arrays and scalars: everything a proof needs, nothing it must not see."""

    seed: int
    arrays: Mapping[str, torch.Tensor] = field(default_factory=dict)
    meta: Mapping[str, float | int | str] = field(default_factory=dict)

    def tensor(self, name: str) -> torch.Tensor:
        """Return a named array, refusing a fixture that forgot it."""
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise ProofError(
                f"fixture seed={self.seed} has no array {name!r}",
                "build the fixture with that array before registering the contract",
            ) from exc

    def number(self, name: str) -> float:
        """Return a named scalar (a head width, a rank, a scale)."""
        try:
            return float(self.meta[name])
        except KeyError as exc:
            raise ProofError(
                f"fixture seed={self.seed} has no scalar {name!r}",
                "state the paper's constant in the fixture's meta",
            ) from exc
