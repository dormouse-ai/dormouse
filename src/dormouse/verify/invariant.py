"""Named properties of a mechanism's output, each checked by a runnable test."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .errors import ProofError
from .fixture import Fixture


@dataclass(frozen=True)
class Invariant:
    """One property — finite, shape, order, conservation — with its predicate."""

    name: str
    test: Callable[[torch.Tensor, Fixture], bool]

    def check(self, output: torch.Tensor, fixture: Fixture) -> None:
        """Refuse the run if this property does not hold, naming it."""
        if not bool(self.test(output, fixture)):
            raise ProofError(
                f"invariant {self.name!r} failed",
                "the mechanism's output broke a stated property; fix the mechanism, "
                "do not drop the invariant",
            )


def finite() -> Invariant:
    """Check that every output entry is a finite number."""
    return Invariant("finite", lambda output, _: bool(torch.isfinite(output).all()))


def shape(*dims: int) -> Invariant:
    """Require the output to have the shape the paper's tensor should have."""
    return Invariant("shape", lambda output, _: tuple(output.shape) == dims)


def bounded(cap: float) -> Invariant:
    """Require the output's norm to stay under a stated ceiling (bounded state)."""
    return Invariant(
        f"bounded<{cap:g}>", lambda output, _: bool(torch.linalg.vector_norm(output) < cap)
    )
