"""An anchor is an external reference: a paper formula or an independent fp64 evaluation.

The API refuses golden capture structurally and explicitly:

* `reference` is called with a `Fixture` only — no model, no forward, no saved
  output can reach it;
* `Anchor.from_golden` exists solely to raise, so the obvious shortcut is a
  named refusal rather than a tempting constructor;
* the gate evaluates every anchor on two different seeds, so a constant baked
  from one of our own runs fails even if someone hides it in a callable.

The `source` is a declaration of where the number came from; an empty `formula`
is refused because "it matches the implementation" is not an external check.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import torch
from dormouse_core.errors import LoudError

from .errors import ProofError
from .fixture import Fixture

__concepts__ = ("Anchor", "AnchorSource")


class AnchorSource(StrEnum):
    """Where an external reference comes from — never our own forward path."""

    PAPER = "paper formula"
    FP64 = "independent fp64 implementation"
    NUMPY = "numpy reference"
    IDENTITY = "independently derived identity"


class Anchor:
    """A formula over a `Fixture`, with its provenance and the equation it implements."""

    def __init__(
        self,
        reference: Callable[[Fixture], torch.Tensor],
        *,
        source: AnchorSource,
        formula: str,
    ) -> None:
        """Keep the callable and require both a source and a written formula."""
        if not callable(reference):
            raise ProofError("an anchor must be callable", "pass a function of a Fixture")
        if not isinstance(source, AnchorSource):
            raise ProofError(
                f"anchor source {source!r} is not an external reference",
                "use AnchorSource.PAPER, FP64, NUMPY or IDENTITY",
            )
        if not formula.strip():
            raise LoudError(
                "an anchor without a formula is not an external reference",
                "name the paper equation or the identity the callable computes",
            )
        self.reference = reference
        self.source = source
        self.formula = formula

    @classmethod
    def from_golden(cls, output: torch.Tensor, *, formula: str = "") -> Anchor:
        """Refuse a golden captured from our own run — the forbidden anchor."""
        raise LoudError(
            "an output of our own forward cannot be an anchor",
            "write the reference from the paper (fp64, numpy or an identity) over the "
            "fixture; a saved tensor only proves we reproduced our last run",
        )

    def evaluate(self, fixture: Fixture) -> torch.Tensor:
        """Compute the reference on raw inputs; a non-tensor answer is a broken proof."""
        result = self.reference(fixture)
        if not isinstance(result, torch.Tensor):
            raise ProofError(
                f"anchor {self.formula!r} returned {type(result).__name__}, not a tensor",
                "an anchor must return one tensor comparable to the mechanism's output",
            )
        return result

    def __repr__(self) -> str:
        """Print the provenance, not the callable's address."""
        return f"Anchor({self.source.value}: {self.formula})"
