"""The four proofs a mechanism registers before it can be enabled.

Counter, degeneration, anchor, invariants.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from dormouse_core.errors import LoudError

from .anchor import Anchor
from .errors import ProofError
from .fixture import Fixture
from .invariant import Invariant


@dataclass(frozen=True)
class Contract:
    """What makes one technology checkable, declared once by its own module.

    * `counter` — the `Counters` key that must be non-zero while the flag is on;
    * `degeneration` — `(flag_on, flag_off)` on a fixture, the off arm being the
      stack this mechanism replaced, equal within `epsilon`;
    * `anchor` — an external reference computed from the fixture outside our
      forward path, equal to the on arm within `epsilon`;
    * `invariants` — named properties the on arm must satisfy.

    The constructor refuses an incomplete contract: a missing anchor, no
    degeneration, no invariants, a blank counter or an unusable epsilon are
    programming errors, not warnings.
    """

    name: str
    counter: str
    degeneration: Callable[[Fixture], tuple[torch.Tensor, torch.Tensor]]
    anchor: Anchor
    invariants: tuple[Invariant, ...]
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """Refuse a contract that cannot fail — the proof would be theatre."""
        if not self.name.strip():
            raise LoudError("a mechanism needs a name", "register the contract with its own name")
        if not self.counter.strip():
            raise LoudError(
                f"mechanism {self.name!r} has no counter",
                "name the Counters key that moves when this mechanism runs",
            )
        if not callable(self.degeneration):
            raise ProofError(
                f"mechanism {self.name!r} has no degeneration",
                "pass a callable returning (flag_on, flag_off) on a fixture",
            )
        if not isinstance(self.anchor, Anchor):
            raise ProofError(
                f"mechanism {self.name!r} has no external anchor",
                "pass an Anchor built from a paper formula or an independent computation; "
                "an output captured from our own run is refused",
            )
        if not self.invariants:
            raise ProofError(
                f"mechanism {self.name!r} has no invariants",
                "name at least one property its output must satisfy",
            )
        if len({inv.name for inv in self.invariants}) != len(self.invariants):
            raise ProofError(
                f"mechanism {self.name!r} repeats an invariant name",
                "give each property a distinct name",
            )
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise LoudError(
                f"mechanism {self.name!r} has an unusable epsilon {self.epsilon!r}",
                "use a finite non-negative float; 0 means bit-identical",
            )

    def __repr__(self) -> str:
        """Print the four proofs so a registry dump is readable."""
        names = ",".join(inv.name for inv in self.invariants)
        return f"Contract({self.name}: {self.counter} | {self.anchor} | [{names}])"
