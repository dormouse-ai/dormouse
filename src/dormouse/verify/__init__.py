"""Proofs, not prose: a mechanism cannot be enabled without proving it works.

Nine shipped defects were all the same shape — a config flag was on, the run
reported a loss, and the mechanism either never executed or computed a different
function from the one its paper names. Counters catch the first half; this
package closes the second. Every mechanism declares a `Contract` with four
proofs: the counter that must move, a degeneration to the stack it replaced, an
external anchor computed outside our forward path, and named invariants.

`register` is process-wide and happens when a mechanism module is imported;
`require` is the loud refusal an active mechanism hits *before* it allocates.
Nothing here imports `dormouse.model`, so `dorm doctor` can report the registry
on a machine with no torch.
"""

from __future__ import annotations

from .anchor import Anchor, AnchorSource
from .contract import Contract
from .errors import ProofError
from .fixture import Fixture
from .invariant import Invariant, bounded, finite, shape
from .registry import REGISTRY, MechanismRegistry, register, registry_table, require

__all__ = [
    "REGISTRY",
    "Anchor",
    "AnchorSource",
    "Contract",
    "Fixture",
    "Invariant",
    "MechanismRegistry",
    "ProofError",
    "bounded",
    "finite",
    "register",
    "registry_table",
    "require",
    "shape",
]
