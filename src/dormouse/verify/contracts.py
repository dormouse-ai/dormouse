"""The three demonstrated contracts, with anchors written from the papers.

These are data: four proofs per mechanism, registered when the mechanism module
is imported. `active_names` is the pre-allocation question assemble asks — a
config that names one of these mechanisms must find its contract in the registry
before a single parameter is built.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .anchor import Anchor, AnchorSource
from .anchors import dsa_paper, kda_paper, tsct_paper
from .contract import Contract
from .degenerations import dsa_degeneration, kda_degeneration, tsct_degeneration
from .fixture import Fixture
from .invariant import Invariant, finite, shape
from .registry import register


def _variance() -> Invariant:
    """Check the TSCT init identity `Var(u·v) = r·σ⁴`, the dense draw's variance."""

    def holds(_: torch.Tensor, fixture: Fixture) -> bool:
        product = fixture.tensor("u") @ fixture.tensor("v")
        want = fixture.number("rank") * fixture.number("sigma") ** 4
        return abs(float(product.var(unbiased=False)) - want) < 0.25 * want

    return Invariant("variance=r·σ⁴", holds)


KDA_CONTRACT = Contract(
    name="kda",
    counter="kda_fwd",
    degeneration=kda_degeneration,
    anchor=Anchor(
        kda_paper,
        source=AnchorSource.PAPER,
        formula="Kimi Linear eq.1: S=(I-βkkᵀ)diag(α)S+βkvᵀ, o=Sᵀq",
    ),
    invariants=(finite(), shape(1, 2, 16, 8)),
    epsilon=1e-9,
)
DSA_CONTRACT = Contract(
    name="dsa",
    counter="dsa_fwd",
    degeneration=dsa_degeneration,
    anchor=Anchor(
        dsa_paper,
        source=AnchorSource.PAPER,
        formula="DeepSeek-V3.2 §2.1: softmax(q·k/√d over S(t)∪W(t))·v, y=W_proj·o",
    ),
    invariants=(finite(), shape(1, 16, 16)),
    epsilon=1e-8,
)
TSCT_CONTRACT = Contract(
    name="tsct",
    counter="tsct_fwd",
    degeneration=tsct_degeneration,
    anchor=Anchor(
        tsct_paper,
        source=AnchorSource.FP64,
        formula="TSCT: y=(x·u)·v, fp64 from the factors; Var(u·v)=r·σ⁴",
    ),
    invariants=(finite(), shape(4, 12), _variance()),
    epsilon=1e-9,
)

#: Register at import: importing a mechanism module imports its contract with it.
for _contract in (KDA_CONTRACT, DSA_CONTRACT, TSCT_CONTRACT):
    register(_contract)


def active_names(pattern: Sequence[str], tsct_rank: int) -> tuple[str, ...]:
    """Return the proved mechanisms a config turns on, before anything is allocated."""
    names = []
    if "kda" in pattern:
        names.append("kda")
    if "dsa" in pattern:
        names.append("dsa")
    if tsct_rank > 0:
        names.append("tsct")
    return tuple(names)
