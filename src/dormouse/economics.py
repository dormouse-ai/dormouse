"""The trade rule: time saved is paid back in data, and sometimes it is not worth it.

A lever that makes a step faster almost always costs quality. The owner's rule is that
the trade is acceptable only when the time saved exceeds the time the *lost quality*
costs to buy back with more data:

    k = 1.25^(loss_pct / (elasticity·100·ln 1.25)) = exp(loss_pct / (100·elasticity))
    verdict = "take" if k < speedup else "refuse"

`k` is the data multiple needed to recover the loss (the two forms are equal because
`1.25^x = exp(x·ln 1.25)`; the 1.25 base in the original note just made "one more
quarter of data" the unit).

ELASTICITY. Loss falls as data grows: `loss ∝ data^-e`. Our own runs fit
`ln(1.1871/1.5021)/ln(2.5/0.12) = 0.0775` over a 20.8x range, while the three
break-even points the owner decided on (2.2 % at 1.30x, 5.9 % at 2.00x, 7.7 % at 2.50x)
imply `e = 0.0845`. Those three anchors are what the plan's verdicts were decided on,
so 0.0845 is the constant here and 0.0775 is the other reading of the same fit; the two
differ by 9 % on `k`, and a measured elasticity from a journal replaces both.

THE RANK LADDER IS COST-NEUTRAL. The one measured rung, `expert_tsct_rank=192` against
256, buys 1.30x for 3.0 % — and 3.0 % costs 1.43x of data to buy back, so it refuses:
each rung's speedup is roughly cancelled by its own quality loss. Rank is a capacity
knob, not a speed lever, and the plan must not print it as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

#: The loss exponent the verdicts are calibrated to (see the module docstring).
ELASTICITY = 0.0845

#: What each lever value has actually measured: name → (speedup, loss in percent).
#: Only measured pairs live here; a lever without a measurement is refused by name.
MEASURED: dict[str, tuple[float, float]] = {
    "expert_tsct_rank=192": (1.30, 3.0),
}


@dataclass(frozen=True)
class Lever:
    """One proposed trade: a name, the step-time multiple it buys, the quality it costs."""

    name: str
    speedup: float | None = None
    loss_pct: float | None = None

    def verdict(self, elasticity: float = ELASTICITY) -> str:
        """`take` when the speedup pays for the data the loss needs, `refuse` otherwise."""
        if self.speedup is None or self.loss_pct is None:
            return "unmeasured"
        return verdict(self.speedup, self.loss_pct, elasticity)

    def row(self, elasticity: float = ELASTICITY) -> str:
        """One plan row: `lever | speedup | loss | verdict`."""
        speedup = "—" if self.speedup is None else f"{self.speedup:.2f}\u00d7"
        loss = "—" if self.loss_pct is None else f"{self.loss_pct:.1f} %"
        return f"{self.name} | {speedup} | {loss} | {self.verdict(elasticity)}"


def recovery_multiple(loss_pct: float, elasticity: float = ELASTICITY) -> float:
    """Return the data multiple that buys `loss_pct` percent of quality back."""
    if loss_pct < 0:
        raise LoudError(
            f"loss_pct={loss_pct}: a lever that gains quality is not a trade",
            "price only levers that lose quality; the gain case needs no rule",
        )
    if elasticity <= 0:
        raise LoudError(
            f"elasticity={elasticity}: loss cannot fall as data grows",
            "pass the measured exponent (> 0), or the constant",
        )
    return math.exp(loss_pct / (100.0 * elasticity))


def break_even_loss(speedup: float, elasticity: float = ELASTICITY) -> float:
    """Return the largest quality loss `speedup` can take and still pay for itself."""
    if speedup <= 0:
        raise LoudError(
            f"speedup={speedup}: a lever that is not a multiple of the old step time",
            "pass the step-time ratio, e.g. 1.30 for a 30 % faster step",
        )
    return 100.0 * elasticity * math.log(speedup)


def verdict(speedup: float, loss_pct: float, elasticity: float = ELASTICITY) -> str:
    """`take` when the saved time exceeds the data the loss costs, `refuse` otherwise."""
    return "take" if recovery_multiple(loss_pct, elasticity) <= speedup else "refuse"


def measured_elasticity(journal: Journal) -> float | None:
    """Fit `-Δln(loss)/Δln(tokens)` over the journal's eval events, or None.

    Two `eval` events carrying `loss` and `tokens` are enough for an exponent; one is
    not, and a journal whose loss did not fall is not an elasticity. None is returned
    rather than a guess, so the caller prints which value it used.
    """
    points = [
        (float(record["tokens"]), float(record["loss"]))
        for record in journal.records
        if record.get("kind") == "eval" and "tokens" in record and "loss" in record
    ]
    if len(points) < 2:
        return None
    (tokens_lo, loss_hi), (tokens_hi, loss_lo) = points[0], points[-1]
    if tokens_hi <= tokens_lo or loss_lo <= 0 or loss_hi <= loss_lo:
        return None
    return math.log(loss_hi / loss_lo) / math.log(tokens_hi / tokens_lo)
