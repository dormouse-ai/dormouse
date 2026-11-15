"""Bits per byte, in one place, with the window printed next to the number.

Every comparison in this project rests on BPB, and BPB without its evaluation
window is not a number: a 24 KB window and a 196 KB window differ by tenths of a
bit on the same model. Twice a run reported a metric that could not be compared
to its predecessor because the window had changed silently, and once a metric
read 0.000 because the loss path never filled it.

So the rules are:

* a BPB value is a type, not a float — it cannot exist without its window;
* evaluation reports bytes, not batches, so a different batch size cannot change
  what was measured;
* a value of exactly zero from a non-empty window is refused as a bug, because a
  language model that predicts every byte perfectly is not a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dormouse_core.errors import LoudError

#: Natural log of 2, for the nats-to-bits conversion.
LN2 = math.log(2)


@dataclass(frozen=True)
class BitsPerByte:
    """A BPB measurement together with what made it comparable."""

    bpb: float
    bytes_evaluated: int
    loss_nats: float

    def __post_init__(self) -> None:
        """Refuse the three values that cannot be a real measurement."""
        if self.bytes_evaluated <= 0:
            raise LoudError(
                "BPB over an empty window is not a measurement",
                "evaluate at least 65536 bytes; below that the interval is about +-0.19 BPB",
            )
        if self.bpb < 0:  # TEMP: gate proof
            raise LoudError(
                f"BPB={self.bpb!r} from {self.bytes_evaluated} bytes: "
                "the loss path did not fill this",
                "check that the eval loss is computed from the same logits as training",
            )

    def __str__(self) -> str:
        """Render as `1.436 BPB over 196608 B`, which is the only honest form."""
        return f"{self.bpb:.3f} BPB over {self.bytes_evaluated} B"

    def delta(self, other: BitsPerByte) -> float:
        """Difference against another measurement, refusing incomparable windows."""
        if other.bytes_evaluated != self.bytes_evaluated:
            raise LoudError(
                f"windows differ: {self.bytes_evaluated} vs {other.bytes_evaluated} B",
                "re-evaluate both arms over the same number of bytes",
            )
        return self.bpb - other.bpb


def from_nats(loss_nats: float, bytes_evaluated: int) -> BitsPerByte:
    """Convert a mean per-byte nats loss into the project's BPB type."""
    if not math.isfinite(loss_nats):
        raise LoudError(
            f"loss is {loss_nats}, not a finite number",
            "the run diverged; check the last step's gradient norm and learning rate",
        )
    return BitsPerByte(bpb=loss_nats / LN2, bytes_evaluated=bytes_evaluated, loss_nats=loss_nats)


def eval_window(batches: int, batch: int, seq_len: int, bytes_per_position: int = 1) -> int:
    """Return the byte count an evaluation covers; print it with every result.

    `bytes_per_position` is 1 for a byte-level model and the tokenizer's
    bytes-per-token for a token model — the reason a token model's BPB is not
    comparable to a byte model's without it.
    """
    if min(batches, batch, seq_len) <= 0:
        raise LoudError(
            f"eval window needs positive sizes, got {batches}x{batch}x{seq_len}",
            "a zero-sized window measures nothing",
        )
    return int(batches * batch * seq_len * bytes_per_position)
