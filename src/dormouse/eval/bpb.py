"""Bits per byte that cannot lie: the window is measured, not assumed.

BPB is the number every change in this project is argued over, and it has lied
twice. Once a token model scaled by a constant bytes-per-token, so the same model
scored differently on a corpus with more accents. Once a loss path never filled the
field and `0.000` was reported as a result.

The rules that follow:

* a token model's BPB is divided by the bytes of *the evaluation text it saw*,
  measured as UTF-8, not by a tokenizer's average;
* when that text is not available the answer is refused — no number beats a wrong
  number;
* `entropy_bits_per_byte` is the anchor the conversion is checked against by hand:
  256 equiprobable bytes are exactly 8 bits, to 1e-6.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from dormouse_core.errors import LoudError

from dormouse.metrics import LN2, BitsPerByte


def entropy_bits_per_byte(probabilities: Sequence[float]) -> float:
    """Return the entropy of a byte distribution, in bits per byte.

    The anchor: `[1/256] * 256` is exactly 8.0, and a distribution with 128
    equiprobable bytes followed by 128 impossible ones is exactly 7.0. A
    distribution that does not sum to 1 is refused rather than silently normalised.
    """
    if not probabilities:
        raise LoudError(
            "an empty byte distribution has no entropy",
            "pass at least one probability",
        )
    if any(p < 0.0 or p > 1.0 for p in probabilities):
        raise LoudError(
            "a byte probability lies outside [0, 1]",
            "the distribution is not a distribution; fix the softmax",
        )
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-9:
        raise LoudError(
            f"byte probabilities sum to {total!r}, not 1",
            "a distribution that does not sum to 1 has no entropy to report",
        )
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def _utf8(eval_text: str | None) -> bytes:
    """Return the evaluation text's bytes, refusing the two ways they go missing."""
    if eval_text is None:
        raise LoudError(
            "the evaluation text is unknown, so the byte count is unknown",
            "keep the eval text next to the checkpoint and pass it here",
        )
    data = eval_text.encode("utf-8")
    if not data:
        raise LoudError(
            "the evaluation text is empty, so the byte count is zero",
            "evaluate on the fixture you scored; an empty window is not a measurement",
        )
    return data


def bytes_per_token(eval_text: str | None, token_count: int) -> float:
    """Return the measured UTF-8 bytes per token of `eval_text`.

    Measured, because a tokenizer's average over a general corpus is a constant
    that is simply wrong for a corpus of code or of non-ASCII text — and that
    constant is what the previous project's comparison silently used.
    """
    if token_count <= 0:
        raise LoudError(
            f"token_count={token_count} cannot carry any bytes",
            "evaluate at least one token",
        )
    return len(_utf8(eval_text)) / token_count


def from_token_nats(
    mean_nats_per_token: float,
    *,
    eval_text: str | None,
    token_count: int,
) -> BitsPerByte:
    """Convert a token model's mean nats per token into BPB over the bytes it saw.

    `mean_nats_per_token * token_count` is the total loss in nats over a window
    holding `token_count` tokens; dividing by the UTF-8 bytes of the same text is
    the only conversion that survives a tokenizer change.
    """
    if not math.isfinite(mean_nats_per_token):
        raise LoudError(
            f"loss is {mean_nats_per_token}, not a finite number",
            "the run diverged; check the last step's gradient norm",
        )
    data = _utf8(eval_text)
    if token_count <= 0:
        raise LoudError(
            f"token_count={token_count} cannot carry any bytes",
            "evaluate at least one token",
        )
    loss_nats = mean_nats_per_token * token_count
    return BitsPerByte(
        bpb=loss_nats / LN2 / len(data),
        bytes_evaluated=len(data),
        loss_nats=loss_nats,
    )
