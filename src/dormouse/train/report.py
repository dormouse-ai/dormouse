"""What a run reports: BPB from the bytes it consumed, the eval event, the audit.

Three measurements, one rule: the number is computed from bytes the batch declared and
`recipe.check_batch` verified, never from a token count times an assumed ratio.

`step_bpb` is one step's loss converted per byte. `evaluate` runs the training weights
(under `optimizer.eval_weights()`, so an `sf` run evaluates `X_t` and not `Y_t`) over a
fixed window of held-out batches and returns the project's `BitsPerByte`, which cannot
exist without the window it was measured over.

`audit` is the run's own consistency check: the journal's step lines must add up to the
tokens and bytes the run reports, and the byte/token ratio must be the manifest's. A run
whose journal cannot explain its byte count is a run whose BPB cannot be compared to
anything, including itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import torch
from dormouse_core.errors import LoudError
from dormouse_core.journal import KINDS, Journal
from torch import nn

from ..metrics import BitsPerByte, from_nats
from .optim import MixOptimizer
from .precision import Precision
from .recipe import BatchStream, check_batch
from .step import chunked_cross_entropy

__all__ = [
    "audit",
    "digest",
    "eval_event",
    "evaluate",
    "finite",
    "note",
    "step_bpb",
    "total_bpb",
]


def note(journal: Journal, kind: str, **fields: Any) -> None:
    """Append one journal event; `checkpoint` is ours because core's `KINDS` is closed.

    `dorm status` reads lines, not the kind list, so an event it does not know is a row
    it ignores — which is what a checkpoint note should be. Bypassing the validator for
    one kind is the smaller lie than recording a checkpoint as a swap.
    """
    if kind in KINDS:
        journal.write(kind, **fields)
        return
    record: dict[str, Any] = {"kind": kind, "t": round(time.time(), 3), **fields}
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    journal.records.append(record)


def digest(ids: torch.Tensor) -> str:
    """Return a short hash of a batch, so two runs can be compared batch by batch."""
    return hashlib.blake2b(ids.detach().cpu().numpy().tobytes(), digest_size=8).hexdigest()


def finite(value: float) -> float | None:
    """Return a journalable number or None: JSON has no NaN, and a status row is not for one."""
    return float(value) if math.isfinite(value) else None


def step_bpb(loss: float, tokens: int, payload_bytes: int) -> float | None:
    """Return one step's BPB from the bytes consumed, or None if the loss is not finite."""
    if not math.isfinite(loss) or loss <= 0.0 or payload_bytes <= 0:
        return None
    return from_nats(loss * tokens / payload_bytes, payload_bytes).bpb


def total_bpb(nats: float, payload_bytes: int) -> float | None:
    """Return the run's BPB over every byte it consumed, or None before the first step."""
    if payload_bytes <= 0 or nats <= 0.0 or not math.isfinite(nats):
        return None
    return from_nats(nats / payload_bytes, payload_bytes).bpb


def evaluate(
    model: nn.Module,
    optim: MixOptimizer,
    stream: BatchStream,
    *,
    batches: int,
    tokens_per_step: int,
    first: int,
) -> BitsPerByte:
    """Return the BPB over `batches` held-out batches starting at probe `first`.

    The weights are the ones the arms are training, not the EMA: evaluating the EMA
    mid-run needs a second full copy of the model, and the run's own weights are the
    comparable number. `Trainer.final_model()` is the EMA, and a final eval runs on that.
    """
    nats = 0.0
    tokens = payload = 0
    policy = Precision.of(model)
    with optim.eval_weights(), torch.no_grad():
        for index in range(first, first + batches):
            batch = stream.batch(index, tokens_per_step)
            check_batch(batch, stage=None, stream=stream, step=index)
            with policy.autocast(batch.ids.device):
                logits = model(batch.ids[:, :-1], None)
            loss = chunked_cross_entropy(logits, batch.ids[:, 1:])
            nats += float(loss) * batch.tokens
            tokens += batch.tokens
            payload += batch.payload_bytes
    if payload <= 0:
        raise LoudError(
            f"the eval window covers {payload} bytes",
            "an eval over no bytes is not a measurement; raise eval_batches",
        )
    return from_nats(nats / payload, payload)


def eval_event(
    journal: Journal,
    step: int,
    model: nn.Module,
    optim: MixOptimizer,
    stream: BatchStream | None,
    *,
    batches: int,
    tokens_per_step: int,
    first: int,
) -> BitsPerByte:
    """Measure the held-out window and write its `eval` line; refuse a missing stream."""
    if stream is None:
        raise LoudError(
            "an eval without an eval_stream",
            "pass eval_stream to the Trainer, or drop eval_every: an eval on the training "
            "stream measures memorisation",
        )
    bpb = evaluate(
        model,
        optim,
        stream,
        batches=batches,
        tokens_per_step=tokens_per_step,
        first=first,
    )
    journal.write("eval", step=step, bpb=bpb.bpb, bytes=bpb.bytes_evaluated, weights="training")
    return bpb


def audit(
    journal: Journal,
    *,
    steps: int,
    tokens: int,
    payload: int,
    bytes_per_token: float,
) -> dict[str, int]:
    """Refuse a run whose journal cannot explain the tokens and bytes it reports."""
    lines = [record for record in journal.records if record.get("kind") == "step"]
    recorded = sum(int(record.get("tokens", 0)) for record in lines)
    covered = sum(int(record.get("bytes", 0)) for record in lines)
    if (len(lines), recorded, covered) != (steps, tokens, payload):
        raise LoudError(
            f"the journal explains {len(lines)} steps / {recorded} tokens / {covered} B "
            f"and the run reports {steps} / {tokens} / {payload}",
            "the journal is the run's artifact: restore the missing lines or start a new "
            "run, because a BPB whose bytes cannot be reconstructed is not a measurement",
        )
    ratio = payload / tokens if tokens else bytes_per_token
    if abs(ratio - bytes_per_token) > 1.0:
        raise LoudError(
            f"the journal's bytes/token is {ratio:.3f} and the manifest says {bytes_per_token}",
            "one of the two changed mid-run; a run that cannot say which is a run "
            "without a comparable number",
        )
    return {"steps": len(lines), "tokens": recorded, "bytes": covered}
