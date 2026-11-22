"""What the run reads: the training mixture and the held-out window, as the loop asks.

`train/recipe.py` declares the interface (`BatchStream.batch(step, tokens)` and
`bytes_per_token`); `data/mix.py` schedules the draws and reads the windows. The two do not
quite meet, and this module is the seam rather than a second implementation:

* the mixture hands back `(inputs, targets)` and the loop wants `[rows, seq+1]`, so the
  last column is stitched back on — a view, not a second read of the corpus;
* the byte ratio is not a property `MixtureStream` has, because for a tokenizer it is a
  *measurement*: a byte-level stream carries exactly one byte per token, and a token stream
  carries the number its manifest recorded. An assumed 4.4 B/token is how the previous
  project's BPB became incomparable to its own.

THE EVAL WINDOW IS HELD OUT BY CONSTRUCTION. The training mixture reads from the START of
its sources by arithmetic; `HeldOut` reads from the END of its file and is addressed by
probe index. They overlap only if the run is longer than the corpus minus the window, and
`entry._heldout` refuses that instead of reporting a training number as a held-out one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dormouse_core.errors import LoudError

from ..data.bin import Stream
from ..data.mix import MixtureStream
from .recipe import Batch
from .runconfig import RunRecipe

__all__ = ["HeldOut", "Mixture", "bytes_per_token"]

__concepts__ = ("HeldOut", "Mixture")


class Mixture:
    """The mixture as the loop wants it: a batch, the bytes behind it, and one number."""

    def __init__(
        self,
        mix: MixtureStream,
        bytes_per_token: float,
        *,
        batch: int,
        seq_len: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Hold the mixture, the row shape, and the device the model reads ids on."""
        self.mix = mix
        self.bytes_per_token = bytes_per_token
        self.batch_rows = batch
        self.seq_len = seq_len
        self.device = device

    def batch(self, step: int, tokens: int) -> Batch:
        """Return `step`'s batch, rebuilt as `[rows, seq+1]` without a second read."""
        inputs, targets = self.mix.step(step, batch=self.batch_rows, seq_len=self.seq_len)
        ids = np.concatenate([inputs, targets[:, -1:]], axis=1)
        covered = int(ids.shape[0] * (ids.shape[1] - 1))
        if covered != tokens:
            raise LoudError(
                f"the loop asked for {tokens} tokens and the mixture carries {covered}",
                "tokens_per_step must be batch x seq_len: a step that declares one shape "
                "and reads another prices its own cost wrong",
            )
        return Batch(
            ids=torch.from_numpy(ids).to(self.device),
            tokens=covered,
            payload_bytes=round(covered * self.bytes_per_token),
            stage=self.mix.stage_at(step).name,
        )


class HeldOut:
    """The eval stream: a file's last windows, addressed by probe index."""

    def __init__(
        self,
        path: str | Path,
        bytes_per_token: float,
        *,
        batch: int,
        seq_len: int,
        probes: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Open the stream and place the windows at its tail, on the model's device."""
        self.stream = Stream(path)
        self.bytes_per_token = bytes_per_token
        self.batch_rows = batch
        self.seq_len = seq_len
        self.device = device
        self.per_probe = batch * (seq_len + 1)
        need = probes * self.per_probe
        if need > self.stream.tokens:
            raise LoudError(
                f"{path}: {probes} eval probes need {need} tokens, the stream has "
                f"{self.stream.tokens}",
                "lower eval_batches or eval_every, or point `eval` at a longer held-out "
                "shard: an eval that runs off the end is not a measurement",
            )
        self.start = self.stream.tokens - need

    def batch(self, step: int, tokens: int) -> Batch:
        """Return probe `step`'s window; `tokens` is the count `check_batch` verifies."""
        at = self.start + step * self.per_probe
        window = np.asarray(self.stream.take(at, self.per_probe)).reshape(
            self.batch_rows, self.seq_len + 1
        )
        ids = torch.from_numpy(window.astype(np.int64)).to(self.device)
        covered = int(ids.shape[0] * (ids.shape[1] - 1))
        if covered != tokens:
            raise LoudError(
                f"an eval probe carries {covered} tokens and the loop asked for {tokens}",
                "the eval window is batch x seq_len positions; fix eval_batches or the "
                "config's batch and seq_len",
            )
        return Batch(
            ids=ids,
            tokens=covered,
            payload_bytes=round(covered * self.bytes_per_token),
            stage="eval",
        )


def bytes_per_token(recipe: RunRecipe) -> float:
    """Return how many bytes sit behind each token: exact for bytes, measured for a tokenizer."""
    if recipe.spec.unit == "byte":
        return 1.0
    measured = set()
    for source in recipe.sources:
        path = Path(source.path)
        meta = path.with_suffix(path.suffix + ".json")
        raw = json.loads(meta.read_text(encoding="utf-8"))
        value = raw.get("bytes_per_token")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise LoudError(
                f"{meta}: the manifest does not carry the bytes per token it measured",
                "a token model's BPB divides by the bytes behind its tokens; measure them "
                "on the slice you tokenized (len(text.encode()) / len(ids)) and add "
                "'bytes_per_token' to the manifest — an assumed 4.4 is how the previous "
                "project's BPB became incomparable",
            )
        measured.add(round(float(value), 6))
    if len(measured) > 1:
        raise LoudError(
            f"the sources disagree on bytes per token: {sorted(measured)}",
            "one number is the run's; retokenize the outliers, or give them a run of their "
            "own — a per-source ratio makes one BPB an average of two",
        )
    ratio = measured.pop()
    if recipe.spec.unit == "bpe" and ratio <= 1.01:
        raise LoudError(
            f"unit=bpe with bytes_per_token={ratio}: BPB would be the loss divided by ln 2",
            "a ratio of 1 says the manifest was written for a byte stream: retokenize the "
            "corpus and record the measured len(text.encode())/len(ids), or run unit=byte "
            "with vocab=256. A BPB that is a rescaled loss is not a measurement",
        )
    return ratio
