"""One training step, in one place: forward, chunked CE, backward, clip, step.

The order is the contract, and it is the order because each item exists to make the
next one safe:

    zero_grad → (forward → chunked CE → backward) × accum → global norm → clip
              → optimizer.step()

CHUNKED CROSS-ENTROPY. `F.cross_entropy` over a `[batch, seq, vocab]` fp32-upcast
copy is `4·B·T·V` bytes in one allocation: 3.2 GiB at the live 24×1024×32768, which
is the number the architecture decision records as 3101 MiB. The same mean loss is
computed here in chunks of `CHUNK_TOKENS` tokens, so the largest temporary is
`4·CHUNK_TOKENS·V` (256 MiB at 2048) and the value is identical up to the order of a
fp32 sum. The `[B, T, V]` logits themselves are the stack's own buffer — chunking the
*head* is a change to `model/` and is not this module's.

GRAD ACCUMULATION. `ids` carries `accum` micro-batches of equal row count, and each
micro-batch's mean loss is divided by `accum` before `backward()`, so the accumulated
gradient is the mean over the whole batch rather than `accum` times it. The equal split
is a refusal, not a rounding: an uneven split would make the last micro-batch a
different-sized estimator under the same 1/accum.

THE NaN FIREWALL. A loss that is not finite, or a gradient whose global norm is not
finite, means the step cannot be taken: the moments and the momentum would be poisoned
by one bad batch and every later step would be wrong. So the step is **skipped**, the
weights are untouched (gradients are cleared), the event is counted as `nan_guard`, and
it is logged at ERROR. It is not raised, because one bad batch in a 10 GB stream is a
data event, not a programming defect; the policy for how many skips end a run belongs
to the loop, and returning `skipped=True` is what lets that loop exist.

COUNTERS. `fwd`, `bwd`, `clip`, `opt_muon`, `opt_adamw`, `nan_guard` — and each is
enabled only when the mechanism is about to run, so a firewalled step reports
`nan_guard` instead of claiming an optimizer step that never happened. `check()` closes
the step; an enabled mechanism with a zero counter raises `MissingReport`.

THE ER-JEPA TERM (optional). With `jepa=JepaAux(...)` the batch is a *packed pair*:
`ids = [view A | view B]` (the paper's `x^full = x^u || x^a`) and `pair` is view B, so
the width of `pair` is the split. One forward then produces both embeddings — the
pooled view-A columns and the pooled view-B columns — and the token loss, which with
the term on is read on the view-B columns only (eq. 3: the model has to carry the
source into the target to predict it, which is also what stops the encoder from
collapsing into a constant that the alignment term would reward). That forward is the
baseline forward of a paired step: the extra cost is the *replay*, one more packed
forward over `jepa_replay_rows(batch)` stored pairs, run once per optimizer step after
the micro-batch loop because the store's entries are not micro-batches of this batch.
With the term off, `ids` is a plain window and the step is what it was.

Loss dropout is decided once per step, before any of it runs, so a skipped step pays
nothing — not the replay, not its head, not its backward. Counters: `jepa_seen` (every
step with the term on), `jepa_fwd` (steps that computed it), `jepa_skipped`,
`jepa_replay_rows` (rows replayed this step) and `jepa_memory_size` (the store's size
after this step; a gauge). `jepa_fwd + jepa_skipped == jepa_seen`, and `jepa_seen` is
what the enable/check discipline binds to, because a mechanism that is *defined* to
skip cannot require a non-zero run count. The term's own value is
`StepResult.jepa_loss`, logged beside the counters; it is a float, and counters are
integers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from .jepa import JepaAux, er_jepa_loss
from .optim import MixOptimizer

__all__ = ["StepResult", "chunked_cross_entropy", "train_step"]

_LOG = logging.getLogger(__name__)


class _Stack(Protocol):
    """What a step needs from the stack: an embedding, logits, and parameters.

    A protocol rather than `Transformer`, because `train` must not import `model`: the
    stack is whatever the step is handed, and the two methods below are the whole
    contract. It exists because `nn.Module.__getattr__` is typed to return a
    `Tensor | Module` union, so `model.hidden(...)` on an `nn.Module` does not
    type-check even though it is the call.
    """

    def hidden(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor: ...

    def head(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


#: Tokens per cross-entropy chunk. 2048 is 256 MiB of fp32 logits at vocab 32768 and
#: 8 MiB at vocab 256; the module docstring prices the alternative.
CHUNK_TOKENS = 2048

#: Global gradient-norm ceiling. 1.0 is the recipe's; a step that exceeds it is
#: scaled down, never skipped.
CLIP = 1.0


@dataclass(frozen=True)
class StepResult:
    """What one step did: the loss it saw, the norm it clipped, and whether it ran.

    `skipped=True` means the NaN firewall fired and **no parameter changed**. `loss`
    is then the non-finite value that tripped it, and `grad_norm` is NaN because the
    norm was never computed — a skipped step has no norm to report.

    `jepa_loss` is the alignment term over the micro-batches that computed it plus the
    replay term, and 0.0 when the term is off or this step dropped it; it is logged
    beside the counters so the term's own curve is readable without diffing the total
    against a baseline.
    """

    loss: float
    grad_norm: float
    skipped: bool
    jepa_loss: float = 0.0


def chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, chunk: int = CHUNK_TOKENS
) -> torch.Tensor:
    """Mean token-level cross-entropy, accumulated in chunks of `chunk` tokens."""
    if logits.shape[:-1] != targets.shape:
        raise LoudError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} disagree "
            "on every axis but the vocabulary",
            "targets are the same tokens shifted by one: logits[:, :-1] against ids[:, 1:]",
        )
    if chunk < 1:
        raise LoudError(
            f"chunk={chunk} tokens holds no token",
            "1 is one token per call; 2048 is the recipe's",
        )
    flat, flat_targets = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    total = flat.new_zeros((), dtype=torch.float32)
    for start in range(0, flat.shape[0], chunk):
        total = total + functional.cross_entropy(
            flat[start : start + chunk], flat_targets[start : start + chunk], reduction="sum"
        )
    return total / flat.shape[0]


def _skip(counters: Counters | None, loss: float, why: str) -> StepResult:
    """Log, count and report a step the firewall refused."""
    _LOG.error(
        "NaN firewall: %s (loss=%r). The optimizer step is skipped and no parameter "
        "changed; this step is counted as nan_guard.",
        why,
        loss,
    )
    if counters is not None:
        counters.count("nan_guard")
        counters.check()
    return StepResult(loss=loss, grad_norm=float("nan"), skipped=True)


def _replay(
    model: _Stack,
    jepa: JepaAux,
    *,
    step: int,
    rows: int,
    counters: Counters | None,
) -> float:
    """Return the replayed half of eq. (4): `β·(λ·d(rep) + the stored pairs' token loss)`.

     One packed forward over the replayed pairs, its own backward, once per optimizer
    step: the store's entries are not micro-batches of the current batch, and eq. (4)
    adds the replay term whole. Returns the value for `StepResult.jepa_loss`; the
    backward runs only when that value is finite, so the caller can still trip the
    firewall.
    """
    budget = jepa.replay_rows(rows)
    if budget < 1:
        return 0.0
    view_a, view_b = jepa.memory.retrieve(budget, step=step)
    packed = torch.cat([view_a, view_b], dim=1)
    x, y = packed[:, :-1], packed[:, 1:]
    width = view_a.shape[1]
    if counters is not None:
        counters.enable("fwd", counter="fwd", why="a forward runs in the step")
        counters.count("fwd", x.numel())
    h = model.hidden(x, counters)
    logits = model.head(h, counters)
    loss_token = chunked_cross_entropy(logits[:, width - 1 :], y[:, width - 1 :])
    replay = er_jepa_loss(
        None,
        (h[:, :width], h[:, width:]),
        jepa.head,
        lam=jepa.lam,
        beta=jepa.replay_weight,
        token_replay=loss_token,
    )
    if counters is not None:
        counters.enable("bwd", counter="bwd", why="the backward runs in the step")
        counters.count("bwd", x.numel())
        counters.count("jepa_replay_rows", budget)
    value = float(replay.detach())
    if math.isfinite(value):
        replay.backward()  # type: ignore[no-untyped-call]
    return value


def train_step(
    model: _Stack,
    ids: torch.Tensor,
    *,
    optimizer: MixOptimizer,
    accum: int = 1,
    clip: float = CLIP,
    counters: Counters | None = None,
    jepa: JepaAux | None = None,
    pair: torch.Tensor | None = None,
    step: int = 0,
) -> StepResult:
    """One optimizer step over `ids [accum·batch, seq+1]`, as the docstring orders it.

    `pair` is view B `[accum·batch, W]` from `JepaPair.at(step)`, and `ids` is the packed
    `[view A | view B]` of width `2·W`: the step reads the token loss on the view-B
    columns and pools the two halves for the alignment term. Both or neither. `step` is
    the run's step number: it positions the replay order and nothing else, so it is
    unused when the term is off.
    """
    if ids.dim() != 2:
        raise LoudError(
            f"a step takes ids [batch, seq+1], got shape {tuple(ids.shape)}",
            "the +1 is the target shift: inputs ids[:, :-1], targets ids[:, 1:]",
        )
    if ids.shape[1] < 2:
        raise LoudError(
            f"ids has {ids.shape[1]} positions: no input has a target",
            "a batch needs at least two tokens to predict one from another",
        )
    if accum < 1 or ids.shape[0] % accum:
        raise LoudError(
            f"accum={accum} does not split {ids.shape[0]} rows evenly",
            "pass ids with a row count that is a multiple of accum; an uneven split "
            "weights the last micro-batch differently under the same 1/accum",
        )
    if clip <= 0.0:
        raise LoudError(
            f"clip={clip} scales every gradient to zero",
            "use the ceiling in loss units (1.0 for this recipe) or math.inf for none",
        )
    if (jepa is None) != (pair is None):
        raise LoudError(
            "the JEPA term and its paired batch come together",
            "pass jepa=JepaAux(...) with pair=JepaPair.at(step)[1], or neither",
        )
    inputs, targets = ids[:, :-1], ids[:, 1:]
    width = 0
    if pair is not None:
        width = pair.shape[1]
        if pair.shape[0] != ids.shape[0] or ids.shape[1] != 2 * width:
            raise LoudError(
                f"ids {tuple(ids.shape)} is not [view A | view B] for view B {tuple(pair.shape)}",
                "pack the pair as cat([view_a, view_b], dim=1): both rows the same width",
            )
        if width < 2:
            raise LoudError(
                f"a view of {width} token(s) leaves no view-B position after the shift",
                "give the pair at least two tokens per view",
            )
    rows = ids.shape[0] // accum
    keep = jepa is not None and jepa.keep()
    if jepa is not None and counters is not None:
        counters.enable("jepa", counter="jepa_seen", why="the ER-JEPA term is on for this run")
        counters.count("jepa_seen")
        if not keep:
            counters.count("jepa_skipped")
    if jepa is not None and pair is not None:
        # Stored on skipped steps too: the store is what a later step replays, so its
        # contents must not depend on the dropout coin.
        jepa.memory.insert(ids[:, :width], ids[:, width:])
    if jepa is not None:
        # The head is its own module, so the optimizer's `zero_grad` does not reach it
        # when the caller optimizes the stack alone: its gradient would accumulate
        # across steps until the clip sees an inf. Zeroed here, and clipped with the
        # stack below, so a head the caller does not optimize stays the frozen random
        # predictor it was initialised to instead of an unbounded one. Add
        # `jepa.head` to the optimizer to train it (the paper's `Pred` is trained).
        jepa.head.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    jepa_total = 0.0
    for micro in range(accum):
        x = inputs[micro * rows : (micro + 1) * rows]
        y = targets[micro * rows : (micro + 1) * rows]
        if counters is not None:
            # Counted before the call: the stack checks its own counters inside
            # `forward`, so a counter this module owns must already be non-zero there.
            counters.enable("fwd", counter="fwd", why="a forward runs in the step")
            counters.count("fwd", x.numel())
        h = model.hidden(x, counters)
        logits = model.head(h, counters)
        if jepa is not None and pair is not None:
            # eq. (3): the language loss over the *target* view, conditioned on the
            # source view that shares the window. Read on the same columns whether or
            # not this step computed the term: loss dropout drops the alignment, never
            # the token objective.
            loss = chunked_cross_entropy(logits[:, width - 1 :], y[:, width - 1 :])
        else:
            loss = chunked_cross_entropy(logits, y)
        if keep and jepa is not None and pair is not None:
            term = er_jepa_loss((h[:, :width], h[:, width:]), None, jepa.head, lam=jepa.lam)
            jepa_total += float(term.detach()) / accum
            loss = loss + term
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            return _skip(counters, float(loss.detach()), "the loss is not finite")
        if counters is not None:
            counters.enable("bwd", counter="bwd", why="the backward runs in the step")
        # torch's stub for `Tensor.backward` is unannotated, and this is the call.
        (loss / accum).backward()  # type: ignore[no-untyped-call]
        if counters is not None:
            counters.count("bwd", x.numel())
        loss_total += float(loss.detach()) / accum
    if counters is not None and keep:
        counters.count("jepa_fwd")
    if keep and jepa is not None:
        replayed = _replay(model, jepa, step=step, rows=rows * accum, counters=counters)
        if not math.isfinite(replayed):
            optimizer.zero_grad(set_to_none=True)
            return _skip(counters, replayed, "the replayed term is not finite")
        jepa_total += replayed
    if jepa is not None and counters is not None:
        counters.count("jepa_memory_size", jepa.memory.size)
    if counters is not None:
        counters.enable("clip", counter="clip", why="the gradient norm is bounded")
    bounded = list(model.parameters())
    if jepa is not None:
        bounded += list(jepa.head.parameters())
    norm = float(torch.nn.utils.clip_grad_norm_(bounded, clip))
    if counters is not None:
        counters.count("clip")
    if not torch.isfinite(torch.tensor(norm)):
        optimizer.zero_grad(set_to_none=True)
        return _skip(counters, loss_total, f"the gradient norm is {norm}")
    sizes = optimizer.sizes
    if counters is not None:
        if sizes["muon"]:
            counters.enable("opt_muon", counter="opt_muon", why="matrices step under Muon")
        if sizes["adamw"]:
            counters.enable("opt_adamw", counter="opt_adamw", why="vectors step under AdamW")
    optimizer.step()
    if counters is not None:
        counters.count("opt_muon", sizes["muon"])
        counters.count("opt_adamw", sizes["adamw"])
        counters.check()
    return StepResult(loss=loss_total, grad_norm=norm, skipped=False, jepa_loss=jepa_total)
