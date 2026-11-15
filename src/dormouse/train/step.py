"""One training step, in one place: forward, chunked CE, backward, clip, step.

The order is the contract, and it is the order because each item exists to make the
next one safe:

    zero_grad → (forward → chunked CE → backward) × accum → global norm → clip
              → pre_step (the loop's policy) → optimizer.step()

CHUNKED CROSS-ENTROPY. `F.cross_entropy` over a `[batch, seq, vocab]` fp32-upcast
copy is `4·B·T·V` bytes in one allocation: 3.2 GiB at the live 24×1024×32768, which
is the number the architecture decision records as 3101 MiB. The same mean loss is
computed here in chunks of `CHUNK_TOKENS` tokens, so the largest temporary is
`4·CHUNK_TOKENS·V` (256 MiB at 2048) and the value is identical up to the order of a
fp32 sum. The `[B, T, V]` logits themselves are the stack's own buffer — chunking the
*head* is a change to `model/` and is not this module's.

fp32 WHERE THE LOSS IS. Each chunk is upcast before `cross_entropy`, and the sum is
fp32: the loss is the number the run is judged by, and a bf16 log-softmax over 32 000
rows is the quality leak `train/precision.py` refuses. The upcast costs one chunk of
fp32 (256 MiB at CHUNK_TOKENS=2048), which this module already priced above.

THE FORWARD section of the policy. The forward runs under `Precision.autocast`, so a
bf16 run's matmuls are bf16 and the context's entry is counted as `autocast_ops`; the
cross-entropy and `optimizer.step()` stay outside it, in fp32, which is the rule
`train/precision.py` writes down.

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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from .optim import MixOptimizer
from .precision import COUNTER, Precision

__all__ = ["PreStep", "StepResult", "chunked_cross_entropy", "train_step"]

_LOG = logging.getLogger(__name__)

#: Tokens per cross-entropy chunk. 2048 is 256 MiB of fp32 logits at vocab 32768 and
#: 8 MiB at vocab 256; the module docstring prices the alternative.
CHUNK_TOKENS = 2048

#: Global gradient-norm ceiling. 1.0 is the recipe's; a step that exceeds it is
#: scaled down, never skipped.
CLIP = 1.0

#: The seam between the backward pass and `optimizer.step()`. `pre_step(loss)` may scale
#: gradients and returns whether the step may be taken at all; the loop owns both
#: policies (per-tensor clipping and the spike skip) because they need state that lives
#: across steps, and a policy applied after the step is a rollback, not a skip.
PreStep = Callable[[float], bool]


@dataclass(frozen=True)
class StepResult:
    """What one step did: the loss it saw, the norm it clipped, and whether it ran.

    `skipped=True` means the NaN firewall fired and **no parameter changed**. `loss`
    is then the non-finite value that tripped it, and `grad_norm` is NaN because the
    norm was never computed — a skipped step has no norm to report.
    """

    loss: float
    grad_norm: float
    skipped: bool


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
            flat[start : start + chunk].float(),
            flat_targets[start : start + chunk],
            reduction="sum",
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


def train_step(
    model: nn.Module,
    ids: torch.Tensor,
    *,
    optimizer: MixOptimizer,
    accum: int = 1,
    clip: float = CLIP,
    counters: Counters | None = None,
    pre_step: PreStep | None = None,
) -> StepResult:
    """One optimizer step over `ids [accum·batch, seq+1]`, as the docstring orders it.

    `pre_step(loss)` is the loop's last word between the backward pass and
    `optimizer.step()`: it may scale the gradients in place (the loop's per-tensor clip)
    and returns `False` to refuse the step, which clears the gradients and leaves every
    parameter and moment untouched — a skip, not a rollback.
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
    inputs, targets = ids[:, :-1], ids[:, 1:]
    rows = ids.shape[0] // accum
    policy = Precision.of(model)
    if counters is not None and policy.mixed:
        # Enabled before the first forward for the same reason `fwd` is: the context counts
        # itself on entry and the stack checks its own counters inside `forward`.
        counters.enable("autocast", counter=COUNTER, why="the forward's matmuls are bf16")
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    for micro in range(accum):
        x = inputs[micro * rows : (micro + 1) * rows]
        y = targets[micro * rows : (micro + 1) * rows]
        if counters is not None:
            # Counted before the call: the stack checks its own counters inside
            # `forward`, so a counter this module owns must already be non-zero there.
            counters.enable("fwd", counter="fwd", why="a forward runs in the step")
            counters.count("fwd", x.numel())
        with policy.autocast(x.device, counters=counters):
            logits = model(x, counters)
        loss = chunked_cross_entropy(logits, y)
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
    if counters is not None:
        counters.enable("clip", counter="clip", why="the gradient norm is bounded")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if counters is not None:
        counters.count("clip")
    if not torch.isfinite(torch.tensor(norm)):
        optimizer.zero_grad(set_to_none=True)
        return _skip(counters, loss_total, f"the gradient norm is {norm}")
    if pre_step is not None and not pre_step(loss_total):
        # The policy refused the step: the gradients go, the weights and the moments stay.
        optimizer.zero_grad(set_to_none=True)
        return StepResult(loss=loss_total, grad_norm=norm, skipped=True)
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
    return StepResult(loss=loss_total, grad_norm=norm, skipped=False)
