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

GRAD ACCUMULATION. `ids` carries `accum` micro-batches of equal row count, and each
micro-batch's mean loss is divided by `accum` before `backward()`, so the accumulated
gradient is the mean over the whole batch rather than `accum` times it. The equal split
is a refusal, not a rounding: an uneven split would make the last micro-batch a
different-sized estimator under the same 1/accum.

THE NaN FIREWALL IS A DEVICE DECISION. A loss that is not finite, or a gradient whose
global norm is not finite, means the step cannot be taken: the moments and the momentum
would be poisoned by one bad batch and every later step would be wrong. All of that is
decided here on the device — `torch.where` masks the backward seed, `nan_to_num_` and
a multiply-by-zero gate the gradients, and `isfinite` on the norm builds one 0/1 flag.
The host reads that scalar ONCE, through `_host_scalar`, and never reads a loss or a
norm from inside the step. That single seam is what makes a whole step capturable: no
`bool(tensor)`/`float(tensor)` branch survives in the arithmetic. On a healthy step the
mask and the gate are exact identities (`where(True, x, 0) == x`, `x * 1.0 == x`), so
the numpicked parameters are bit-identical to the eager step this replaces.

COUNTERS. `fwd`, `bwd`, `clip`, `opt_muon`, `opt_adamw`, `nan_guard` — and each is
enabled only when the mechanism is about to run, so a firewalled step reports
`nan_guard` instead of claiming an optimizer step that never happened. `check()`
closes the step; an enabled mechanism with a zero counter raises `MissingReport`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from .optim import MixOptimizer

__all__ = [
    "PreStep",
    "StepResult",
    "chunked_cross_entropy",
    "commit_step",
    "host_scalar",
    "train_step",
]

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


def host_scalar(value: torch.Tensor) -> float:
    """Return the ONE declared host read of a step scalar, shared by both paths.

    Everything the step *decides* is device-resident; this is the auditable crossing
    where an already-decided scalar reaches the host. A 0-sync gate patches this one
    seam (and grants it the cadence read) instead of chasing reads across modules.
    """
    return float(value)


@dataclass(frozen=True)
class StepResult:
    """What one step did: the loss it saw, the norm it clipped, and whether it ran.

    `skipped=True` means the NaN firewall fired and **no parameter changed**. `loss`
    is then the non-finite value that tripped it, and `grad_norm` is NaN because the
    norm was never computed — a skipped step has no norm to report. The two `*_device`
    fields are the same numbers before the seam crosses them: a log at a slower cadence
    reads those instead of re-running the step.
    """

    loss: float
    grad_norm: float
    skipped: bool
    loss_device: torch.Tensor | None = None
    norm_device: torch.Tensor | None = None
    #: On a NaN-loss skip, the first forward stage seen non-finite (`logits` / `ce`).
    bad_stage: str = ""


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


def _skip(
    counters: Counters | None,
    loss: float,
    why: str,
    grad_norm: float = float("nan"),
    *,
    bad_loss: bool = False,
    stage: str = "",
) -> StepResult:
    """Log, count and report a step the firewall refused."""
    where = f" first non-finite forward stage: {stage}" if stage else ""
    _LOG.error(
        "NaN firewall: %s (loss=%r)%s. The optimizer step is skipped and no parameter "
        "changed; this step is counted as nan_guard.",
        why,
        loss,
        where,
    )
    if counters is not None:
        counters.count("nan_guard")
        # N11: a NaN loss is a forward receipt, a NaN norm is a backward one. Name
        # them apart so a run can say which path died without a checkpoint to replay.
        counters.count("nan_loss" if bad_loss else "nan_norm")
        counters.check()
    return StepResult(loss=loss, grad_norm=grad_norm, skipped=True, bad_stage=stage)


def commit_step(
    model: nn.Module,
    optimizer: MixOptimizer,
    loss: torch.Tensor,
    *,
    counters: Counters | None = None,
    clip: float = CLIP,
    pre_step: PreStep | None = None,
    read: bool = True,
    forward_stages: tuple[tuple[str, torch.Tensor], ...] = (),
) -> StepResult:
    """Run the device firewall, the clip and the step for eager and replayed steps.

    `loss` is a device scalar that already carries every micro-batch's contribution.
    Nothing here reads it to the host except through `host_scalar`, and the step/no-step
    decision is a device flag crossed once. With `read=False` the reported numbers stay
    device-resident on the result and the caller reads them at log cadence — the step
    itself still crosses the single flag, because a skip must not move a moment.
    """
    if counters is not None:
        counters.enable("clip", counter="clip", why="the gradient norm is bounded")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if counters is not None:
        counters.count("clip")
    finite = torch.isfinite(loss.detach()).all() & torch.isfinite(norm)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    # A NaN that came through an activation leaves a NaN gradient even when the loss
    # seed was masked. Sanitise first, then gate: *1.0 is the identity on a healthy
    # step, *0 makes every direction exactly zero on a skipped one.
    for grad in grads:
        grad.nan_to_num_(0.0, 0.0, 0.0)
    gate = finite.to(loss.dtype)
    for grad in grads:
        grad.mul_(gate)
    ok = host_scalar(finite.to(torch.float32)) > 0.5
    loss_value = host_scalar(loss.detach()) if read else 0.0
    norm_value = host_scalar(norm) if read else 0.0
    if not ok:
        optimizer.zero_grad(set_to_none=True)
        bad_loss = not math.isfinite(loss_value)
        why = "the loss is not finite" if bad_loss else f"the gradient norm is {norm_value}"
        stage = ""
        if bad_loss:
            # Device flags were captured before the backward; read them only on the
            # skip path, so a healthy step pays no extra host sync.
            stage = next((name for name, flag in forward_stages if not bool(flag)), "ce")
        # A NaN loss has no norm to report (NaN); an inf norm is itself the receipt.
        return _skip(
            counters,
            loss_value,
            why,
            float("nan") if bad_loss else norm_value,
            bad_loss=bad_loss,
            stage=stage,
        )
    if pre_step is not None and not pre_step(loss_value):
        # The policy refused the step: the gradients go, the weights and the moments stay.
        optimizer.zero_grad(set_to_none=True)
        return StepResult(loss=loss_value, grad_norm=norm_value, skipped=True)
    sizes = optimizer.sizes
    if counters is not None:
        if sizes["muon"]:
            counters.enable("opt_muon", counter="opt_muon", why="matrices step under Muon")
        if sizes["adamw"]:
            counters.enable("opt_adamw", counter="opt_adamw", why="vectors step under AdamW")
    optimizer.step(counters=counters)
    if counters is not None:
        counters.count("opt_muon", sizes["muon"])
        counters.count("opt_adamw", sizes["adamw"])
        counters.check()
    return StepResult(
        loss=loss_value,
        grad_norm=norm_value,
        skipped=False,
        loss_device=loss.detach() if not read else None,
        norm_device=norm if not read else None,
    )


def train_step(
    model: nn.Module,
    ids: torch.Tensor,
    *,
    optimizer: MixOptimizer,
    accum: int = 1,
    clip: float = CLIP,
    counters: Counters | None = None,
    pre_step: PreStep | None = None,
    read: bool = True,
) -> StepResult:
    """One optimizer step over `ids [accum·batch, seq+1]`, as the docstring orders it.

    `pre_step(loss)` is the loop's last word between the backward pass and
    `optimizer.step()`: it may scale the gradients in place (the loop's per-tensor clip)
    and returns `False` to refuse the step, which clears the gradients and leaves every
    parameter and moment untouched — a skip, not a rollback.

    The forward/backward half accumulates the loss on the device. The firewall, the
    norm and the step decision are `commit_step`'s, so the eager step and a replayed
    `StepGraph` cannot drift apart in what "take a step" means.
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
    optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), dtype=torch.float32, device=ids.device)
    forward_stages: list[tuple[str, torch.Tensor]] = []
    for micro in range(accum):
        x = inputs[micro * rows : (micro + 1) * rows]
        y = targets[micro * rows : (micro + 1) * rows]
        if counters is not None:
            # Counted before the call: the stack checks its own counters inside
            # `forward`, so a counter this module owns must already be non-zero there.
            counters.enable("fwd", counter="fwd", why="a forward runs in the step")
            counters.count("fwd", x.numel())
        logits = model(x, counters)
        forward_stages.append(("logits", torch.isfinite(logits.detach()).all()))
        loss = chunked_cross_entropy(logits, y)
        # The reported sum keeps the raw value (a NaN step reports NaN). The backward
        # seed is masked: on a finite step `where(True, loss, 0)` is bit-identical to
        # `loss`, on a bad one the seed is exactly zero and no gradient is injected.
        loss_total = loss_total + loss.detach() / accum
        finite = torch.isfinite(loss.detach()).all()
        seed = torch.where(finite, loss, torch.zeros_like(loss))
        if counters is not None:
            counters.enable("bwd", counter="bwd", why="the backward runs in the step")
        # torch's stub for `Tensor.backward` is unannotated, and this is the call.
        (seed / accum).backward()  # type: ignore[no-untyped-call]
        if counters is not None:
            counters.count("bwd", x.numel())
    return commit_step(
        model,
        optimizer,
        loss_total,
        counters=counters,
        clip=clip,
        pre_step=pre_step,
        read=read,
        forward_stages=tuple(forward_stages),
    )
