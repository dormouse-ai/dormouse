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
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from .aux import LightMTP
from .jepa import JepaAux, er_jepa_loss
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
    #: The auxiliary terms, reported separately from `loss`: two runs must not disagree
    #: for a reason hidden inside the token loss.
    aux_loss: float | None = None
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


def _skip(
    counters: Counters | None, loss: float, why: str, grad_norm: float = float("nan")
) -> StepResult:
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
    return StepResult(loss=loss, grad_norm=grad_norm, skipped=True)


class _Encoder(Protocol):
    """The encoder half the paired JEPA term reads; the stack provides it."""

    def encode(
        self, ids: torch.Tensor, counters: Counters | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def hidden(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor: ...

    def head(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor: ...


def _replay(
    model: nn.Module,
    jepa: JepaAux,
    step: int,
    rows: int,
    counters: Counters | None,
) -> float:
    """Return the replayed half of ER-JEPA's eq. (4), from one packed forward.

    The store holds whole pairs, not micro-batches of the current batch, so the term is
    added whole after the micro-batch loop. Only its value is returned; the backward
    runs here, and a non-finite value leaves the caller to skip the step.
    """
    budget = jepa.replay_rows(rows)
    if budget < 1:
        return 0.0
    view_a, view_b = jepa.memory.retrieve(budget, step=step)
    x = view_b[:, :-1]
    if counters is not None:
        counters.enable("fwd", counter="fwd", why="a forward runs in the step")
        counters.count("fwd", x.numel())
        counters.enable("jepa_replay_rows", counter="jepa_replay_rows", why="rows were replayed")
        counters.count("jepa_replay_rows", budget)
    encoder = cast(_Encoder, model)
    # The two views are encoded in separate passes: the live stack's KDA requires a
    # sequence that tiles its 64-token chunk, and a packed view is an odd length.
    h_a, _logits_a = encoder.encode(view_a[:, :-1], counters)
    h_b, logits_b = encoder.encode(view_b[:, :-1], counters)
    token = chunked_cross_entropy(logits_b, view_b[:, 1:])
    replay = er_jepa_loss(
        None,
        (h_a, h_b),
        jepa.head,
        lam=jepa.lam,
        beta=jepa.replay_weight,
        token_replay=token,
    )
    if counters is not None:
        counters.enable("bwd", counter="bwd", why="the backward runs in the step")
        counters.count("bwd", x.numel())
    value = float(replay.detach())
    if math.isfinite(value):
        replay.backward()  # type: ignore[no-untyped-call]
    return value


def commit_step(
    model: nn.Module,
    optimizer: MixOptimizer,
    loss: torch.Tensor,
    *,
    counters: Counters | None = None,
    clip: float = CLIP,
    pre_step: PreStep | None = None,
    read: bool = True,
    bounded: list[nn.Parameter] | None = None,
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
    norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) if bounded is None else bounded, clip
    )
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
        # A NaN loss has no norm to report (NaN); an inf norm is itself the receipt.
        return _skip(counters, loss_value, why, float("nan") if bad_loss else norm_value)
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
    optimizer.step()
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
    aux: LightMTP | None = None,
    jepa: JepaAux | None = None,
    pair: torch.Tensor | None = None,
    step: int = 0,
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
    if aux is not None and getattr(model, "aux", None) is not aux:
        raise LoudError(
            "the auxiliary head is not attached to this model, so no optimizer owns it",
            "call aux.attach(model) before MixOptimizer(model): an unattached head is "
            "routed to no parameter group and never trains",
        )
    if (jepa is None) != (pair is None):
        raise LoudError(
            "the JEPA term and its paired batch come together",
            "pass jepa=JepaAux(...) with pair=view_b, or neither",
        )
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
    keep = jepa is not None and jepa.keep()
    if jepa is not None and counters is not None:
        counters.enable("jepa", counter="jepa_seen", why="the ER-JEPA term is on for this run")
        counters.count("jepa_seen")
        if keep:
            counters.enable("jepa_fwd", counter="jepa_fwd", why="the term computed this step")
            counters.count("jepa_fwd")
        else:
            counters.enable("jepa_skipped", counter="jepa_skipped", why="loss dropout fired")
            counters.count("jepa_skipped")
    if jepa is not None and pair is not None:
        # Stored on skipped steps too: the store a later step replays must not depend
        # on this step's dropout coin.
        jepa.memory.insert(ids[:, :width], ids[:, width:])
    if jepa is not None:
        jepa.head.zero_grad(set_to_none=True)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    rows = ids.shape[0] // accum
    optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), dtype=torch.float32, device=ids.device)
    aux_total = 0.0
    jepa_total = 0.0
    for micro in range(accum):
        x = inputs[micro * rows : (micro + 1) * rows]
        y = targets[micro * rows : (micro + 1) * rows]
        if counters is not None:
            # Counted before the call: the stack checks its own counters inside
            # `forward`, so a counter this module owns must already be non-zero there.
            counters.enable("fwd", counter="fwd", why="a forward runs in the step")
            counters.count("fwd", x.numel())
        seen: dict[int, torch.Tensor] = {}
        h: torch.Tensor | None = None
        # One context for both mechanisms: the hooks exist only while the aux head is
        # on, and they are removed in `finally` whatever the encoder does.
        captures = aux.capture(model) if aux is not None else nullcontext({})
        with captures as captured:
            if jepa is not None and pair is not None:
                # Two passes over one encoder: view A is the source, view B the target.
                # The stack's chunked attention cannot take the two views packed into
                # one odd-length window; the objective, its stop-gradient and the replay
                # are unchanged. The token loss still reads view B only.
                view_a, view_b = ids[:, :width], ids[:, width:]
                h_a, _ = cast(_Encoder, model).encode(view_a[:, :-1], counters)
                h_b, logits = cast(_Encoder, model).encode(view_b[:, :-1], counters)
                h = torch.cat([h_a, h_b], dim=0)
            else:
                logits = model(x, counters)
            seen = captured
        if jepa is not None and pair is not None:
            # eq. (3)'s token term stays on view B; dropping the alignment never drops
            # it, and a skipped step pays only this one forward's extra view.
            loss = chunked_cross_entropy(logits, y[:, width:])
        else:
            loss = chunked_cross_entropy(logits, y)
        total = loss
        if aux is not None:
            extra = aux.loss(seen)
            if not torch.isfinite(extra).all():
                optimizer.zero_grad(set_to_none=True)
                return _skip(counters, float(extra.detach()), "the aux loss is not finite")
            total = total + aux.weight * extra
            aux_total += float(extra.detach()) / accum
            if counters is not None:
                # Enabled after the forward: the stack checks its own counters inside
                # it, so a counter the step owns must not be declared before it returns.
                counters.enable("aux_head", counter="aux_fwd", why="the head predicts a state")
                counters.enable("aux", counter="aux_loss", why="the aux loss is added in")
                counters.count("aux_fwd", x.shape[0] * (x.shape[1] - aux.k))
                counters.count("aux_loss")
            _LOG.info(
                "aux lightmtp loss=%.6f weighted=%.6f k=%d layer=%d",
                float(extra.detach()),
                float(extra.detach()) * aux.weight,
                aux.k,
                aux.layer,
            )
        if keep and jepa is not None and h is not None:
            term = er_jepa_loss(
                (h[: h.shape[0] // 2], h[h.shape[0] // 2 :]),
                None,
                jepa.head,
                lam=jepa.lam,
            )
            if not torch.isfinite(term).all():
                optimizer.zero_grad(set_to_none=True)
                return _skip(counters, float(term.detach()), "the JEPA term is not finite")
            total = total + term
            jepa_total += float(term.detach()) / accum
        # The reported sum keeps the raw NTP value (a NaN step reports NaN). The
        # backward seed is masked: on a finite step `where(True, total, 0)` is
        # bit-identical to `total`; on a bad one no gradient is injected at all.
        loss_total = loss_total + loss.detach() / accum
        finite = torch.isfinite(total.detach()).all()
        seed = torch.where(finite, total, torch.zeros_like(total))
        if counters is not None:
            counters.enable("bwd", counter="bwd", why="the backward runs in the step")
        # torch's stub for `Tensor.backward` is unannotated, and this is the call.
        (seed / accum).backward()  # type: ignore[no-untyped-call]
        if counters is not None:
            counters.count("bwd", x.numel())
    if keep and jepa is not None:
        replayed = _replay(model, jepa, step=step, rows=rows * accum, counters=counters)
        if not math.isfinite(replayed):
            optimizer.zero_grad(set_to_none=True)
            return _skip(counters, replayed, "the replayed term is not finite")
        jepa_total += replayed
    if jepa is not None and counters is not None:
        counters.enable("jepa_memory_size", counter="jepa_memory_size", why="the store has a size")
        counters.count("jepa_memory_size", jepa.memory.size)
    bounded = list(model.parameters())
    if jepa is not None:
        attached = {id(p) for p in bounded}
        bounded.extend(p for p in jepa.head.parameters() if id(p) not in attached)
    result = commit_step(
        model,
        optimizer,
        loss_total,
        counters=counters,
        clip=clip,
        pre_step=pre_step,
        read=read,
        bounded=bounded,
    )
    if aux is None and jepa is None:
        return result
    return StepResult(
        loss=result.loss,
        grad_norm=result.grad_norm,
        skipped=result.skipped,
        loss_device=result.loss_device,
        norm_device=result.norm_device,
        aux_loss=aux_total if aux is not None else None,
        jepa_loss=jepa_total,
    )
