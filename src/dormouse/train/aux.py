"""LightMTP-S: a latent auxiliary objective that predicts the trunk's own future state.

The objective (Czinczoll, Kallini, de Melo & Shani, *LightMTP: Lightweight Latent
Multi-Token Prediction*, arXiv:2610.06031, §3.2) adds one term to next-token
prediction, for a horizon `k` and every position `t` with `t+k < T`:

    v̂_t     = norm(g_φ(h^L_t))                     the head reads the last hidden state
    v_{t+k} = norm(sg(h^l_{t+k}))                  the trunk's OWN state k steps on, at
                                                   layer l, sg(·) = stop-gradient
    L_MTP   = 1 - (1/|T_k|) Σ_t ⟨v̂_t, v_{t+k}⟩                             (eq. 1)
    L_total = L_NTP + λ·L_MTP [+ λ_var·L_VarReg]                           (eq. 3)

with `g_φ` a two-layer GELU MLP `d → width → d`, `λ = 0.1`. The "-S" is single-target:
ONE vector, `k` steps on. The "-C" variant concatenates the states at `t+1 … t+k` into
one `k·d` output and costs about 1% of the trunk against 0.4%; it is not implemented
here because nothing in the paper's tables buys it back (LightMTP-S beats it at every
`k` on Countdown: 56.7 ± 0.9 against 56.1 ± 1.4 at k=8).

WHY THIS OBJECTIVE. The target needs no second view, no teacher, no second model and no
diffusion backbone — it is the same trunk, `k` positions later, at the same layer, with
the gradient cut. Measured at 286M in the paper (§5.2-5.4), at the same GPU-hours:
Countdown 50.6 → 56.7 at k=8, GSM8K strict 4.94 → 5.69, FineWeb-Edu PPL 1.683 → 1.684.
A token-level MTP head instead adds a vocabulary projection per offset — `(k-1)·|V|·d`
parameters, plus the NTP regression the paper measures — which is exactly what `-S`
avoids: at d=832 and width=d/2 the whole head is 692k.

WHY THE SOURCE IS `h^L` AND THE TARGET IS `h^l`. The source is the state the token head
reads, so the gradient shapes the trunk where its output is consumed. The target may be
any intermediate block (`aux_layer`); the paper's default is two-thirds of depth (l=16
of 24) and this module's default is the final block, the cheapest honest choice and the
one an A/B should start from (one layer to blame instead of two). Both vectors are taken
AFTER the block and BEFORE the final norm: the cosine loss normalises anyway, and routing
the target through the final norm's learnable gain would put that gain inside a loss it
was never trained against.

WHY A FORWARD HOOK AND NOT A SECOND FORWARD. `train/step.py` already runs the forward and
the backward; collecting hidden states with a second pass would double the step's FLOPs
and, worse, measure a different batch. `LightMTP.capture` registers one hook per watched
block for exactly one forward and removes it in a `finally`, so `aux="none"` registers
nothing at all and the step stays bit-identical.

VARIANCE REGULARIZATION — READ THIS BEFORE ENABLING IT. Eq. 2 penalises the *norm of the
mean* of the target layer's normalised states, per batch element:

    ĥ_{b,t} = norm(h^l_{b,t}),   L_VarReg = (1/B) Σ_b ‖(1/T) Σ_t ĥ_{b,t}‖₂    (eq. 2)

A large `‖mean‖` means every position points the same way, so minimising it drives that
mean to zero: it collapses the SHARED component of the target, leaving the per-position
directions mutually orthogonal (the per-position norms are pinned to 1 by the
normalisation, so those cannot collapse). The paper measures what that costs — +VarReg
improves future-token probing and degrades every language-modelling column (FineWeb-Edu
PPL 1.684 → 1.694 for -S, LAMBADA PPL 16.28 → 17.53, LAMBADA acc 43.31 → 42.33) — which
is why `aux_varreg` defaults to false, and why `tests/test_aux.py` shows the collapse
instead of asserting it is harmless.

BUDGET. The head is `2·d·width` parameters and `LightMTP` refuses more than 1% of the
trunk. The paper's own head is `d → d → d`, `2·d² = 1.2M` at d=768 on a 286.3M d12 trunk
(+0.4%); the same form on this repo's 94.8M byte stack is +1.46%, over the ceiling, so
`width` defaults to `d/2` (692k, +0.73%). Pass `width=d_model` when the trunk is big
enough to afford it — the assertion is what stops a head from becoming a second model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from ..model.model import Transformer

__concepts__ = ("LightMTP", "LightMTPHead")

__all__ = ["LightMTP", "LightMTPHead", "varreg"]

#: The paper's λ_MTP (eq. 3).
WEIGHT = 0.1

#: The paper's λ_VarReg: every LightMTP variant is trained with and without it (§5).
VARREG_WEIGHT = 0.1

#: The head's ceiling as a fraction of the trunk's parameters: the paper's "at most 1%".
MAX_FRACTION = 0.01

#: Projection init, shared with `model/`: `N(0, 0.02²)`.
STD = 0.02


def varreg(target: torch.Tensor) -> torch.Tensor:
    """Eq. 2: `(1/B) Σ_b ‖mean_t norm(h^l_{b,t})‖₂`, the target's shared direction.

    Read as a measurement rather than a penalty: this value is 1 when every position
    carries the same direction (a target that is trivial to predict, and that carries no
    per-position information) and 0 when the positions are mutually orthogonal. Disabled
    by default; see the module docstring for the cost the paper measures when it is on.
    """
    unit: torch.Tensor = functional.normalize(target.float(), dim=-1)
    shared: torch.Tensor = unit.mean(dim=1)
    mean_direction: torch.Tensor = shared.norm(dim=-1).mean()
    return mean_direction


class LightMTPHead(nn.Module):
    """`g_φ`: `d → width → d` with a GELU between, bias-free, one vector out.

    No bias, like every projection in `model/`: a bias-free map is what the residual
    stream's own parameterisation is written against. The width defaults to `d/2` for
    the budget's sake, not the paper's — see the module docstring.
    """

    def __init__(
        self,
        d_model: int,
        width: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the two matrices, refusing a width that projects into nothing."""
        super().__init__()
        resolved = d_model // 2 if width is None else width
        if resolved < 1:
            raise LoudError(
                f"the LightMTP head's width={resolved} projects into no dimension",
                "width is the MLP's hidden width; d_model//2 is the default and 1 the floor",
            )
        self.d_model = d_model
        self.width = resolved
        self.up = nn.Linear(d_model, resolved, bias=False, device=device)
        self.down = nn.Linear(resolved, d_model, bias=False, device=device)
        nn.init.normal_(self.up.weight, std=STD)
        nn.init.normal_(self.down.weight, std=STD)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`h […, d]` → the predicted state `k` positions on, same shape."""
        return cast(torch.Tensor, self.down(functional.gelu(self.up(h))))

    def extra_repr(self) -> str:
        """Print the two widths the budget is argued in."""
        return f"d_model={self.d_model}, width={self.width}"


class LightMTP(nn.Module):
    """The LightMTP-S objective: the head, its horizon, and the loss it produces."""

    def __init__(
        self,
        model: Transformer,
        *,
        k: int = 8,
        layer: int = 0,
        weight: float = WEIGHT,
        varreg: bool = False,
        width: int | None = None,
    ) -> None:
        """Check the horizon, build the head, refuse a head over 1% of the trunk.

        `layer` is 1-based over the blocks; `0` (and `n_layers`) mean the final block's
        output. The trunk's parameter count is read BEFORE the head is attached, so the
        fraction is the one a config prices: head over trunk, no circularity.
        """
        super().__init__()
        if k < 1:
            raise LoudError(
                f"aux k={k} looks zero positions ahead",
                "k=8 is the horizon the A/B starts from; k=1 is a next-state prediction, "
                "which is not what the paper measures",
            )
        if weight <= 0.0:
            raise LoudError(
                f"aux weight={weight} gives the head a zero (or reversed) gradient",
                f"{WEIGHT} is the paper's λ_MTP; a head trained at 0 is 692k parameters "
                "that change nothing and are still paid for",
            )
        self.k = k
        self.n_layers = model.n_layers
        target = model.n_layers if layer == 0 else layer
        if not 1 <= target <= self.n_layers:
            raise LoudError(
                f"aux layer={layer} is not a block of this {self.n_layers}-layer stack",
                "0 (the default) and n_layers both mean the final block's output; "
                "1..n_layers-1 index an intermediate block, which is where the paper "
                "takes its targets (l=16 of 24)",
            )
        self.layer = target
        self.weight = weight
        self.varreg = varreg
        self.head = LightMTPHead(model.d_model, width, device=next(model.parameters()).device)
        self.head_params = sum(param.numel() for param in self.head.parameters())
        trunk = sum(param.numel() for param in model.parameters())
        self.fraction = self.head_params / trunk
        if self.fraction > MAX_FRACTION:
            raise LoudError(
                f"the LightMTP head is {self.head_params:,} parameters, "
                f"{self.fraction:.2%} of the trunk's {trunk:,}",
                f"a head may cost at most {MAX_FRACTION:.0%} of the trunk: pass a "
                "narrower width (d_model//2 is the default) or run it on a larger stack; "
                "width=d_model is 2·d² — the paper's own form, affordable on its 286.3M "
                "d12 trunk and not on this one",
            )

    def attach(self, model: Transformer) -> None:
        """Register the head on `model` as `model.aux`, so the step owns it.

        A submodule and not a module beside the model, because two things read
        `model.named_parameters()`: `MixOptimizer`'s router (an unattached head is in no
        param group and silently never trains) and `clip_grad_norm_` (one global norm
        over the step). Attach it BEFORE the optimizer is built and AFTER `quantize`:
        this head is a training-only fp32 path, never a weight a checkpoint ships.
        """
        if getattr(model, "aux", None) is not None:
            raise LoudError(
                "this model already carries an aux head",
                "one aux slot per model: call `assemble` once per run",
            )
        model.add_module("aux", self)

    @contextmanager
    def capture(self, model: Transformer) -> Iterator[dict[int, torch.Tensor]]:
        """Record the watched blocks' outputs for one forward, then remove the hooks.

        Yields `{layer_index: h}`, 1-based over the blocks, holding every position: the
        `k`-step shift belongs to `objective`, where it can be reasoned about once. The
        `finally` is the contract — a forward that raises leaves no hook on the model.
        """
        seen: dict[int, torch.Tensor] = {}

        def record(index: int) -> Callable[..., None]:
            """Build the hook for one watched block."""

            def hook(module: nn.Module, inputs: object, output: torch.Tensor) -> None:
                seen[index] = output

            return hook

        handles = [
            model.blocks[index - 1].register_forward_hook(record(index))
            for index in sorted({self.layer, self.n_layers})
        ]
        try:
            yield seen
        finally:
            for handle in handles:
                handle.remove()

    def loss(self, seen: dict[int, torch.Tensor]) -> torch.Tensor:
        """Return the auxiliary loss from the hidden states one `capture` recorded."""
        source = seen.get(self.n_layers)
        target = seen.get(self.layer)
        if source is None or target is None:
            raise LoudError(
                f"no hidden state was captured for layer {self.layer} or {self.n_layers}",
                "run the forward inside `with aux.capture(model)`; a head whose target "
                "never arrived would report a loss that trains nothing",
            )
        return self.objective(source, target)

    def objective(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Eq. 1 (+ eq. 2 when enabled) from `h^L` and `h^l`, each `[batch, seq, d]`.

        The shift, in one line: the prediction at `t` reads `source[:, t]` only, the
        target is `target[:, t+k]`, and `detach()` is what makes it a target — no
        gradient flows back through the trunk's future positions, only through the
        present one the token loss also reads.
        """
        if source.dim() != 3 or source.shape != target.shape:
            raise LoudError(
                "the aux head needs two hidden states [batch, seq, d] of one shape, got "
                f"{tuple(source.shape)} and {tuple(target.shape)}",
                "both come from one forward: the source is the last block's output and "
                "the target is the block `aux_layer` names",
            )
        if source.shape[1] <= self.k:
            raise LoudError(
                f"a {source.shape[1]}-position sequence has no position {self.k} ahead",
                "the horizon is k steps on: keep k below seq_len and above 0",
            )
        prediction: torch.Tensor = functional.normalize(
            self.head(source[:, : -self.k]).float(), dim=-1
        )
        future: torch.Tensor = functional.normalize(target[:, self.k :].detach().float(), dim=-1)
        loss: torch.Tensor = 1.0 - (prediction * future).sum(dim=-1).mean()
        if self.varreg:
            loss = loss + VARREG_WEIGHT * varreg(target)
        return loss

    def extra_repr(self) -> str:
        """Print the four numbers a run's log line should repeat."""
        return (
            f"k={self.k}, layer={self.layer}, weight={self.weight}, varreg={self.varreg}, "
            f"params={self.head_params}"
        )
