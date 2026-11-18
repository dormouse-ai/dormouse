"""The two fake-quantizations the stack can be built with, and the block that proves both.

A quantization is a mechanism like any other here: it changes the arithmetic of every
forward, so it must move a counter that the step reads. The protocol is the same as
`model/attn_variants`': the number in the plan is only a claim, the counter is the
proof.

WEIGHTS: TERNARY, AND WHY NOT `NN.UTILS.PARAMETRIZE`. `ternary` is the BitNet b1.58
straight-through estimator — round the weight to `s·{-1,0,1}` with `s` the per-matrix
mean magnitude, and pass the gradient through unchanged. The textbook way to apply it
is a parametrization, which renames the parameter to
`parametrizations.weight.original` — and `train/optim.py`'s Muon routing reads names
(`_head_split` looks for `.attn.qkv.weight`, `route` for `embed.weight`), so a
parametrized run would silently lose the qkv head split. Hence the class swap:
`QuantLinear` is an `nn.Linear` that ternarises its weight in `forward`, so the
arithmetic changes and the parameter names do not.

ACTIVATIONS: INT4, PER TOKEN. `int4` clips to the symmetric range `[-7, 7]` with one
scale per token (the activation's own `amax`), straight-through as well. Per token,
not per tensor: a per-tensor scale over a 24·1024 window is dominated by the loudest
row and rounds the quiet ones to zero, which is the failure a fake-quant exists to
predict. The conflict with `torch.compile` is measured rather than assumed: on the
four-layer stack `torch._dynamo.explain` reports 12 graphs and 11 graph breaks with
int4 against 11 and 10 without, so the fake-quant boundary splits the compiled graph
— the numbers agree (max |Δ| = 3.8e-5 over a forward), the fusion does not happen, and
`spec.validate` refuses the pair instead of pricing a kernel that never ran.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.counters import Counters
from torch import nn
from torch.nn import functional

__concepts__ = ("QuantBlock", "QuantLinear")


def ternary(weight: torch.Tensor) -> torch.Tensor:
    """`s·round(w/s)` clipped to `{-1, 0, 1}`, with the gradient passed through."""
    scale = weight.detach().abs().mean().clamp_min(1e-8)
    return weight + ((weight / scale).round().clamp(-1, 1) * scale - weight).detach()


def int4(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int4 fake-quantization, straight-through."""
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7
    return x + ((x / scale).round().clamp(-7, 7) * scale - x).detach()


class QuantLinear(nn.Linear):
    """An `nn.Linear` whose weight is ternarised for the forward, never in the state."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run `nn.Linear`'s contract, with one `ternary` call on the weight."""
        return functional.linear(x, ternary(self.weight), self.bias)


def ternarize(model: nn.Module) -> int:
    """Swap every plain `nn.Linear` for `QuantLinear`; return how many were swapped."""
    swapped = 0
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = QuantLinear
            swapped += 1
    return swapped


class QuantBlock(nn.Module):
    """A block with the quantizations applied where they belong, and counted.

    `act_int4` quantizes the activations entering it, and the ternary counter is
    read from the LIVE module tree at forward time — not from the `matrices`
    snapshot taken at build. An undone swap then reports zero and an enabled
    quantization that never ran raises `MissingReport`, which is the defect this
    file exists to make impossible.
    """

    def __init__(self, inner: nn.Module, *, matrices: int = 0, act_int4: bool = False) -> None:
        """Wrap a block; `matrices` is a claim about the inner module's weights."""
        super().__init__()
        self.inner = inner
        self.matrices = matrices
        self.act_int4 = act_int4

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Quantize what the block reads, then run it."""
        if self.act_int4:
            x = int4(x)
            if counters is not None:
                counters.count("act_int4", x.shape[0] * x.shape[1])
        if counters is not None:
            #: Live use, not the build-time claim: a swap that never happened (or was
            #: undone) must leave the counter at zero so `check()` can refuse it.
            live = sum(type(m) is QuantLinear for m in self.inner.modules())
            if live:
                counters.count("weight_ternary", live)
        return cast(torch.Tensor, self.inner(x, counters))
