"""The fake-quantizations the stack can be built with, and the block that proves them.

A quantization is a mechanism like any other here: it changes the arithmetic of every
forward, so it must move a counter that the step reads. The protocol is the same as
`model/attn_variants`': the number in the plan is only a claim, the counter is the
proof.

WEIGHTS: TERNARY, FP4, AND WHY NOT `NN.UTILS.PARAMETRIZE`. `ternary` is the BitNet
b1.58 straight-through estimator — round the weight to `s·{-1,0,1}` with `s` the
per-matrix mean magnitude, and pass the gradient through unchanged. `fp4` is
`fp4.fake_quant` (NVFP4: e2m1 codes, an e4m3 scale per 16 values, an fp32 tensor
scale). The textbook way to apply either is a parametrization, which renames the
parameter to `parametrizations.weight.original` — and `train/optim.py`'s Muon
routing reads names (`_head_split` looks for `.attn.qkv.weight`, `route` for
`embed.weight`), so a parametrized run would silently lose the qkv head split.
Hence the class swap: `QuantLinear` is an `nn.Linear` whose forward reads a
fake-quantized weight, so the arithmetic changes and the parameter names do not.

ONE QUANTIZED WEIGHT, TWO READERS. A matrix is read two ways in this stack: plain
(`functional.linear`) and through its rank-`r` spectrum (`tsct.low_rank`, which
never calls the module's forward). So the swap puts the scheme on the module as
`effective_weight()`, and BOTH readers ask for it — a swap that only rewrote
`forward` would leave every expert in the bank reading its unquantized master
while the block still counted the matrix as quantized. That was the shape of the
defect this file's counters exist for, and `tests/test_fp4.py` reads a weight
through both paths to pin it.

ACTIVATIONS: INT4 OR FP4, PER TOKEN AT THE BLOCK BOUNDARY. `int4` clips to the
symmetric range `[-7, 7]` with one scale per token (the activation's own `amax`);
`fp4` is the format above, one e4m3 scale per 16 channels. Per token, not per
tensor: a per-tensor scale over a 24·1024 window is dominated by the loudest row
and rounds the quiet ones to zero, which is the failure a fake-quant exists to
predict. The conflict with `torch.compile` is measured rather than assumed: on the
four-layer stack `torch._dynamo.explain` reports 12 graphs and 11 graph breaks with
int4 against 11 and 10 without, so the fake-quant boundary splits the compiled graph
— the numbers agree (max |Δ| = 3.8e-5 over a forward), the fusion does not happen,
and `spec.validate` refuses the pair instead of pricing a kernel that never ran.

WHAT THE FP4 COUNTERS CLAIM, AND WHAT THEY DO NOT. `weight_fp4`/`act_fp4` are this
block's own reads — the same claim `weight_ternary`/`act_int4` make. The kernel is
a different question: `fp4.tally()` says whether `torch._scaled_mm` served any
matmul on this build, and a run whose tally has `fp4_scaled_mm=0` is the reference
format, not the format's tensor cores. Both numbers are reported, never merged.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from . import fp4

__concepts__ = ("QuantBlock", "QuantLinear")

#: The weight schemes a module can carry; `"none"` is the absence of a swap.
SCHEMES: tuple[str, ...] = ("none", "ternary", "fp4")


def ternary(weight: torch.Tensor) -> torch.Tensor:
    """`s·round(w/s)` clipped to `{-1, 0, 1}`, with the gradient passed through."""
    scale = weight.detach().abs().mean().clamp_min(1e-8)
    return weight + ((weight / scale).round().clamp(-1, 1) * scale - weight).detach()


def int4(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int4 fake-quantization, straight-through."""
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7
    return x + ((x / scale).round().clamp(-7, 7) * scale - x).detach()


class QuantLinear(nn.Linear):
    """An `nn.Linear` whose forward reads a fake-quantized weight, never the state.

    `scheme` picks the fake-quant (`"ternary"` or `"fp4"`) and is written by
    `quantize_weights` when the class is swapped, so the parameter names, the
    `state_dict()` and every name-based rule in `train/optim.py` are untouched.
    """

    scheme: str = "ternary"

    #: Whether the block quantized the activation before this layer read it, which
    #: is what makes the fp4 kernel (both operands at one format) a candidate.
    activations_fp4: bool = False

    def effective_weight(self) -> torch.Tensor:
        """Return the weight a forward reads: the fake-quantized master, straight-through."""
        if self.scheme == "fp4":
            return fp4.fake_quant(self.weight)
        return ternary(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`nn.Linear`'s contract on the quantized weight; the kernel when it can."""
        if self.scheme == "fp4":
            return fp4.linear(x, self.weight, self.bias, activations=self.activations_fp4)
        return functional.linear(x, ternary(self.weight), self.bias)


def quantize_weights(model: nn.Module, scheme: str, *, activations_fp4: bool = False) -> int:
    """Swap every plain `nn.Linear` for a `QuantLinear` on `scheme`; return the count."""
    if scheme not in SCHEMES or scheme == "none":
        raise LoudError(
            f"scheme={scheme!r} swaps no matrix",
            f"use one of: {', '.join(SCHEMES[1:])} (a slot set to 'none' is not swapped)",
        )
    swapped = 0
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = QuantLinear
            rung: QuantLinear = cast(QuantLinear, module)
            rung.scheme = scheme
            rung.activations_fp4 = activations_fp4 and scheme == "fp4"
            swapped += 1
    return swapped


class QuantBlock(nn.Module):
    """A block with the quantizations applied where they belong, and counted.

    `matrices` is how many quantized matrices this block's forward reads (shared bank
    matrices are read by every block, so each block reports its own read) and
    `weight_key` is the counter that proves which scheme was applied. `act_int4` and
    `act_fp4` quantize the activations entering the block. Every counter here is the
    proof the plan's bits are real: an enabled quantization that never ran raises
    `MissingReport`, which is the defect this file exists to make impossible.
    """

    def __init__(
        self,
        inner: nn.Module,
        *,
        matrices: int = 0,
        weight_key: str = "weight_ternary",
        act_int4: bool = False,
        act_fp4: bool = False,
    ) -> None:
        """Wrap a block; `matrices` is a claim about the inner module's weights."""
        super().__init__()
        self.inner = inner
        self.matrices = matrices
        self.weight_key = weight_key
        self.act_int4 = act_int4
        self.act_fp4 = act_fp4

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Quantize what the block reads, then run it."""
        tokens = x.shape[0] * x.shape[1]
        if self.act_int4:
            x = int4(x)
            if counters is not None:
                counters.count("act_int4", tokens)
        if self.act_fp4:
            x = fp4.fake_quant(x)
            if counters is not None:
                counters.count("act_fp4", tokens)
        if counters is not None and self.matrices:
            counters.count(self.weight_key, self.matrices)
        return cast(torch.Tensor, self.inner(x, counters))
