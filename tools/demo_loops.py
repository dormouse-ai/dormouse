"""F1 inference-time loop graft for the Friday demo (S6, arXiv 2605.23872 equations).

`graft` patches ONLY the model's forward: the contiguous mid-block window is applied
as a damped Euler fixed point ``x <- (1-beta) x + beta g(x)`` K times. No kernel, no
weight, no optimizer changes; `state_dict()` stays bit-for-bit the original. The
implementation is from the update equation, not copied from any CC BY-NC-ND code.

    model = graft(model, window=(7, 10), K=2, beta=0.5, mode="layer")
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch
from dormouse_core.errors import LoudError
from torch import nn


def _check(
    model: nn.Module,
    window: tuple[int, int],
    K: int,  # noqa: N803
    beta: float,
    mode: str,
) -> None:
    """Refuse a loop the stack cannot express, by name."""
    if mode != "layer":
        raise LoudError(f"mode={mode!r} is not the layer loop", "use mode='layer'")
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise LoudError(
            "graft needs model.blocks", "the loop is over contiguous Transformer blocks"
        )
    lo, hi = window
    if not (0 <= lo < hi < len(blocks)):
        raise LoudError(
            f"window={window} is outside {len(blocks)} blocks",
            "the loop needs at least one interior layer after a prefix and before a suffix",
        )
    if K < 1 or not isinstance(K, int):
        raise LoudError(f"K={K} is not a positive iteration count", "K counts fixed-point steps")
    if not 0.0 < beta <= 1.0:
        raise LoudError(f"beta={beta} is outside (0, 1]", "the damping coefficient must shrink x")


def graft(
    model: nn.Module,
    *,
    window: tuple[int, int] = (7, 10),
    K: int = 2,  # noqa: N803
    beta: float = 0.5,
    mode: str = "layer",
) -> nn.Module:
    """Wrap forward with K damped applications of blocks[window[0]:window[1]+1].

    The prefix and suffix blocks keep their original order and run exactly once. The
    tied head and final norm are the model's own, called once at the end. The wrapper
    adds no parameters and no buffers, so state_dict is unchanged.
    """
    _check(model, window, K, beta, mode)
    blocks = model.blocks
    lo, hi = window
    tail = range(hi + 1, len(blocks))

    def looped_forward(self: nn.Module, ids: torch.Tensor, counters: Any = None) -> torch.Tensor:
        """Run the stack's forward with the mid-block window iterated K times."""
        if ids.dim() != 2 or ids.is_floating_point():
            raise LoudError("loop expects token ids [batch, seq]", "cast to a 2-D long tensor")
        h = model.embed(ids)
        for index in range(lo):
            h = blocks[index](h, counters)
        for _ in range(K):
            updated = h
            for index in range(lo, hi + 1):
                updated = blocks[index](updated, counters)
            h = (1.0 - beta) * h + beta * updated
        for index in tail:
            h = blocks[index](h, counters)
        return model.head(model.norm(h, counters), counters)

    model.forward = types.MethodType(looped_forward, model)
    return model


def polynomial_weights(K: int = 3) -> tuple[float, ...]:  # noqa: N803
    """F4: binomial/C(K-1,k)/2^(K-1) weights for a K-1 loop polynomial."""
    if K < 2:
        raise LoudError(f"K={K} has no polynomial ablation", "K>=2 gives at least two states")
    denom = 2.0 ** (K - 1)
    return tuple(math.comb(K - 1, k) / denom for k in range(K))


def self_test() -> str:
    """Check on CPU that the wrapper changes logits and never touches weights."""
    from dormouse.model.assemble import assemble
    from dormouse.model.spec import StackSpec

    spec = StackSpec(
        d_model=64,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        c_lat=32,
        d_ffn=128,
        n_experts=4,
        topk=1,
        expert_tsct_rank=8,
        attn_k=64,
        attn_window=32,
        seq_len=128,
        batch=2,
        torch_compile=False,
    )
    model = assemble(spec, device="cpu", smoke=True).model
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    ids = torch.randint(0, spec.vocab, (2, 128))
    with torch.no_grad():
        base = model(ids)
    grafted = graft(model, window=(3, 5), K=2, beta=0.5, mode="layer")
    after = grafted.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    with torch.no_grad():
        looped = grafted(ids)
    assert looped.shape == base.shape
    assert not torch.equal(looped, base)
    assert polynomial_weights(3) == (0.25, 0.5, 0.25)
    return f"graft self-test PASS: {len(before)} tensors bitwise, logits changed"


if __name__ == "__main__":
    print(self_test())
