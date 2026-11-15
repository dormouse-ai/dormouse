"""Root-mean-square norm, and the pre-norm residual block that uses it twice.

Two concepts live here because a block is defined by its normalisation: separating
them would put the norm in a file the block's docstring has to point at, for a class
of 20 lines. `__concepts__` is the recipe's own declaration of that.

`RMSNorm`, as implemented (Zhang & Sennrich 2019, eq. 4):

    y = x · rsqrt(mean(x²) + ε) · g

WHY RMS AND NOT LAYER NORM. LayerNorm subtracts the mean, which costs a reduction
whose result is thrown away by the very next linear layer (a bias-free projection is
invariant to a constant shift of its input only if the input is normalised — the
mean removal buys nothing once the bias is gone, and a bias is the thing every
projection here refuses). RMSNorm keeps the variance normalisation, drops the mean
and the shift: `(1 + d)/d` fewer parameters per norm and one reduction instead of two.

WHY ONE ε, COMPUTED IN fp32. The archive carries 1e-5 in the blocks and 1e-3 in the
final norm, with a config knob that reached only the second — so an ε sweep measured
nothing in 12 of 13 norms and the two scales differed by 100x for no stated reason.
Here every norm in the stack shares one ε, `1e-6`: small enough that a collapsed
stream (the failure an ε must not hide) still reads as collapsed, large enough that
an fp32 `rsqrt` never divides by zero. The statistic is reduced in fp32 regardless of
the residual dtype, because `mean(x²)` over 832 bf16 values loses more than the norm's
own error budget.

WHY PRE-NORM, AND NOT SANDWICH. The block normalises the INPUT of each sublayer and
adds its raw output to the stream (`x + attn(norm(x))`). Sandwich norming
(`x + norm(attn(norm(x)))`) adds a parameter and a FLOP per sublayer and has no
measurement in this project behind it; pre-norm is what both reference implementations
of this recipe converge on and it is the form the gradient-flow invariant is written
against.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .attn import PROJ_STD, Attention
from .ffn import GatedFFN

__concepts__ = ("RMSNorm", "Block")

#: One ε for every norm in the stack; see the module docstring for why 1e-6.
EPS = 1e-6


class RMSNorm(nn.Module):
    """`x · rsqrt(mean(x²) + ε) · g`, with a learnable gain `g` initialised to 1."""

    def __init__(
        self, d_model: int, eps: float = EPS, device: torch.device | str | None = None
    ) -> None:
        """Build the gain; refuse a window that the statistic cannot survive."""
        super().__init__()
        if eps <= 0.0:
            raise LoudError(
                f"norm eps={eps} is not a positive floor",
                "use 1e-6; an eps of 0 divides by the stream's own zero rows",
            )
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Normalise the last axis, returning the input's dtype."""
        xf = x.to(torch.float32)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        if counters is not None:
            counters.count("norm_fwd", x.shape[0] * x.shape[1])
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        """Print the two numbers that define this norm."""
        return f"d_model={self.d_model}, eps={self.eps}"


class Block(nn.Module):
    """`x + attn(norm1(x))`, then `x + ffn(norm2(x))` — one decoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        d_ffn: int,
        *,
        n_layers: int,
        rope: bool = True,
        max_seq_len: int = 1024,
        eps: float = EPS,
        device: torch.device | str | None = None,
    ) -> None:
        """Build both pre-norms and both sublayers.

        `n_layers` is the depth of the STACK and is here for one reason: the two
        projections that write the residual stream (`attn.proj`, `ffn.down`) draw
        from `N(0, (0.02/√(2L))²)` instead of `N(0, 0.02²)`. A block adds a delta of
        variance `v` to the stream, so `L` blocks leave `Var(x_L) ≈ Var(x_0) + L·v`;
        scaling every write by `1/√(2L)` makes the accumulated variance `≈ 1.5·Var(x_0)`
        at any depth, which is the difference between a 12-layer model and a stack
        whose last layer sees a stream 12x the first one's. A block cannot know `L`,
        so its caller passes it.
        """
        super().__init__()
        if n_layers < 1:
            raise LoudError(
                f"n_layers={n_layers}: a block cannot be the 0th of nothing",
                "pass the depth of the stack this block belongs to",
            )
        write_std = PROJ_STD / (2.0 * n_layers) ** 0.5
        self.norm1 = RMSNorm(d_model, eps, device=device)
        self.attn = Attention(
            d_model,
            n_heads,
            head_dim,
            rope=rope,
            max_seq_len=max_seq_len,
            write_std=write_std,
            device=device,
        )
        self.norm2 = RMSNorm(d_model, eps, device=device)
        self.ffn = GatedFFN(d_model, d_ffn, write_std=write_std, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` in, the residual stream out, same shape."""
        x = x + self.attn(self.norm1(x, counters), counters)
        x = x + self.ffn(self.norm2(x, counters), counters)
        if counters is not None:
            counters.count("block_fwd")
        return x

    def extra_repr(self) -> str:
        """Print what one layer costs, without materialising it."""
        return f"d_model={self.attn.d_model}, d_ffn={self.ffn.d_ffn}, rope={self.attn.rope}"
