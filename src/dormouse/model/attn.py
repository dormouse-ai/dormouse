"""Causal scaled dot-product attention, with rotary positions written out.

The formula, as implemented (Vaswani et al. 2017 eq. 1; Su et al. 2021 eq. 1-3):

    q, k, v = W_qkv · x                        # one packed [3d, d] matrix, no bias
    θ_j     = base^(-2j/head_dim)              # j = 0 .. head_dim/2 - 1
    R(p, j) = (cos(p·θ_j), sin(p·θ_j))         # rotate channels (2j, 2j+1) of q, k
    a_i     = softmax_j≤i (q'_i · k'_j / √head_dim) · v_j
    y       = W_proj · a

WHY ROPE IS THE DEFAULT AND NOT A SWITCH NOBODY FLIPS. A byte-level model carries
no tokenizer prior on distance, so without a positional signal every position sees
the same multiset of bytes: the 2026-10 record of a full-attention block with
positions stripped measured loss 4.28 at step 0 climbing to 16.4. `rope=False` is
here for one reason — the permutation test has to show that the rotations, and not
the mask, are what break position symmetry — it is the wrong model, not an arm.

WHY ONE PACKED `qkv`. Three `[d, d]` linears and one `[d, 3d]` linear hold the same
parameters and do the same arithmetic; one GEMM is one launch instead of three and
the split is a view, not a copy.

WHY THE CAUSAL MASK IS ALSO A KEYWORD. `causal=False` exists so a test can separate
"the mask ties outputs to index order" from "the rotations tie them to absolute
position" — two different facts that one causal configuration would confound. It
is the identity of a non-causal attention, not a training option.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

#: Std of the position-independent projections. 0.02 is GPT-2's; at d=832 a
#: pre-activation has variance `d·0.02² ≈ 0.33`, a third of the unit-RMS stream it
#: is added to. The residual-writing projection takes the caller's `write_std`.
PROJ_STD = 0.02


class Attention(nn.Module):
    """Multi-head attention: `W_proj(SDPA(RoPE(W_qkv x)))`."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        rope: bool = True,
        max_seq_len: int = 1024,
        rope_base: float = 10_000.0,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the two projections and, when `rope`, the position tables."""
        super().__init__()
        if n_heads < 1 or head_dim < 2:
            raise LoudError(
                f"attention needs n_heads >= 1 and head_dim >= 2, got {n_heads}x{head_dim}",
                "head_dim 1 has no pair for RoPE to rotate",
            )
        if head_dim % 2:
            raise LoudError(
                f"head_dim={head_dim} is odd: RoPE rotates pairs of channels",
                "use an even head_dim (64 in the live recipe)",
            )
        if n_heads * head_dim != d_model:
            raise LoudError(
                f"n_heads·head_dim = {n_heads}·{head_dim} = {n_heads * head_dim} "
                f"≠ d_model={d_model}",
                "the split into heads must tile the residual width exactly",
            )
        if max_seq_len < 1:
            raise LoudError(
                f"max_seq_len={max_seq_len} holds no positions",
                "set max_seq_len to the window the run will train on",
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope = rope
        self.max_seq_len = max_seq_len
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        self.proj = nn.Linear(d_model, d_model, bias=False, device=device)
        nn.init.normal_(self.qkv.weight, std=PROJ_STD)
        nn.init.normal_(self.proj.weight, std=write_std)
        #: `[max_seq_len, head_dim/2]`, not persistent: they are a function of the
        #: config (base, head_dim, window), so a checkpoint carrying them could only
        #: disagree with it. Allocated once here — a per-forward `outer` is 24 KiB of
        #: trig per block per step, which is exactly the kind of cost that hides.
        half = head_dim // 2
        freq = rope_base ** (-2.0 * torch.arange(half, dtype=torch.float32) / head_dim)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        ang = torch.outer(pos, freq)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        *,
        causal: bool = True,
    ) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the attention write, same shape."""
        batch, seq, _ = x.shape
        if seq > self.max_seq_len:
            raise LoudError(
                f"a {seq}-token forward exceeds max_seq_len={self.max_seq_len}: "
                "positions past the table have no rotation",
                f"train on <= {self.max_seq_len} tokens, or rebuild with a longer window",
            )
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        shape = (batch, seq, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        if self.rope:
            cos, sin = self.cos[:seq], self.sin[:seq]
            q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=self.scale)
        y = y.transpose(1, 2).reshape(batch, seq, self.d_model)
        if counters is not None:
            counters.count("attn_fwd", batch * seq)
            if self.rope:
                counters.count("rope_fwd", batch * seq)
        return self.proj(y)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent channel pairs of `x [b, h, t, head_dim]` in place-of-value."""
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
