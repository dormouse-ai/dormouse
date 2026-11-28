"""Chunked sparse attention with a selector, and the coarse HCA level inside it.

WHAT IT IS. Cut the keys and values into blocks of `chunk` tokens. For query `t`
the selector scores every ELIGIBLE block `c ≤ ⌊t/chunk⌋` by its mean key,

    s[t,c] = ⟨q_t, k̄_c⟩ / √d_k,        k̄_c = mean of block c's keys,

keeps the top-k past blocks, adds `t`'s own block unconditionally (a token must
be able to read itself), and attention runs only over the tokens those blocks
contain. That is O(T·k·chunk) comparisons instead of O(T²), and it is the
"selected" branch of Native Sparse Attention (DeepSeek 2502.11089 §3.2) with a
content-only summary.

HCA IS A PARAMETER, NOT A CLASS. `hca_group=g` groups `g` fine blocks into one
coarse block and runs the same selector one level up first: one strictly past
coarse block is kept, plus the query's own coarse block (forced), then the fine
selector chooses inside them (`hca_group=1` is plain CSA; `g=8` is the
128-token coarse arm). It is a parameter because both levels share every weight
— a second class would be this file with a different loop.

THE COARSE BLOCK IS FORCED, NOT SCORED. The mean of a coarse block containing
the query includes positions after it, so letting its score decide eligibility
lets a later token change an earlier output. The branch this was ported from
scored that block; here it is always kept and only strictly past coarse blocks
are ranked. Their means contain only tokens the query may read.

THE SELECTOR IS NOT A SWITCH. The top-k mask decides which blocks are READ; the
selector's own score is also added to the logits of the blocks it kept, scaled by
a per-head `selector_gain`. Without that term the scores would receive no
gradient (top-k is piecewise constant) and the selector would stay at its random
initialisation for the whole run — the failure a previous arm measured as exactly
chance selection.

CAUSALITY. Blocks after `t`'s block are ineligible, positions after `t` are
masked, and the current block's selector score is dropped from the logits because
its mean key contains positions later than `t`. A tie among ineligible slots is
resolved by pointing them at the current block (deterministically, not by
`topk`'s choice among `-inf`), so a later perturbation leaves earlier outputs
bit-identical.

This layer owns its q/k/v projections (one head each, no GQA); the newer
`LatentProjections` bottleneck belongs to KDA and DSA and is left untouched.

Ceiling: the gather materialises `k·chunk` keys per query. That is the point of
the mechanism at 4k tokens and the reason `topk` is a knob, not a constant.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.coeffects import Coeffects
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..blocks import PROJ_STD

#: The recipe: 16-token blocks, two past blocks, one coarse level of eight.
CHUNK = 16
TOPK = 2
HCA_GROUP = 8


class CSA(nn.Module):
    """Top-k block attention with a content selector, optionally behind HCA."""

    #: CSA is only CSA with a selector on; see `__init__`.
    COEFFECTS = Coeffects(
        requires={"attn.pos": "none", "attn.selector": "true"},
        provides={"attn.state": "sparse", "attn.selector": "true"},
        exclusive=("attn.state", "attn.selector"),
    )

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        chunk: int = CHUNK,
        topk: int = TOPK,
        hca_group: int = 1,
        selector: bool = True,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the projections and the selector gain; refuse a selector-less CSA.

        `selector=False` is refused for the same reason `KDA(delta=False)` is:
        block attention with every block kept is the dense `Attention` control
        arm, and calling it CSA would report a sparse arm that never selected.
        """
        super().__init__()
        if not selector:
            raise LoudError(
                "CSA(selector=False): without the selector this is dense block attention",
                "keep selector=True, or use the dense attention control arm",
            )
        if d_model < 1 or n_heads < 1 or head_dim < 1:
            raise LoudError(
                f"CSA needs positive widths, got d_model={d_model}, "
                f"n_heads={n_heads}, head_dim={head_dim}",
                "pass the residual width and the per-head width the projections write",
            )
        if chunk < 1:
            raise LoudError(
                f"chunk={chunk} holds no tokens",
                "a block must hold at least one position; 16 is the recipe",
            )
        if topk < 1:
            raise LoudError(
                f"topk={topk} keeps no past block",
                "topk is the number of past blocks a query may read; use >= 1",
            )
        if hca_group < 1:
            raise LoudError(
                f"hca_group={hca_group} groups no blocks",
                "use 1 for plain CSA, or the number of fine blocks per coarse block",
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk = chunk
        self.topk = topk
        self.hca_group = hca_group
        self.d_kv = n_heads * head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * self.d_kv, bias=False, device=device)
        #: One number per head: how much a kept block's selector score moves the
        #: logits, and therefore the only path by which the selector learns.
        self.selector_gain = nn.Parameter(torch.full((n_heads,), 0.1, device=device))
        #: `d_kv x d_model`, as in `kda.py`: the heads need not tile the stream.
        self.proj = nn.Linear(self.d_kv, d_model, bias=False, device=device)
        nn.init.normal_(self.qkv.weight, std=PROJ_STD)
        nn.init.normal_(self.proj.weight, std=write_std)

    def coeffects(self) -> Coeffects:
        """Declare the selector this mechanism needs and provides."""
        return self.COEFFECTS

    def extra_repr(self) -> str:
        """Print the shape of the sparsity, not the weights."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, chunk={self.chunk}, topk={self.topk}, "
            f"hca_group={self.hca_group}"
        )

    # --------------------------------------------------------------- the blocks
    def select(
        self, q: torch.Tensor, k: torch.Tensor, *, return_aux: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Return kept block ids `[b, s, h, keep]` and their selector scores.

        Public because the selector is the part worth measuring on its own: a
        query with a needle in one block must keep that block far above the
        `topk / n_blocks` chance rate, and no amount of end-to-end loss shows it.

        `return_aux=True` also returns the coarse bias and its pre-softmax
        scores, which is how the coarse-learning test observes the gradient of
        a block the fine selector did not keep.
        """
        batch, seq, heads, width = q.shape
        chunk, group = self.chunk, self.hca_group
        if seq % (chunk * group):
            raise LoudError(
                f"a {seq}-token selection leaves {seq % (chunk * group)} tokens "
                f"outside chunk={chunk} x hca_group={group}",
                f"pad the window to a multiple of {chunk * group}, or build CSA "
                "with a chunk and group that divide it",
            )
        blocks = seq // chunk
        means = k.view(batch, blocks, chunk, heads, width).mean(2)
        score = torch.einsum("bshd,bnhd->bshn", q, means) * self.scale
        position = torch.arange(seq, device=q.device) // chunk
        index = torch.arange(blocks, device=q.device)
        eligible = position.view(1, seq, 1, 1) >= index.view(1, 1, 1, blocks)
        if group > 1:
            coarse = means.view(batch, blocks // group, group, heads, width).mean(2)
            cscore = torch.einsum("bshd,bnhd->bshn", q, coarse) * self.scale
            coarse_index = torch.arange(blocks // group, device=q.device)
            #: Only strictly past coarse blocks are ranked. The query's own
            #: coarse block is kept unconditionally below: its mean looks into
            #: the future, so its score may not affect an earlier output.
            cel = position.view(1, seq, 1, 1) >= ((coarse_index + 1) * group).view(1, 1, 1, -1)
            keep_coarse = min(coarse_index.numel(), max(1, -(-self.topk // group)))
            picked = cscore.masked_fill(~cel, -torch.inf).topk(keep_coarse, dim=-1)
            chosen = torch.zeros(
                batch, seq, heads, coarse_index.numel(), dtype=torch.bool, device=q.device
            ).scatter_(-1, picked.indices, True)
            own_coarse = (position // group).view(1, seq, 1, 1).expand(batch, seq, heads, 1)
            chosen.scatter_(-1, own_coarse, True)
            eligible = eligible & chosen.repeat_interleave(group, dim=-1)
            #: Straight-through top-k: forward keeps the hard choice, backward
            #: flows through a softmax over every eligible coarse block. Without
            #: it the coarse selector sees only the blocks it already kept and
            #: can never learn that it should have kept another one.
            coarse_bias = self._coarse_bias(cscore, cel, picked)
            score = score + coarse_bias.repeat_interleave(group, dim=-1)
        else:
            coarse_bias = None
            cscore = None
        keep = min(blocks, self.topk + 1)
        own = position.view(1, seq, 1, 1).expand(batch, seq, heads, 1)
        ranked = score.scatter(-1, own, torch.inf).masked_fill(~eligible, -torch.inf)
        ids = ranked.topk(keep, dim=-1).indices
        picked_score = score.gather(-1, ids)
        #: `topk` over `-inf` ties is not a defined order; point the unused slots
        #: at the current block, whose duplicate contributes exactly its own mass.
        dead = torch.isneginf(ranked.gather(-1, ids))
        ids = torch.where(dead, own.expand_as(ids), ids)
        picked_score = torch.where(dead, torch.zeros_like(picked_score), picked_score)
        if return_aux:
            return ids, picked_score, dead, coarse_bias, cscore
        return ids, picked_score, dead

    def _coarse_eligibility(self, seq: int, device: torch.device | str) -> torch.Tensor:
        """Whether a coarse block is strictly past the query, `[1, seq, 1, n_coarse]`."""
        chunk, group = self.chunk, self.hca_group
        blocks = seq // chunk
        position = torch.arange(seq, device=device) // chunk
        coarse_index = torch.arange(blocks // group, device=device)
        return position.view(1, seq, 1, 1) >= ((coarse_index + 1) * group).view(1, 1, 1, -1)

    def _coarse_bias(
        self, cscore: torch.Tensor, cel: torch.Tensor, picked: torch.Tensor
    ) -> torch.Tensor:
        """Straight-through coarse selection, spread to coarse blocks only.

        Forward is the hard top-k: only the picked coarse blocks are nonzero.
        Backward is a softmax over every eligible coarse block, so an unselected
        block receives the gradient that says "this one should have been kept".
        """
        logits = cscore.masked_fill(~cel, -torch.inf)
        soft = torch.softmax(logits, dim=-1)
        #: A query in the first coarse block has no strictly past coarse block;
        #: `softmax` over all `-inf` is NaN and would poison every fine logit.
        soft = torch.nan_to_num(soft, nan=0.0)
        kept = soft.gather(-1, picked.indices)
        straight = 1.0 + kept - kept.detach()
        bias = torch.zeros_like(cscore)
        return bias.scatter(-1, picked.indices, straight)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the attention write, same shape."""
        batch, seq, _ = x.shape
        span = self.chunk * self.hca_group
        if seq % span:
            raise LoudError(
                f"a {seq}-token forward leaves {seq % span} tokens outside "
                f"chunk={self.chunk} x hca_group={self.hca_group}",
                f"pad the window to a multiple of {span}, or build CSA with a "
                "chunk and group that divide it",
            )
        shape = (batch, seq, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape) for z in self.qkv(x).split(self.d_kv, dim=-1))
        ids, picked, dead = self.select(q, k)
        chunk = self.chunk
        place = torch.arange(chunk, device=x.device)
        at = (ids.unsqueeze(-1) * chunk + place).reshape(batch, seq, self.n_heads, -1)
        kept = at.shape[-1]
        source = k.transpose(1, 2).unsqueeze(-2).expand(batch, self.n_heads, seq, kept, -1)
        index = (
            at.transpose(1, 2).unsqueeze(-1).expand(batch, self.n_heads, seq, kept, self.head_dim)
        )
        keys = torch.gather(source, 2, index)
        values = torch.gather(v.transpose(1, 2).unsqueeze(-2).expand_as(source), 2, index)
        logits = torch.einsum(
            "bhsd,bhskd->bhsk", q.transpose(1, 2), keys
        ) * self.scale + self._bias(ids, picked, x)
        visible = at.transpose(1, 2) <= torch.arange(seq, device=x.device).view(1, 1, seq, 1)
        #: A dead slot aliases the own block for indexing only. It must not
        #: contribute a second copy of that block's mass to the softmax.
        visible = visible & ~dead.transpose(1, 2).repeat_interleave(chunk, dim=-1)
        weights = torch.softmax(logits.masked_fill(~visible, -torch.inf), dim=-1)
        out = torch.einsum("bhsk,bhskd->bhsd", weights, values).transpose(1, 2)
        y = self.proj(out.reshape(batch, seq, self.d_kv))
        if counters is not None:
            counters.count("csa_fwd", batch * seq)
            counters.count("csa_selector", batch * seq * self.n_heads)
            counters.count("csa_kept", int((~dead).sum()))
            if y.requires_grad:
                y.register_hook(lambda _: counters.count("csa_bwd", batch * seq))
        return cast(torch.Tensor, y)

    def _bias(
        self,
        ids: torch.Tensor,
        picked: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Spread each kept block's selector score over its tokens, current block 0.

        The current block is zeroed because its mean key was computed from
        positions later than the query: letting that score into the logits would
        make a token's output depend on tokens it may not read.
        """
        chunk = self.chunk
        spread = picked.transpose(1, 2).repeat_interleave(chunk, dim=-1)
        own = ids.transpose(1, 2) == (torch.arange(x.shape[1], device=x.device) // chunk).view(
            1, 1, -1, 1
        )
        #: With every block kept the selector bias is not a sparse choice and
        #: dense attention has no such term; keeping it would make the topk=all
        #: anchor differ from dense attention by construction.
        if self.topk >= x.shape[1] // chunk:
            return torch.zeros_like(spread)
        gain = self.selector_gain.view(1, self.n_heads, 1, 1)
        return (gain * spread).masked_fill(own.repeat_interleave(chunk, dim=-1), 0.0)
