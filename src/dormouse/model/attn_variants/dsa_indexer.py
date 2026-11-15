"""The lightning indexer: one scalar per cached token, and the prefix top-k over it.

WHAT IT IS (the indexer of DeepSeek Sparse Attention, DeepSeek-V3.2 §2.1; GLM-5
runs the same shape — the paper that describes it is 2611.03768 §2):

    c_s     = ReLU(W_idx x_s)                  W_idx: R^d -> R^d_idx
    I_s     = <u, c_s>                         u in R^d_idx, one vector per layer
    S(t)    = top-k { s <= t : I_s }           the tokens query t reads exactly

`c_s` is the small projection, `I_s` is one cached token's score, and `S(t)` is a
PREFIX top-k: a query can never select a position after itself, by construction
rather than by a mask applied afterwards.

THE COST IS THE POINT. Scoring the window costs `T·d_idx` MACs and the projection
`T·d·d_idx`: both linear in T. A per-query scan of every eligible position is
`T²·head_dim`, quadratic in the selector: at T=1M and head_dim=64 that is 6.4e16
MAC in the selector alone, two orders of magnitude more than the attention it
chooses.

THE DEPARTURE FROM THE PAPER, SAID OUT LOUD. V3.2 scores every PAIR,
`I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_{s,j})`, which is `T²·d_idx` MACs, and
that pairwise form is exactly what makes its selection query-dependent. Here the
query factor is folded into the layer's vector `u`, so every query of a layer
shares one selection and the cost is `T·d_idx`. The two properties cannot both
hold: a per-query score for every cached token is a `T x T` matrix by definition.
What is kept is exactness (the k highest-scoring eligible tokens, no blocks, no
approximation); what is given up is per-query choice; the local window in
`dsa.py` re-introduces per-query locality over the last `w` positions.
`ponytail:` a pairwise indexer costs `T²·d_idx` (1.3e14 MAC at T=1M, d_idx=128,
three orders over the layer's own bill) — swap `scores` for the pairwise form and
keep `select`'s prefix invariant if a prefill budget ever pays for it.

EXACT PREFIX TOP-K IN LINEAR TIME. A per-query `topk` over the whole window would
materialise `T²` scores, which is the cost this file exists to avoid. Instead the
window is walked in chunks of `chunk` queries and the candidate set for a chunk is
(previous chunk's last-query top-k) ⊎ (this chunk). A token outside that carry
scores no better than the carry's k-th, and all k carry tokens are visible to
every query of this chunk and later, so nothing outside the candidate set can be
in the prefix top-k: the result is EXACT, at `O(T·(k + chunk))` comparisons and
`O(chunk·(k + chunk))` memory.

MLI: REUSE MUST BE MEASURED, NOT ASSUMED. The paper's multi-layer indexer reuse
claims adjacent layers select nearly the same tokens. `overlap` measures it (the
mean fraction of the k slots two selections share; chance is `k/T`), and `reuse`
REFUSES a previous selection whose measured overlap is under `MLI_MIN_OVERLAP`
instead of silently degrading the layer. A run cannot re-derive the overlap
without paying for the scan it is avoiding, so the floor is applied where both
selections are in hand: a calibration pass, a debug run, or a test.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..blocks import PROJ_STD

#: The overlap under which reuse is refused. Chance for a k-of-T selection is
#: `k/T` (0.125 at k=512, T=4096); half the slots is the floor at which "the two
#: layers are not selecting the same tokens" is unambiguous rather than noise.
MLI_MIN_OVERLAP = 0.5


class LightningIndexer(nn.Module):
    """`ReLU(W_idx x)` per token, one score per cached token, exact prefix top-k."""

    def __init__(
        self,
        d_model: int,
        d_idx: int = 128,
        *,
        chunk: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the projection and the layer's scoring vector; refuse empty widths."""
        super().__init__()
        if d_model < 1:
            raise LoudError(
                f"d_model={d_model} is not a residual width",
                "pass the width the stream is written at",
            )
        if d_idx < 1:
            raise LoudError(
                f"d_idx={d_idx} projects to nothing: an indexer with no channels cannot "
                "separate a needle from a haystack",
                "use a positive bottleneck, 128 is the V3.2 recipe",
            )
        if chunk < 1:
            raise LoudError(
                f"chunk={chunk} holds no query",
                "the chunk is how many queries share one candidate set; use >= 1",
            )
        self.d_model = d_model
        self.d_idx = d_idx
        self.chunk = chunk
        self.key = nn.Linear(d_model, d_idx, bias=False, device=device)
        #: `u`: the scoring vector. It is a parameter and not a second projection
        #: because the score must stay one number per cached token (see the module
        #: docstring: a query-dependent score is the `T x T` form).
        self.weight = nn.Parameter(torch.empty(d_idx, device=device))
        nn.init.normal_(self.key.weight, std=PROJ_STD)
        nn.init.normal_(self.weight, std=PROJ_STD)

    def extra_repr(self) -> str:
        """Print the bottleneck, not the matrix."""
        return f"d_model={self.d_model}, d_idx={self.d_idx}, chunk={self.chunk}"

    def macs_per_token(self) -> int:
        """Return the indexer's multiplies per token: the key projection and one dot."""
        return self.key.in_features * self.key.out_features + self.d_idx

    # ------------------------------------------------------------------ scoring
    def scores(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d]` → one score per cached token, `[batch, seq]`.

        `T·d_idx` MACs: this is the implementation of the indexer's cost claim, and
        `dsa_index_macs` is counted from it (not from a formula).
        """
        batch, seq, _ = x.shape
        scored = torch.relu(self.key(x)) @ self.weight
        if counters is not None:
            counters.count("dsa_index_macs", batch * seq * self.d_idx)
        return scored

    # --------------------------------------------------------------- selection
    def select(self, x: torch.Tensor, k: int, counters: Counters | None = None) -> torch.Tensor:
        """Return `S(t)` for every query: `[batch, seq, k]` positions, ascending.

        Ascending order is not cosmetic: it is the layout the dense anchor in
        `DSA.dense` uses, so `window >= T` makes the two bit-identical.
        """
        batch, seq, _ = x.shape
        if k < 1:
            raise LoudError(
                f"k={k} selects nothing",
                "k is the per-query budget; use at least 1 (512 is the recipe)",
            )
        if k > seq:
            raise LoudError(
                f"k={k} > seq={seq}: no query has that many eligible tokens",
                "lower k, or run a longer window",
            )
        scored = self.scores(x, counters)
        span = min(self.chunk, seq)
        chosen = torch.zeros(batch, seq, k, dtype=torch.long, device=x.device)
        #: The carry starts empty (-inf), so the first chunk's top-k is the prefix
        #: top-k it would compute anyway.
        carry = torch.zeros(batch, k, dtype=torch.long, device=x.device)
        carry_score = x.new_full((batch, k), -torch.inf)
        for start in range(0, seq, span):
            stop = min(start + span, seq)
            here = torch.arange(start, stop, device=x.device)
            cand = torch.cat(
                [carry, here.view(1, -1).expand(batch, -1)],
                dim=-1,
            )
            cand_score = torch.cat([carry_score, scored[:, start:stop]], dim=-1)
            eligible = cand.unsqueeze(1) <= here.view(1, -1, 1)
            ranked = cand_score.unsqueeze(1).masked_fill(~eligible, -torch.inf)
            picked = ranked.topk(k, dim=-1)
            alive = torch.isfinite(ranked.gather(-1, picked.indices))
            ids = torch.where(
                alive,
                torch.gather(cand.unsqueeze(1).expand(-1, stop - start, -1), -1, picked.indices),
                here.view(1, -1, 1),
            )
            chosen[:, start:stop] = ids.sort(-1).values
            #: The last query's selection is the carry: every later query sees all
            #: of it, and anything it dropped scores worse than its k-th.
            carry = chosen[:, stop - 1]
            carry_score = scored.gather(-1, carry)
        return chosen

    # --------------------------------------------------------------------- MLI
    def overlap(self, prev: torch.Tensor, own: torch.Tensor) -> float:
        """Mean fraction of the k slots two selections share, `[batch, seq, k]` each.

        Chance for a k-of-T selection is `k/T`. Both tensors are expected to hold
        `k` real positions (no padding), which is the case whenever `T >= k`.
        """
        width = prev.shape[-1]
        hits = (prev.unsqueeze(-1) == own.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        return float(hits.float().mean() / width)

    def reuse(
        self,
        prev: torch.Tensor,
        *,
        k: int,
        own: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the previous layer's selection, refusing an unusable one.

        Refused: the wrong budget, a position after its query, and — when `own` is
        supplied — an overlap under `MLI_MIN_OVERLAP`. The last one is the paper's
        premise checked instead of assumed: two layers that select different tokens
        cannot share a selection without changing what the second layer reads.
        """
        if prev.ndim != 3:
            raise LoudError(
                f"MLI prev has shape {tuple(prev.shape)}: a selection is [batch, seq, k]",
                "pass the tensor `select` (or `forward(..., prev=...)`) returned",
            )
        _batch, seq, width = prev.shape
        if width != k:
            raise LoudError(
                f"MLI prev keeps {width} tokens, this layer's k is {k}",
                "reuse is a saving only when the budget matches; give every layer of "
                "the group the same k",
            )
        future = int((prev > torch.arange(seq, device=prev.device).view(1, seq, 1)).sum())
        if future:
            raise LoudError(
                f"MLI prev has {future} slots pointing after their query",
                "a reused selection must be causal too; check the layer it came from",
            )
        if own is not None:
            rate = self.overlap(prev, own)
            if rate < MLI_MIN_OVERLAP:
                raise LoudError(
                    f"MLI refused: overlap {rate:.3f} is under the {MLI_MIN_OVERLAP} floor "
                    f"(chance is k/T = {k / seq:.3f})",
                    "the adjacent layers are selecting different tokens, so reusing this "
                    "selection buys a saving and pays for it in quality; let this layer scan",
                )
        return prev.sort(-1).values
