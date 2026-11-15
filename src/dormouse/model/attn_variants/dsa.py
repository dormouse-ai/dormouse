"""DeepSeek sparse attention: exact attention over k selected tokens plus a window.

THE MECHANISM (DeepSeek Sparse Attention, DeepSeek-V3.2 §2.1; the same layer is in
GLM-5, whose indexer 2611.03768 §2 describes):

    q, k, v = W_qkv x                                the heads full attention uses
    S(t)    = LightningIndexer(x, k)                 prefix top-k, `dsa_indexer.py`
    W(t)    = { max(0, t-w+1), ..., t }              the local window, always kept
    o_t     = softmax over S(t) plus W(t) of q_t·k_s/√d_k · v_s, for s <= t only
    y       = W_proj · o

`o_t` is EXACT attention — no block summary, no mean key, no approximation of the
softmax — over at most `k + w` positions instead of `t`. That is the whole mechanism:
the selector chooses, and the arithmetic over what it chose is the arithmetic dense
attention would have done.

WHY THIS AND NOT DENSE ATTENTION. The bill per attention layer at T=1M, as this
stack is priced (d_model=832, 8 heads of 64):

    layer           MAC per layer       linear in T?
    dense           512e12              no (T²) — 371% of the whole FFN
    KDA             69e9                yes (bounded state)
    DSA, k=512      550e9               yes (T·k·d_kv + T·d_idx)

KDA alone has no long-range lookup: its state is a rolling memory of bounded size,
which is why the 3:1 stack keeps one exact-attention layer per period. DSA is the
partner that makes that layer affordable at 1M instead of impossible.

THE WINDOW IS NOT AN OPTIMISATION. The indexer selects by content, and content is
not locality: a token whose recent context is unremarkable can be dropped from
every query that follows it, including the query one position later. Without the
window the layer cannot read the tokens it just wrote, and the needle gate in
`tests/gates/` measures exactly that collapse. `k < window` is refused because a
budget under the window leaves the selection nothing to choose.

CAUSALITY is structural: `S(t)` is a prefix top-k (a position after `t` is not a
candidate, not a masked candidate), the window ends at `t`, and every gathered key
is checked against its own query before the softmax.

THE SELECTION PATHS ARE THE ONLY DIFFERENCE. `dense()` runs the same projections and
the same `_attend` with the selection replaced by a contiguous prefix; with
`window >= T` the window alone covers every visible position, so a sparse forward
must equal `dense()` bit for bit. That equality is what proves the sparse path adds
no arithmetic of its own — only an index list.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.coeffects import Coeffects
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..blocks import PROJ_STD
from .dsa_indexer import LightningIndexer
from .projections import LatentProjections

#: Per-query window, in tokens. The window is what carries locality that a
#: content score cannot; 128 is a quarter of the k=512 budget.
WINDOW = 128

#: Selected tokens per query. V3.2's 2048 is a serving budget; 512 is the one this
#: stack's 4k/32k pricing is written at.
K = 512


class DSA(LatentProjections):
    """Sparse attention: exact over the indexer's top-k plus the local window."""

    #: What this mechanism needs and provides. `attn.state` is exclusive, so a
    #: layer cannot be KDA and DSA at once; different layers is the hybrid.
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
        n_kv_heads: int | None = None,
        c_lat: int | None = None,
        k: int = K,
        window: int = WINDOW,
        d_idx: int = 128,
        chunk: int = 256,
        mli: bool = False,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the projections and the indexer; refuse a budget the window eats.

        `n_kv_heads`/`c_lat` are the GQA and bottleneck widths (`projections.py`);
        their degenerate setting is the pre-change packed `qkv`. `mli=True` is a
        declaration, not a performance hint: the layer will refuse a forward that has
        no previous selection to reuse, because the alternative (silently scanning)
        is MLI being off while the run reports it on.
        """
        super().__init__(
            d_model,
            n_heads,
            head_dim,
            n_kv_heads=n_kv_heads,
            c_lat=c_lat,
            device=device,
        )
        if k < 1:
            raise LoudError(
                f"k={k} selects nothing",
                "k is the tokens a query reads from the indexer; 512 is the recipe",
            )
        if window < 1:
            raise LoudError(
                f"window={window} keeps no recent token",
                "the window is the layer's locality; use at least 1 (128 is the recipe)",
            )
        if k < window:
            raise LoudError(
                f"k={k} < window={window}: the budget is inside the window, so the "
                "indexer would select tokens the layer already reads",
                "keep k >= window, or shrink the window to what the selection pays for",
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k = k
        self.window = window
        self.mli = mli
        self.scale = head_dim**-0.5
        #: `d_kv x d_model`: the heads need not tile the stream (KDA does the same).
        self.proj = nn.Linear(self.d_kv, d_model, bias=False, device=device)
        self.indexer = LightningIndexer(d_model, d_idx, chunk=chunk, device=device)
        nn.init.normal_(self.proj.weight, std=write_std)

    def coeffects(self) -> Coeffects:
        """Declare the selector this mechanism needs and provides."""
        return self.COEFFECTS

    def extra_repr(self) -> str:
        """Print the sparsity and the indexer's width, not the weights."""
        return (
            f"k={self.k}, window={self.window}, "
            f"d_idx={self.indexer.d_idx}, mli={self.mli}, " + super().extra_repr()
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        *,
        prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the attention write, same shape.

        `prev` is the layer below's selection `[batch, seq, k]`; when it is given the
        indexer does not run and `dsa_index_macs` stays at zero. `mli=True` without
        it is refused rather than absorbed.
        """
        batch, seq, _ = x.shape
        if self.k >= seq and self.window < seq:
            raise LoudError(
                f"k={self.k} >= seq={seq}: every eligible token is selected, so this "
                "forward is dense attention with extra steps",
                "lower k to the budget you meant; with window >= T the layer IS the dense "
                "control, and dense() runs it under that name",
            )
        if self.mli and prev is None:
            raise LoudError(
                "mli=True but no previous selection was passed",
                "MLI reuses the layer below: pass prev=..., or build the first layer of "
                "the group with mli=False so it scans",
            )
        q, key, value = self._project(x)
        at, visible = self.attended(x, counters, prev=prev)
        reused = 0 if prev is None else batch * seq
        y = self._attend(q, key, value, at, visible)
        if counters is not None:
            counters.count("dsa_fwd", batch * seq)
            counters.count("dsa_selected", batch * seq * self.k)
            if reused:
                counters.count("dsa_mli_reuse", reused)
            if y.requires_grad:
                y.register_hook(  # type: ignore[no-untyped-call]
                    lambda _: counters.count("dsa_bwd", batch * seq)
                )
        return y

    def attended(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        *,
        prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the positions each query reads and their visibility, both `[b, s, w+k]`.

        Public because the read set is the part worth measuring on its own: whether
        the needle is in it, whether any position is after its query, and whether the
        window alone already covers the sequence are questions about these two
        tensors, and no end-to-end loss answers them.
        """
        tail = (
            self.indexer.select(x, self.k, counters)
            if prev is None
            else self.indexer.reuse(prev, k=self.k)
        )
        return self._layout(tail, x.shape[1])

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        """Read every visible position with the indexer bypassed: the control arm.

        It reads `window + k` positions, so it is the whole visible window exactly
        when `window >= seq` — which is the configuration the degeneracy gate
        compares against a sparse forward, bit for bit.
        """
        batch, seq, _ = x.shape
        q, key, value = self._project(x)
        tail = torch.arange(self.k, device=x.device).view(1, 1, self.k).expand(batch, seq, self.k)
        at, visible = self._layout(tail, seq)
        return self._attend(q, key, value, at, visible)

    # -------------------------------------------------------------- internals
    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x` → `q, k, v` at `[batch, seq, heads, head_dim]`.

        GQA: a shared key/value head is broadcast to the query heads that read it,
        because the kernel keeps one score row per QUERY head either way. Off by
        default (`n_kv_heads == n_heads`), which is what keeps the uncompressed
        layer bit-identical to the pre-change one.
        """
        q, k, v = self.split(self.project(x))
        groups = self.n_heads // self.n_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=2)
            v = v.repeat_interleave(groups, dim=2)
        return q, k, v

    def _layout(self, tail: torch.Tensor, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `at [b, s, w+k]` and the visibility mask, window first.

        The window comes first and the selected tail after it, both ascending, so
        two layouts that read the same positions are the same tensor and the
        arithmetic over them is identical. Early queries pad the window (clamped at
        0) and the tail keeps its own slots marked invisible; a padded slot is
        `-inf` in the softmax, which contributes exactly nothing.
        """
        batch = tail.shape[0]
        span = torch.arange(seq, device=tail.device).view(seq, 1) - self.window + 1
        win = span + torch.arange(self.window, device=tail.device)
        alive = win >= 0
        at = torch.cat([win.clamp_min(0).expand(batch, -1, -1), tail], dim=-1)
        ends = torch.arange(seq, device=tail.device).view(1, seq, 1) - self.window
        visible = torch.cat([alive.expand(batch, -1, -1), tail <= ends], dim=-1)
        return at, visible

    def _attend(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        at: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Exact causal attention over the gathered positions, then the write."""
        batch, seq, heads, width = q.shape
        keep = at.shape[-1]
        place = at.unsqueeze(1).unsqueeze(-1).expand(batch, heads, seq, keep, width)
        keys = torch.gather(
            key.transpose(1, 2).unsqueeze(-2).expand(batch, heads, seq, keep, width), 2, place
        )
        values = torch.gather(
            value.transpose(1, 2).unsqueeze(-2).expand(batch, heads, seq, keep, width), 2, place
        )
        logits = torch.einsum("bhsd,bhskd->bhsk", q.transpose(1, 2), keys) * self.scale
        weights = torch.softmax(logits.masked_fill(~visible.unsqueeze(1), -torch.inf), dim=-1)
        out = torch.einsum("bhsk,bhskd->bhsd", weights, values).transpose(1, 2)
        y = self.proj(out.reshape(batch, seq, self.d_kv))
        return cast(torch.Tensor, y)

    def macs_per_token(self, seq_len: int) -> int:
        """Return the layer's multiplies per token: projections, indexer, kernel.

        The kernel reads at most `k + window` positions per query (`_layout`), and
        `_attend` multiplies each of them twice: q·k and the weighted sum of v.
        """
        keep = min(seq_len, self.k + self.window)
        kernel = 2 * self.n_heads * self.head_dim * keep
        return super().macs_per_token() + self.indexer.macs_per_token() + kernel
