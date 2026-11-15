"""The stack: token table → L pre-norm blocks → RMSNorm → tied head.

    logits = W_emb · norm_L+1( block_L( … block_1( W_emb[ids] ) … ) )

Every parameterisation choice below is stated with its reason, because each one is a
number a later measurement will be compared against and a silent default here is a
silent difference in every run.

TIED HEAD. The logits' matrix IS the token table, transposed. At `vocab=256` a head
of its own is 0.2M parameters (0.22% of a 12-layer d=832 model) that learn "predict
byte b" separately from "read byte b" — for a 256-row table the two gradients belong
in one matrix (GPT-2's choice at exactly this scale), and an untied head buys nothing
a measurement here has shown.

PLAIN TOKEN TABLE, NOT FACTORISED. `vocab·d = 212 992` is 0.22% of the model; the
`U·Vᵀ` factorisation that pays at 32k tokens would turn a 213k-parameter lookup into
a rank bottleneck to save 100k, which is a rank cut paid for with model quality and
no parameter budget returned. When the tokenizer moves to 32k, this is the line that
changes, and it is one line.

INITIALISATION. Token rows and every projection draw `N(0, 0.02²)`, except the two
projections that write the residual (`attn.proj`, `ffn.down`), which draw
`N(0, (0.02/√(2L))²)`; see `blocks.Block`. The variance argument: a pre-activation of
width `d` has variance `d·0.02² ≈ 0.33` at d=832 — a third of the unit-RMS stream it
is added to, so the residual starts as the signal and the deltas start loud enough to
learn; and the `1/√(2L)` on the writes keeps the stream's variance `≈ 1.5x` its init
at any depth instead of growing with L. Token rows are `N(0, 1)` because RMSNorm is
defined against a unit-RMS input: the first block's pre-norm is near-identity at step
0, which is the state every anchor below is written against.

ROPE. The live recipe is a byte-level model over a 1024-token window with no
positional prior in the tokenizer; without rotations every position sees the same
multiset of bytes (measured: loss 4.28 → 16.4). `Attention` owns the formula; this
class only decides that the stack uses it, and `rope=True` is the recipe.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .blocks import EPS, Block, RMSNorm


class Transformer(nn.Module):
    """A decoder-only transformer with the live recipe's defaults."""

    def __init__(
        self,
        *,
        vocab: int = 256,
        d_model: int = 832,
        n_layers: int = 12,
        n_heads: int = 13,
        head_dim: int = 64,
        d_ffn: int = 2048,
        max_seq_len: int = 1024,
        rope: bool = True,
        eps: float = EPS,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the table, the stack, the final norm and the (tied) head."""
        super().__init__()
        if n_heads * head_dim != d_model:
            raise LoudError(
                f"n_heads·head_dim = {n_heads}·{head_dim} = {n_heads * head_dim} "
                f"≠ d_model={d_model}",
                "the split into heads must tile the residual width exactly",
            )
        if vocab < 2:
            raise LoudError(
                f"vocab={vocab}: a model that predicts one symbol predicts nothing",
                "byte-level is 256",
            )
        if n_layers < 1:
            raise LoudError(
                f"n_layers={n_layers} builds no blocks",
                "a stack needs at least one layer",
            )
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ffn = d_ffn
        self.max_seq_len = max_seq_len
        self.rope = rope
        self.embed = nn.Embedding(vocab, d_model, device=device)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    head_dim,
                    d_ffn,
                    n_layers=n_layers,
                    rope=rope,
                    max_seq_len=max_seq_len,
                    eps=eps,
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps, device=device)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`ids [batch, seq]` token ids → `logits [batch, seq, vocab]`."""
        if ids.dim() != 2:
            raise LoudError(
                f"forward takes ids [batch, seq], got shape {tuple(ids.shape)}",
                "flatten a longer input to two axes; the stack carries no batch-in-batch",
            )
        if ids.is_floating_point():
            raise LoudError(
                "token ids are integers",
                "cast the batch to torch.long before the forward",
            )
        if counters is not None:
            self._enable(counters)
        h = self.embed(ids)
        if counters is not None:
            counters.count("embed_fwd", ids.numel())
        for block in self.blocks:
            h = block(h, counters)
        logits = self.head(self.norm(h, counters), counters)
        if counters is not None:
            counters.check()
        return logits

    def head(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Logits from the token table, transposed: no parameter of its own.

        The counter is the proof that the tying is live: a head that quietly grew its
        own matrix would report `head_fwd` and never `tied_head`, which is the defect
        class `dormouse_core.counters` exists for.
        """
        if counters is not None:
            counters.count("head_fwd", x.shape[0] * x.shape[1])
            counters.count("tied_head")
        return nn.functional.linear(x, self.embed.weight)

    def _enable(self, counters: Counters) -> None:
        """Declare the mechanisms this stack runs, so silence becomes an error.

        Idempotent, and called at the top of every `forward`: `Counters.check()` at
        the end of the same call is then a statement about THIS forward — a mechanism
        that was enabled and never counted raises `MissingReport` before the step's
        loss is ever returned.
        """
        counters.enable("embedding", counter="embed_fwd", why="the token table is read")
        counters.enable("blocks", counter="block_fwd", why="the residual stack runs")
        counters.enable("attention", counter="attn_fwd", why="every block attends")
        counters.enable("ffn", counter="ffn_fwd", why="every block has a gated FFN")
        counters.enable("norm", counter="norm_fwd", why="two pre-norms per block + final")
        counters.enable("head", counter="head_fwd", why="logits come from the tied head")
        counters.enable("tied_head", counter="tied_head", why="the head has no parameter")
        if self.rope:
            counters.enable("rope", counter="rope_fwd", why="byte-level 1024 window, no prior")

    # ------------------------------------------------------------------- report
    def param_report(self) -> dict[str, int]:
        """Parameters by group, counted from the module tree — never guessed.

        The groups are the ones a budget is argued in: `embedding`, `attention`,
        `ffn`, `norms`, `head` (0 while the head is tied, which is the point of
        printing it), and `total`. `total` is summed from `parameters()`, so a group
        that forgets a parameter shows up as a disagreement instead of a lost number.
        """
        groups = dict.fromkeys(("embedding", "attention", "ffn", "norms", "head"), 0)
        for name, param in self.named_parameters():
            if name.startswith("embed."):
                key = "embedding"
            elif name == "norm.weight" or ".norm" in name:
                key = "norms"
            elif ".attn." in name:
                key = "attention"
            elif ".ffn." in name:
                key = "ffn"
            else:
                key = "head"
            groups[key] += param.numel()
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups
