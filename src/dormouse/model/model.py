"""The stack: token table → L pre-norm blocks → RMSNorm → tied head.

    logits = W_emb · norm_L+1( block_L( … block_1( W_emb[ids] ) … ) )

Every parameterisation choice below is stated with its reason, because each one is a
number a later measurement will be compared against and a silent default here is a
silent difference in every run.

NO DEFAULT BLOCKS. The blocks come prebuilt from `assemble.assemble`, which is the
one place that knows which attention mechanism a layer holds. A shell that also
built its own would be a second model, and the two would disagree the first time a
mechanism moved.

TIED HEAD. The logits' matrix IS the token table, transposed. At the tokenizer's
32 768 rows a head of its own is 27M parameters (2.6% of a 1B model) that learn
"predict token t" separately from "read token t"; for a table this size the two
gradients belong in one
matrix, and the head's own bill stays 0 so a translation of the tying to a second
matrix shows up as a number that moved.

PLAIN TOKEN TABLE, NOT FACTORISED. `vocab·d = 27.3M` is 2.6% of the model; the
`U·Vᵀ` factorisation would turn a lookup into a rank bottleneck to save half of it,
which is a rank cut paid for with model quality and no parameter budget returned.

INITIALISATION. Token rows and every projection draw `N(0, 0.02²)`, except the two
projections that write the residual (`attn.proj`, `ffn.down`), which draw
`N(0, (0.02/√(2L))²)`; see `blocks.Block`. The variance argument: a pre-activation of
width `d` has variance `d·0.02² ≈ 0.33` at d=832 — a third of the unit-RMS stream it
is added to, so the residual starts as the signal and the deltas start loud enough to
learn; and the `1/√(2L)` on the writes keeps the stream's variance `≈ 1.5x` its init
at any depth instead of growing with L. Token rows are `N(0, 1)` because RMSNorm is
defined against a unit-RMS input: the first block's pre-norm is near-identity at step
0, which is the state every anchor below is written against.

NO ROTARY TABLE. Neither attention mechanism uses one (KDA's decay and DSA's
selection are their own positional signal), so the stack has no `rope` switch and no
position table to disagree with a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..data.tokenizer import DEFAULT_VOCAB
from .blocks import EPS, RMSNorm
from .residual import Residual


class Transformer(nn.Module):
    """A decoder-only transformer over a stack of prebuilt blocks."""

    def __init__(
        self,
        *,
        vocab: int = DEFAULT_VOCAB,
        d_model: int = 832,
        d_ffn: int = 6144,
        max_seq_len: int = 1024,
        eps: float = EPS,
        device: torch.device | str | None = None,
        blocks: Sequence[nn.Module],
        declarations: Sequence[tuple[str, str, str]],
        mixer: Residual | None = None,
    ) -> None:
        """Build the table, the stack, the final norm and the (tied) head.

        `mixer` owns the residual stream when the spec names one; `None` is the
        block's own two adds and leaves the forward exactly as it was.
        """
        super().__init__()
        if vocab < 2:
            raise LoudError(
                f"vocab={vocab}: a model that predicts one symbol predicts nothing",
                "BPE takes the tokenizer's vocabulary (data.tokenizer.DEFAULT_VOCAB); "
                "a byte model has 256 symbols and is the data layer's arm",
            )
        if d_model < 1 or d_ffn < 1:
            raise LoudError(
                f"the stack needs positive widths, got d_model={d_model}, d_ffn={d_ffn}",
                "pass the residual and FFN widths the blocks were built at",
            )
        if not blocks:
            raise LoudError(
                "the stack needs at least one block",
                "build them with assemble.assemble: a block's attention arm is not a default",
            )
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = len(blocks)
        self.d_ffn = d_ffn
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab, d_model, device=device)
        nn.init.normal_(self.embed.weight, std=1.0)
        #: `declarations` is the list of mechanisms whose counters prove this stack
        #: ran; the constructor cannot infer it from the modules, and a mechanism the
        #: forward cannot count is a mechanism nobody can see was skipped.
        self.declarations = tuple(declarations)
        self.blocks = nn.ModuleList(blocks)
        self.mixer = mixer
        self.norm = RMSNorm(d_model, eps, device=device)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`ids [batch, seq]` token ids → `logits [batch, seq, vocab]`."""
        self._check_ids(ids)
        if counters is not None:
            self._enable(counters)
        _h, logits = self.encode(ids, counters)
        return logits

    def encode(
        self, ids: torch.Tensor, counters: Counters | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the final-normed stream the paired JEPA term reads.

        One forward, one set of counters and one `check()`: the token head and the
        alignment term must see the same encoder pass, or an A/B would measure two
        different batches.
        """
        self._check_ids(ids)
        if counters is not None:
            self._enable(counters)
        h: torch.Tensor = self.norm(self._blocks(ids, counters), counters)
        logits = self.head(h, counters)
        if counters is not None:
            counters.check()
        return h, logits

    def hidden(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`ids [batch, seq]` → the final-normed `[batch, seq, d_model]` stream.

        This is the encoder embedding the ER-JEPA term reads: the paired batch is
        forwarded once, split into its two views, and no `[B, T, V]` logit buffer is
        allocated for a replay whose logits nobody scores. The head's counters are not
        enabled here, so this call's `check()` is a statement about the encoder only.
        """
        self._check_ids(ids)
        if counters is not None:
            self._enable_encoder(counters)
        h: torch.Tensor = self.norm(self._blocks(ids, counters), counters)
        if counters is not None:
            counters.check()
        return h

    @staticmethod
    def _check_ids(ids: torch.Tensor) -> None:
        """Check the one validation the forward and the encoder share."""
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

    def _blocks(self, ids: torch.Tensor, counters: Counters | None) -> torch.Tensor:
        """Embed and run the stack; the mixer, when set, owns the stream."""
        h: torch.Tensor = self.embed(ids)
        if counters is not None:
            counters.count("embed_fwd", ids.numel())
        if self.mixer is None:
            for block in self.blocks:
                h = block(h, counters)
        else:
            self.mixer.begin(h)
            for block in self.blocks:
                block(h, counters, self.mixer)
            h = self.mixer.read()
        return h

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
        loss is ever returned. Only the parts every stack shares are declared here;
        the ones that depend on which arm a block holds come from `declarations`.
        """
        self._enable_encoder(counters)
        counters.enable("head", counter="head_fwd", why="logits come from the tied head")
        counters.enable("tied_head", counter="tied_head", why="the head has no parameter")

    def _enable_encoder(self, counters: Counters) -> None:
        """Declare what the encoder half runs; `hidden()` checks only these."""
        counters.enable("embedding", counter="embed_fwd", why="the token table is read")
        counters.enable("blocks", counter="block_fwd", why="the residual stack runs")
        counters.enable("norm", counter="norm_fwd", why="two pre-norms per block + final")
        for mechanism, counter, why in self.declarations:
            counters.enable(mechanism, counter=counter, why=why)

    # ------------------------------------------------------------------- report
    def param_report(self) -> dict[str, int]:
        """Parameters by group, counted from the module tree — never guessed.

        The groups are the ones a budget is argued in: `embedding`, `attention`,
        `ffn`, `norms`, `head` (0 while the head is tied, which is the point of
        printing it), and `total`. `total` is summed from `parameters()`, so a group
        that forgets a parameter shows up as a disagreement instead of a lost number.
        """
        groups = dict.fromkeys(("embedding", "attention", "ffn", "norms", "residual", "head"), 0)
        for name, param in self.named_parameters():
            if name.startswith("embed."):
                key = "embedding"
            elif name == "norm.weight" or ".norm" in name:
                key = "norms"
            elif name.startswith("mixer."):
                key = "residual"
            elif ".attn." in name:
                key = "attention"
            elif ".ffn." in name:
                key = "ffn"
            else:
                key = "head"
            groups[key] += param.numel()
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups
