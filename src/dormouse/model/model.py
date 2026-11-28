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
at any depth instead of growing with L. `DORMOUSE_LNS=1` replaces that depth init
with plain `0.02` and scales both pre-norm outputs of layer `l` by `1/sqrt(l)`; the
combination is measured worse than LNS alone. Token rows are `N(0, 1/sqrt(d_model))`:
the head is TIED to this table, so unit-variance rows would make the logits'
standard deviation `sqrt(d_model)` and the init CE ~900 at d=2816 instead of
`ln(vocab) ~ 10.4`. RMSNorm is scale-free, so shrinking the rows does not change
what the blocks see.

NO ROTARY TABLE. Neither attention mechanism uses one (KDA's decay and DSA's
selection are their own positional signal), so the stack has no `rope` switch and no
position table to disagree with a checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..data.tokenizer import DEFAULT_VOCAB
from .blocks import EPS, LNS_COUNTER, RMSNorm, lns_enabled
from .loops import LoopGates


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
        loop_passes: int = 1,
        loop_start_frac: float = 0.25,
        loop_end_frac: float = 0.75,
        loop_rank: int = 8,
        block_checkpoint: bool = False,
        chunked_loss: bool = False,
    ) -> None:
        """Build the table, the stack, the final norm and the (tied) head."""
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
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d_model))
        #: `declarations` is the list of mechanisms whose counters prove this stack
        #: ran; the constructor cannot infer it from the modules, and a mechanism the
        #: forward cannot count is a mechanism nobody can see was skipped.
        self.declarations = tuple(declarations)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model, eps, device=device)
        self.block_checkpoint = bool(block_checkpoint)
        self.chunked_loss = bool(chunked_loss)
        self.loop_passes = int(loop_passes)
        self.loop_rank = int(loop_rank)
        self.loop_start = round(loop_start_frac * len(blocks))
        self.loop_end = max(self.loop_start + 1, round(loop_end_frac * len(blocks)))
        self.loop_start = min(self.loop_start, len(blocks) - 1)
        self.loop_end = min(self.loop_end, len(blocks))
        if self.loop_passes < 1:
            raise LoudError(
                f"loop_passes={self.loop_passes}: a loop runs at least once",
                "use 1 for the dense control, 2 for one extra pass",
            )
        if self.loop_passes > 1 and not (0 <= self.loop_start < self.loop_end <= len(blocks)):
            raise LoudError(
                f"loop window [{self.loop_start}, {self.loop_end}) is outside the stack",
                "set the loop fractions to a non-empty block range",
            )
        self.loop_gates: LoopGates | None = None
        self.loop_emb: nn.Parameter | None = None
        self.loop_mem: nn.Parameter | None = None
        self.loop_read: nn.Parameter | None = None
        self.loop_a: nn.ModuleList | None = None
        self.loop_b: nn.ModuleList | None = None
        if self.loop_passes > 1:
            dtype = self.embed.weight.dtype
            self.loop_gates = LoopGates(d_model, self.loop_passes, device=device, dtype=dtype)
            self.loop_emb = nn.Parameter(
                torch.zeros(self.loop_passes - 1, d_model, device=device, dtype=dtype)
            )
            self.loop_mem = nn.Parameter(
                torch.zeros(self.loop_passes - 1, d_model, device=device, dtype=dtype)
            )
            self.loop_read = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
            self.loop_a = nn.ModuleList(
                [
                    nn.Linear(d_model, self.loop_rank, bias=False, device=device, dtype=dtype)
                    for _ in range(self.loop_passes - 1)
                ]
            )
            self.loop_b = nn.ModuleList(
                [
                    nn.Linear(self.loop_rank, d_model, bias=False, device=device, dtype=dtype)
                    for _ in range(self.loop_passes - 1)
                ]
            )
            for a, b in zip(self.loop_a, self.loop_b, strict=True):
                assert isinstance(a.weight, torch.Tensor) and isinstance(b.weight, torch.Tensor)
                nn.init.normal_(a.weight, std=1e-4)
                nn.init.normal_(b.weight, std=1e-4)

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
        pre_loop = h
        for index, block in enumerate(self.blocks):
            if self.block_checkpoint and torch.is_grad_enabled() and h.requires_grad:
                h = checkpoint(block, h, counters, use_reentrant=False, preserve_rng_state=False)
            else:
                h = block(h, counters)
            if index + 1 == self.loop_end and self.loop_passes > 1:
                assert self.loop_gates is not None and self.loop_emb is not None
                assert self.loop_mem is not None and self.loop_read is not None
                assert self.loop_a is not None and self.loop_b is not None
                previous = h
                for pass_index in range(self.loop_passes - 1):
                    gate = self.loop_gates(h, pass_index)
                    if counters is not None:
                        counters.count("loop_fwd")
                    h = h + self.loop_emb[pass_index]
                    h = h + torch.tanh(self.loop_mem[pass_index]) * previous
                    h = h + self.loop_b[pass_index](self.loop_a[pass_index](h))
                    for repeated in self.blocks[self.loop_start : self.loop_end]:
                        out = repeated(h, counters)
                        h = h + gate * (out - h)
                    previous = h
                    if counters is not None:
                        h.register_hook(lambda grad: counters.count("loop_bwd"))
                h = h + torch.tanh(self.loop_read) * pre_loop
        logits = self.head(self.norm(h, counters), counters)
        if counters is not None:
            counters.check()
        return logits

    def loss(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor,
        counters: Counters | None = None,
        *,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Return the token-mean CE without materialising the full logits tensor.

        The head is the widest activation in the step (`[b, s, vocab]`); at b64 it is
        the tensor that makes the batch miss the card. Each sequence chunk computes
        its own logits, and `checkpoint` recomputes that chunk in the backward, so
        only `[b, s, d]` hidden states survive between the passes.
        """
        if counters is not None:
            self._enable(counters)
        h = self.embed(ids)
        if counters is not None:
            counters.count("embed_fwd", ids.numel())
        pre_loop = h
        for index, block in enumerate(self.blocks):
            if self.block_checkpoint and torch.is_grad_enabled() and h.requires_grad:
                h = checkpoint(block, h, counters, use_reentrant=False, preserve_rng_state=False)
            else:
                h = block(h, counters)
            if index + 1 == self.loop_end and self.loop_passes > 1:
                assert self.loop_gates is not None and self.loop_emb is not None
                assert self.loop_mem is not None and self.loop_read is not None
                assert self.loop_a is not None and self.loop_b is not None
                previous = h
                for pass_index in range(self.loop_passes - 1):
                    gate = self.loop_gates(h, pass_index)
                    if counters is not None:
                        counters.count("loop_fwd")
                    h = h + self.loop_emb[pass_index]
                    h = h + torch.tanh(self.loop_mem[pass_index]) * previous
                    h = h + self.loop_b[pass_index](self.loop_a[pass_index](h))
                    for repeated in self.blocks[self.loop_start : self.loop_end]:
                        out = repeated(h, counters)
                        h = h + gate * (out - h)
                    previous = h
                    if counters is not None:
                        h.register_hook(lambda grad: counters.count("loop_bwd"))
                h = h + torch.tanh(self.loop_read) * pre_loop
        h = self.norm(h, counters)
        seq = h.shape[1]
        span = max(1, min(int(chunk), seq))
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        for start in range(0, seq, span):
            stop = min(start + span, seq)
            hidden = h[:, start:stop]
            label = targets[:, start:stop]

            def _piece(hidden_piece: torch.Tensor, label_piece: torch.Tensor) -> torch.Tensor:
                logits = self.head(hidden_piece, counters)
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    label_piece.reshape(-1),
                    reduction="sum",
                )

            total = total + checkpoint(
                _piece, hidden, label, use_reentrant=False, preserve_rng_state=False
            )
        return total / max(1, targets.numel())

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
        counters.enable("embedding", counter="embed_fwd", why="the token table is read")
        counters.enable("blocks", counter="block_fwd", why="the residual stack runs")
        counters.enable("norm", counter="norm_fwd", why="two pre-norms per block + final")
        counters.enable("head", counter="head_fwd", why="logits come from the tied head")
        counters.enable("tied_head", counter="tied_head", why="the head has no parameter")
        if lns_enabled():
            counters.enable(
                "lns", counter=LNS_COUNTER, why="DORMOUSE_LNS=1: every block must scale"
            )
        if self.loop_passes > 1:
            counters.enable("loop", counter="loop_fwd", why="K>1: every extra pass must run")
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
        groups = dict.fromkeys(("embedding", "attention", "ffn", "norms", "head", "loops"), 0)
        for name, param in self.named_parameters():
            if name.startswith("embed."):
                key = "embedding"
            elif name == "norm.weight" or ".norm" in name:
                key = "norms"
            elif ".attn." in name:
                key = "attention"
            elif ".ffn." in name:
                key = "ffn"
            elif name.startswith("loop_"):
                key = "loops"
            else:
                key = "head"
            groups[key] += param.numel()
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups
