"""The MoE FFN: 64 routed MoA experts plus one always-on shared expert.

THE BILL, BY ARITHMETIC (d=832, f=6144, three bias-free matrices per expert, each
stored as its rank-256 factors `d·r + r·f`):

    per expert      3·(832·256 + 256·6144) =   5 357 568
    routed stored   64 · per               = 342 884 352
    shared stored   1 · per                =   5 357 568
    router          64 · 832               =      53 248
    stored total                           = 348 295 168
    active (top-2 + shared)                =  16 072 704

`3·(d·r + r·f)` is the gated SwiGLU's bill at rank `r`: (gate, up, down), no bias, so an
expert IS an FFN arm (`GatedFFN` or, as the one architecture names it, `MoaFFN` over the
six-entry dictionary) and the bank reuses that class instead of re-implementing the
formula. The dense bill it replaces is `3·d·f = 15 335 424` per expert; `rank=0` builds
that arm (`tsct.project`), it is the control, and `param_report` bills whichever one was
built. One number in the brief does not survive the arithmetic: the dense `3·832·6144`
is 15.34M, not 15.7M — at this width an expert would need `f=6291` to bill 15.7M, and the
brief itself fixes `f=6144`. The refused configuration is the one that would make an
expert a rounding error: below 10M dense parameters, or below 1% of the active set
(`topk + n_shared > 100`), and both refusals name the escape.

DROPLESS, AND ONE PADDED DISPATCH PER LAYER. `_dispatch` lays every routed row into ONE
contiguous `[E·capacity, d]` bank — a row's slot is `expert·capacity + its rank inside
that expert` — and reads each group of experts as a 3-D batch: `[g, capacity, d] @
[g, d, r] @ [g, r, f]`. The old arm ran the same arithmetic one expert at a time and paid
a Python call and ~77 kernel launches per expert per bank call: 4 919 per call and 59 028
per step at twelve layers, which was 81% of a measured 403 387 launches per step. The
batched path issues `6·ceil(E/g)` matmuls for the up/gate/down trio plus the mixture's
elementwise chain, i.e. a few hundred dispatches per bank call, and no Python loop over
experts exists to scale with `n_experts`.

CAPACITY IS A FORMULA OVER SHAPES, MASKED, AND NEVER A DROP. `capacity` is
`CAPACITY_RATIO·mean` (2x), floored by `mean + DISPATCH_SIGMA·√mean` (4 sigma of the
assignment's own noise, which is what a small batch needs) with `mean = n·topk/E` —
integers the caller already knows (`n`, `topk`, `E`). That is the whole point: a capacity
that depends on the assignment's own maximum has to be read back from the device to size a
tensor, and `counts.tolist()` was that read. Padding rows are MASKED, not
executed-and-reported: the output is gathered by the slot each real row owns, so a padded
slot is never read, and its gradient is zero, so no expert's gradient moves. The padded
total is bounded by `MAX_PAD_RATIO` by construction (the capacity is clamped by it) and
reported as `moe_pad_rows`: 2.0x the routed rows at b8 and b24 (1.0x would be a ragged
bank, and the old arm's 2.0 was never reported at all). A batch whose largest
group is past the capacity REFUSES loudly — a device-side assert on the card, `LoudError`
on the CPU — because this bank has no drop policy to fall back on; `dispatch_sigma` and
`CAPACITY_RATIO` are the knobs, and the refusal names the second.

MEMORY IS ONE GROUP'S SLAB, NOT THE BATCH'S. Two buffers are proportional to the routed
rows `R = n·topk`: the `[E·capacity, d]` layout (the `[R, d]` gather the loop did per
group, plus the padding) and the slots it is gathered from. The wide buffers are
`[group, capacity, d_ffn]`, and the group is CHOSEN to keep one of them under
`DISPATCH_SLAB_BYTES` (256 MiB): four of them are live in the mixture at once, so the
transient is bounded by an expert group's slab and never by `E·capacity·f` — the padded
2.0 bank the old docstring refused, 1.38 GiB at b24 in fp32. At b8 in fp32 the group is
32 experts (2 groups per call), at b16 17 (4), and the slab in flight is ~250 MiB.

NO HOST READ, SO THE FORWARD IS CAPTURABLE. The router was already device-side
(`scatter_add_`, no `bincount`, no `.item()`); the histogram crossing to the host was the
last thing that kept a CUDA graph out, and it is gone: `tests/test_moe.py` runs the
forward under `torch.cuda.set_sync_debug_mode("error")` and captures it in a graph. The
one Python int the path needs — the capacity — comes from the batch's shape, which is
free.

COUNTERS (a zero must be a measurement, not a default):

* `moe_fwd` — bank passes (one per forward);
* `moe_rows_run` — routed expert rows executed, must be `n_tokens · topk`;
* `moe_dropped` - `routed - executed`, counted even when 0 so a run that never
  drops proves it measured (the key is in `snapshot()`; `Counters.line()` prints
  non-zero values only, so a zero is read from the snapshot, not from the log);
* `moe_capacity` — rows per expert slot this forward sized, from the formula above;
* `moe_pad_rows` — the padded total `E·capacity`, i.e. what the slabs executed against
  `moe_rows_run` (the ratio is itself the padding's report);
* `moe_experts_used` — expert slots executed: the bank's width, because a padded
  dispatch reads every expert (a dead expert is a routing fact, and the router's own
  `[E]` histogram is the device-side place it is read);
* `moe_bank_builds` — expert-group slabs materialised, `ceil(E/group)`; a batch-wide
  bank would make it 1;
* `moe_shared`, `moe_qb` — the always-on expert and the balance update.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ffn import PROJ_STD, LaFFN, MoaFFN, Stacked, _Glu, stack_matrices
from .ffn_spec import FFNSpec
from .router import Router
from .tsct import TSCT_RANK

#: Rows per expert slot, as a multiple of the mean load `routed/E`. Two is what the
#: bank's own run needs: measured on the card at b8 x s1024, the largest group is 1.43x
#: the mean on the first step (router bias zero, before QB has equalised anything) and
#: 1.87x on the second; beyond that the ROUTER is unbalanced, not the dispatch, and a
#: capacity that followed it would have to be recomputed from the device — which is the
#: host read this path exists to remove. 2.0 is also the multiple the old arm multiplied
#: into the bank and never reported; here it is the slot width, the padding is masked,
#: `moe_pad_rows` reports it, and a wider group REFUSES instead of being truncated.
CAPACITY_RATIO = 2.0

#: Counting-noise floor on the same width, in standard deviations of the assignment
#: (`sd ~ √mean` of a multinomial): 4 sigma is what a small batch needs where `mean` is
#: a handful of rows and `CAPACITY_RATIO` alone would be a slot of two.
DISPATCH_SIGMA = 4.0

#: The padded bank may not exceed this multiple of the routed rows. The capacity above is
#: CLAMPED by it, so the padding is bounded by construction and reported (`moe_pad_rows`);
#: it binds only below `mean ≈ 16` rows per expert, i.e. a 64-expert bank fed under 1024
#: routed rows (b1 x s1024), where the √mean term is the whole slack.
MAX_PAD_RATIO = 3.0

#: Bytes one `[group, capacity, d_ffn]` slab may hold. The mixture keeps ~4 of them live
#: (gate, up, their weighted sum and the write's input), so an expert group is chosen to
#: keep that set near a GiB; the trio's dispatch count is `6·ceil(E/group)` per call.
DISPATCH_SLAB_BYTES = 128 * 2**20

#: An expert below this is not an expert at this project's scale; see the
#: module docstring for why 10M and the live config that satisfies it. Measured against
#: the DENSE bill `3·d·f` even when the expert is factorised: TSCT changes what an
#: expert stores, not the width the design chose, so a rank-256 expert storing 5.36M is
#: still the 15.34M expert the architecture is priced at.
MIN_EXPERT_PARAMS = 10_000_000

#: An expert must be at least this share of the active set. The design's
#: headline pair is 5.36M stored against 16.07M active: 33% per expert, not 1%.
MIN_EXPERT_SHARE = 0.01


def expert_parameters(d_model: int, d_ffn: int, rank: int = 0) -> int:
    """Return the expert's stored bill: `3·d·f` dense, `3·(d·r + r·f)` factorised."""
    if rank < 1:
        return 3 * d_model * d_ffn
    return 3 * rank * (d_model + d_ffn)


def bank_bill(
    d_model: int,
    d_ffn: int,
    *,
    n_experts: int,
    topk: int,
    n_shared: int,
    rank: int = 0,
    extra_per_expert: int = 0,
) -> dict[str, int]:
    """Return the stored and active bill by formula, before a single tensor exists.

    `active` is the design's own arithmetic: `topk·per_expert + shared`, with
    the router (a `[E, d]` projection) reported separately. A caller that wants
    the number with the router adds `router`.
    """
    per = expert_parameters(d_model, d_ffn, rank) + extra_per_expert
    router = n_experts * d_model
    return {
        "per_expert": per,
        "routed_stored": per * n_experts,
        "shared_stored": per * n_shared,
        "router": router,
        "stored": per * (n_experts + n_shared) + router,
        "routed_active": per * topk,
        "shared_active": per * n_shared,
        "active": per * (topk + n_shared),
        "active_with_router": per * (topk + n_shared) + router,
    }


def _group_write(bank: MoEFFN, arms: list[_Glu], slab: torch.Tensor) -> torch.Tensor:
    """One expert group's write, recomputed in the backward rather than retained.

    Two things are inside the recomputed region, not outside it: the `[group, capacity,
    d_ffn]` slabs the mixture reads, and the `Stacked` COPY of the group's factors. Keeping
    either alive per layer for the backward retains `E·capacity·d_ffn` (1.2 GiB at b8) and
    `E·r·d_ffn` (470 MiB) — measured as 10 GiB of `torch.stack` blocks over twelve layers
    before this was moved in — where recomputing them keeps one group's slab as the
    graph's whole routed activation.
    """
    return arms[0].batch_write(bank._stacked(arms), slab)


class MoEFFN(nn.Module):
    """`x [n, d] → [n, d]`: top-k routed experts + one shared expert.

    `[n, d]` and not `[batch, seq, d]` on purpose: the bank routes POSITIONS,
    and a caller that hands it a 3-D tensor would be asking the dispatch to
    guess which axis the tokens are on. Flatten at the call site.
    """

    def __init__(
        self,
        d_model: int = 832,
        d_ffn: int = 6144,
        *,
        n_experts: int = 64,
        topk: int = 2,
        n_shared: int = 1,
        act: str = "moa",
        d_ffn_gelu: int | None = None,
        expert_tsct_rank: int = TSCT_RANK,
        write_std: float = PROJ_STD,
        qb: bool = True,
        min_expert_params: int = MIN_EXPERT_PARAMS,
        dispatch_sigma: float = DISPATCH_SIGMA,
        capacity_ratio: float = CAPACITY_RATIO,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the router, the routed experts and the shared expert, in that order."""
        super().__init__()
        if d_model < 1:
            raise LoudError(
                f"d_model={d_model} is not a width",
                "pass the residual width the bank reads",
            )
        if n_experts < 1:
            raise LoudError(
                f"n_experts={n_experts} builds no bank",
                "one expert is the dense control; top-k of a one-expert bank is parity",
            )
        if n_shared not in (0, 1):
            raise LoudError(
                f"n_shared={n_shared}: the design has ONE always-on expert",
                "use 0 (the parity control) or 1; a bank of shared experts is a "
                "second dense path wearing the shared expert's name",
            )
        self.expert_spec = FFNSpec(
            d_model, d_ffn, act=act, d_ffn_gelu=d_ffn_gelu, tsct_rank=expert_tsct_rank
        )
        per = expert_parameters(d_model, d_ffn, expert_tsct_rank) + (
            self.expert_spec.added_parameters()
        )
        active = per * (topk + n_shared)
        dense = expert_parameters(d_model, d_ffn) + self.expert_spec.added_parameters()
        if dense < min_expert_params:
            raise LoudError(
                f"an expert bills 3·{d_model}·{d_ffn}={dense:,} parameters dense, below "
                f"the {min_expert_params:,} floor",
                "a routed expert at this project's scale is 3·832·6144=15,335,424; "
                "raise d_ffn, or pass min_expert_params only for a smoke arm and "
                "say so in the report",
            )
        if per < MIN_EXPERT_SHARE * active:
            raise LoudError(
                f"an expert is {per / active:.4%} of the active set "
                f"(topk={topk} + n_shared={n_shared}): below "
                f"{MIN_EXPERT_SHARE:.0%}",
                "the routed experts are the model at that point; lower topk or "
                "n_shared so each expert is more than 1% of what runs",
            )
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_experts = n_experts
        self.topk = topk
        self.n_shared = n_shared
        self.expert_tsct_rank = expert_tsct_rank
        self.dispatch_sigma = dispatch_sigma
        self.capacity_ratio = capacity_ratio
        self.router = Router(d_model, n_experts, topk, qb=qb, device=device)
        self.experts = nn.ModuleList(
            self.expert_spec.build(write_std=write_std, device=device) for _ in range(n_experts)
        )
        self.shared = (
            self.expert_spec.build(write_std=write_std, device=device) if n_shared else None
        )

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [n, d]` → the bank's write, same shape: routed assignments + shared."""
        if x.dim() != 2:
            raise LoudError(
                f"MoEFFN takes x [tokens, d_model], got {tuple(x.shape)}",
                "flatten batch and sequence at the call site; the bank routes positions",
            )
        if x.shape[1] != self.d_model:
            raise LoudError(
                f"x is [_, {x.shape[1]}] but the bank is d_model={self.d_model}",
                "build the bank at the residual width it reads",
            )
        if x.shape[0] < 1:
            raise LoudError(
                "a forward with no tokens routes nothing",
                "skip empty micro-batches; an all-empty step is not a training step",
            )
        if counters is not None:
            self._enable(counters)
        index, weight, counts = self.router(x, counters)
        y = self._dispatch(x, index, weight, counts, counters)
        if self.shared is not None:
            y = y + self.shared(x, counters)
            if counters is not None:
                counters.count("moe_shared")
        if counters is not None:
            counters.count("moe_fwd")
        return y

    def capacity(self, routed: int) -> int:
        """Rows per expert slot: `CAPACITY_RATIO` of the mean, floored by counting noise.

        A function of `routed = n·topk` alone — the property that lets the dispatch size
        its layout without reading the routing back from the device. `MAX_PAD_RATIO` is
        what keeps the clamp a bound and not a wish: the padded total `E·capacity` cannot
        exceed it, so the waste is stated before the tensors exist, and a batch whose
        largest group is wider than the slot is REFUSED (`_refuse_overflow`), not truncated.
        """
        mean = routed / self.n_experts
        slack = max(self.capacity_ratio * mean, mean + self.dispatch_sigma * math.sqrt(mean))
        return max(1, min(math.ceil(slack), int(MAX_PAD_RATIO * mean)))

    def group(self, capacity: int, itemsize: int) -> int:
        """Experts per slab: the most whose transients fit the budget.

        Two allocations scale with the group, and the smaller limit governs: the padded
        `[group, capacity, d_ffn]` slab (the mixture's own wide buffer) and the stacked
        `[group, r, d_ffn]` factors (the batched read's operands, a copy of that group's
        parameters). At the live bank the first is 1.5x the second, so the slab governs;
        at a tiny capacity the factors do, and this keeps their stack off the peak.
        """
        factors = self.expert_tsct_rank * max(self.d_model, self.d_ffn)
        per_expert = max(capacity * self.d_ffn, factors or self.d_model * self.d_ffn)
        return max(1, min(self.n_experts, DISPATCH_SLAB_BYTES // (per_expert * itemsize)))

    def _dispatch(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        counts: torch.Tensor,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """Lay the routed rows into one padded bank and read each expert group in a batch."""
        n, k = index.shape
        routed = n * k
        capacity = self.capacity(routed)
        order = torch.argsort(index.reshape(-1), stable=True)
        tokens = order // k
        gates = weight.reshape(-1)[order]
        expert = index.reshape(-1)[order]
        #: The rank of a row inside its own expert: the sorted position minus the number
        #: of rows the experts before it took. Device-only, like everything below.
        starts = counts.cumsum(0) - counts
        slot = expert * capacity + (
            torch.arange(routed, device=x.device) - starts.index_select(0, expert)
        )
        self._refuse_overflow(counts, capacity, routed)
        pad = self.n_experts * capacity
        bank = x.new_zeros((pad, x.shape[1]))
        bank.index_copy_(0, slot, x.index_select(0, tokens))
        out = x.new_empty((pad, x.shape[1]))
        group = self.group(capacity, x.element_size())
        for start in range(0, self.n_experts, group):
            stop = min(start + group, self.n_experts)
            span = slice(start * capacity, stop * capacity)
            arms = self._arms(start, stop)
            slab = bank[span].view(stop - start, capacity, x.shape[1])
            written = (
                checkpoint(_group_write, self, arms, slab, use_reentrant=False)
                if torch.is_grad_enabled() and slab.requires_grad
                else _group_write(self, arms, slab)
            )
            out[span] = written.view(-1, x.shape[1])
        y = torch.zeros_like(x)
        y.index_add_(0, tokens, out.index_select(0, slot) * gates.unsqueeze(1))
        if counters is not None:
            counters.count("moe_rows_run", routed)
            counters.count("moe_experts_used", self.n_experts)
            counters.count("moe_bank_builds", math.ceil(self.n_experts / group))
            counters.count("moe_capacity", capacity)
            counters.count("moe_pad_rows", pad)
            counters.count("moe_dropped", 0)
            #: The read belongs to the arm, so it reports the rows the DISPATCH read — the
            #: assignments, not their padding (that is `moe_pad_rows`).
            self._arms(0, 1)[0].count_read(routed, counters)
        return y

    def _arms(self, start: int, stop: int) -> list[_Glu]:
        """Return experts `[start, stop)` as the arms they are: one `nn.ModuleList` slice."""
        return cast("list[_Glu]", list(self.experts[start:stop]))

    def _stacked(self, arms: list[_Glu]) -> Stacked:
        """Stack one group's parameters into the `[group, ...]` batches `batch_write` reads."""
        first = arms[0]
        mix = None
        coef = None
        if isinstance(first, MoaFFN):
            moa = cast("list[MoaFFN]", arms)
            mix = (
                torch.stack([arm.mix.weight for arm in moa]).transpose(1, 2),
                torch.stack([arm.mix.bias for arm in moa]),
            )
        if isinstance(first, (LaFFN, MoaFFN)):
            mixture = cast("list[LaFFN | MoaFFN]", arms)
            coef = torch.stack([arm.coef for arm in mixture])
        return Stacked(
            gate=stack_matrices([arm.gate for arm in arms]),
            up=stack_matrices([arm.up for arm in arms]),
            down=stack_matrices([arm.down for arm in arms]),
            mix=mix,
            coef=coef,
        )

    def _refuse_overflow(self, counts: torch.Tensor, capacity: int, routed: int) -> None:
        """LOUD: the largest group is past the capacity, so a slot would be shared.

        A device-side assert on the card — `torch._assert_async` needs no host read, which
        is the property this path exists for, and it poisons the context at the next
        synchronisation instead of writing two experts' rows into one slot. On the CPU a
        host read is free, so the same check raises the project's own error.
        """
        what = (
            f"an expert group is wider than the dispatch capacity of {capacity} rows "
            f"({capacity / (routed / self.n_experts):.2f}x the mean of "
            f"{routed}/{self.n_experts})"
        )
        escape = (
            f"raise `capacity_ratio` (now {self.capacity_ratio:g}) or `dispatch_sigma` "
            f"(now {self.dispatch_sigma:g}) on the bank, or feed it more rows per forward; "
            "this bank never drops an assignment"
        )
        if counts.is_cuda:
            #: A device-side assert carries a bounded message (torch caps it at 255 bytes),
            #: so it names the overflow and the knob; the escape above is the CPU's copy.
            torch._assert_async(
                counts.max() <= capacity,
                f"{what}; raise capacity_ratio (now {self.capacity_ratio:g})",
            )
        elif int(counts.max()) > capacity:
            raise LoudError(what, escape)

    def _enable(self, counters: Counters) -> None:
        """Declare the mechanisms this forward runs, so silence becomes an error."""
        counters.enable("moe", counter="moe_fwd", why="the routed bank runs")
        counters.enable("moe_dropless", counter="moe_rows_run", why="every assignment is executed")
        counters.enable("moe_experts", counter="moe_experts_used", why="every expert slot ran")
        counters.enable(
            "moe_bank", counter="moe_bank_builds", why="the expert-group slab was built"
        )
        counters.enable("moe_capacity", counter="moe_capacity", why="the capacity sized")
        if self.shared is not None:
            counters.enable("moe_shared", counter="moe_shared", why="the shared expert")
        if self.router.qb and self.training:
            counters.enable("moe_qb", counter="moe_qb", why="the load controller ran")

    def param_report(self) -> dict[str, int]:
        """Return stored and active counts measured from the tree and asserted by formula.

        The formula side is `bank_bill` (`3·(d·r + r·f)` per expert at rank `r`,
        `topk·per + shared` active); the measured side is `parameters()`. A disagreement
        is a LOUD failure, not a table with two numbers in it.
        """
        bill = bank_bill(
            self.d_model,
            self.d_ffn,
            n_experts=self.n_experts,
            topk=self.topk,
            n_shared=self.n_shared,
            rank=self.expert_tsct_rank,
            extra_per_expert=self.expert_spec.added_parameters(),
        )
        routed = sum(p.numel() for p in self.experts.parameters())
        shared = sum(p.numel() for p in self.shared.parameters()) if self.shared else 0
        router = sum(p.numel() for p in self.router.parameters())
        if routed != bill["routed_stored"] or shared != bill["shared_stored"]:
            raise LoudError(
                f"the tree bills routed={routed:,} shared={shared:,}, the formula "
                f"routed={bill['routed_stored']:,} shared={bill['shared_stored']:,}",
                f"an expert must store 3·(d·r + r·f)="
                f"{bill['per_expert']:,} parameters; a biased or extra matrix is a "
                f"different expert than the one the bill quotes",
            )
        return {
            **bill,
            "stored": routed + shared + router,
            "measured_per_expert": routed // self.n_experts,
        }

    def extra_repr(self) -> str:
        """Print the routing and the widths the bill is quoted at."""
        return (
            f"d_model={self.d_model}, d_ffn={self.d_ffn}, n_experts={self.n_experts}, "
            f"topk={self.topk}, n_shared={self.n_shared}"
        )
