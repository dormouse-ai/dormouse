"""The MoE FFN: 64 routed gated-SwiGLU experts plus one always-on shared expert.

THE BILL, BY ARITHMETIC (d=832, f=6144, three bias-free matrices per expert):

    per expert      3·832·6144            =  15 335 424
    routed stored   64 · per              = 981 467 136
    shared stored   1 · per               =  15 335 424
    router          64 · 832               =      53 248
    stored total                          = 996 855 808   (~1B)
    active (top-2 + shared)               = 46 059 520    (~46M)

`3·d·f` is the gated SwiGLU's own bill: (gate, up, down), no bias, so an expert
IS a `GatedFFN` and the bank reuses that class instead of re-implementing the
formula. One number in the brief does not survive the arithmetic: `3·832·6144`
is 15.34M, not 15.7M — at this width an expert would need `f=6291` to bill 15.7M,
and the brief itself fixes `f=6144`. The refused configuration is the one that
would make an expert a rounding error: below 10M parameters, or below 1% of the
active set (`topk + n_shared > 100`), and both refusals name the escape.

DROPLESS, SO THERE IS NOTHING FOR A CAPACITY TO PAD. The old arm multiplied a
fixed capacity of `2.0` into the per-expert bank; at b24 it executed 2.52x the
rows the router assigned, which doubled the FLOPs and made every training-cost
number an upper bound on a different model. Here `capacity` is not a knob: the
largest group is whatever the router assigned THIS batch (reported as
`moe_max_group`), and `_dispatch` executes exactly `n_tokens · topk` expert rows
— no zero rows, no dropped assignments. `check_row_budget` fails LOUDLY if the
rows the loop executed disagree with the router's own count, which is the check
the 2.0 arm would have failed.

NO `[R, f]` BANK. Two wide buffers made the old arm unmeasurable: the gathered
per-row activation bank `[R, f]` (R = n·topk) and a zero-padded
`[E, 2·R/E, f]` one. At b24 x s1024 in fp32 those are 1.21 GB and 2.42 GB PER
BUFFER. This module never builds either: the router multiplies its `[E, d]`
weight once (`[n, E]`, 6.3 MB at b24) and never gathers winner weights into
`[R, d]`; `_dispatch` gathers one expert's TOKENS (`[r, d]`, r = that group's
size) and runs the three matmuls on `[r, ·]`, so the activation peak is the
largest single group (~850 rows at b24, ~63 MB across gate/up/h) and not the
batch's. `scripts/bench_moe.py --peak` measures the real peak; the analytic
numbers are in the tests.

ONE HOST READ, AND IT IS NOT ON THE COUNTING PATH. The router scores, selects
and counts entirely on the device (`router.count` is a `scatter_add_`, not
`bincount`), so a CUDA-graph capture of the routing is possible — the test
compiles it with `fullgraph=True`. `_dispatch` then reads the `[E]` histogram
once (`counts.tolist()`): a ragged per-expert GEMM needs the group sizes as
Python ints, and PyTorch has no CPU grouped-GEMM that takes device offsets.
That is one transfer of 64 integers per forward, the same read the reference
loop has always needed — `ponytail:` the ceiling is that the dispatch cannot be
CUDA-graph captured until a grouped GEMM (`torch._grouped_mm` on CUDA, a Triton
kernel on the roadmap) replaces the loop; the ROUTER, which is what `.item()`
and `bincount` broke, is already capture-safe.

COUNTERS (a zero must be a measurement, not a default):

* `moe_fwd` — bank passes (one per forward);
* `moe_rows_run` — expert rows executed, must be `n_tokens · topk`;
* `moe_experts_used` — groups with at least one row;
* `moe_bank_builds` — per-group activation buffers materialised; equal to
  `moe_experts_used`, and a batch-wide bank would make it 1;
* `moe_dropped` - `routed - executed`, counted even when 0 so a run that never
  drops proves it measured (the key is in `snapshot()`; `Counters.line()` prints
  non-zero values only, so a zero is read from the snapshot, not from the log);
* `moe_max_group` — the largest group this batch, i.e. the capacity that is
  actually used (no 2.0 multiplier);
* `moe_shared`, `moe_qb` — the always-on expert and the balance update.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .ffn import PROJ_STD, GatedFFN
from .router import Router

#: An expert below this is not an expert at this project's scale; see the
#: module docstring for why 10M and the live config that satisfies it.
MIN_EXPERT_PARAMS = 10_000_000

#: An expert must be at least this share of the active set. The design's
#: headline pair is 15.34M stored against 46.06M active: 33% per expert, not 1%.
MIN_EXPERT_SHARE = 0.01


def expert_parameters(d_model: int, d_ffn: int) -> int:
    """`3·d·f` — the gated SwiGLU's bill: gate, up and down, no bias."""
    return 3 * d_model * d_ffn


def bank_bill(
    d_model: int, d_ffn: int, *, n_experts: int, topk: int, n_shared: int
) -> dict[str, int]:
    """Return the stored and active bill by formula, before a single tensor exists.

    `active` is the design's own arithmetic: `topk·per_expert + shared`, with
    the router (a `[E, d]` projection) reported separately. A caller that wants
    the number with the router adds `router`.
    """
    per = expert_parameters(d_model, d_ffn)
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


def check_row_budget(*, executed: int, routed: int, topk: int, largest_group: int) -> None:
    """LOUD: a dispatch that executed a different number of rows than it routed.

    `executed` is the row count the DISPATCH itself accumulated, `routed` the
    router's `n_tokens · topk`. This is the gate the capacity-2.0 arm fails:
    a padded bank executes `E · max_group ≥ routed` rows (2.52x measured), a
    capacity rule at 1.0 drops rows and executes fewer, and both make the
    counter a lie about the model that ran.
    """
    if executed != routed:
        raise LoudError(
            f"executed {executed} expert rows against {routed} routed assignments "
            f"(topk={topk}, largest group={largest_group})",
            "dropless dispatch executes exactly n_tokens·topk rows: more means a "
            "padded capacity bank, fewer means dropped assignments",
        )


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
        act: str = "silu",
        write_std: float = PROJ_STD,
        qb: bool = True,
        min_expert_params: int = MIN_EXPERT_PARAMS,
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
        per = expert_parameters(d_model, d_ffn)
        active = per * (topk + n_shared)
        if per < min_expert_params:
            raise LoudError(
                f"an expert bills 3·{d_model}·{d_ffn}={per:,} parameters, below the "
                f"{min_expert_params:,} floor",
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
        self.router = Router(d_model, n_experts, topk, qb=qb, device=device)
        self.experts = nn.ModuleList(
            GatedFFN(d_model, d_ffn, act=act, write_std=write_std, device=device)
            for _ in range(n_experts)
        )
        self.shared: GatedFFN | None = (
            GatedFFN(d_model, d_ffn, act=act, write_std=write_std, device=device)
            if n_shared
            else None
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
            y = y + self.shared(x)
            if counters is not None:
                counters.count("moe_shared")
        if counters is not None:
            counters.count("moe_fwd")
        return y

    def _dispatch(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        counts: torch.Tensor,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """Execute each routed row on its own expert, expert-major, row-exact."""
        n, k = index.shape
        routed = n * k
        sizes = counts.tolist()  # the ONE host read; see the module docstring
        order = torch.argsort(index.reshape(-1), stable=True)
        tokens = order // k
        gates = weight.reshape(-1)[order]
        y = torch.zeros_like(x)
        pos = 0
        used = 0
        largest = 0
        for expert, size in zip(self.experts, sizes, strict=True):
            if size == 0:
                continue
            tok = tokens[pos : pos + size]
            out = expert(x.index_select(0, tok))
            y.index_add_(0, tok, out * gates[pos : pos + size, None])
            pos += size
            used += 1
            largest = max(largest, size)
        check_row_budget(executed=pos, routed=routed, topk=k, largest_group=largest)
        if counters is not None:
            counters.count("moe_rows_run", pos)
            counters.count("moe_experts_used", used)
            counters.count("moe_bank_builds", used)
            counters.count("moe_max_group", largest)
            counters.count("moe_dropped", routed - pos)
        return y

    def _enable(self, counters: Counters) -> None:
        """Declare the mechanisms this forward runs, so silence becomes an error."""
        counters.enable("moe", counter="moe_fwd", why="the routed bank runs")
        counters.enable("moe_dropless", counter="moe_rows_run", why="every assignment is executed")
        counters.enable("moe_experts", counter="moe_experts_used", why="an expert with rows ran")
        counters.enable("moe_bank", counter="moe_bank_builds", why="the per-group bank was built")
        counters.enable("moe_max_group", counter="moe_max_group", why="the capacity used")
        if self.shared is not None:
            counters.enable("moe_shared", counter="moe_shared", why="the shared expert")
        if self.router.qb and self.training:
            counters.enable("moe_qb", counter="moe_qb", why="the load controller ran")

    def param_report(self) -> dict[str, int]:
        """Return stored and active counts measured from the tree and asserted by formula.

        The formula side is `bank_bill` (`3·d·f` per expert, `topk·per + shared`
        active); the measured side is `parameters()`. A disagreement is a LOUD
        failure, not a table with two numbers in it.
        """
        bill = bank_bill(
            self.d_model,
            self.d_ffn,
            n_experts=self.n_experts,
            topk=self.topk,
            n_shared=self.n_shared,
        )
        routed = sum(p.numel() for p in self.experts.parameters())
        shared = sum(p.numel() for p in self.shared.parameters()) if self.shared else 0
        router = sum(p.numel() for p in self.router.parameters())
        if routed != bill["routed_stored"] or shared != bill["shared_stored"]:
            raise LoudError(
                f"the tree bills routed={routed:,} shared={shared:,}, the formula "
                f"routed={bill['routed_stored']:,} shared={bill['shared_stored']:,}",
                f"an expert must be 3·{self.d_model}·{self.d_ffn}="
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
