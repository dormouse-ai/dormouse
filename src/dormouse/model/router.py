"""Routing: `sqrt(softplus(logits))` scores, top-k, a device-side count, and QB.

THE SCORE IS `sqrt(softplus(logit))` (the design's recorded choice, DeepSeek-V4
§2.1). Two properties matter here. It is non-negative and finite for every
logit, so a selection weight can never be negative or a NaN; and it grows
SUB-linearly (a logit of 20 scores 4.5, not 5·10⁸), so one over-confident
projection direction cannot dominate a token's top-k the way softmax's `exp`
does. `softplus` keeps a gradient at very negative logits (`sigmoid(z) → 0` but never
0 in fp32), which is what lets a routed-off expert's row still move when a
later step selects it.

THE BIAS IS A LOGIT OFFSET; THE GATE WEIGHT IS THE UNBIASED SCORE. Quantile
Balancing steers *selection*; it must not change the magnitude the expert's
write is scaled by — a bias on the weight would re-scale the residual stream by
a quantity that is a function of the batch's load. So `index =
topk(logits + bias)` and `weight = sqrt(softplus(logits))[index] / Σ …`. The
bias lives in LOGIT space (the recorded implementation's convention) because
the score is a compressive map: a ±0.02 offset in score space is most of the
top-k margin, while the same offset in logit space is a small, well-scaled
nudge.

THE WEIGHT IS NORMALISED BY THE SELECTED SUM, which makes `topk=1` exactly
`1.0` (a number divided by itself), not `0.999…`. That is the property the
dense-FFN parity gate in `moe.py` rests on: a bank of one expert at top-1 must
add nothing to the arithmetic, and a weight that is merely close would make the
gap a rounding claim instead of an exact one.

COUNTING IS ONE `scatter_add_`, NOT `bincount`. `torch.bincount` on CUDA pulls a
host value and aborts a CUDA-graph capture (measured in this project: a bare
`torch.cuda.graph` over `bincount` raises "Cannot copy between CPU and CUDA
tensors during CUDA graph capture"). The old MoE arm ran uncompiled partly for
this reason. `count()` is a zeroed `[E]` buffer plus one scatter of ones: no
host read, no `.item()`, no `bincount`, so the router is capturable as one
graph (the test compiles it with `fullgraph=True`).

QUANTILE BALANCING (the recorded rule, sign included). For expert `i` with
target load `p = topk/E` per token, `q_i` is the batch's LOGIT quantile at
level `1 - p`: the logit above which only a `p` fraction of the batch lies.
The bias that puts every expert's target quantile on a common level is
`b_i = mean(q) - q_i` — the recorded sign convention. The rule is STATELESS:
`q` depends on the batch's logits alone, so the same batch writes the same
bias, an already-balanced batch moves the bias by 0, and a cumulative
integrator's drift cannot happen. The quantile is located with a `256`-bin
per-expert histogram (one `scatter_add_`, one `cumsum`) instead of `[n, E]`
sorts, and the whole update is device-side: no scalar extraction, no list
conversion, no CPU copy.

WHAT THE RULE IS MEASURED TO DO. It equalises each expert's THRESHOLD at the
level above which the target share lies, which is what makes a bank with
unequal columns converge. Measured, one update, CPU: the recorded skewed
fixture (`N(0,1) + linspace(1.5, 0)`, E=8, top-2) goes from a worst load 103%
off the target to 4.4%; a RANDOM LINEAR router over `x ~ N(0,1)` (d=64, E=16,
top-2) goes from a max/mean load ratio 1.22 to 1.14, and E=32 from 1.54 to
1.17; a bank whose experts' logit columns are permutations of one another is a
fixed point (the bias moves by 0). It is a threshold rule and not a proof of
top-k membership — those three fixtures are the evidence, and a run whose load
drifts is a new measurement. `ponytail:` 256 bins put the located quantile
within `1/256` of the batch's own range (the anchor test compares it against
`torch.quantile`); a per-expert sort is the exact upgrade if a run ever shows
the bin error matters.

A bias is a BUFFER, not a parameter: no optimizer sees it, and a checkpoint
that carries it would resume a load controller whose input batch no longer
exists. `mean(q) - q` is centred by construction, so the gauge cannot drift.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

#: Std of the router projection. The same `PROJ_STD` the FFN's `d → f` halves
#: draw: at `d=832` a token's logit has variance `d·σ² ≈ 0.33`, so the initial
#: top-k is close to uniform and no expert is dead at step 0 by scale alone.
ROUTER_STD = 0.02

#: Histogram resolution of the QB quantile. 256 bins put the located quantile
#: within `1/256` of the batch's score range — the ±10% load band the
#: mechanism is held to is far wider, and the cost is one `[256, E]` tensor.
QB_BINS = 256


class Router(nn.Module):
    """`x [n, d]` → `(index [n, k], weight [n, k], counts [E])`, and the QB bias.

    The bias is updated by `forward` only in `training` mode; `eval` leaves it
    at the value the last training batch set. That is the controller's contract:
    it reads the routing of a batch it is allowed to change.
    """

    #: Declared for the type checker: `register_buffer` returns nothing.
    bias: torch.Tensor

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        topk: int = 2,
        *,
        qb: bool = True,
        qb_bins: int = QB_BINS,
        std: float = ROUTER_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the projection and the zero bias; refuse an impossible routing."""
        super().__init__()
        if d_model < 1:
            raise LoudError(
                f"d_model={d_model} is not a width",
                "pass the residual width the router reads",
            )
        if n_experts < 1:
            raise LoudError(
                f"n_experts={n_experts} is not a bank",
                "a router needs at least one expert to route to",
            )
        if not 1 <= topk <= n_experts:
            raise LoudError(
                f"topk={topk} of n_experts={n_experts}: a token cannot be sent to "
                f"{topk} distinct experts out of {n_experts}",
                "use 1 <= topk <= n_experts; 2 is the design's width",
            )
        if qb_bins < 2:
            raise LoudError(
                f"qb_bins={qb_bins} cannot locate a quantile",
                "use at least 2 bins; 256 is the measured default",
            )
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.qb = qb
        self.qb_bins = qb_bins
        self.weight = nn.Parameter(torch.empty(n_experts, d_model, device=device))
        nn.init.normal_(self.weight, std=std)
        #: Quantile Balancing's load offset, added to the SCORE before the top-k.
        #: A buffer: nothing optimises it, and `mean(q) - q` keeps it centred.
        self.register_buffer("bias", torch.zeros(n_experts, device=device))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return the projection `x [n, d] @ Wᵀ`, `[n, E]` — one `[n, E]` output."""
        return nn.functional.linear(x, self.weight)

    def scores(self, logits: torch.Tensor) -> torch.Tensor:
        """`sqrt(softplus(logit))` — non-negative, sub-linear, differentiable."""
        return torch.sqrt(nn.functional.softplus(logits))

    def count(self, index: torch.Tensor) -> torch.Tensor:
        """Return the assignment histogram `[E]` by one device-side `scatter_add_`.

        No `bincount` (not capturable on CUDA) and no host round trip: the
        router's own count is the number every row-budget check is made against.
        """
        flat = index.reshape(-1)
        out = torch.zeros(self.n_experts, dtype=torch.long, device=flat.device)
        out.scatter_add_(0, flat, torch.ones_like(flat))
        return out

    def select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k over `logits + bias`; the gate weight from the UNBIASED scores.

        The weight divides by the selected scores' own sum (clamped only at the
        dtype's tiny floor, so a zero-sum row is the only case the clamp
        touches), which makes `topk=1` exactly `1.0`.
        """
        _, index = (logits + self.bias).topk(self.topk, dim=-1)
        gate = torch.gather(self.scores(logits), 1, index)
        total = gate.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gate.dtype).tiny)
        return index, gate / total, self.count(index)

    def qb_update(self, logits: torch.Tensor) -> None:
        """Set each expert's bias from the logit quantile at its target load.

        `q_i` is the upper edge of the first bin whose cumulative count reaches
        `p·n` (`p = topk/E`), i.e. `quantile(logits[:, i], 1 - p)`. The bias is
        `mean(q) - q`, the recorded sign: an expert whose threshold sits HIGH
        (it takes more than its share) gets a negative bias and is selected
        less. Stateless by construction, so repeated calls on one batch are a
        fixed point rather than an integrator.
        """
        with torch.no_grad():
            s = logits.detach().float()
            n, e = s.shape
            lo = s.min(dim=0).values
            hi = s.max(dim=0).values
            span = (hi - lo).clamp_min(1e-12)
            idx = ((s - lo) * (self.qb_bins / span)).to(torch.int64)
            idx.clamp_(0, self.qb_bins - 1)
            hist = torch.zeros(self.qb_bins, e, dtype=torch.float32, device=s.device)
            hist.scatter_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))
            cdf = hist.cumsum(dim=0)
            target = n * (1.0 - self.topk / e)
            j = (cdf < target).sum(dim=0).clamp_(0, self.qb_bins - 1)
            q = lo + (j.to(lo.dtype) + 1.0) * span / self.qb_bins
            self.bias.copy_(q.mean() - q)

    def forward(
        self, x: torch.Tensor, counters: Counters | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route `x [n, d]`; in `training` mode also move the QB bias and count it."""
        logits = self.logits(x)
        index, weight, counts = self.select(logits)
        if self.qb and self.training:
            self.qb_update(logits)
            if counters is not None:
                counters.count("moe_qb")
        return index, weight, counts

    def extra_repr(self) -> str:
        """Print the routing width and the controller's state."""
        return f"d_model={self.d_model}, n_experts={self.n_experts}, topk={self.topk}, qb={self.qb}"
