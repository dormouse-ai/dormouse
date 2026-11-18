"""The residual write as a mechanism: one plain add, or a mixture chosen by a query.

`blocks.Block` writes its two sublayers into the stream with `x = x + w`. This module
owns the two mechanisms that replace that write, and nothing else: a plain `+` is not
here (it is the block's own line, and it stays the block's line so `residual="plain"`
cannot move).

ONE SOURCE PER SUBLAYER, `N = L` in the paper's counting. arXiv:2603.15031 §2 counts
each self-attention and each MLP as an individual layer, so our twelve blocks are 24
layers and `N = 2·n_layers` here. The block variant (`N = 8` over 93 layers) is a
different mechanism: it exists to keep the source count under the saturation knee at
depth, and with one source per layer there is no knee, so depth is irrelevant. This is
also the only form whose identity init can be the block's own `x + u` bit for bit: a
per-BLOCK source would have to write `x + (a + f)`, which is not `(x + a) + f` in
fp32. At our width the full variant costs `N·(d+1)` = 24·833 = 19,992 parameters —
0.002 % of the 1.05B stored, 0.020 % of the 98.4M active — and it is EXACTLY zero at
init (see `AttnResFull`).

WHY THE ANCHOR IS `n` AND NOT `log n`. The paper's softmax is over the sources with a
zero-initialised query, so its own init is an equal-weight AVERAGE of the sources. An
A/B arm that starts by averaging is measuring the averaging, not the mechanism — the
same failure this file's mHC init gate exists to catch. So the mixture carries a zero
source whose logit is `log n`, which under the house `S·softmax` convention
(`sum_j w_j = S`) makes the current sublayer's weight exactly 1 and leaves the older
sources at `e^{-30}`. The zero source's contribution to the denominator is written as
the integer `n`, not as `exp(log n)`: `exp(fl(log n))` is not `n` (for n = 12 it is
11.9999997 in fp32), and that one-ulp error is exactly what would put the write one ulp
off the plain residual. Written as `n`, at the zero-query init the denominator is
`n + 1` exactly, so `w_self = (n+1)/(n+1) = 1.0` bit-exactly and the write is the plain
add.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.utils.checkpoint import checkpoint

__concepts__ = ("AttnResFull", "MHC", "Residual")

#: The log-prior floor for every source but the current one. `e^{-30} = 9.4e-14`, far
#: below the fp32 ulp of a residual element, so the older sources round out of the
#: write at init while their gradient is still non-zero (a floor of `-inf` would be
#: exact and permanently dead: `d softmax / d logit = 0` at a zero weight).
PRIOR_FLOOR = -30.0

#: The mixture's `ε` for the score's RMSNorm, the same one every other norm uses.
EPS = 1e-6

#: Sinkhorn-Knopp iterations: mHC App. A.1's `t_max = 20`.
SINKHORN_ITERS = 20

#: The hyper-network's gating factor at init (mHC App. A.1: `alpha = 0.01`).
ALPHA_INIT = 0.01

#: mHC's read-in bias: `sigmoid(10) = 0.99995`, the full read-in gate.
STATIC_INIT = 10.0

#: mHC's residual-mixing bias DIAGONAL. The Sinkhorn-Knopp projection of
#: `b·I + O(1e-4)` leaks `(n-1)·e^{-b}` off the diagonal, so the init is the identity
#: corner of the Birkhoff polytope only while `b` is large. `b = 12` leaks
#: `e^{-12} = 6.1e-6` at any `n` (the per-lane diagonal is `12 + log(n-1)`), which is
#: inside the `1e-5` gate; the port's `b_res = 0` was the OPPOSITE corner — `J/n`,
#: uniform mixing — and the two contradictory mHC verdicts on record were both
#: measured on that init (`docs/records/reviews/2026-11-11-mhc-attnres-4bit-verdicts.md`).
B_RES = 12.0


class Residual(Protocol):
    """What a block's residual write needs from a stack-level mixer.

    Structural, so the two mechanisms are the only implementations and neither block
    nor stack knows which one it holds: `begin` seeds the state once per forward,
    `enter`/`write` move it one sublayer at a time, and `read` is what the next block
    (or the final norm) sees.
    """

    def begin(self, x: torch.Tensor) -> None:
        """Seed the state from the stack's embedding."""
        ...

    def enter(self, x: torch.Tensor) -> torch.Tensor:
        """Return the first sublayer's input."""
        ...

    def write(self, u: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Apply one sublayer's output and return the next sublayer's input."""
        ...

    def read(self) -> torch.Tensor:
        """Return the stream the next block (or the final norm) reads."""
        ...


def sinkhorn_knopp(logits: torch.Tensor, iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Project `logits [..., n, n]` onto the Birkhoff polytope, in the log domain.

    mHC Eq. 9: `iters` alternating row/column normalisations, each written as
    `m -= logsumexp(m)` so it is shift-invariant and never overflows on the
    hyper-network's unbounded output.
    """
    m = logits
    for _ in range(iters):
        m = m - torch.logsumexp(m, dim=-1, keepdim=True)
        m = m - torch.logsumexp(m, dim=-2, keepdim=True)
    return m.exp()


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`x · rsqrt(mean(x²) + ε)` on the last axis: the score's key norm, no gain."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _score(src: torch.Tensor, q: torch.Tensor, prior: torch.Tensor, eps: float) -> torch.Tensor:
    """One source's `q · RMSNorm(u) + prior`, in fp32 — a checkpoint SEGMENT.

    Called through `checkpoint`, so the fp32 upcast and the normalised copy are
    recomputed in the backward one source at a time instead of being retained for it.
    """
    return (_rms(src.float(), eps) * q).sum(-1) + prior


class AttnResFull(nn.Module):
    """Attention Residuals, full depth: one pseudo-query per sublayer (arXiv:2603.15031).

    The write at sublayer `i` is `x + a` with

        a = sum_j w_j u_j,   w = S·softmax([log n, z_1, …, z_n]),
        z_j = q_i · RMSNorm(u_j) + prior[dist(i, j)],   S = n + 1,

    where `u_1 … u_{n-1}` are the earlier sublayers' outputs (`u_0 = 0` is the zero
    source that carries `log n`) and `u_n` is this sublayer's own output. At the
    zero-query init the weights are `(n, …, 1)`, so `a = u_n` and the write is the
    plain residual — bit for bit, which `tests/test_residual.py` asserts with
    `torch.equal`.

    MEMORY — the scores are one `[b, t]` column per source, never a `[T, S, d]` stack.
    The old arm materialised `[T, S, d]` (507 MiB at b24x1024, layer 12) and OOMed at
    b24, b20 and b16 with no verdict. Here the only source-shaped tensor alive is the
    one `[b, t, d]` fp32 RMSNorm transient inside `_score`, and it is dropped before the
    next source; the persistent tensors are the `[b, t, n+1]` logits (1.3 MiB at the
    live shape) and one `[b, t, d]` accumulator. Same arithmetic, same order of
    accumulation, so the values are unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_sources: int,
        *,
        eps: float = EPS,
        device: torch.device | str | None = None,
    ) -> None:
        """One zero query per sublayer, and the `(0, -30, …)` log-prior."""
        super().__init__()
        if n_sources < 1:
            raise LoudError(
                f"AttnResFull needs at least one source, got n_sources={n_sources}",
                "pass 2·n_layers: every sublayer writes one source, including the first",
            )
        self.d_model = int(d_model)
        self.n_sources = int(n_sources)
        self.eps = float(eps)
        #: Zero, as §5 mandates: the query contributes nothing at step 0 and the prior
        #: alone decides the weights. A non-zero query would make the arm's init an
        #: average of the sources, which is the state this mechanism must NOT start in.
        self.query = nn.Parameter(torch.zeros(self.n_sources, self.d_model, device=device))
        #: `prior[0]` is this sublayer's own output, `prior[k]` the write `k`
        #: sublayers back. `(0, -30, …)` puts the whole write on the current source.
        prior = torch.full((self.n_sources,), PRIOR_FLOOR, device=device)
        prior[0] = 0.0
        self.prior = nn.Parameter(prior)
        self._history: list[torch.Tensor] = []
        self._stream: torch.Tensor | None = None

    # ------------------------------------------------------------------- state
    def begin(self, x: torch.Tensor) -> None:
        """Start a forward: the stream is the embedding, the history is empty."""
        self._history = []
        self._stream = x

    def enter(self, x: torch.Tensor) -> torch.Tensor:
        """Return the first sublayer's input: the stream itself (AttnRes adds, not mixes)."""
        if self._stream is None:
            self._stream = x
        return self._stream

    def write(self, u: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Mix this sublayer's output with the ones below it and add the mixture."""
        if self._stream is None:
            raise LoudError(
                "AttnResFull.write before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the stream",
            )
        i = len(self._history)
        if i >= self.n_sources:
            raise LoudError(
                f"AttnResFull was built for {self.n_sources} sources and the stack wrote {i + 1}",
                "build it with n_sources=2·n_layers: one source per sublayer, no blocks",
            )
        n = i + 1
        q = self.query[i].float()
        #: `[b, t, n]` fp32; column `j` is `u_j`, the last column is this sublayer.
        z = u.new_empty((*u.shape[:-1], n), dtype=torch.float32)
        for j, src in enumerate(self._history):
            z[..., j] = checkpoint(
                _score, src, q, self.prior[i - j].float(), self.eps, use_reentrant=False
            )
        z[..., n - 1] = checkpoint(
            _score, u, q, self.prior[0].float(), self.eps, use_reentrant=False
        )
        # The max-shift is clamped at 0 because the zero source's logit is `log n >= 0`:
        # with `m = max(z, 0)` its denominator term is the exact `n`, and no exponent
        # can overflow (`e^{z-m} <= 1` either way). The shift cancels in the ratio, so
        # the function is the paper's softmax whichever `m` is used.
        m = z.max(-1, keepdim=True).values.clamp_min(0.0)
        denom = n * torch.exp(-m) + torch.exp(z - m).sum(-1, keepdim=True)
        w = (n + 1) * torch.exp(z - m) / denom
        a = u.new_zeros(u.shape, dtype=torch.float32)
        for j, src in enumerate(self._history):
            a = a + w[..., j].unsqueeze(-1) * src
        # The current update last: its weight is exactly 1.0 at init, so the older
        # sources' `e^{-30}` tail rounds away in this final add instead of perturbing
        # the stream by a sub-ulp of its own first element.
        a = a + w[..., n - 1].unsqueeze(-1) * u
        self._history.append(u)
        self._stream = self._stream + a.to(self._stream.dtype)
        if counters is not None:
            counters.count("attnres_sources", n)
        return self._stream

    def read(self) -> torch.Tensor:
        """Return the residual stream the next block — or the final norm — reads."""
        if self._stream is None:
            raise LoudError(
                "AttnResFull.read before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the stream",
            )
        return self._stream

    def extra_repr(self) -> str:
        """Print the two numbers that define the arm."""
        return f"d_model={self.d_model}, n_sources={self.n_sources}"


class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections: `n` residual lanes mixed by Eq. 3.

    Per write, for `x_l` of shape `[n, d]` and the sublayer's output `u`:

        H_pre  = sigmoid(xn @ phi_pre + b_pre)              (the read, made convex)
        H_post = 2 sigmoid(xn @ phi_post + b_post)
        H_res  = Sinkhorn-Knopp(xn @ phi_res + b_res)       (Eq. 8/9, the polytope)
        x_{l+1} = H_res @ x_l + H_post^T @ u                (Eq. 3)

    with `xn = RMSNorm(vec(x_l))` and `phi_* ~ N(0, 0.01²)`, `alpha = 0.01` (Eq. 7).

    INIT. `b_pre = 10`, `b_post = 0` and `b_res = B_RES·I`. The residual bias is the
    one that decides whether this mechanism starts at the identity or at its opposite:
    `b_res = 0` gives `H_res = J/n` — every lane the average of all lanes, measured
    `||H - J/n|| = 0.0000` and `||H - I|| = 1.0000` at n = 2 — so a model with it starts by
    averaging all its residual streams, which is a different model. At `b_res = 12·I`
    the Sinkhorn projection is the identity to `e^{-12} = 6.1e-6`.

    READ. The lanes are `n` full-width copies of one state, so the read-out is their
    MEAN; the paper's `n` lanes are `d/n`-wide slices that re-sum to `d`, and the port
    had to divide by `n` to emulate that (its mHC-only arm summed lanes that carried
    `1/n` of the state, and its first block read `n·h` when they did not). Carrying the
    state per lane and reading the mean is the same arithmetic without the bookkeeping.
    `H_pre` is normalised to a convex combination for the same reason: `n` copies of the
    stream would otherwise be read as `n·x`.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 2,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the hyper-network and the three bias vectors at their identity init."""
        super().__init__()
        if n_streams < 2:
            raise LoudError(
                f"n_streams={n_streams}: one lane is the plain residual with extra parameters",
                "use n_streams >= 2, or residual='plain'",
            )
        d, n = int(d_model), int(n_streams)
        self.d_model = d
        self.n_streams = n
        total = n * d
        #: Eq. 7's three projections, `N(0, 0.01²)` — the paper states no init and the
        #: crate's 0.01 is ours, named as ours (`docs/records/reviews/mhc-2026-09-30.md`).
        self.phi_pre = nn.Parameter(torch.randn(total, n, device=device) * 0.01)
        self.phi_post = nn.Parameter(torch.randn(total, n, device=device) * 0.01)
        self.phi_res = nn.Parameter(torch.randn(total, n * n, device=device) * 0.01)
        self.alpha_pre = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_post = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_res = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.b_pre = nn.Parameter(torch.full((n,), STATIC_INIT, device=device))
        self.b_post = nn.Parameter(torch.zeros(n, device=device))
        #: The identity anchor, with the `log(n-1)` that keeps the leakage `(n-1)e^{-b}`
        #: equal to `e^{-B_RES}` for any lane count (see `B_RES`).
        anchor = B_RES + math.log(n - 1)
        self.b_res = nn.Parameter(anchor * torch.eye(n, device=device))
        self._lanes: torch.Tensor | None = None

    # ------------------------------------------------------------------- state
    def begin(self, x: torch.Tensor) -> None:
        """Start a forward: `n` full-width lanes, each a copy of the embedding."""
        self._lanes = x.unsqueeze(-2).expand(*x.shape[:-1], self.n_streams, x.shape[-1])

    def enter(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sublayer's input: `H_pre x_l`, the convex read of the lanes."""
        if self._lanes is None:
            self.begin(x)
        return self.read_in()

    def mappings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eq. 8's constrained mappings `(H_pre, H_post, H_res)` for the current lanes."""
        if self._lanes is None:
            raise LoudError(
                "MHC.mappings before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the lanes",
            )
        lanes = self._lanes
        flat = lanes.reshape(*lanes.shape[:-2], self.n_streams * self.d_model).float()
        xn = _rms(flat, EPS)
        pre = self.alpha_pre * (xn @ self.phi_pre) + self.b_pre
        post = self.alpha_post * (xn @ self.phi_post) + self.b_post
        res = self.alpha_res * (xn @ self.phi_res) + self.b_res.reshape(-1)
        shape = (*lanes.shape[:-2], self.n_streams, self.n_streams)
        return torch.sigmoid(pre), 2.0 * torch.sigmoid(post), sinkhorn_knopp(res.reshape(shape))

    def read_in(self) -> torch.Tensor:
        """Return `H_pre x_l`, a convex mix: the lanes are copies of one state, not slices."""
        if self._lanes is None:
            raise LoudError(
                "MHC.read_in before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the lanes",
            )
        lanes = self._lanes
        flat = lanes.reshape(*lanes.shape[:-2], self.n_streams * self.d_model).float()
        pre = torch.sigmoid(self.alpha_pre * (_rms(flat, EPS) @ self.phi_pre) + self.b_pre)
        pre = pre / pre.sum(-1, keepdim=True)
        return (pre.unsqueeze(-1) * lanes).sum(-2)

    def write(self, u: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Eq. 3: `x_{l+1} = H_res x_l + H_post^T u`, then the next read."""
        if self._lanes is None:
            raise LoudError(
                "MHC.write before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the lanes",
            )
        _pre, post, h_res = self.mappings()
        lanes = self._lanes.float()
        mixed = torch.einsum("...ij,...jd->...id", h_res, lanes)
        self._lanes = mixed + post.unsqueeze(-1) * u.float().unsqueeze(-2)
        if counters is not None:
            counters.count("mhc_mix")
        return self.read_in()

    def read(self) -> torch.Tensor:
        """Return the read-out: the mean of the lanes, `sum_j x_l[j] / n`."""
        if self._lanes is None:
            raise LoudError(
                "MHC.read before begin",
                "call begin(x) at the top of the stack's forward: the mixer owns the lanes",
            )
        return self._lanes.float().mean(-2)

    def extra_repr(self) -> str:
        """Print the expansion rate the lanes are priced at."""
        return f"d_model={self.d_model}, n_streams={self.n_streams}"
