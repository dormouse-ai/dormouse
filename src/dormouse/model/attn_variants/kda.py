"""Kimi Delta Attention: the delta rule with a per-channel diagonal decay.

THE MECHANISM, per head (Kimi Linear 2510.26692 eq. 1; Kimi K3 2607.24653 §3.1;
the rule itself is Gated DeltaNet 2412.06464 §2, after DeltaNet 2102.11174):

    S_t = (I - β_t k_t k_tᵀ) Diag(alpha_t) S_{t-1} + β_t k_t v_tᵀ     S_t ∈ R^{d_kxd_v}
    o_t = S_tᵀ q_t                                               o_t ∈ R^{d_v}
    alpha_t = exp(-softplus(W_alpha x_t + b_alpha)) ∈ (0, 1)^{d_k}           per-channel decay
    β_t = sigmoid(W_β x_t) ∈ (0, 1)                              write strength
    q_t, k_t = L2Norm(Swish(ShortConv(W_q x_t)))                 eq. 1's stability fix

Two things make this KDA and not linear attention, and both are load-bearing:

* the DELTA RULE. `I - β k kᵀ` is the gradient step of
  `L_t(S) = ½‖Sᵀk_t - v_t‖²` (Kimi eq. 2.2), so the state is corrected toward
  `k_t ↦ v_t` instead of merely accumulated. A plain `S ← S + β k vᵀ` is a
  different operator with different interference behaviour.
* the DECAY `Diag(alpha_t)` on the key channels — Kimi's refinement of GDN's scalar
  gate, and the reason this layer needs no RoPE (§6.1: the decay is a learnable,
  data-dependent position signal).

`delta=False` or `decay=False` is therefore refused in `__init__` rather than
silently training something else. That is the defect class this project paid for
once already: a `use_kda=false` arm that measured "linear attention wins" while
the mechanism was not running at all.

THE CHUNK FORM. Chunks of C tokens: every product inside a chunk is a matmul and
only the state carry is sequential (Kimi eq. 2-9). With `a_t = ∏_{i≤t} alpha_i` the
cumulative decay inside a chunk, the unrolled state is

    S_t = Diag(a_t) S_0 + Σ_{i≤t} Diag(a_t / a_i) k_i y_iᵀ

and substituting it into eq. 1 gives a lower-triangular system for the "pseudo"
values `y` (Kimi eq. 4-7; their Γ^{i→r} ⊙ K is exactly (a_r / a_i) ⊙ k):

    A[t,i] = β_t ⟨a_t ⊙ k_t, k_i / a_i⟩   for i < t, unit diagonal
    y      = A⁻¹ · Diag(β) · (v - (a ⊙ k) S_0)                  eq. 7: U - W S_0
    o_t    = (a_t ⊙ q_t) S_0 + Σ_{i≤t} ⟨a_t ⊙ q_t, k_i / a_i⟩ y_i    eq. 9
    S_C    = Diag(a_C) S_0 + Σ_i Diag(a_C / a_i) k_i y_iᵀ           eq. 8

The one departure from the paper's algebra is numerical, not mathematical: eq. 6
and 9 divide by Γ, and §3.2 names that division as the reason the DPLR
formulation needs secondary chunking in fp32. Here the ratio `a_t / a_i ≤ 1` is
formed directly on the strictly-lower pairs (one CxCxd_k tensor per chunk), so no
`1/Γ` is ever materialised: same number, without the exponent range.

`recurrent` is eq. 1 verbatim, one token at a time. It is the anchor the chunk
path is tested against, and it is what a sampler runs.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.coeffects import Coeffects
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..blocks import PROJ_STD
from .projections import LatentProjections

#: Per-channel decay bias at init. `alpha = exp(-softplus(-4)) ≈ 0.982` per step, a
#: half-life of ~38 tokens: long enough that a 64-token chunk carries information
#: across itself, short enough that the state is a rolling memory and not a sum
#: that grows with the window (the bounded-state gate measures exactly this).
A_BIAS_INIT = -4.0

#: Recompute the chunk graph in the backward instead of retaining it. The graph is
#: `O(T/chunk)` tensors of `[batch, heads, chunk, chunk, head_dim]` — measured 987 MiB
#: per layer at b1 x s1024, 8.7 GiB over the nine KDA layers, which is what does not fit
#: on a 16 GiB card; the trade is one extra forward of the chunk loop in the backward
#: (`tests/test_kda.py` pins the arithmetic under both settings).
CHECKPOINT_CHUNKS = True


class KDA(LatentProjections):
    """Chunkwise KDA: `W_proj(gate · RMSNorm(chunk_delta_rule(q, k, v, alpha, β))))`."""

    #: What this mechanism needs and provides (Kimi §6.1: no RoPE inside it).
    COEFFECTS = Coeffects(
        requires={"attn.pos": "none", "attn.rule": "delta"},
        provides={"attn.state": "recurrent", "attn.decay": "diagonal"},
        exclusive=("attn.state", "attn.decay"),
    )

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        n_kv_heads: int | None = None,
        c_lat: int | None = None,
        chunk: int = 64,
        conv_width: int = 4,
        delta: bool = True,
        decay: bool = True,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the projections, the short conv and the two gates.

        `n_kv_heads`/`c_lat` are the GQA and bottleneck widths (`projections.py`);
        their degenerate setting is the pre-change layer. `delta` and `decay` exist
        only so that the degeneracy is refusable: both `False` values raise here,
        because neither is a configuration of KDA.
        """
        super().__init__(
            d_model,
            n_heads,
            head_dim,
            n_kv_heads=n_kv_heads,
            c_lat=c_lat,
            device=device,
        )
        if not delta:
            raise LoudError(
                "KDA(delta=False): the delta rule is not a switch, it is the mechanism",
                "keep delta=True; a state written as `S ← S + β k vᵀ` is plain linear "
                "attention, which belongs to another arm (and to a paper that measures it)",
            )
        if not decay:
            raise LoudError(
                "KDA(decay=False): without Diag(alpha_t) this is not Kimi Delta Attention",
                "keep decay=True (Kimi Linear §3 refines GDN's gate to a diagonal one); "
                "an undecayed recurrence is Gated DeltaNet with alpha=1, not KDA",
            )
        if chunk < 1:
            raise LoudError(
                f"chunk={chunk} holds no positions",
                "one chunk is the smallest unit the chunkwise form can run; 64 is the recipe",
            )
        if conv_width < 2:
            raise LoudError(
                f"conv_width={conv_width}: a short conv of one tap is a scale",
                "use 4 (Kimi's ShortConv), or drop the conv if that is the arm you mean",
            )
        self.chunk = chunk
        self.scale = head_dim**-0.5
        #: Swish, as a module so that the call is typed rather than `Any`.
        self.activation = nn.SiLU()
        #: Depthwise, left-padded: tap t reads t-3..t, never t+1 (causal by shape).
        #: It runs on the PACKED `q|k|v` width, i.e. before the heads are split.
        self.conv = nn.Conv1d(
            self.packed,
            self.packed,
            conv_width,
            groups=self.packed,
            padding=conv_width - 1,
            device=device,
        )
        #: Kimi's decay is a low-rank projection of rank = head_dim, then f(·).
        self.a_down = nn.Linear(d_model, head_dim, bias=False, device=device)
        self.a_up = nn.Linear(head_dim, self.d_kv, bias=False, device=device)
        self.a_bias = nn.Parameter(torch.full((n_heads, head_dim), A_BIAS_INIT, device=device))
        self.beta = nn.Linear(d_model, n_heads, bias=True, device=device)
        #: Kimi's output gate: `sigmoid(W_g x)` before the write projection.
        self.gate = nn.Linear(d_model, self.d_kv, bias=False, device=device)
        self.head_norm = nn.Parameter(torch.ones(head_dim, device=device))
        #: `d_kv x d_model`: the attention width is n_heads·head_dim, which is not
        #: d_model (at d=832 with 8 heads of 64 it is 512).
        self.proj = nn.Linear(self.d_kv, d_model, bias=False, device=device)
        for linear in (self.a_down, self.a_up, self.gate, self.beta):
            nn.init.normal_(linear.weight, std=PROJ_STD)
        nn.init.zeros_(self.beta.bias)
        nn.init.normal_(self.conv.weight, std=PROJ_STD)
        nn.init.normal_(self.proj.weight, std=write_std)

    def coeffects(self) -> Coeffects:
        """Declare the context this mechanism needs and the state it provides."""
        return self.COEFFECTS

    def extra_repr(self) -> str:
        """Print the kernel and the compression (the base class prints the widths)."""
        return f"chunk={self.chunk}, " + super().extra_repr()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the attention write, same shape."""
        return self.run(x, counters)[0]

    def run(
        self, x: torch.Tensor, counters: Counters | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the write and the final recurrent state `S_T` of every head.

        The state is returned, not hidden, because "the state does not grow with
        the window" is a property of this tensor and nothing else.
        """
        batch, seq, _ = x.shape
        if seq % self.chunk:
            raise LoudError(
                f"a {seq}-token forward does not tile chunk={self.chunk}: "
                f"{seq % self.chunk} tokens are left over",
                f"pad the window to a multiple of {self.chunk}, or build KDA with a "
                "chunk that divides it",
            )
        q, k, v, alpha, beta = self._project(x)
        #: GQA: the kernel keeps one state per QUERY head, so a shared key/value head
        #: is broadcast to the heads that read it. `n_kv_heads == n_heads` skips this
        #: entirely, which is what makes the uncompressed layer bit-identical.
        groups = self.n_heads // self.n_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=2)
            v = v.repeat_interleave(groups, dim=2)
        out, state = self._chunkwise(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            alpha.transpose(1, 2),
            beta.transpose(1, 2),
        )
        y = self._write(x, out.transpose(1, 2))
        if counters is not None:
            counters.count("kda_fwd", batch * seq)
            counters.count("kda_chunks", batch * (seq // self.chunk))
            if y.requires_grad:
                y.register_hook(  # type: ignore[no-untyped-call]
                    lambda _: counters.count("kda_bwd", batch * seq)
                )
        return y, state

    def recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 1 verbatim: the O(T) reference the chunk path is checked against.

        Shapes are the kernel's: `q, k, v, alpha [batch, heads, seq, head_dim]`
        and `beta [batch, heads, seq]`.
        """
        batch, heads, seq, width = q.shape
        state = q.new_zeros(batch, heads, width, width)
        out = []
        for t in range(seq):
            kt = k[:, :, t]
            step = beta[:, :, t].view(batch, heads, 1, 1)
            state = state * alpha[:, :, t].unsqueeze(-1)
            state = state - step * kt.unsqueeze(-1) * (kt.unsqueeze(-2) @ state)
            state = state + step * kt.unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
            out.append(torch.einsum("bhdv,bhd->bhv", state, q[:, :, t]))
        return torch.stack(out, dim=2), state

    # -------------------------------------------------------------- internals
    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x` → `q, k, v [b, s, h, d]`, `alpha [b, s, h, d]`, `beta [b, s, h]`.

        `q` is at `n_heads` and `k`/`v` at `n_kv_heads`; the decay and the write
        strength are per query head, as they were before the compression existed.
        """
        batch, seq, _ = x.shape
        packed = self.project(x).transpose(1, 2)
        packed = self.activation(self.conv(packed)[..., :seq]).transpose(1, 2)
        q, k, v = self.split(packed)
        q = nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = nn.functional.normalize(k, dim=-1, eps=1e-6)
        decay = (
            self.a_up(self.activation(self.a_down(x))).view(batch, seq, self.n_heads, self.head_dim)
            + self.a_bias
        )
        alpha = torch.exp(-nn.functional.softplus(decay))
        return q, k, v, alpha, torch.sigmoid(self.beta(x))

    def _write(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Head-wise RMSNorm, the output gate, then the residual-writing projection."""
        batch, seq, _ = x.shape
        normed = nn.functional.rms_norm(out, (self.head_dim,), self.head_norm, 1e-6)
        gated = normed * torch.sigmoid(self.gate(x)).view_as(normed)
        return cast(torch.Tensor, self.proj(gated.reshape(batch, seq, self.d_kv)))

    def _chunkwise(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 2-9 in a loop over chunks: matmuls inside, one carry between."""
        batch, heads, seq, width = q.shape
        chunk = self.chunk
        incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
        strict = torch.tril(incl, -1)
        #: `masked_fill` does not right-align a 2-D mask against a 4-D/5-D
        #: operand, so the masks carry their broadcast axes explicitly.
        keep_pair = incl.view(1, 1, chunk, chunk)
        keep_strict = strict.view(1, 1, chunk, chunk)
        keep_ratio = incl.view(1, 1, chunk, chunk, 1)
        state = q.new_zeros(batch, heads, width, width)
        #: ONE CHECKPOINT PER CHUNK, not one for the loop. The forward keeps the five
        #: inputs and the carry; the backward recomputes a chunk's graph, backprops
        #: through it, and frees it before the next — a whole-loop checkpoint would
        #: rebuild all `seq/chunk` chunk graphs at once in the backward and move the peak
        #: there instead of removing it (measured at b8 x s1024: 2.0 GiB of `ratio` live
        #: with the loop-wide form, and the per-chunk transient is 134 MiB).
        keep = CHECKPOINT_CHUNKS and torch.is_grad_enabled() and q.requires_grad
        masks = (keep_ratio, keep_pair, keep_strict)
        out = []
        for start in range(0, seq, chunk):
            window = slice(start, start + chunk)
            rows = (state, q[..., window, :], k[..., window, :], v[..., window, :])
            gates = (alpha[..., window, :], beta[..., window].unsqueeze(-1))
            if keep:
                carried_out, state = checkpoint(
                    self._chunk, *rows, *gates, *masks, use_reentrant=False
                )
            else:
                carried_out, state = self._chunk(*rows, *gates, *masks)
            out.append(carried_out)
        return torch.cat(out, dim=2), state

    def _chunk(
        self,
        state: torch.Tensor,
        qi: torch.Tensor,
        ki: torch.Tensor,
        vi: torch.Tensor,
        alpha: torch.Tensor,
        step: torch.Tensor,
        keep_ratio: torch.Tensor,
        keep_pair: torch.Tensor,
        keep_strict: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of eq. 2-9: `(o_chunk, S_next)` from `(S_in, q, k, v, a, β)`."""
        a = torch.cumprod(alpha, dim=-2)
        #: The floor is 1e-12 so that a fully decayed channel gives a finite
        #: log and an exact `a_t/a_i = 1` pair instead of `-inf - (-inf)`.
        log_a = torch.log(a.clamp_min(1e-12))
        diff = (log_a.unsqueeze(-2) - log_a.unsqueeze(-3)).masked_fill(~keep_ratio, 0.0)
        #: `a_t / a_i ≤ 1` on the lower triangle: the paper's Γ ratio, taken
        #: pairwise so that no `1/Γ` reaches a matmul (see the module docstring).
        ratio = diff.exp().masked_fill(~keep_ratio, 0.0)
        keyed = a * ki
        as_key = ki.unsqueeze(-3)  # [b, h, 1, i, d]
        gram = (ratio * ki.unsqueeze(-2) * as_key).sum(-1)
        qk = (ratio * qi.unsqueeze(-2) * as_key).sum(-1).masked_fill(~keep_pair, 0.0)
        lower = step * gram.masked_fill(~keep_strict, 0.0)
        pseudo = torch.linalg.solve_triangular(
            lower,
            step * (vi - keyed @ state),
            upper=False,
            unitriangular=True,
        )
        written = (a * qi) @ state + qk @ pseudo
        carried = torch.exp(log_a[..., -1:, :] - log_a)
        next_state = a[..., -1, :].unsqueeze(-1) * state + torch.einsum(
            "bhtd,bhtv->bhdv", carried * ki, pseudo
        )
        return written, next_state
