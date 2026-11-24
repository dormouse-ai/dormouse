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
formulation needs secondary chunking in fp32. Here the division is taken INSIDE one
chunk, where `a` spans at most `chunk` decays (`0.982⁶⁴ ≈ 0.31` at the init, so
`1/a ≤ 3.2`), and it reaches the arithmetic as a matmul operand — `gram = (a ⊙ k)·(k/a)ᵀ`
— rather than as a pairwise `[chunk, chunk, head_dim]` tensor. Same number, no exponent
range, and no 5-D tensor at any size.

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

from ...kernels import mfu40 as mfu40_kernels
from ...kernels.kda_fla import chunk_kda_fla, fla_requested
from ..blocks import PROJ_STD
from .projections import LatentProjections

#: Per-channel decay bias at init. `alpha = exp(-softplus(-4)) ≈ 0.982` per step, a
#: half-life of ~38 tokens: long enough that a 64-token chunk carries information
#: across itself, short enough that the state is a rolling memory and not a sum
#: that grows with the window (the bounded-state gate measures exactly this).
A_BIAS_INIT = -4.0

#: Per-channel decay floor. Below this the chunk's Gamma division reaches
#: `1/alpha >= 1e6` and the triangular solve is ill-conditioned (N2/N6 measured
#: `alpha_min -> 0`, i.e. `1e12`, after ~300 steps at high lr). A collapse is a
#: defect in the decay projection/lr, not a precision edge case to clamp away.
ALPHA_FLOOR = 1e-6

#: Recompute the batched core in the backward instead of retaining it. OFF by default, and
#: that is a change of regime rather than of taste: the graph the per-chunk form could not
#: retain was `[batch, heads, chunk, chunk, head_dim]` PER CHUNK — measured 987 MiB per
#: layer at b1 x s1024, 8.7 GiB over the nine KDA layers. The batched core's graph is
#: `[batch, heads, chunks, chunk, chunk]`: 67 MiB per layer at b8 x s1024, 0.4 GiB over the
#: nine, so keeping it costs 0.4 GiB of peak and saves one recomputed forward per layer
#: (measured at b8 x s1024: 81.1 ms per layer recomputing, 53.6 ms retaining). Turn it on
#: for a card that cannot spare those 0.4 GiB.
CHECKPOINT_CHUNKS = False


class KDA(LatentProjections):
    """Chunkwise KDA: `W_proj(gate · RMSNorm(chunk_delta_rule(q, k, v, alpha, β))))`."""

    #: Declared for the type checker: `register_buffer` returns nothing.
    _alpha_floor_hit: torch.Tensor

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
        #: Sticky device-side record of a collapsed decay, set without a host read so a
        #: captured step stays capturable. The eager half reads it and refuses; a graph
        #: reads it after replay (train/graph.py), the same seam `_overflow` uses.
        self.register_buffer(
            "_alpha_floor_hit",
            torch.zeros((), dtype=torch.bool, device=device),
            persistent=False,
        )
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
        #: Eager forwards refuse a collapsed decay before feeding 1/alpha into the
        #: solve. A capture records the flag instead: no device read may sit inside
        #: it, so graph.py reads the same flag after replay.
        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            self.check_alpha_floor()
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
        #: The log-decay is exponentiated by the recurrence, so it must be fp32
        #: BEFORE anything rounds it (D01 B4). `decay.float()` after an autocast
        #: `a_up` only upcasts a bf16 value; run the projection itself with autocast
        #: off and promote a low-precision INPUT. A fp64 anchor then stays fp64
        #: end to end rather than being cast down to fp32.
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            work_x = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
            work = (
                self.a_up(self.activation(self.a_down(work_x))).view(
                    batch, seq, self.n_heads, self.head_dim
                )
                + self.a_bias
            )
            alpha = torch.exp(-nn.functional.softplus(work))
        #: No host read here: capture records the flag, the eager half refuses.
        self._alpha_floor_hit |= (alpha < ALPHA_FLOOR).any()
        return q, k, v, alpha, torch.sigmoid(self.beta(x))

    def check_alpha_floor(self) -> None:
        """Named refusal for a decay that collapsed past `ALPHA_FLOOR`.

        Reads the device flag on the eager side only; the captured forward must
        never cross that seam (graph.py checks it after replay instead).
        """
        if bool(self._alpha_floor_hit):
            raise LoudError(
                f"KDA alpha fell below the {ALPHA_FLOOR:g} floor: the Gamma division "
                "reaches 1e6+ and the triangular solve is ill-conditioned",
                "the decay projection is growing without bound at this lr; lower the "
                "lr, freeze a_bias, or bound a_up(a_down(x)) — raising the 1e-12 clamp "
                "in _chunks would only hide the collapse",
            )

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
        """Eq. 2-9 with the chunk axis batched: the matmuls once, the carry per chunk.

        WHY BATCHED AND NOT A LOOP. The only sequential part of the chunk form is the
        state carry; every other term is a function of one chunk alone. Written per
        chunk, the Γ ratio `a_t/a_i` materialises as a `[batch, heads, chunk, chunk,
        head_dim]` tensor that is multiplied, masked and summed several times inside
        every chunk — measured at b8 x s1024: 27 GB of elementwise traffic per layer,
        164 ms of a 6.1 s step for one of the nine KDA layers, 3 133 launches. Batched,
        the same term is `(a ⊙ k) @ (k / a)ᵀ`: one matmul per chunk, and no
        `[·, chunk, chunk, head_dim]` tensor exists at any size.
        """
        #: The Triton backend keeps the state and the log-decay in fp32. CPU (the fp64
        #: anchor) and fp64 stay on the eager path; on CUDA a missing FLA is a LoudError,
        #: never a silent timing of the eager path.
        if (
            fla_requested()
            and q.is_cuda
            and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and self.chunk in (32, 64)
        ):
            return chunk_kda_fla(q, k, v, alpha, beta, chunk=self.chunk)
        batch, heads, _, _ = q.shape
        chunk = self.chunk
        count = q.shape[2] // chunk

        def split(tile: torch.Tensor) -> torch.Tensor:
            """`[b, h, seq, ·]` → `[b, h, chunks, chunk, ·]`, a view."""
            return tile.reshape(*tile.shape[:2], count, chunk, tile.shape[-1])

        gates = beta.reshape(batch, heads, count, chunk)
        core = (split(q), split(k), split(v), split(alpha), gates)
        keep = CHECKPOINT_CHUNKS and torch.is_grad_enabled() and q.requires_grad
        #: ONE CHECKPOINT FOR THE BATCHED CORE, not one per chunk. It retains the five
        #: inputs; what the backward recomputes is a handful of matmuls, where the
        #: per-chunk form recomputed thirteen elementwise passes over a 5-D tensor per
        #: chunk. The graph it removes is `[b, h, chunks, chunk, chunk]` (16.8 MiB at the
        #: live shape) instead of the 5-D tensor (8.7 GiB over the nine layers).
        written: torch.Tensor
        state: torch.Tensor
        #: FLA's WY solve runs in fp32. Under autocast the eager matmuls would
        #: silently drop to bf16, so their gradients diverge by 3-24% and the A/B
        #: would be a dtype comparison, not a kernel comparison. Kimi §3.2 asks for
        #: fp32 here; keep the eager core in the same arithmetic as the Triton one.
        with torch.autocast("cuda", enabled=False):
            if keep:
                written, state = checkpoint(
                    self._chunks, *core, use_reentrant=False, preserve_rng_state=False
                )
            else:
                written, state = self._chunks(*core)
        return written.reshape(batch, heads, -1, written.shape[-1]), state

    def _chunks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 2-9 over every chunk at once, with the carry as the one loop.

        `q, k, v, alpha` are `[batch, heads, chunks, chunk, head_dim]`, `beta` is
        `[batch, heads, chunks, chunk]`. With `a_t = ∏_{i≤t} α_i`:

            A[t, i] = β_t ⟨a_t ⊙ k_t, k_i / a_i⟩          strictly lower, unit diagonal
            pseudo  = A⁻¹ · β(v - (a ⊙ k) S_0)             eq. 7, linear in S_0
            o_t     = (a_t ⊙ q_t) S_0 + A_qt · pseudo      eq. 9
            S_C     = Diag(a_C) S_0 + Σ_i Diag(a_C/a_i) k_i y_iᵀ   eq. 8

        The paper's Γ appears here as `1/a_i`, and it is bounded because it is taken
        inside ONE chunk: `a` spans at most `chunk` decays (0.982⁶⁴ ≈ 0.31 at the init),
        so `1/a <= 3.2` and no exponent range is reached. The 1e-12 floor is the guard
        the logged form carried, so a fully decayed channel still gives a finite ratio
        and an exact `a_t/a_i = 1` pair instead of `-inf - (-inf)`.
        """
        if mfu40_kernels.enabled("KO1"):
            return self._chunks_batched_solves(q, k, v, alpha, beta)
        chunk = q.shape[-2]
        incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
        strict = torch.tril(incl, -1)
        a = torch.cumprod(alpha, dim=-2)
        #: `k_i / a_i`: the Γ division, bounded inside one chunk, and the operand that
        #: turns the pairwise ratio into a matmul (see the module docstring).
        divided = k * a.clamp_min(1e-12).reciprocal()
        keyed = a * k  # `a_t ⊙ k_t`, the left operand of the Γ gram
        as_query = a * q
        divided_t = divided.transpose(-1, -2)
        gram = (keyed @ divided_t).masked_fill(~strict, 0.0)
        qk = (as_query @ divided_t).masked_fill(~incl, 0.0)
        #: `(a_C / a_i) ⊙ k_i`: the carry's operand, per chunk, `[b, h, chunks, chunk, d]`.
        carried = a[..., -1, :].unsqueeze(-2) * divided
        last = a[..., -1, :]
        width = q.shape[-1]
        state = q.new_zeros(*q.shape[:2], width, width)
        #: The solve runs in at least fp32, which is what Kimi §3.2 asks of this
        #: arithmetic and what `triangular_solve` has a kernel for in every backend. The
        #: carry itself stays in the stream's dtype, as it was before.
        work = torch.promote_types(q.dtype, torch.float32)
        out: list[torch.Tensor] = []
        for index in range(q.shape[-3]):
            step = beta[..., index, :, None]
            qi, vi, ki = as_query[..., index, :, :], v[..., index, :, :], keyed[..., index, :, :]
            pseudo = torch.linalg.solve_triangular(
                (step * gram[..., index, :, :]).to(work),
                (step * (vi - ki @ state)).to(work),
                upper=False,
                unitriangular=True,
            ).to(q.dtype)
            out.append(qi @ state + qk[..., index, :, :] @ pseudo)
            state = last[..., index, :].unsqueeze(-1) * state + torch.einsum(
                "bhtd,bhtv->bhdv", carried[..., index, :, :], pseudo
            )
        return torch.stack(out, dim=-3), state

    def _chunks_batched_solves(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KO1: two batched solves; ``A⁻¹(β(v-keyed·S)) = A⁻¹(βv) − (A⁻¹β·keyed)·S``."""
        mfu40_kernels.count("ko1_attn_batched_solves")
        chunk = q.shape[-2]
        incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
        strict = torch.tril(incl, -1)
        a = torch.cumprod(alpha, dim=-2)
        divided = k * a.clamp_min(1e-12).reciprocal()
        keyed = a * k
        as_query = a * q
        divided_t = divided.transpose(-1, -2)
        gram = (keyed @ divided_t).masked_fill(~strict, 0.0)
        qk = (as_query @ divided_t).masked_fill(~incl, 0.0)
        carried = a[..., -1, :].unsqueeze(-2) * divided
        last = a[..., -1, :]
        width = q.shape[-1]
        work = torch.promote_types(q.dtype, torch.float32)
        step = beta.unsqueeze(-1)
        system = (step * gram).to(work)
        part_v = torch.linalg.solve_triangular(
            system, (step * v).to(work), upper=False, unitriangular=True
        ).to(q.dtype)
        part_k = torch.linalg.solve_triangular(
            system, (step * keyed).to(work), upper=False, unitriangular=True
        ).to(q.dtype)
        state = q.new_zeros(*q.shape[:2], width, width)
        out: list[torch.Tensor] = []
        for index in range(q.shape[-3]):
            pseudo = part_v[..., index, :, :] - part_k[..., index, :, :] @ state
            out.append(as_query[..., index, :, :] @ state + qk[..., index, :, :] @ pseudo)
            state = last[..., index, :].unsqueeze(-1) * state + torch.einsum(
                "bhtd,bhtv->bhdv", carried[..., index, :, :], pseudo
            )
        return torch.stack(out, dim=-3), state
