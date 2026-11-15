"""The gated SwiGLU feed-forward, and the two learned-activation arms.

The baseline formula (Shazeer 2020, "GLU variants improve transformer", §2
SwiGLU; Touvron et al. 2023 LLaMA §2):

    h = silu(W_gate · x) ⊙ (W_up · x)          # ⊙ is elementwise
    y = W_down · h

Three matrices, no biases, no rank: `3·d·d_ffn` parameters. The bias-free form is
the measured recipe's, and a bias on a projection that is immediately normalised
upstream (`norm2`) and added downstream (`x + y`) only moves the residual's mean,
which the next pre-norm removes again.

WHY 2048 AT d=832 — THE PARAMETER-NEUTRAL WIDTH. A GELU block of the same bill is
`2·d·d_ffn` (two matrices). Swapping in a third matrix at `d_ffn = 3072` would be
`3·832·3072 = 7.67M` against the GELU block's `2·832·3072 = 5.11M`: a 50% bigger
model reporting the gate's gain. `neutral_width` holds the bill equal —
`3·d·f = 2·d·f_gelu` gives `f = 2048`, and `3·832·2048 == 2·832·3072 == 5 111 808`
exactly, so a BPB difference is the activation and not the parameter count.

WHY `silu` AND NOT A SWITCH. SwiGLU's `silu(g) = g·sigmoid(g)` is the variant the
2026 measurement is on (-0.0145 BPB, t=-19, at 0% parameter cost). SiTU/MemGLU
were untested arms of the old code whose constants (`beta1`, `beta2`, `c0`)
reached a forward that the live config never selected. A parameter that no live
recipe sets is a place for a silent default, so the live activation is one line,
not a table — plus the two arms below, which exist only when a config names them.

WHY `la` AND `moa` (arXiv 2605.26647 Tables 1-2, 0.12B dense, terminal loss
against SwiGLU): a learnable linear combination of a fixed dictionary is worth
one-LA -0.015, bi-LA -0.016, one-MoA -0.011, bi-MoA -0.016, and bi-MoA with
sigmoid gating -0.029. The dictionary is the bill's whole trick: LA adds `K`
scalars and MoA adds `d·K + 2K` (a `d → K` gate, its bias, and `K` priors), so
the extra parameters are O(d) with `K` a small constant. The same weights at
`K=1` must be the plain FFN bit for bit — `dictionary=("silu",)` or `("situ",)`
is the degeneracy gate below, and it is `torch.equal`, not `allclose`, because a
mixture that is merely close is a mixture wired wrong.

THE TWO ARMS, IN ONE BREATH. `LaFFN` mixes the dictionary with coefficients that
depend on training step only (`A(z) = Σ cₖ·fₖ(z)`, `cₖ = 1` at init). `MoaFFN`
mixes it with per-token coefficients from a small `d → K` linear through a
sigmoid (`cₖ(x) = αₖ·sigmoid(Wₖ·x + bₖ)`, normalised by their sum), on the gate branch
only (`one`) or on both branches (`bi`, the paper's best and the default).
Normalising the sigmoid gate is what makes the single-entry dictionary collapse
to `1·f(z)` exactly and keeps the `bi` product within the dictionary's own
bounds — both are assertions in `tests/test_ffn_variants.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .activations import DEFAULT_DICTIONARY, activation, resolve
from .tsct import low_rank

__concepts__ = ("GatedFFN", "LaFFN", "MoaFFN")

#: Std of the two `d -> d_ffn` halves; the residual-writing `down` takes the
#: caller's `write_std` (the depth-dependent scale, see `model.Transformer`).
PROJ_STD = 0.02

#: How many branches the mixture is applied to: the gate only, or gate and up.
FFN_SIDES: tuple[str, ...] = ("one", "bi")


def neutral_width(d_ffn_gelu: int) -> int:
    """Return the gated width whose bill equals a GELU block's: `2·d_ffn_gelu / 3`.

    Rounded DOWN so the gated block can be at most the GELU block's size — a
    rounding error may cost a few parameters, never add a claim.
    """
    if d_ffn_gelu < 3:
        raise LoudError(
            f"d_ffn_gelu={d_ffn_gelu} is too narrow to compare against",
            "pass the GELU control's own width, e.g. 3072 at d_model=832",
        )
    return 2 * d_ffn_gelu // 3


def _enabled(coef: torch.Tensor) -> list[int]:
    """Return the indices of coefficients that are switched on (non-zero)."""
    return [i for i, c in enumerate(coef.detach()) if float(c) != 0.0]


class _Glu(nn.Module):
    """The three bias-free projections every arm shares, built and drawn once.

    Private because it is the bill, not a concept: `GatedFFN`, `LaFFN` and
    `MoaFFN` differ only in what happens between `gate`/`up` and `down`, so the
    parameter-neutral width argument applies to all of them by construction.

    `tsct_rank` is the compute path, not a second bill: above zero the three
    matrices are read through their rank-`rank` spectrum (`tsct.py`) while the
    parameters stay the dense `d·f` masters, so `state_dict()` and the parameter
    count are the dense ones on both settings.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        *,
        write_std: float,
        device: torch.device | str | None,
        tsct_rank: int = 0,
    ) -> None:
        """Build the three matrices and draw their weights."""
        super().__init__()
        if d_ffn < 1:
            raise LoudError(
                f"d_ffn={d_ffn} is not a width",
                "use a positive width; 2048 is the parameter-neutral one at d_model=832",
            )
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.tsct_rank = tsct_rank
        self.gate = nn.Linear(d_model, d_ffn, bias=False, device=device)
        self.up = nn.Linear(d_model, d_ffn, bias=False, device=device)
        self.down = nn.Linear(d_ffn, d_model, bias=False, device=device)
        nn.init.normal_(self.gate.weight, std=PROJ_STD)
        nn.init.normal_(self.up.weight, std=PROJ_STD)
        nn.init.normal_(self.down.weight, std=write_std)

    def _read(self, linear: nn.Linear, x: torch.Tensor, counters: Counters | None) -> torch.Tensor:
        """One of the three matrices, dense or through its rank-`r` factors."""
        return low_rank(linear, x, self.tsct_rank, counters)


class GatedFFN(_Glu):
    """`x [n, d] -> [n, d]` through `down(act(gate(x)) * act?(up(x)))`.

    `act="silu", sides="one"` is the live SwiGLU and the control every arm is
    measured against. `sides="bi"` applies the activation to the up branch too;
    it exists here so a bi-sided mixture has a plain FFN to degenerate to.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int = 2048,
        *,
        act: str = "silu",
        sides: str = "one",
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
        tsct_rank: int = 0,
    ) -> None:
        """Build the three matrices and resolve the activation by name."""
        super().__init__(d_model, d_ffn, write_std=write_std, device=device, tsct_rank=tsct_rank)
        if sides not in FFN_SIDES:
            raise LoudError(
                f"sides={sides!r} is not an FFN topology",
                f"use one of: {', '.join(FFN_SIDES)}",
            )
        self.act = act
        self.sides = sides
        self._fn: Callable[[torch.Tensor], torch.Tensor] = activation(act)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the FFN write, same shape."""
        g, u = self._read(self.gate, x, counters), self._read(self.up, x, counters)
        h = self._fn(g) * (self._fn(u) if self.sides == "bi" else u)
        if counters is not None:
            counters.count("ffn_fwd", x.shape[0] * x.shape[1])
        return self._read(self.down, h, counters)

    def extra_repr(self) -> str:
        """Show the shape the parameter bill is quoted for."""
        return f"d_model={self.d_model}, d_ffn={self.d_ffn}, act={self.act}, sides={self.sides}"


class LaFFN(_Glu):
    """Learnable-activation FFN: `down(A(gate(x)) * up(x))`, `A = Σ cₖ·fₖ`.

    The coefficients are input-independent and start at 1 (the paper's choice),
    so the arm starts as the sum of the dictionary rather than a random mixture.
    `K` parameters per layer, on the gate branch only: applying the mixture to
    the up branch as well is bi-LA, which costs a second `K` and is not in this
    task's parameter gate.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int = 2048,
        *,
        dictionary: Sequence[str] = DEFAULT_DICTIONARY,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
        tsct_rank: int = 0,
    ) -> None:
        """Build the three matrices, resolve the dictionary, and set every `cₖ = 1`."""
        super().__init__(d_model, d_ffn, write_std=write_std, device=device, tsct_rank=tsct_rank)
        self.dictionary = resolve(dictionary)
        self._fns: tuple[Callable[[torch.Tensor], torch.Tensor], ...] = tuple(
            activation(name) for name in self.dictionary
        )
        self.coef = nn.Parameter(torch.ones(len(self.dictionary), device=device))

    def _mix(self, z: torch.Tensor) -> torch.Tensor:
        """`Σ cₖ·fₖ(z)` — a running sum, so `K=1` returns `1·f(z)` untouched."""
        out = self.coef[0] * self._fns[0](z)
        for k in range(1, len(self._fns)):
            out = out + self.coef[k] * self._fns[k](z)
        return out

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the FFN write, same shape."""
        h = self._mix(self._read(self.gate, x, counters)) * self._read(self.up, x, counters)
        if counters is not None:
            tokens = x.shape[0] * x.shape[1]
            counters.count("ffn_fwd", tokens)
            counters.count("ffn_la", tokens)
            counters.count("ffn_dict_entries_used", len(_enabled(self.coef)))
        return self._read(self.down, h, counters)

    def dead_entries(self) -> tuple[str, ...]:
        """Report entries that are switched on but received no gradient.

        The `MissingReport` discipline applied to the mixture: an entry with a
        non-zero coefficient whose gradient is exactly zero is a silently dead
        entry — the mixture claims it and no step can ever move it. Call after
        `backward()`; before it, every enabled entry is reported as dead because
        there is no gradient yet to prove otherwise.
        """
        enabled = _enabled(self.coef)
        if self.coef.grad is None:
            return tuple(self.dictionary[k] for k in enabled)
        return tuple(self.dictionary[k] for k in enabled if float(self.coef.grad[k]) == 0.0)

    def extra_repr(self) -> str:
        """Show the width and the dictionary the coefficients index."""
        return f"d_model={self.d_model}, d_ffn={self.d_ffn}, dict={','.join(self.dictionary)}"


class MoaFFN(_Glu):
    """Mixture-of-activations FFN with per-token, input-dependent coefficients.

    `cₖ(x) = αₖ·sigmoid(Wₖ·x + bₖ)`, normalised by their sum, so the mixture is a
    convex combination of the dictionary. `sides="bi"` (the default, and the
    paper's best) applies it to both branches: `down(A(x,g) · A(x,u))`.

    THE SIGMOID IS NORMALISED, AND WHY THAT MATTERS. The paper's ablation has
    sigmoid gating at -0.029 against SwiGLU, softmax at -0.016 and tanh at
    -0.002; a raw sigmoid gives `K` coefficients in (0,1) that do not sum to 1.
    Dividing by their sum costs nothing, makes `K=1` exactly `1·f(z)` (so a
    one-entry dictionary is the plain FFN, `torch.equal`), and keeps the `bi`
    product inside the dictionary's own bound — the property that makes the arm
    a candidate for low-bit activations at all. `αₖ` starts at 1 so the initial
    mixture is a uniform prior over the dictionary.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int = 2048,
        *,
        dictionary: Sequence[str] = DEFAULT_DICTIONARY,
        sides: str = "bi",
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
        tsct_rank: int = 0,
    ) -> None:
        """Build the projections, the `d → K` gate, and the `K` priors."""
        super().__init__(d_model, d_ffn, write_std=write_std, device=device, tsct_rank=tsct_rank)
        if sides not in FFN_SIDES:
            raise LoudError(
                f"sides={sides!r} is not a mixture topology",
                f"use one of: {', '.join(FFN_SIDES)}",
            )
        self.dictionary = resolve(dictionary)
        self._fns: tuple[Callable[[torch.Tensor], torch.Tensor], ...] = tuple(
            activation(name) for name in self.dictionary
        )
        self.sides = sides
        k = len(self.dictionary)
        self.mix = nn.Linear(d_model, k, bias=True, device=device)
        nn.init.normal_(self.mix.weight, std=PROJ_STD)
        nn.init.zeros_(self.mix.bias)
        self.coef = nn.Parameter(torch.ones(k, device=device))

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token mixture weights `[..., K]`, a convex combination."""
        w = torch.sigmoid(self.mix(x)) * self.coef
        return w / w.sum(-1, keepdim=True)

    def _mix(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """`Σₖ wₖ·fₖ(z)` with `w [..., K]` — a running sum, exact at `K=1`."""
        out = w[..., 0:1] * self._fns[0](z)
        for k in range(1, len(self._fns)):
            out = out + w[..., k : k + 1] * self._fns[k](z)
        return out

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the FFN write, same shape."""
        w = self._weights(x)
        g, u = self._read(self.gate, x, counters), self._read(self.up, x, counters)
        h = self._mix(w, g) * (self._mix(w, u) if self.sides == "bi" else u)
        if counters is not None:
            tokens = x.shape[0] * x.shape[1]
            counters.count("moa_fwd", tokens)
            counters.count("ffn_dict_entries_used", len(_enabled(self.coef)))
            counters.count("ffn_fwd", tokens)
        return self._read(self.down, h, counters)

    def dead_entries(self) -> tuple[str, ...]:
        """Report entries switched on but receiving no gradient, through either path.

        Both paths are checked: the entry's prior `αₖ` and its gate row `Wₖ`.
        A gate row at zero gradient is a token-independent entry wearing an
        input-dependent name — the defect class the counters exist for.
        """
        enabled = _enabled(self.coef)
        if self.coef.grad is None or self.mix.weight.grad is None:
            return tuple(self.dictionary[k] for k in enabled)
        return tuple(
            self.dictionary[k]
            for k in enabled
            if float(self.coef.grad[k]) == 0.0 or float(self.mix.weight.grad[k].abs().sum()) == 0.0
        )

    def extra_repr(self) -> str:
        """Show the width, the topology and the dictionary the gates index."""
        return (
            f"d_model={self.d_model}, d_ffn={self.d_ffn}, "
            f"dict={','.join(self.dictionary)}, sides={self.sides}"
        )
