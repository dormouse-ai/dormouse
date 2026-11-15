"""The attention stack: nine KDA layers and three DSA layers, in a 3:1 pattern.

THE PATTERN, AND WHY IT IS NOT A CONFIG VALUE. One period is

    [kda, kda, kda, dsa]

and the stack is that period repeated to `n_layers` (three quarters KDA, one quarter
DSA). KDA carries the short-range work with a fixed-size recurrent state and no
selector that can miss; DSA adds the arbitrary long-range lookup KDA cannot do. The
same 3:1 ratio is what Kimi Linear 2510.26692 §5.2 and Olmo Hybrid 2604.03444
measure. It is stated here and nowhere else: a config cannot rename it, reorder it or
ask for a fourth mechanism, because the previous version of this file offered exactly
that menu and a menu is how an arm gets measured without running.

`n_layers` must be a multiple of the period. A trailing partial repetition is
refused, not absorbed: "10 layers of 3:1" would mean 3:1 for eight layers and
something else for two, which is not the architecture anyone measured.

WHY COEFFECTS AND NOT A BOOLEAN. Each mechanism declares what it needs from the
context and what it provides (`Coeffects`, `dormouse_core/coeffects.py`). The stack
checks those declarations before it allocates a single parameter, so an impossible
combination dies at construction with a named conflict instead of training: a
mechanism that wants RoPE (none here does), or two mechanisms claiming one layer's
state, are both refused with the names in the message.

NO ROTARY TABLE. Both mechanisms carry their own ordering — KDA's decay is a
learnable, data-dependent position signal and DSA's selection is over content — so
the stack declares `attn.pos="none"` and nobody has to guess which one is inside.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from dormouse_core.coeffects import Coeffects
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..blocks import PROJ_STD
from .dsa import DSA, WINDOW, K
from .kda import KDA

#: The two mechanisms this stack assembles, in the names a refusal prints.
KINDS: tuple[str, ...] = ("kda", "dsa")

#: The 3:1 recipe, one period of it — the only pattern this stack builds.
HYBRID_3_TO_1: tuple[str, ...] = ("kda", "kda", "kda", "dsa")


def _coeffects(name: str) -> Coeffects:
    """Return the declaration of one mechanism by name."""
    if name == "kda":
        return KDA.COEFFECTS
    if name == "dsa":
        return DSA.COEFFECTS
    raise LoudError(
        f"{name!r} is not an attention mechanism",
        f"use one of: {', '.join(KINDS)}",
    )


def check_compatible(mechanisms: Sequence[str], ctx: Mapping[str, str]) -> None:
    """Refuse a combination whose coeffects contradict, naming the conflict.

    Called before anything is allocated, so a refusal costs no memory and leaves no
    half-built module behind.
    """
    claimed: dict[str, str] = {}
    for name in mechanisms:
        declared = _coeffects(name)
        gap = declared.unsatisfied(dict(ctx))
        if gap:
            raise LoudError(
                f"{name} cannot take this context: it {gap}",
                "declare the context the mechanism needs ("
                + ", ".join(f"{k}={v!r}" for k, v in declared.requires.items())
                + "), or pick a mechanism that runs here",
            )
        for key in declared.exclusive:
            owner = claimed.get(key)
            if owner is not None and owner != name:
                raise LoudError(
                    f"{owner} and {name} both claim {key!r} on one layer",
                    "one mechanism per layer: put the other one in a different "
                    "layer (the 3:1 hybrid does exactly that)",
                )
            claimed[key] = name


class HybridStack(nn.Module):
    """`[kda]x3 + [dsa]x1` per period: the attention slot, as twelve layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        n_layers: int = 12,
        n_kv_heads: int | None = None,
        c_lat: int | None = None,
        chunk: int = 64,
        k: int = K,
        window: int = WINDOW,
        context: Mapping[str, str] | None = None,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Validate the pattern against the coeffects, then build the layers.

        `context` is the host's declaration of the shared context. It overrides the
        defaults, which is how a contradiction (`{"attn.pos": "rope"}`) is caught
        rather than quietly ignored.
        """
        super().__init__()
        period = HYBRID_3_TO_1
        if n_layers < 1:
            raise LoudError(
                f"n_layers={n_layers} assembles no attention",
                "a stack needs at least one layer",
            )
        if n_layers % len(period):
            raise LoudError(
                f"n_layers={n_layers} is not a multiple of the {len(period)}-layer "
                f"period {list(period)}: {n_layers % len(period)} layers are left over",
                "change n_layers to a multiple of 4, or the ratio is no longer 3:1",
            )
        required = {k: v for name in period for k, v in _coeffects(name).requires.items()}
        self.context = {**required, **dict(context or {})}
        for name in period:
            check_compatible([name], self.context)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.period = period
        names = period * (n_layers // len(period))
        self.layers = nn.ModuleList(
            [
                KDA(
                    d_model,
                    n_heads,
                    head_dim,
                    n_kv_heads=n_kv_heads,
                    c_lat=c_lat,
                    chunk=chunk,
                    write_std=write_std,
                    device=device,
                )
                if name == "kda"
                else DSA(
                    d_model,
                    n_heads,
                    head_dim,
                    n_kv_heads=n_kv_heads,
                    c_lat=c_lat,
                    k=k,
                    window=window,
                    write_std=write_std,
                    device=device,
                )
                for name in names
            ]
        )

    def layer_types(self) -> tuple[str, ...]:
        """Return the mechanism of every layer, in build order."""
        return self.period * (len(self.layers) // len(self.period))

    def coeffects(self) -> Coeffects:
        """Declare the context the stack needs and the slot it fills."""
        return Coeffects(
            requires={k: v for name in self.period for k, v in _coeffects(name).requires.items()},
            provides={"attn.stack": "hybrid"},
        )

    def extra_repr(self) -> str:
        """Print the period, not twelve module reprs."""
        return f"n_layers={len(self.layers)}, period={list(self.period)}"

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` in, the residual stream out."""
        if counters is not None:
            counters.count("hybrid_fwd", x.shape[0] * x.shape[1])
        for name, layer in zip(self.layer_types(), self.layers, strict=True):
            x = x + layer(x, counters)
            if counters is not None:
                counters.count("hybrid_layer_fwd")
                counters.count(f"hybrid_{name}_layer_fwd")
        return x

    def param_report(self) -> dict[str, int]:
        """Return the bill per layer type and in total, counted from the modules."""
        per = dict.fromkeys(KINDS, 0)
        seen: set[str] = set()
        for name, layer in zip(self.layer_types(), self.layers, strict=True):
            if name not in seen:
                seen.add(name)
                per[name] = sum(p.numel() for p in layer.parameters())
        counts = Counter(self.layer_types())
        report = {f"per_{kind}": per[kind] for kind in KINDS}
        report.update({kind: per[kind] * counts[kind] for kind in KINDS})
        report["total"] = sum(p.numel() for p in self.parameters())
        return report
