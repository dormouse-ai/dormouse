"""The attention slot: layers of KDA and CSA, and the rule that refuses a bad mix.

THE PATTERN. The live recipe is 3:1 — three recurrent layers per sparse-attention
layer (`COMPATIBILITY.md` §1; the same ratio as Kimi Linear 2510.26692 §5.2,
where KDA carries the short-range work and the full-attention layer carries the
long-range lookups, and as Olmo Hybrid 2604.03444 in its Gated DeltaNet form):

    [KDA, KDA, KDA, CSA] x n,     n = n_layers / 4

The pattern must tile `n_layers` exactly. A trailing partial repetition is
refused here, not absorbed: "10 layers of 3:1" means the ratio is 3:1 for eight
layers and something else for two, which is not the arm anyone measured.
`[KDA] x n` and `[CSA] x n` are the two control arms and tile trivially.

WHY COEFFECTS AND NOT A BOOLEAN. Each mechanism declares what it needs from the
context and what it provides (`Coeffects`, `dormouse_core/coeffects.py`). The
stack checks those declarations before it allocates a single parameter, so an
impossible configuration dies at construction with a named conflict instead of
training. Two rules are enforced:

* a requirement the context does not meet — e.g. `attn.pos="rope"` with a KDA
  layer, which has no rotary table and does not want one (Kimi §6.1);
* two mechanisms on one layer claiming the same exclusive key — KDA keeps a
  recurrent state and CSA keeps a sparse cache, and a layer cannot be both
  (the first row of the conflict matrix).

NO ROTARY TABLE. Both mechanisms carry their own ordering: KDA's decay is a
learnable, data-dependent position signal and CSA's blocks are positional. The
control arm is `model/attn.py`'s dense `Attention`, which keeps RoPE; the hybrid
declares `attn.pos="none"` so nobody has to guess which one is in the stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from dormouse_core.coeffects import Coeffects
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..attn import PROJ_STD
from .csa import CSA
from .kda import KDA

#: The mechanisms this slot can assemble, in the names a pattern uses.
KINDS: tuple[str, ...] = ("KDA", "CSA")

#: The 3:1 recipe, one period of it.
HYBRID_3_TO_1: tuple[str, ...] = ("KDA", "KDA", "KDA", "CSA")


def _coeffects(name: str) -> Coeffects:
    """Return the declaration of one mechanism by name."""
    if name == "KDA":
        return KDA.COEFFECTS
    if name == "CSA":
        return CSA.COEFFECTS
    raise LoudError(
        f"{name!r} is not an attention mechanism",
        f"use one of: {', '.join(KINDS)}",
    )


def check_compatible(mechanisms: Sequence[str], ctx: Mapping[str, str]) -> None:
    """Refuse a combination whose coeffects contradict, naming the conflict.

    Called before anything is allocated, so a refusal costs no memory and leaves
    no half-built module behind.
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
    """`[KDA]x3 + [CSA]x1` per period: the attention slot, as layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        n_layers: int = 12,
        pattern: Sequence[str] | None = None,
        chunk: int = 64,
        csa_chunk: int = 16,
        topk: int = 2,
        hca_group: int = 1,
        context: Mapping[str, str] | None = None,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
    ) -> None:
        """Validate the pattern against the coeffects, then build the layers.

        `context` is the host's declaration of the shared context. It overrides
        the defaults, which is how a contradiction (`{"attn.pos": "rope"}`) is
        caught rather than quietly ignored.
        """
        super().__init__()
        chosen = tuple(pattern) if pattern is not None else HYBRID_3_TO_1
        if n_layers < 1:
            raise LoudError(
                f"n_layers={n_layers} assembles no attention",
                "a stack needs at least one layer",
            )
        if not chosen:
            raise LoudError(
                "pattern=() names no mechanism",
                f"pass one period of the stack, e.g. {list(HYBRID_3_TO_1)}",
            )
        if n_layers % len(chosen):
            raise LoudError(
                f"n_layers={n_layers} is not a multiple of the {len(chosen)}-layer "
                f"pattern {list(chosen)}: {n_layers % len(chosen)} layers are left over",
                "the pattern is one period of the stack; change n_layers to a "
                "multiple of it, or shorten the pattern so it tiles",
            )
        required = {k: v for name in chosen for k, v in _coeffects(name).requires.items()}
        self.context = {**required, **dict(context or {})}
        for name in chosen:
            check_compatible([name], self.context)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.pattern = chosen
        self.layers = nn.ModuleList(
            [
                KDA(
                    d_model,
                    n_heads,
                    head_dim,
                    chunk=chunk,
                    write_std=write_std,
                    device=device,
                )
                if name == "KDA"
                else CSA(
                    d_model,
                    n_heads,
                    head_dim,
                    chunk=csa_chunk,
                    topk=topk,
                    hca_group=hca_group,
                    write_std=write_std,
                    device=device,
                )
                for name in self._period(n_layers)
            ]
        )

    def _period(self, n_layers: int) -> tuple[str, ...]:
        """Return the per-layer names: the pattern repeated to `n_layers`."""
        return self.pattern * (n_layers // len(self.pattern))

    def layer_types(self) -> tuple[str, ...]:
        """Return the mechanism of every layer, in build order."""
        return self._period(len(self.layers))

    def coeffects(self) -> Coeffects:
        """Declare the context the stack needs and the slot it fills."""
        return Coeffects(
            requires={k: v for name in self.pattern for k, v in _coeffects(name).requires.items()},
            provides={"attn.stack": "hybrid" if len(set(self.pattern)) > 1 else "uniform"},
        )

    def extra_repr(self) -> str:
        """Print the pattern, not twelve module reprs."""
        return f"n_layers={len(self.layers)}, pattern={list(self.pattern)}"

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` in, the residual stream out."""
        if counters is not None:
            counters.count("hybrid_fwd", x.shape[0] * x.shape[1])
        for name, layer in zip(self.layer_types(), self.layers, strict=True):
            x = x + layer(x, counters)
            if counters is not None:
                counters.count("hybrid_layer_fwd")
                counters.count(f"hybrid_{name.lower()}_layer_fwd")
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
        report = {f"per_{kind.lower()}": per[kind] for kind in KINDS}
        report.update({kind.lower(): per[kind] * counts[kind] for kind in KINDS})
        report["total"] = sum(p.numel() for p in self.parameters())
        return report
