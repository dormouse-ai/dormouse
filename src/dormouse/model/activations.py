"""The fixed activation dictionary that `la` and `moa` mix over.

WHY A DICTIONARY, AND WHY FIXED. "More Expressive Feedforward Layers: Part I"
(arXiv 2605.26647, Tables 1-2) measures a *learnable linear combination of a
fixed set of activations* at 0.12B: one-LA -0.015, bi-LA -0.016, one-MoA -0.011,
bi-MoA -0.016, and bi-MoA with sigmoid gating -0.029 against SwiGLU, all in
terminal loss. The dictionary is the paper's `gsr2lr` — gelu, silu, relu², leaky
ReLU, tanh — and the cost is the point: the combination adds `K` scalars, not
`K` matrices, so the bill grows by O(d), not O(d²).

EVERY ENTRY IS ELEMENTWISE. A mixture `A(z) = Σₖ cₖ·fₖ(z)` is only well defined
if each `fₖ` maps one tensor to one tensor. A GLU is two-operand and cannot be
an entry, so K3's SiTU (arXiv 2607.24653 §2.3.2 Eq. 12,
`[β₁·tanh(g/β₁)·sigmoid(g)] ⊙ [β₂·tanh(u/β₂)]`) enters through its gate factor
`β₁·tanh(z/β₁)·sigmoid(z)`: the bounded half, which is the property quantization cares
about. The up-side cap `β₂·tanh(u/β₂)` belongs to the two-operand GLU and is
therefore not part of this dictionary; `ffn_act="situ"` is exactly the one-sided
form `down(situ(gate(x))·up(x))`, not the full K3 GLU.

BOUNDEDNESS, STATED HONESTLY. `situ` is bounded by `β₁ = 4` and `tanh` by 1;
gelu, silu, relu² and leaky ReLU are not bounded. `relu²` is the worst of them
(quadratic above zero), which is why a `situ` entry is available at all: a
mixture that must survive low-bit activation quantization needs at least one
bounded member in the span, and `tanh`'s bound of 1 is too tight to stand in for
a unit-scale pre-activation on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from dormouse_core.errors import LoudError
from torch import nn

#: K3's SiTU soft-cap scale on the gate branch (arXiv 2607.24653 §2.3.2).
SITU_BETA = 4.0

#: The paper's `gsr2lr` dictionary, and the default for both mixtures.
DEFAULT_DICTIONARY: tuple[str, ...] = ("gelu", "silu", "relu2", "leaky", "tanh")


def _gelu(z: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, the same call the GELU control block makes."""
    return nn.functional.gelu(z)


def _silu(z: torch.Tensor) -> torch.Tensor:
    """`z·sigmoid(z)`, SwiGLU's gate and the baseline this paper is measured against."""
    return nn.functional.silu(z)


def _relu2(z: torch.Tensor) -> torch.Tensor:
    """Squared ReLU, `relu(z)²` (So et al. 2021)."""
    return nn.functional.relu(z).square()


def _leaky(z: torch.Tensor) -> torch.Tensor:
    """Leaky ReLU at PyTorch's own slope, 0.01 — no constant invented here."""
    return nn.functional.leaky_relu(z)


def _tanh(z: torch.Tensor) -> torch.Tensor:
    """`tanh(z)`: bounded by 1, saturating on both sides."""
    return torch.tanh(z)


def _situ(z: torch.Tensor) -> torch.Tensor:
    """K3's SiTU gate factor, `β₁·tanh(z/β₁)·sigmoid(z)`, bounded by `β₁ = 4`."""
    return SITU_BETA * torch.tanh(z / SITU_BETA) * torch.sigmoid(z)


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": _gelu,
    "silu": _silu,
    "relu2": _relu2,
    "leaky": _leaky,
    "tanh": _tanh,
    "situ": _situ,
}


def activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the dictionary's entry `name`; an unknown name is refused with the list."""
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise LoudError(
            f"unknown activation {name!r} in the FFN dictionary",
            f"known entries: {', '.join(_ACTIVATIONS)}",
        ) from None


def resolve(names: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze a dictionary declaration.

    An empty dictionary is refused because a mixture of nothing is the zero
    function — a FFN that writes zero and trains no faster. A duplicate is
    refused because it buys a coefficient that is not a second concept and makes
    `ffn_dict_entries_used` count an entry twice.
    """
    frozen = tuple(names)
    if not frozen:
        raise LoudError(
            "ffn_dict is empty: a mixture of no activations is the zero function",
            f"name at least one of: {', '.join(_ACTIVATIONS)}",
        )
    for name in frozen:
        activation(name)
    if len(set(frozen)) != len(frozen):
        raise LoudError(
            f"ffn_dict repeats an entry: {frozen}",
            "each dictionary entry gets one coefficient; list it once",
        )
    return frozen
