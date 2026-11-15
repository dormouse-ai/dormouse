"""The gated SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.

The formula (Shazeer 2020, "GLU variants improve transformer", §2 SwiGLU; Touvron
et al. 2023 LLaMA §2):

    h = silu(W_gate · x) ⊙ (W_up · x)          # ⊙ is elementwise
    y = W_down · h

Three matrices, no biases, no rank: `3·d·d_ffn` parameters. The bias-free form is
the measured recipe's, and a bias on a projection that is immediately normalised
upstream (`norm2`) and added downsteam (`x + y`) only moves the residual's mean,
which the next pre-norm removes again.

WHY 2048 AT d=832 — THE PARAMETER-NEUTRAL WIDTH. A GELU block of the same bill is
`2·d·d_ffn` (two matrices). Swapping in a third matrix at `d_ffn = 3072` would be
`3·832·3072 = 7.67M` against the GELU block's `2·832·3072 = 5.11M`: a 50% bigger
model reporting the gate's gain. `neutral_width` holds the bill equal —
`3·d·f = 2·d·f_gelu` gives `f = 2048`, and `3·832·2048 == 2·832·3072 == 5 111 808`
exactly, so a BPB difference is the activation and not the parameter count.

WHY `silu` AND NOT A SWITCH. SwiGLU's `silu(g) = g·sigmoid(g)` is the variant the
2026 measurement is on (-0.0145 BPB, t=-19, at 0% parameter cost); SiTU/MemGLU were
untested arms of the old code whose constants (`beta1`, `beta2`, `c0`) reached a
forward that the live config never selected. A parameter that no live recipe sets
is a place for a silent default, so the activation is one line, not a table.
"""

from __future__ import annotations

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

#: Std of the two `d -> d_ffn` halves; the residual-writing `down` takes the
#: caller's `write_std` (the depth-dependent scale, see `model.Transformer`).
PROJ_STD = 0.02


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


class GatedFFN(nn.Module):
    """`x [n, d] -> [n, d]` through `down(silu(gate(x)) * up(x))`."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int = 2048,
        *,
        write_std: float = PROJ_STD,
        device: torch.device | str | None = None,
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
        self.gate = nn.Linear(d_model, d_ffn, bias=False, device=device)
        self.up = nn.Linear(d_model, d_ffn, bias=False, device=device)
        self.down = nn.Linear(d_ffn, d_model, bias=False, device=device)
        nn.init.normal_(self.gate.weight, std=PROJ_STD)
        nn.init.normal_(self.up.weight, std=PROJ_STD)
        nn.init.normal_(self.down.weight, std=write_std)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the FFN write, same shape."""
        h = nn.functional.silu(self.gate(x)) * self.up(x)
        if counters is not None:
            counters.count("ffn_fwd", x.shape[0] * x.shape[1])
        return self.down(h)

    def extra_repr(self) -> str:
        """Show the shape the parameter bill is quoted for."""
        return f"d_model={self.d_model}, d_ffn={self.d_ffn}, act=silu"
