"""Per-token, per-pass write-back gates for the core loop architecture.

The gate is the G1 form already proven in `tools/demo_loops.py`, moved into the
model so the training path owns it: `beta_p(h) = clamp(sigmoid(W_p h + b_p), 0.05, 1)`.
It starts neutral (`W=0`), with the first pass conservative and later passes full,
and it is registered in the module tree so `state_dict`, checkpointing and the
optimizer all see it.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.errors import LoudError
from torch import nn

__all__ = ["GATE_BIAS", "GATE_CEIL", "GATE_FLOOR", "LoopGates"]

GATE_BIAS = (-2.2, 2.2)
GATE_FLOOR = 0.05
GATE_CEIL = 1.0


class LoopGates(nn.Module):
    """One scalar gate per token and pass, clamped away from closed and above one."""

    def __init__(
        self,
        d_model: int,
        passes: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Build one gate per extra pass; the weights start neutral."""
        super().__init__()
        if d_model < 1 or passes < 1:
            raise LoudError(
                f"LoopGates(d_model={d_model}, passes={passes})",
                "positive width and at least one pass",
            )
        self.passes = passes
        self.gates = nn.ModuleList(
            [
                nn.Linear(d_model, 1, bias=True, device=device, dtype=dtype)
                for _ in range(max(1, passes - 1))
            ]
        )
        for index, gate in enumerate(self.gates):
            nn.init.zeros_(cast(torch.Tensor, gate.weight))
            bias = GATE_BIAS[index] if index < len(GATE_BIAS) else GATE_BIAS[-1]
            nn.init.constant_(cast(torch.Tensor, gate.bias), bias)

    def forward(self, h: torch.Tensor, pass_index: int) -> torch.Tensor:
        """Return `[batch, seq, 1]` in `[GATE_FLOOR, GATE_CEIL]` for one pass."""
        if not 0 <= pass_index < len(self.gates):
            raise LoudError(
                f"pass_index={pass_index} outside {len(self.gates)} gates",
                "one gate per extra pass",
            )
        return torch.sigmoid(self.gates[pass_index](h)).clamp(GATE_FLOOR, GATE_CEIL)

    def extra_repr(self) -> str:
        """Print the pass count and the clamp."""
        return f"passes={self.passes}, floor={GATE_FLOOR}, ceil={GATE_CEIL}"
