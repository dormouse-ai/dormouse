"""The run's numeric policy: the dtype the parameters are, and the three places fp32 stays.

`spec.dtype` is one of two literals, and this module turns it into a decision:

    fp32   the reference: parameters, gradients, moments and every kernel in fp32. A run
           of it is the run this project measured before the policy existed, and a spec
           built in code (a test, a notebook) gets it.
    bf16   a real run: parameters and gradients are 2 bytes each instead of 4, which is
           the point — a 1.05B stack's weights+gradients+fp32 moments do not fit 16 GB at
           batch 2 in fp32 and do at a batch worth training in bf16.

WHERE bf16 RUNS, AND WHERE IT MUST NOT. The rule, as the forward is written:

    bf16   every matmul: the attention q/kv projections and the writes, KDA's chunk
           products and its triangular solve, DSA's scores and gathered values, the MoA
           experts' three matrices, the token table and the tied head. `Precision.autocast`
           is the context they run under and `autocast_ops` counts its entries, so a
           forward that ran the policy says so in the step's counters.
    fp32   (1) RMSNorm's statistic. `blocks.RMSNorm` reduces `mean(x²)` in fp32 whatever
           the residual dtype is, and returns the input's dtype.
           (2) The cross-entropy and the loss accumulation. `step.chunked_cross_entropy`
           upcasts each chunk and sums in fp32: a bf16 log-softmax over 32 000 rows is
           the quality leak this policy exists to refuse, and the loss is the number the
           run is judged by.
           (3) Every optimizer moment and the update arithmetic. Muon's momentum and
           AdamW's two moments are fp32, and each arm casts the parameter to fp32 for the
           update and writes the result back. A bf16 moment loses the sub-ulp increment a
           ~1e-3 learning rate lives on, and `polar` is a chain of five products on it.
           (4) The TSCT basis (`model/tsct.py`): the rank-r SVD of an expert's weight is
           taken in fp32 and cast back, because bf16 has no LAPACK path and a bf16 basis
           is a different rank-r approximation of the matrix.

WHY THERE IS NO GradScaler. A scaler exists to keep fp16 gradients out of the subnormal
range: fp16 has 5 exponent bits and a smallest normal of 2⁻¹⁴, so a small gradient flushes
to zero without a 2^k scale on the loss. bf16 has 8 exponent bits — fp32's range, 2⁻¹²⁶ —
and a mantissa loss instead, which no scale can repair. Multiplying the loss by 2¹⁶ and
dividing it back would be cargo cult with a state dict to carry.

THE POLICY IS VISIBLE ON THE MODEL. `Precision.of(model)` reads the parameters' dtype, so
`train/step.py`, `train/report.py` and an eval derive the regime from the model they were
handed rather than from a second copy of the spec that can disagree with it. The spec is
what builds the parameters (`assemble`); the dtype of the parameters is what everything
downstream reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from ..model.spec import DTYPES

__all__ = ["COUNTER", "Precision"]

#: The counter that proves the autocast region was entered. Named in `step.py`'s enable.
COUNTER = "autocast_ops"

#: The torch dtype each literal names.
_TORCH: dict[str, torch.dtype] = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _kind(device: torch.device | str | None) -> str:
    """Return the autocast device type for a device, defaulting to where torch runs."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device).type


@dataclass(frozen=True)
class Precision:
    """The run's numeric policy: one literal, and the three things it decides."""

    dtype: str = "fp32"

    def __post_init__(self) -> None:
        """Refuse a literal with no path through this module."""
        if self.dtype not in DTYPES:
            raise LoudError(
                f"dtype={self.dtype!r} is not a numeric policy this stack has",
                f"use one of: {', '.join(DTYPES)}; fp32 is the reference and bf16 is what "
                "a real run gets (parameters and gradients in bf16, moments in fp32)",
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the dtype the parameters and gradients are stored in."""
        return _TORCH[self.dtype]

    @property
    def mixed(self) -> bool:
        """Whether this policy mixes precisions, i.e. whether it needs the autocast region."""
        return self.dtype != "fp32"

    @classmethod
    def of(cls, model: nn.Module) -> Precision:
        """Return the policy a built model is running, read off its own parameters."""
        first = next(model.parameters(), None)
        if first is None:
            raise LoudError(
                "a model with no parameters has no numeric policy",
                "build the stack with assemble(), which sets it from the spec's dtype",
            )
        for name, dtype in _TORCH.items():
            if first.dtype == dtype:
                return cls(name)
        raise LoudError(
            f"the parameters are {first.dtype}, which is not a policy this stack runs",
            "assemble() builds fp32 or bf16 from spec.dtype; cast the model rather than "
            "handing a mixed or half-precision tree to the step",
        )

    def cast[T: nn.Module](self, module: T) -> T:
        """Put `module`'s floating parameters in the policy's dtype; fp32 is a no-op."""
        return module.to(self.torch_dtype) if self.mixed else module

    def autocast(
        self, device: torch.device | str | None = None, *, counters: Counters | None = None
    ) -> AbstractContextManager[None]:
        """Run the matmuls inside in bf16, and count the entry where a step asks for it.

        A no-op under fp32 — no region, no counter, which is what makes `autocast_ops`
        a statement about a bf16 run rather than about the code path being imported.
        """
        if not self.mixed:
            return nullcontext()
        return _region(self.torch_dtype, _kind(device), counters)


@contextmanager
def _region(dtype: torch.dtype, kind: str, counters: Counters | None) -> Iterator[None]:
    """Enter the autocast region; count it first, so the forward's own check sees it."""
    if counters is not None:
        counters.count(COUNTER)
    with torch.autocast(kind, dtype=dtype):
        yield
