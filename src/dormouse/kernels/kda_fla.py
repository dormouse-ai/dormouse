"""FP32-state FLA adapter for ``chunk_kda`` (MIT dependency, not vendored code).

The FlashKDA failure class in NVIDIA's KDA² study is precision, not the chunked
algorithm: a bf16 recurrent state gives 3.45% final-state relative RMSE at 8k
tokens and an fp16 cumulative-decay LUT gives 9% decay error (23/24 long real
sequences fail). Gated DeltaNet-2's remedy is the one taken here: keep the state
fp32, compute the log-decay in fp32 outside the kernel, and use the real decay
rather than a quantized table.

Translation from our kernel layout to FLA's:

* ``q, k, v, alpha [b, h, s, d]`` and ``beta [b, h, s]`` become FLA's
  ``[b, s, h, d]`` / ``[b, s, h]``;
* the per-channel decay ``alpha = exp(-softplus(decay))`` becomes FLA's log-space
  ``g = log(alpha)``, computed in fp32. FLA is called with
  ``use_gate_in_kernel=False`` (its default), so no gate LUT is involved;
* all five operands share fp32: FLA's Triton ``tl.dot`` rejects mixed dtypes, and
  fp32 is what keeps its WY solve and its final state in the arithmetic the eager
  chunk form uses;
* ``scale=1.0`` (our eq. 1 has no ``1/sqrt(d)``) and ``chunk_size=64``.

``DORMOUSE_KDA_FLA`` selects the backend (default off until the full-width paired
gate passes). On CPU the eager path always runs; on CUDA a missing FLA is a
LoudError, never a silent fall back to the eager timing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import torch
from dormouse_core.errors import LoudError

ChunkKDA = Callable[..., tuple[torch.Tensor, torch.Tensor]]

_DORMOUSE_KDA_FLA = "DORMOUSE_KDA_FLA"

#: The default is deliberately OFF. `tests/test_kda_precision.py` proves the
#: full-width paired gate before this literal is flipped to "1".
_DEFAULT = "0"

_ESCAPE = (
    "install flash-linear-attention (MIT): `uv add flash-linear-attention`; "
    "it is pinned in pyproject/uv.lock. To keep the eager `_chunks` path instead, "
    "set DORMOUSE_KDA_FLA=0 (that path is slower and will be timed as such)."
)


def fla_requested() -> bool:
    """Whether the caller asked for the FLA backend (default: off)."""
    return os.getenv(_DORMOUSE_KDA_FLA, _DEFAULT).strip().lower() not in {"0", "off", "false", "no"}


def load_chunk_kda() -> ChunkKDA:
    """Return FLA's ``chunk_kda`` or fail loudly with the install escape."""
    try:
        from fla.ops.kda import chunk_kda  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LoudError(
            "KDA/FLA backend requested but flash-linear-attention is not installed",
            _ESCAPE,
        ) from exc
    return cast(ChunkKDA, chunk_kda)


def chunk_kda_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FLA's KDA kernel with an fp32 state and a real fp32 log-decay.

    ``q, k, v, alpha`` are ``[b, h, s, d]`` and ``beta`` is ``[b, h, s]``; the
    write comes back in the same layout and the state as ``[b, h, d, d]`` fp32.
    """
    chunk_kda = load_chunk_kda()
    # Under bf16 autocast q/k are fp32 (L2Norm promotes) while v/beta are bf16.
    # FLA's Triton dots require one dtype for all operands; fp32 is both the
    # lossless promotion for v/beta and the dtype of its WY solve and state.
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    # The real log-decay in fp32: `log(exp(-softplus(decay))) = -softplus(decay)`.
    # No fp16 cumulative-decay table, and no upcast of an already-quantized log.
    g = alpha.float().log()
    out, state = chunk_kda(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        g.transpose(1, 2),
        beta.transpose(1, 2),
        scale=1.0,
        output_final_state=True,
        use_gate_in_kernel=False,
        chunk_size=chunk,
    )
    if state.dtype != torch.float32:
        raise LoudError(
            f"FLA returned a {state.dtype} recurrent state, expected fp32",
            "the precision fix requires fp32 state; do not accept a bf16 kernel here",
        )
    return out.transpose(1, 2), state
