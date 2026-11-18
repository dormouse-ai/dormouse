"""External references for the three demonstrated mechanisms: paper formulas only.

Nothing in this module imports `dormouse.model` or calls a mechanism. Each
function is a fresh fp64 evaluation of the published equation over a `Fixture`;
if the model's own path is wrong, the number here stays right.
"""

from __future__ import annotations

import torch

from .fixture import Fixture


def kda_paper(fixture: Fixture) -> torch.Tensor:
    """Kimi Linear eq. 1 / Gated DeltaNet eq. 2, evaluated one token at a time in fp64.

        S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ
        o_t = S_tᵀ q_t

    `q, k, v, alpha [b, h, T, d]`, `beta [b, h, T]`; the returned write is
    `[b, h, T, d]`, the same tensor the chunk algebra must produce.
    """
    q, k, v, alpha, beta = (
        fixture.tensor(name).double() for name in ("q", "k", "v", "alpha", "beta")
    )
    batch, heads, _, width = q.shape
    state = torch.zeros(batch, heads, width, width, dtype=torch.float64)
    out: list[torch.Tensor] = []
    for t in range(q.shape[2]):
        key = k[:, :, t]
        step = beta[:, :, t].view(batch, heads, 1, 1)
        state = state * alpha[:, :, t].unsqueeze(-1)
        state = state - step * key.unsqueeze(-1) @ (key.unsqueeze(-2) @ state)
        state = state + step * key.unsqueeze(-1) @ v[:, :, t].unsqueeze(-2)
        out.append(torch.einsum("bhdv,bhd->bhv", state, q[:, :, t]))
    return torch.stack(out, dim=2)


def dsa_paper(fixture: Fixture) -> torch.Tensor:
    """DeepSeek-V3.2 §2.1 exact attention over the selected keys, fp64 from the formula.

        o_t = softmax_{s ∈ S(t) ∪ W(t), s ≤ t}(q_t·k_s / √d_k) · v_s
        y   = W_proj · o

    The gather is over the fixture's own `at`/`visible` arrays, so this is the
    arithmetic the sparse layer claims to perform, not a copy of its kernel.
    """
    q, key, value, proj = (fixture.tensor(name).double() for name in ("q", "k", "v", "proj"))
    at = fixture.tensor("at").long()
    visible = fixture.tensor("visible").bool()
    scale = fixture.number("head_dim") ** -0.5

    def gather(source: torch.Tensor) -> torch.Tensor:
        """Read the fixture's positions: `source [b,s,h,d]` → `[b,h,s,keep,d]`."""
        wide = source.transpose(1, 2)[:, :, None, None].expand(
            -1, -1, at.shape[1], at.shape[2], -1, -1
        )
        place = at[:, None, :, :, None, None].expand(-1, q.shape[2], -1, -1, 1, source.shape[-1])
        return torch.gather(wide, 4, place).squeeze(4)

    keys, values = gather(key), gather(value)
    logits = torch.einsum("bhsd,bhskd->bhsk", q.transpose(1, 2), keys) * scale
    weights = torch.softmax(logits.masked_fill(~visible.unsqueeze(1), -torch.inf), dim=-1)
    out = torch.einsum("bhsk,bhskd->bhsd", weights, values).transpose(1, 2)
    batch, seq, heads, width = out.shape
    return out.reshape(batch, seq, heads * width) @ proj.T


def tsct_paper(fixture: Fixture) -> torch.Tensor:
    """TSCT's stated map `y = (x·u)·v`, evaluated in fp64 direct from the factors."""
    x, u, v = (fixture.tensor(name).double() for name in ("x", "u", "v"))
    return torch.einsum("...i,ir,ro->...o", x, u, v)
