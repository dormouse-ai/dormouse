"""The projection block both attention mechanisms share: GQA through a latent bottleneck.

THE SHAPE. `q` is `n_heads x head_dim` wide and `k`/`v` are `n_kv_heads x head_dim`,
with `n_heads = 8, n_kv_heads = 2` in the live recipe (GQA, Ainslie et al. 2023 §2:
four query heads share one key/value head, which cuts the KV cache and the `k`/`v`
matmuls by 4x). Both are read and written through a bottleneck of `c_lat = 512`:

    q  = W_q_up · W_q_lat · x        W_q_lat  [c_lat, d_model]
    kv = W_kv_up · W_kv_lat · x      W_kv_up  [2·n_kv_heads·head_dim, c_lat]

WHY A BOTTLENECK ON TOP OF GQA. The projection block is the attention layer's cost —
`d_model·d_kv` per stream, three streams, against O(head_dim) for the kernel itself.
One `[c_lat, d_model]` matrix shared by both reads plus one `[·, c_lat]` write per
stream replaces `[·, d_model]` with `[·, c_lat] + [c_lat, d_model]`: the write side
drops by `1 - c_lat/d_model` and the read side pays one matrix the two streams share.
Measured on the layer at live widths (d=832, 8 heads of 64, 2 KV heads, c_lat=512):
2.77M → 1.67M parameters on the projection block, **39.6%**, and the same ratio on
the multiplies (`bill()` prints both sides; `tests/test_kda.py` asserts the saving).

THE COMPRESSION OFF IS THE PRE-CHANGE LAYER, NOT A SECOND ONE. `n_kv_heads=n_heads`
and `c_lat=d_model` build ONE packed `[3·d_kv, d_model]` matrix — the form every
checkpoint written before this module existed has — and no latent projections at all.
That is the degeneracy gate: the two settings are not two layers, they are one layer
with the compression turned off, so a checkpoint from the dense recipe loads and runs
bit-identically. `tests/test_kda.py` pins the parameter inventory and the equality.

WHY IT IS A BASE CLASS AND NOT A WRAPPER. `q_lat`/`q_up`/... have to live on the
layer so the names (`attn.qkv.weight`, `attn.q_up.weight`) stay the ones the optimizer
routes on and a checkpoint keeps them; delegating to a submodule would rename every
matrix it holds.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.errors import LoudError
from torch import nn

from ..blocks import PROJ_STD


class LatentProjections(nn.Module):
    """`project(x)` → the packed `q|k|v` tensor, and `split()` into heads."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        *,
        n_kv_heads: int | None = None,
        c_lat: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Build the packed matrix, or the two bottleneck pairs when compressing."""
        super().__init__()
        kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        latent = d_model if c_lat is None else c_lat
        if d_model < 1 or n_heads < 1 or head_dim < 1:
            raise LoudError(
                f"attention needs positive widths, got d_model={d_model}, "
                f"n_heads={n_heads}, head_dim={head_dim}",
                "pass the residual width and the per-head width the projections write",
            )
        if kv_heads < 1 or n_heads % kv_heads:
            raise LoudError(
                f"n_heads={n_heads} does not tile {kv_heads} key/value heads",
                "GQA shares one key/value head among a whole number of query heads: "
                "pass n_kv_heads dividing n_heads (2 of 8 in the live recipe)",
            )
        if latent < 1:
            raise LoudError(
                f"c_lat={latent} is not a width",
                "the bottleneck is at least one channel; c_lat=d_model turns it off",
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = kv_heads
        self.head_dim = head_dim
        self.c_lat = latent
        self.d_kv = n_heads * head_dim
        self.d_kv_kv = kv_heads * head_dim
        #: The packed width `q|k|v`; the conv in KDA reads exactly this many channels.
        self.packed = self.d_kv + 2 * self.d_kv_kv
        #: Off means "the pre-change layer": one matrix, no bottleneck.
        self.latent = not (kv_heads == n_heads and latent == d_model)
        if self.latent:
            self.q_lat = nn.Linear(d_model, latent, bias=False, device=device)
            self.q_up = nn.Linear(latent, self.d_kv, bias=False, device=device)
            self.kv_lat = nn.Linear(d_model, latent, bias=False, device=device)
            self.kv_up = nn.Linear(latent, 2 * self.d_kv_kv, bias=False, device=device)
            for linear in (self.q_lat, self.q_up, self.kv_lat, self.kv_up):
                nn.init.normal_(linear.weight, std=PROJ_STD)
        else:
            self.qkv = nn.Linear(d_model, 3 * self.d_kv, bias=False, device=device)
            nn.init.normal_(self.qkv.weight, std=PROJ_STD)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """`x [batch, seq, d_model]` → the packed `[batch, seq, packed]`."""
        if not self.latent:
            return cast(torch.Tensor, self.qkv(x))
        written: torch.Tensor = torch.cat(
            (self.q_up(self.q_lat(x)), self.kv_up(self.kv_lat(x))), dim=-1
        )
        return written

    def split(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`[batch, seq, packed]` → `q [b, s, n_heads, head_dim]` and `k, v` at `n_kv`."""
        batch, seq, _ = packed.shape
        q = packed[..., : self.d_kv].view(batch, seq, self.n_heads, self.head_dim)
        rest = packed[..., self.d_kv :].view(batch, seq, 2 * self.n_kv_heads, self.head_dim)
        return q, rest[:, :, : self.n_kv_heads], rest[:, :, self.n_kv_heads :]

    def extra_repr(self) -> str:
        """Print the compression, which is the only thing that varies here."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, c_lat={self.c_lat}"
        )

    def macs_per_token(self, _seq_len: int) -> int:
        """Return the block's multiplies per token, from `bill()` — not a second formula."""
        return bill(self.d_model, self.n_heads, self.head_dim, self.n_kv_heads, self.c_lat)["macs"]


def bill(
    d_model: int,
    n_heads: int,
    head_dim: int,
    n_kv_heads: int,
    c_lat: int,
) -> dict[str, int]:
    """Parameters and multiplies of the projection block, by arithmetic.

    The block is the q/kv read and the residual write (`proj`, `d_kv → d_model`); the
    kernel and the gates are the same on both sides of the comparison, so they are not
    in it. `params` is what a step stores, `macs` what a token costs.
    """
    d_kv, d_kv_kv = n_heads * head_dim, n_kv_heads * head_dim
    if n_kv_heads == n_heads and c_lat == d_model:
        read = 3 * d_kv * d_model
    else:
        read = d_model * c_lat + c_lat * d_kv + d_model * c_lat + c_lat * 2 * d_kv_kv
    write = d_kv * d_model
    return {"params": read + write, "macs": read + write, "read": read, "write": write}
