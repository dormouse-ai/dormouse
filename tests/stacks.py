"""Test-only stacks: the reference Attention, and the block that carries it.

The dense causal attention with RoPE lives in `reference_attn.py` because it is not
part of the model any more — the stack is KDA and DSA (`model/attn_variants/`). It
stays in the tests because it is the anchor those two are checked against: a hand
written `softmax(qkᵀ/√d, causal)v` that nobody claims is fast, and which every
kernels' causality, masking and projection order is compared to.

`reference_stack` builds a `Transformer` out of blocks that use it, so a test that
needs a whole model (an optimizer, a step, a counter) does not have to assemble the
1B stack to get one. `arm_declarations` is what such a stack must declare for
`Counters.check()` to be a statement about the mechanisms it actually runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from reference_attn import Attention

from dormouse.model.blocks import Block
from dormouse.model.ffn_spec import FFNSpec
from dormouse.model.model import Transformer
from dormouse.model.residual import Residual

#: What a reference stack's forward counts, in `(mechanism, counter, why)` form.
REFERENCE: tuple[tuple[str, str, str], ...] = (
    ("attention", "attn_fwd", "the reference attends"),
    ("ffn", "ffn_fwd", "every block has an FFN"),
    ("rope", "rope_fwd", "the reference rotates positions"),
)


def arm_declarations(act: str) -> tuple[tuple[str, str, str], ...]:
    """Return the mixture declarations an `act`-armed reference stack needs."""
    dictionary = ("ffn_dictionary", "ffn_dict_entries_used", "an entry must contribute")
    if act == "la":
        return ("ffn_la", "ffn_la", "la mixes the dictionary"), dictionary
    if act == "moa":
        return ("moa", "moa_fwd", "moa gates the dictionary"), dictionary
    return ()


def reference_stack(
    *,
    vocab: int = 16,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    head_dim: int = 16,
    d_ffn: int = 64,
    max_seq_len: int = 32,
    ffn: FFNSpec | None = None,
    declarations: Sequence[tuple[str, str, str]] | None = None,
    mixer: Residual | None = None,
    seed: int = 0,
) -> Transformer:
    """A `Transformer` whose blocks use the reference attention and `ffn`."""
    if seed:
        torch.manual_seed(seed)
    blocks = [
        Block(
            d_model,
            d_ffn,
            n_layers=n_layers,
            attn=Attention(d_model, n_heads, head_dim, max_seq_len=max_seq_len),
            ffn=ffn,
        )
        for _ in range(n_layers)
    ]
    declared = tuple(declarations) if declarations is not None else REFERENCE
    return Transformer(
        vocab=vocab,
        d_model=d_model,
        d_ffn=d_ffn,
        max_seq_len=max_seq_len,
        blocks=blocks,
        declarations=declared,
        mixer=mixer,
    )
