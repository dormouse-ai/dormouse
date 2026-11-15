"""The model: the stack, and the mechanisms each module specifies.

`attn_variants/` holds the two attention mechanisms (KDA and DSA) on the projection
block they share, `ffn` is the gated FFN and the mixture arms, `tsct` is the rank-`r`
read the experts use, `moe` is the bank and its dispatch, `blocks` is RMSNorm plus the
pre-norm residual block, `model` is the shell over assembled blocks. Each module states
its formula and why the parameterisation is the one this project trains, so a reader
never has to diff two implementations to learn what the model does.

The dense reference — full causal attention with RoPE — is in `tests/reference_attn.py`
on purpose: it is the anchor, not an arm.
"""

from .model import Transformer

__all__ = ["Transformer"]
