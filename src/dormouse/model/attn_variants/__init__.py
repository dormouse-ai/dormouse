"""The attention slot: KDA, DSA, and the 3:1 stack that assembles them.

`kda` is the chunkwise delta rule with a per-channel decay; `dsa` is exact attention
over a lightning indexer's k selected tokens plus a local window; `projections` is the
GQA + latent block both of them read and write through; `hybrid` is the layer pattern
(there is one) plus the coeffects check that refuses an impossible mix. The dense
reference — full causal attention with RoPE — lives in `tests/reference_attn.py`: it
is the anchor these two are checked against, and it is deliberately not importable
from the model.
"""

from .dsa import DSA
from .dsa_indexer import LightningIndexer
from .hybrid import HYBRID_3_TO_1, KINDS, HybridStack, check_compatible
from .kda import KDA
from .projections import LatentProjections, bill

__all__ = [
    "DSA",
    "HYBRID_3_TO_1",
    "KDA",
    "KINDS",
    "HybridStack",
    "LatentProjections",
    "LightningIndexer",
    "bill",
    "check_compatible",
]
