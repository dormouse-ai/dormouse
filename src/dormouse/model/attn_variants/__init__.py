"""The attention slot: KDA, sparse global layers, and the patterns that mix them.

`kda` is the chunkwise delta rule with a per-channel decay; `dsa` is exact attention
over a lightning indexer's k selected tokens plus a local window; `csa` is top-k block
attention whose `hca_group=8` level is HCA; `projections` is the GQA + latent block
KDA and DSA read and write through; `hybrid` is the named layer pattern plus the
coeffects check that refuses an impossible mix. The dense reference — full causal
attention with RoPE — lives in `tests/reference_attn.py`: it is the anchor these
arms are checked against, and it is deliberately not importable from the model.
"""

from .csa import CSA
from .dsa import DSA
from .dsa_indexer import LightningIndexer
from .hybrid import (
    HCA_GROUP,
    HYBRID_3_TO_1,
    HYBRID_6_KDA_CSA_HCA,
    HYBRID_24_KDA_4CSA_4HCA,
    KINDS,
    PATTERN_KINDS,
    HybridStack,
    check_compatible,
)
from .kda import KDA
from .projections import LatentProjections, bill

__all__ = [
    "CSA",
    "DSA",
    "HCA_GROUP",
    "HYBRID_3_TO_1",
    "HYBRID_6_KDA_CSA_HCA",
    "HYBRID_24_KDA_4CSA_4HCA",
    "KDA",
    "KINDS",
    "PATTERN_KINDS",
    "HybridStack",
    "LatentProjections",
    "LightningIndexer",
    "bill",
    "check_compatible",
]
