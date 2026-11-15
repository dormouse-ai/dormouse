"""The attention slot: KDA, CSA, DSA, and the hybrid stack that assembles them.

`kda` is the chunkwise delta rule with a per-channel decay, `csa` is top-k block
attention with a content selector and the coarse HCA level inside it, `dsa` is
exact attention over the indexer's k selected tokens plus a local window, and
`hybrid` is the 3:1 layer pattern plus the coeffects check that refuses an
impossible mix (`kda:dsa` by default, `kda:csa` available). `model/attn.py` stays
the dense reference and the control arm: these four are alternatives to it, not
replacements for it.
"""

from .csa import CSA
from .dsa import DSA
from .dsa_indexer import LightningIndexer
from .hybrid import HYBRID_3_TO_1, MIXES, HybridStack, check_compatible
from .kda import KDA

__all__ = [
    "CSA",
    "DSA",
    "HYBRID_3_TO_1",
    "KDA",
    "MIXES",
    "HybridStack",
    "LightningIndexer",
    "check_compatible",
]
