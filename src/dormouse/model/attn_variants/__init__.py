"""The attention slot: KDA, CSA, and the hybrid stack that assembles them.

`kda` is the chunkwise delta rule with a per-channel decay, `csa` is top-k block
attention with a content selector and the coarse HCA level inside it, `hybrid`
is the 3:1 layer pattern plus the coeffects check that refuses an impossible
mix. `model/attn.py` stays the dense reference and the control arm: these three
are alternatives to it, not replacements for it.
"""

from .csa import CSA
from .hybrid import HYBRID_3_TO_1, HybridStack, check_compatible
from .kda import KDA

__all__ = ["CSA", "HYBRID_3_TO_1", "KDA", "HybridStack", "check_compatible"]
