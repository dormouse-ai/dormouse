"""The model: four mechanisms, each specified in its own module.

`ffn` is the gated SwiGLU, `attn` is causal attention with rotary positions,
`blocks` is RMSNorm plus the pre-norm residual block, `model` is the stack that
assembles them. Each module states its formula and why the parameterisation is
the one this project trains, so a reader never has to diff two implementations
to learn what the model does.
"""

from .model import Transformer

__all__ = ["Transformer"]
