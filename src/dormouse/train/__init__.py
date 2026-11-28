"""Training: the optimizer pair, the schedule shapes, and one step.

`muon` owns the spectral rule and its chain, `adamw` owns the entry-wise one,
`optim` routes a model across them, `schedule` owns the learning rate as two closed
forms, `step` owns the order a step happens in and the refusals that keep a bad
batch out of the weights. Import the public names; nothing else is API.
"""

from .optim import AdamW, MixOptimizer, Muon
from .schedule import warmup_cosine, warmup_stable_decay
from .step import StepResult, train_step

__all__ = [
    "AdamW",
    "MixOptimizer",
    "Muon",
    "StepResult",
    "train_step",
    "warmup_cosine",
    "warmup_stable_decay",
]
