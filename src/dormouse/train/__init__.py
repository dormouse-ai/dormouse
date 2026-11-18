"""Training: the optimizer pair, the schedule shapes, and one step.

`muon` owns the spectral rule and its chain, `adamw` owns the entry-wise one, `sf`
wraps both in the schedule-free iterate average, `optim` routes a model across them,
`schedule` owns the learning rate as two closed forms, `step` owns the order a step
happens in and the refusals that keep a bad batch out of the weights. Import the
public names; nothing else in this package is API.
"""

from .aux import LightMTP, LightMTPHead
from .jepa import JepaAux, JepaConfig, JepaHead, JepaPair
from .optim import AdamW, MixOptimizer, Muon
from .schedule import warmup_cosine, warmup_stable_decay
from .sf import SFAdamW, SFNorMuon
from .step import StepResult, train_step

__all__ = [
    "AdamW",
    "JepaAux",
    "JepaConfig",
    "JepaHead",
    "JepaPair",
    "LightMTP",
    "LightMTPHead",
    "MixOptimizer",
    "Muon",
    "SFAdamW",
    "SFNorMuon",
    "StepResult",
    "train_step",
    "warmup_cosine",
    "warmup_stable_decay",
]
