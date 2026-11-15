"""Training: the optimizer pair, the schedule shapes, one step, and ER-JEPA.

`optim` owns the two update rules and the router between them, `schedule` owns the
learning rate as two closed forms, `step` owns the order a step happens in and the
refusals that keep a bad batch out of the weights, `jepa` owns the optional paired
term with its replay memory and its loss dropout. Import them; nothing else in this
package is API.
"""

from .jepa import EpisodicMemory, JepaAux, JepaConfig, JepaHead, JepaPair, er_jepa_loss, jepa_loss
from .optim import AdamW, MixOptimizer, Muon
from .schedule import warmup_cosine, warmup_stable_decay
from .step import StepResult, train_step

__all__ = [
    "AdamW",
    "EpisodicMemory",
    "JepaAux",
    "JepaConfig",
    "JepaHead",
    "JepaPair",
    "MixOptimizer",
    "Muon",
    "StepResult",
    "er_jepa_loss",
    "jepa_loss",
    "train_step",
    "warmup_cosine",
    "warmup_stable_decay",
]
