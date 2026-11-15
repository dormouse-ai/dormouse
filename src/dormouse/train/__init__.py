"""Training: the optimizer pair, the schedule shapes, one step, one aux objective.

`optim` owns the two update rules and the router between them, `schedule` owns the
learning rate as two closed forms, `step` owns the order a step happens in and the
refusals that keep a bad batch out of the weights, `aux` owns the second objective
(LightMTP-S) the step may add to the token loss. Import the four; nothing else in this
package is API.
"""

from .aux import LightMTP, LightMTPHead
from .optim import AdamW, MixOptimizer, Muon
from .schedule import warmup_cosine, warmup_stable_decay
from .step import StepResult, train_step

__all__ = [
    "AdamW",
    "LightMTP",
    "LightMTPHead",
    "MixOptimizer",
    "Muon",
    "StepResult",
    "train_step",
    "warmup_cosine",
    "warmup_stable_decay",
]
