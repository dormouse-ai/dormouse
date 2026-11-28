"""The router: which parameter obeys which update rule, and the two arms as one.

Two update rules live in `muon.py` (matrices) and `adamw.py` (everything else). This
module decides which parameter is under which, refuses a shape neither rule claims, and
presents the pair as one object: `step`, `zero_grad`, `set_lr`, `state_dict`.

§6. ROUTING. A parameter goes to Muon iff it is a 2-D matrix that is not the token
table; to AdamW iff it is 1-D (norms' gains, mixture coefficients, biases) or the token
table. Anything else — a 3-D table, a convolution — routes to neither, and that is a
refusal, not a default: a fallback is how a table is trained by the wrong rule for
3000 steps and nobody sees it.

§7. THE TWO SCHEDULES. `cosine` and `wsd` differ only in the closed form the loop
writes through `set_lr` (`schedule.py`); the arms do not know what step it is, so a run
that stops at `total` and a run that continues use the same code. `sizes` and the
counter names `step.py` reports are the same in both.

§8. THE SWAP, IN ONE PLACE. An eval that forgets the model's mode reports a loss for a
model that is not the one being trained — silently, because the discrepancy starts at
zero. `with optimizer.eval_weights():` owns both halves: the mode and the model.
`model.train()`/`eval()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .adamw import ADAMW_LR, AdamW
from .muon import (
    COLROW_EPS,
    MUON_LR,
    NS_COEFFS,
    NS_DTYPES,
    NS_EPS,
    NS_POWER_ITERS,
    NS_STEPS,
    Muon,
    colrow,
    muon_direction,
    polar,
    spectral_bound,
)

__all__ = [
    "COLROW_EPS",
    "MUON_LR",
    "NS_COEFFS",
    "NS_DTYPES",
    "NS_EPS",
    "NS_POWER_ITERS",
    "NS_STEPS",
    "AdamW",
    "MixOptimizer",
    "Muon",
    "colrow",
    "muon_direction",
    "polar",
    "route",
    "route_all",
    "spectral_bound",
]

#: The three modes §7 names. `cosine` is the default and the current recipe.
SCHEDULES: tuple[str, ...] = ("cosine", "wsd")


class MixOptimizer:
    """One `step()` over two arms, routed by §6 and refusing anything unrouted."""

    def __init__(
        self,
        model: nn.Module,
        *,
        schedule: str = "cosine",
        lr_muon: float | None = None,
        lr_adamw: float | None = None,
        momentum: float = 0.95,
        steps: int = NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        muon_weight_decay: float = 0.0,
        eps: float = 1e-8,
        muon_grouped: bool = True,
        muon_ns_dtype: str = "fp32",
    ) -> None:
        """Route `model`'s parameters, then build the arm pair `schedule` names.

        A rate of `None` keeps the recipe's `muon_lr=0.02` and `lr_adamw=0.003`.
        `muon_ns_dtype` selects the Muon chain's arithmetic; `fp16` is an opt-in
        tensor-core rung.
        """
        if schedule not in SCHEDULES:
            raise LoudError(
                f"schedule={schedule!r} is not a mode this stack has",
                f"use one of: {', '.join(SCHEDULES)}",
            )
        if muon_ns_dtype not in NS_DTYPES:
            raise LoudError(
                f"muon_ns_dtype={muon_ns_dtype!r} is not a chain dtype",
                f"use one of: {', '.join(NS_DTYPES)}",
            )
        self.schedule = schedule
        self.model = model
        self.table = route_all(model)
        self.base_lr = (
            MUON_LR if lr_muon is None else lr_muon,
            ADAMW_LR if lr_adamw is None else lr_adamw,
        )
        matmuls = [(name, p) for name, p in model.named_parameters() if self.table[name] == "muon"]
        vectors = [(name, p) for name, p in model.named_parameters() if self.table[name] == "adamw"]
        if not matmuls:
            raise LoudError(
                "no parameter routed to Muon: the router or the model is wrong",
                "a stack of matrices must have at least one 2-D non-embedding weight",
            )
        groups: list[dict[str, Any]] = []
        for name, param in matmuls:
            split = _head_split(model, name, param)
            for group in groups:
                if group["head_split"] == split:
                    group["params"].append(param)
                    break
            else:
                groups.append({"params": [param], "head_split": split})
        lr_muon, lr_adamw = self.base_lr
        self.muon = Muon(
            groups,
            lr=lr_muon,
            momentum=momentum,
            steps=steps,
            weight_decay=muon_weight_decay,
            grouped=muon_grouped,
            ns_dtype=muon_ns_dtype,
        )
        self.adamw = AdamW(
            [p for _, p in vectors], lr=lr_adamw, betas=betas, eps=eps, weight_decay=weight_decay
        )

    @property
    def sizes(self) -> dict[str, int]:
        """Tensors per arm — the unit `train_step`'s counters report."""
        return {
            "muon": sum(len(g["params"]) for g in self.muon.param_groups),
            "adamw": sum(len(g["params"]) for g in self.adamw.param_groups),
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both arms' gradients."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, *, counters: Counters | None = None) -> None:
        """Run both arms; the routing already decided which parameter is where."""
        self.muon.step(counters=counters)
        self.adamw.step()

    def set_lr(self, fraction: float) -> None:
        """Scale both peak rates by the schedule's `fraction` (1.0 is the peak)."""
        if fraction < 0.0:
            raise LoudError(
                f"learning-rate fraction={fraction} is negative",
                "a schedule produces [0, 1]; a negative rate climbs the loss",
            )
        for group in self.muon.param_groups:
            group["lr"] = self.base_lr[0] * fraction
        self.adamw.set_lr(self.base_lr[1] * fraction)

    @contextmanager
    def eval_weights(self) -> Iterator[None]:
        """Evaluate at the model that is being trained, then put the run back."""
        training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(training)

    def state_dict(self) -> dict[str, Any]:
        """Both arms' state, keyed by arm, as a snapshot rather than a view.

        `torch.optim`'s `load_state_dict` copies the dict shallowly, so a state dict
        holding the live momentum buffers would be mutated by the run it resumed
        from — two optimizers sharing one momentum, and a resume that diverges. The
        mode is part of the snapshot: an `sf` checkpoint has three buffers per
        parameter where a cosine one has one, and loading one as the other would
        silently restore nothing.
        """
        return {
            "schedule": self.schedule,
            "muon": deepcopy(self.muon.state_dict()),
            "adamw": deepcopy(self.adamw.state_dict()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore both arms from `state_dict`'s layout, refusing a mode mismatch."""
        if state["schedule"] != self.schedule:
            raise LoudError(
                f"this optimizer is schedule={self.schedule!r} and the checkpoint is "
                f"{state['schedule']!r}",
                "resume with the mode that wrote the checkpoint, or start a new run: "
                "the sf arms hold z, m and v where the scheduled ones hold momentum",
            )
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def route_all(model: nn.Module) -> dict[str, str]:
    """`name -> "muon" | "adamw"` for every parameter, refusing an unroutable one."""
    table: dict[str, str] = {}
    for name, param in model.named_parameters():
        table[name] = route(name, param)
    return table


def route(name: str, param: torch.Tensor) -> str:
    """Which arm owns this parameter, by §6; a shape neither rule claims is refused."""
    if name.endswith("embed.weight"):
        return "adamw"
    if param.ndim == 2:
        return "muon"
    if param.ndim == 3 and param.shape[1] == 1:
        #: A depthwise convolution: `[channels, 1, taps]`. KDA's short conv is this
        #: shape, and Muon has no chain to orthogonalise across taps — the update is
        #: a per-channel filter, which is AdamW's arm. A grouped conv of real width
        #: (axis 1 > 1) is still refused below rather than guessed.
        return "adamw"
    if param.ndim <= 1:
        return "adamw"
    raise LoudError(
        f"{name} has shape {tuple(param.shape)}, which neither arm claims",
        "Muon is matrices and AdamW is vectors and tables: add a rule that says which "
        "rule this shape obeys, or it is a cheap tensor that will be trained at the "
        "wrong scale",
    )


def _head_split(model: nn.Module, name: str, param: torch.Tensor) -> tuple[int, int] | None:
    """Return `(groups, rows)` for a per-head projection, or None when it is whole.

    Three names carry heads in this stack: the packed `qkv` of an uncompressed layer
    (`2·n_heads` slices of Q and K, V as the remainder), the latent `q_up` (every row
    is a query head) and the latent `kv_up` (the K half is per key/value head, the V
    half stays whole). The head count comes from the module that owns the projection;
    a name that looks like one of these without an `n_heads` is refused rather than
    quietly given the whole-matrix chain.
    """
    if "." not in name:
        return None
    heads = getattr(model.get_submodule(name.rsplit(".", 2)[0]), "n_heads", None)
    if name.endswith(".attn.qkv.weight"):
        if not isinstance(heads, int):
            raise LoudError(
                f"{name} looks like a packed qkv weight but its owner has no n_heads",
                "either pass a real attention block, or rename the parameter so the "
                "router stops reading it as q/k/v",
            )
        return 2 * heads, 2 * param.shape[0] // 3
    if name.endswith(".attn.q_up.weight"):
        return (heads, param.shape[0]) if isinstance(heads, int) else None
    if name.endswith(".attn.kv_up.weight"):
        owner = model.get_submodule(name.rsplit(".", 2)[0])
        kv_heads = getattr(owner, "n_kv_heads", None)
        head_dim = getattr(owner, "head_dim", 0)
        if not isinstance(kv_heads, int) or not isinstance(head_dim, int):
            return None
        #: The first `n_kv·head_dim` rows are K; V is the remainder and stays whole.
        return kv_heads, kv_heads * head_dim
    return None
