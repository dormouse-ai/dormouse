"""The router: which parameter obeys which update rule, and the two arms as one.

Two update rules live in `muon.py` (matrices) and `adamw.py` (everything else);
`sf.py` is the third mode, the same two rules wrapped in an iterate average. This
module decides which parameter is under which, refuses a shape neither rule claims, and
presents the pair as one object: `step`, `zero_grad`, `set_lr`, `state_dict`.

§6. ROUTING. A parameter goes to Muon iff it is a 2-D matrix that is not the token
table; to AdamW iff it is 1-D (norms' gains, mixture coefficients, biases) or the token
table. Anything else — a 3-D table, a convolution — routes to neither, and that is a
refusal, not a default: a fallback is how a table is trained by the wrong rule for
3000 steps and nobody sees it.

§7. THE THREE SCHEDULES, AND WHAT EACH ONE IS. `cosine` and `wsd` differ only in the
closed form the loop writes through `set_lr` (`schedule.py`); the arms do not know what
step it is, so a run that stops at `total` and a run that continues use the same code.
`sf` builds `sf.py`'s schedule-free pair instead, and then `set_lr` is a **refusal**: the
SF arms compute their own rate from their own counter, and a schedule written on top of
that applies the decay twice — once inside the average's weight and once on the step.
`sizes` and the counter names `step.py` reports are the same in all three, because the
routing is.

§8. THE SWAP, IN ONE PLACE. Under `sf` the parameters hold `Y_t` and the model worth
evaluating is `X_t`, so an eval that forgets to swap reports a loss for a model that is
not the one being trained — silently, because the discrepancy starts at zero. So the
context manager is here, one call for all three modes, and it owns both halves: the
weights and `model.train()`/`eval()`. `with optimizer.eval_weights():` is the whole
contract; under `cosine` and `wsd` it only flips the module's mode.
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
    NS_EPS,
    NS_POWER_ITERS,
    NS_STEPS,
    ROUTER_LR_COUNTER,
    Muon,
    colrow,
    muon_direction,
    polar,
    spectral_bound,
)
from .sf import SF_LR, SF_WEIGHT_DECAY, SFAdamW, SFNorMuon

__all__ = [
    "COLROW_EPS",
    "MUON_LR",
    "NS_COEFFS",
    "NS_EPS",
    "NS_POWER_ITERS",
    "NS_STEPS",
    "ROUTER_LR_COUNTER",
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
SCHEDULES: tuple[str, ...] = ("cosine", "wsd", "sf")


class MixOptimizer:
    """One `step()` over two arms, routed by §6 and refusing anything unrouted."""

    def __init__(
        self,
        model: nn.Module,
        *,
        schedule: str = "cosine",
        lr_muon: float | None = None,
        lr_adamw: float | None = None,
        lr_router: float | None = None,
        momentum: float = 0.95,
        steps: int = NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        muon_weight_decay: float = 0.0,
        eps: float = 1e-8,
        sf_beta: float = 0.9,
        sf_warmup: int = 0,
        sf_weight_decay: float = SF_WEIGHT_DECAY,
        sf_final_decay_frac: float = 0.05,
        muon_grouped: bool = True,
    ) -> None:
        """Route `model`'s parameters, then build the arm pair `schedule` names.

        Under `cosine`/`wsd` a rate of `None` keeps the recipe's `muon_lr=0.02` and
        `lr_adamw=0.003`; under `sf` it keeps the paper's single `η = 0.01` for both,
        because a schedule-free run at the scheduled rate is 6× the tuned one and
        reports the divergence as a property of the method.

        A router's `*.router.weight` gets its own Muon group at `lr_router`, by default
        `lr_muon / √d_model` (read off the row width, ≈7e-4 at d=832). Muon sets every
        row's update magnitude to exactly the group's lr, so at the recipe's 0.02 a
        single step moves a row by the whole init scale `√d·0.02`; the split keeps the
        router's direction and makes its step a small fraction of that scale. Router
        stays in Muon (not AdamW) so the polar direction it inherits is preserved.
        """
        if schedule not in SCHEDULES:
            raise LoudError(
                f"schedule={schedule!r} is not a mode this stack has",
                f"use one of: {', '.join(SCHEDULES)}",
            )
        self.schedule = schedule
        self.model = model
        self.table = route_all(model)
        schedule_free = schedule == "sf"
        #: A rate of `None` is the mode's default: the recipe's per-arm pair, or the
        #: paper's single η for a schedule-free run (both arms, §7).
        self.base_lr = (
            (_sf_lr(lr_muon), _sf_lr(lr_adamw))
            if schedule_free
            else (
                MUON_LR if lr_muon is None else lr_muon,
                ADAMW_LR if lr_adamw is None else lr_adamw,
            )
        )
        # Sorting by name makes the split deterministic: adding the router can never
        # silently reorder the other matrices inside their head-split groups.
        matmuls = sorted(
            ((name, p) for name, p in model.named_parameters() if self.table[name] == "muon"),
            key=lambda item: item[0],
        )
        vectors = sorted(
            ((name, p) for name, p in model.named_parameters() if self.table[name] == "adamw"),
            key=lambda item: item[0],
        )
        if not matmuls:
            raise LoudError(
                "no parameter routed to Muon: the router or the model is wrong",
                "a stack of matrices must have at least one 2-D non-embedding weight",
            )
        routers = [(name, p) for name, p in matmuls if _is_router(name)]
        others = [(name, p) for name, p in matmuls if not _is_router(name)]
        groups: list[dict[str, Any]] = []
        for name, param in others:
            split = _head_split(model, name, param)
            for group in groups:
                if group["head_split"] == split:
                    group["params"].append(param)
                    break
            else:
                groups.append({"params": [param], "head_split": split})
        # The reader's explicit override wins; the default comes off the router row
        # width, so a config that names no rate still gets lr_muon / sqrt(d_model).
        d_router = routers[0][1].shape[1] if routers else 0
        self.base_lr_router = (
            (lr_router if lr_router is not None else self.base_lr[0] / d_router**0.5)
            if routers
            else (lr_router or 0.0)
        )
        self.router_group_enabled = lr_router is not None or bool(routers)
        if routers:
            groups.append(
                {
                    "params": [p for _, p in routers],
                    "head_split": None,
                    "is_router": True,
                    "lr": self.base_lr_router,
                }
            )
        lr_muon, lr_adamw = self.base_lr
        if schedule_free:
            self.muon: Muon | SFNorMuon = SFNorMuon(
                groups,
                lr=lr_muon,
                betas=(sf_beta, betas[1]),
                weight_decay=sf_weight_decay,
                steps=steps,
                warmup=sf_warmup,
                final_decay_frac=sf_final_decay_frac,
            )
            self.adamw: AdamW | SFAdamW = SFAdamW(
                [p for _, p in vectors],
                lr=lr_adamw,
                betas=(sf_beta, betas[1]),
                eps=eps,
                weight_decay=sf_weight_decay,
                warmup=sf_warmup,
                final_decay_frac=sf_final_decay_frac,
            )
            return
        self.muon = Muon(
            groups,
            lr=lr_muon,
            momentum=momentum,
            steps=steps,
            weight_decay=muon_weight_decay,
            grouped=muon_grouped,
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

    @property
    def router_lr(self) -> float:
        """The router group's peak rate; 0 when the model carries no router."""
        return self.base_lr_router

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both arms' gradients."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, *, counters: Counters | None = None) -> None:
        """Run both arms; the routing already decided which parameter is where.

        `counters` is passed only to Muon's scheduled arm: that is where the grouped
        polar runs, and the SF arm has its own (untouched) chain.
        """
        if self.router_group_enabled and counters is not None:
            counters.enable(
                "router_lr_group",
                counter=ROUTER_LR_COUNTER,
                why="router weights step under their own small Muon LR",
            )
        if isinstance(self.muon, Muon):
            self.muon.step(counters=counters)
        else:
            self.muon.step()
        self.adamw.step()

    @property
    def _schedule_free(self) -> tuple[SFNorMuon, SFAdamW] | None:
        """The two SF arms, or None when this is a scheduled optimizer (§8)."""
        if isinstance(self.muon, SFNorMuon) and isinstance(self.adamw, SFAdamW):
            return self.muon, self.adamw
        return None

    def set_lr(self, fraction: float) -> None:
        """Scale both peak rates by the schedule's `fraction` (1.0 is the peak)."""
        if isinstance(self.muon, SFNorMuon) or isinstance(self.adamw, SFAdamW):
            raise LoudError(
                f"set_lr({fraction}) on a schedule-free optimizer",
                "the sf arms derive their rate from their own counter; a schedule on "
                "top of them applies the decay twice — anneal with begin_final_decay()",
            )
        if fraction < 0.0:
            raise LoudError(
                f"learning-rate fraction={fraction} is negative",
                "a schedule produces [0, 1]; a negative rate climbs the loss",
            )
        for group in self.muon.param_groups:
            base = self.base_lr_router if group.get("is_router") else self.base_lr[0]
            group["lr"] = base * fraction
        self.adamw.set_lr(self.base_lr[1] * fraction)

    def begin_final_decay(self, total: int) -> None:
        """Anneals the last `sf_final_decay_frac` of a `total`-step run to zero (§7).

        The only call in this class that knows a horizon, and the reason a corpus can
        grow: not calling it leaves the rate flat, and calling it later only changes
        which steps are the last ones.
        """
        if not isinstance(self.muon, SFNorMuon) or not isinstance(self.adamw, SFAdamW):
            raise LoudError(
                "begin_final_decay() on a scheduled optimizer",
                "cosine and wsd anneal through their own closed form; only sf needs this",
            )
        self.muon.begin_final_decay(total)
        self.adamw.begin_final_decay(total)

    @contextmanager
    def eval_weights(self) -> Iterator[None]:
        """Evaluate at the model that is being trained, then put the run back (§8)."""
        training = self.model.training
        arms = self._schedule_free
        self.model.eval()
        if arms is not None:
            arms[0].eval_weights()
            arms[1].eval_weights()
        try:
            yield
        finally:
            if arms is not None:
                arms[0].train_weights()
                arms[1].train_weights()
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


def _sf_lr(lr: float | None) -> float:
    """Return the SF base rate: the caller's, or the paper's single η for both arms."""
    return SF_LR if lr is None else lr


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


def _is_router(name: str) -> bool:
    """Whether a parameter is a router projection, by its path (not its shape)."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2:] == ["router", "weight"]


def _head_split(model: nn.Module, name: str, param: torch.Tensor) -> tuple[int, int] | None:
    """Return `(groups, rows)` for a per-head projection, or None when it is whole.

    Three names carry heads in this stack: the packed `qkv` of an uncompressed layer
    (`2·n_heads` slices of Q and K, V as the remainder), the latent `q_up` (every row
    is a query head) and the latent `kv_up` (the K half is per key/value head, the V
    half stays whole). The head count comes from the module that owns the projection;
    a name that looks like one of these without an `n_heads` is refused rather than
    quietly given the whole-matrix chain.
    """
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
