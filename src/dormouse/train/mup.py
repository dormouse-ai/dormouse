"""μP for this stack: one table of per-parameter init, LR and decay rules.

WHY A TABLE AND NOT A FLAG. The failure this module exists to prevent is a parameter
that keeps the standard parameterisation because nobody classified it: it trains, the
loss falls, and the only symptom is that the optimal rate at one width does not move to
the next. So every parameter gets a `ParamRule`, and a parameter whose owner is not in
the table is a `LoudError` rather than a default.

THE RULE, AND ITS SOURCE. Width is `n`. `2608.20061` (MoE + MLA + Muon) Table 2 gives the
per-parameter factors relative to a base proxy:

    Init. Var            fan_in_base / fan_in      (every hidden parameter)
    LR factor            fan_in_base / fan_in      (matrix-like hidden parameters)
    LR factor            1                         (vector-like hidden parameters)
    I/O embeddings       LR factor 1, var 1

`2605.21486` Table 2 is the same table in the `n^{-a}, n^{-2b}, n^{-c}, n^{-d}` gauge:

    μP:  embedding  variance n⁻²ᵇ = 1,        LR n⁻ᶜ = 1,        WD n⁻ᵈ = 1
         hidden     variance n⁻²ᵇ = 1/n,      LR n⁻ᶜ = 1/n,      WD n⁻ᵈ = n
         last       variance n⁻²ᵇ = 1/n²,     LR n⁻ᶜ = 1/n,      WD n⁻ᵈ = n
         LayerNorm  LR n⁻ᶜ = 1,               WD —

Two consequences are wired into this repository:

* THE TIED HEAD. `2605.21486` §17.15 ("Weight Tying") cannot satisfy the first-layer and
  last-layer conditions with one exponent; it sets the first layer to `b_u = 0`, `c_u = 0`
  and moves the mismatch into the forward pass, `a_v = 1`, i.e. the logits are multiplied
  by `n⁻¹`. The table there is:

      First (U)      var 1     LR 1     forward multiplier 1
      Middle (W)     var 1/n   LR 1/n
      Last (Uᵀ)      var 1     LR 1     forward multiplier 1/n

  So a tied head does NOT get a second parameter and does NOT get a doubled rate: it
  shares the table's rate and scales the logits by `base_width / d_model`
  (`Transformer.head_scale`). Without that factor the logits are `Θ(√n)` at init, which
  is the instability `2605.21486` §5 names.

* WEIGHT DECAY IS PART OF THE PARAMETRISATION, NOT A REGULARISER. `2605.21486` eq. (17.9)
  requires `c_l + d_l = 0`, i.e.

      η_l · λ_l = η · λ          for every layer l.

  `λ` is therefore specified in the paper's units — the fraction of the weight removed per
  unit of the *base* rate — and the optimizer sets the group coefficient to `λ / lr_mult`.
  A width change that moves `lr_mult` cannot change the decay's contribution; in this
  stack every `lr_mult` is 1, so `λ` is width-invariant by construction. `2510.19093`
  shows this term, not the global normalisation, is what keeps the update dynamics stable
  after the early steps where μP's alignment assumptions hold.

THE MUON ADAPTATION. `2608.20061` Table 2's matrix-like LR factor is `fan_in_base/fan_in`.
Their Muon run first rescales the update to AdamW's RMS (`0.2·√max(A,B)`, Appendix A), so
their update is per-element `Θ(1)` and the `1/fan_in` factor is the only width term. Our
`Muon` (`muon.py` §4) applies `ColRow`, which already makes the update per-element
`Θ(1/√fan_in)`; applying `1/fan_in` again would double-count it, and the step's
preactivation change would vanish as `n⁻¹` instead of staying `Θ(1)`. So the Muon
arm keeps `lr_mult = 1` and gets μP's init scaling; the embedding/head rate is the one
knob `2605.21486` §5 shows is worth sweeping on its own. The gate in
`tests/test_mup.py` is the evidence for this adaptation at this stack's scale; the
paper's factor assumes their Muon is RMS-matched to AdamW (Appendix A), which ours is
not.

DEPTH, BATCH AND DURATION ARE NOT SCALED HERE. `2608.20061` §2.1 keeps depth fixed because
depth-μP transfer is unstable; `2512.22382` is the reference for the other axes and is not
implemented, because a factor that no test can exercise is exactly the flag this project
refuses to ship.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from dormouse_core.errors import LoudError
from torch import nn

from ..model.attn_variants.dsa_indexer import LightningIndexer
from ..model.attn_variants.kda import KDA
from ..model.blocks import PROJ_STD, RMSNorm
from ..model.ffn import MoaFFN
from ..model.router import ROUTER_STD, Router
from ..model.spec import StackSpec

__all__ = [
    "KINDS",
    "ParamRule",
    "Parametrisation",
    "classify",
]

__concepts__ = ("ParamRule", "Parametrisation")

#: The four roles `2608.20061` Table 1/2 and `2605.21486` §17.15 distinguish. A parameter
#: that is none of these is unclassified and refused.
KINDS: tuple[str, ...] = ("hidden", "embedding", "head", "router")

#: Module owners this stack knows how to classify. A module added to the model without
#: being added here raises, which is how the next lane is caught rather than measured.
_HIDDEN_OWNERS = (nn.Linear, nn.Conv1d, RMSNorm, MoaFFN, KDA, LightningIndexer)


@dataclass(frozen=True)
class ParamRule:
    """One parameter's μP rule: its role, its init std, its rate and its decay factor.

    `init_std = 0.0` marks a parameter that is NOT drawn from a width-scaled normal — a
    norm gain at 1, a mixture prior at 1, KDA's `a_bias` at a constant. μP leaves those
    initial values alone (a vector has no width to scale) while still assigning them a
    learning-rate rule.
    """

    kind: str
    init_std: float
    lr_mult: float
    wd_mult: float


def classify(model: nn.Module, name: str, param: torch.Tensor) -> str:
    """Return `name`'s role, or refuse an owner this table has never seen.

    Classification is by the module that OWNS the parameter, not by its name: a name is a
    string a refactor can preserve while the module under it changes, and the owner is the
    object that decides the shape and the initialisation. The one name rule is the head,
    because a tied head has no module of its own and an untied one is a `[vocab, d]`
    `nn.Linear` that would otherwise read as an ordinary hidden matrix.
    """
    owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    if name.endswith(("head.weight", "lm_head.weight")) and _vocab_of(model) == param.shape[0]:
        return "head"
    if isinstance(owner, nn.Embedding):
        return "embedding"
    if isinstance(owner, Router):
        return "router"
    if isinstance(owner, _HIDDEN_OWNERS):
        return "hidden"
    raise LoudError(
        f"{name} is owned by {type(owner).__module__}.{type(owner).__name__}, which μP "
        "has no rule for",
        "add the owner to mup._HIDDEN_OWNERS with the role it plays, or the parameter "
        "silently trains under the standard parameterisation and its width scaling is "
        "never measured",
    )


class Parametrisation:
    """The width geometry, and the rule table `2608.20061` Table 2 implies for it.

    `width` is the model being built; `base_width` is the proxy whose optimal rate was
    tuned. Every factor is a ratio of the two, so at `width == base_width` the table is
    the identity and `apply` reproduces the builder's own initialisation.
    """

    def __init__(
        self,
        *,
        width: int,
        base_width: int,
        vocab: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        embedding_lr: float = 1.0,
        weight_decay: float = 0.1,
    ) -> None:
        """Validate the geometry, then freeze the two width-derived factors."""
        if width < 1 or base_width < 1:
            raise LoudError(
                f"width={width}, base_width={base_width}: a width is at least one channel",
                "pass the model's d_model and the proxy's d_model",
            )
        if embedding_lr <= 0.0:
            raise LoudError(
                f"embedding_lr={embedding_lr} is not a rate",
                "a positive multiplier on the base rate; 2605.21486 §5's μP value is 1.0",
            )
        if weight_decay < 0.0:
            raise LoudError(
                f"weight_decay={weight_decay} is negative",
                "λ in 2605.21486 eq. (17.9) is the fraction removed per unit of the base "
                "rate; use 0 for no decay, or 0.1 for the paper's",
            )
        self.width = width
        self.base_width = base_width
        self.vocab = vocab
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.embedding_lr = embedding_lr
        self.weight_decay = weight_decay

    @classmethod
    def from_spec(cls, spec: StackSpec) -> Parametrisation:
        """Build the table a `parametrisation="mup"` spec asks for."""
        return cls(
            width=spec.d_model,
            base_width=spec.mup_base_width,
            vocab=spec.vocab,
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            head_dim=spec.head_dim,
            n_kv_heads=spec.n_kv_heads,
            embedding_lr=spec.embedding_lr,
            weight_decay=spec.weight_decay,
        )

    # ------------------------------------------------------------------ the factors
    @property
    def width_ratio(self) -> float:
        """`n / n_base`, the factor every fan-in and the tied head's forward scale use."""
        return self.width / self.base_width

    @property
    def head_scale(self) -> float:
        """The tied head's forward multiplier: `2605.21486` §17.15's `n⁻¹` ⟹ base/width."""
        return self.base_width / self.width

    def rule(self, model: nn.Module, name: str, param: torch.Tensor) -> ParamRule:
        """Return the (kind, init, rate, decay) rule for one parameter."""
        kind = classify(model, name, param)
        lr_mult = self.embedding_lr if kind in ("embedding", "head") else 1.0
        return ParamRule(
            kind=kind,
            init_std=self._init_std(model, name, param, kind),
            lr_mult=lr_mult,
            #: `c_l + d_l = 0` (2605.21486 eq. 17.9): the coefficient is chosen so the
            #: optimizer's `lr · weight_decay` product is the base-rate λ, whatever the
            #: rate multiplier is.
            wd_mult=1.0 / lr_mult,
        )

    def _init_std(self, model: nn.Module, name: str, param: torch.Tensor, kind: str) -> float:
        """Return the std `name` is drawn with, or 0.0 when it is not a width-scaled normal."""
        base = self._base_std(model, name, param, kind)
        if base == 0.0:
            return 0.0
        if kind == "embedding":
            #: `2605.21486` §17.15: the tied table's first-layer variance is 1.
            return base
        if kind == "head":
            #: An untied last layer is variance n⁻² (std base/width); tied, this branch
            #: is unreachable and the embedding's 1.0 is the whole table.
            return base * self.head_scale
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        return base / math.sqrt(self._fan_ratio(name, param, owner))

    def _base_std(self, model: nn.Module, name: str, param: torch.Tensor, kind: str) -> float:
        """Return the builder's std at the base width, or 0.0 when it has no normal draw."""
        if kind in ("embedding", "head"):
            return 1.0
        if kind == "router":
            return ROUTER_STD
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        if isinstance(owner, nn.Conv1d) and param.ndim == 3:
            return PROJ_STD
        if isinstance(owner, nn.Linear) and param.ndim == 2:
            return PROJ_STD / math.sqrt(2.0 * self.n_layers) if _writes_residual(name) else PROJ_STD
        return 0.0

    def _fan_ratio(self, name: str, param: torch.Tensor, owner: nn.Module) -> float:
        """`fan_in / fan_in_base` for a matrix, or 1 for a fan-in width does not scale.

        Only the residual width and the attention output's head width scale with width
        (the recipe fixes `head_dim` and scales `n_heads`). `c_lat` and `d_ffn` are fixed
        by the architecture (`2608.20061` §2.1 keeps the MoE intermediate dimension), so
        a `q_up`, `kv_up` or `down` matrix keeps its init and its rate. Reading the
        scaling off the parameter's *owner* rather than off a set of numbers matters at
        tiny widths, where `c_lat` can equal `n_kv_heads·head_dim` by coincidence.
        """
        if isinstance(owner, (nn.Linear, Router)):
            fan = param.shape[1]
        elif isinstance(owner, nn.Conv1d):
            return 1.0
        else:
            return 1.0
        if fan == self.width:
            return self.width_ratio
        if name.endswith(".attn.proj.weight") and fan == self.n_heads * self.head_dim:
            return self.width_ratio
        return 1.0

    # ------------------------------------------------------------------- application
    def apply(self, model: nn.Module) -> None:
        """Rescale every classified parameter to its rule's std and set the head's scale.

        The existing draw is scaled by `init_std / base_std`, not redrawn, so a μP model
        at the base width is the SP model exactly (the ratio is 1) and a μP model at
        another width is the SAME sample with μP's variance — the paired comparison the
        transfer gate needs. Parameters with `init_std = 0` keep the arm's own initial
        value (a gain, a constant bias, a mixture prior).
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                rule = self.rule(model, name, param)
                base = self._base_std(model, name, param, rule.kind)
                if rule.init_std > 0.0 and base > 0.0:
                    param.mul_(rule.init_std / base)
        cast(Any, model).head_scale = self.head_scale

    def table(self, model: nn.Module) -> dict[str, ParamRule]:
        """Return the rule for every parameter, refusing an unclassified one."""
        return {name: self.rule(model, name, param) for name, param in model.named_parameters()}


def _writes_residual(name: str) -> bool:
    """Whether this matrix writes the residual stream (`blocks.Block`'s `1/√(2L)`)."""
    return name.endswith((".proj.weight", ".down.weight"))


def _vocab_of(model: nn.Module) -> int:
    """Return the token-table row count a head's first axis must equal, or -1 when unknown."""
    vocab = getattr(model, "vocab", -1)
    return vocab if isinstance(vocab, int) else -1
