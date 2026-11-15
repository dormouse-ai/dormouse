"""Muon on matrices, AdamW on everything that is not a matrix, one router.

Two update rules, and the router that decides which parameter is under which. The
formulas, as implemented:

    MUON (Jordan 2024, arXiv:2409.16986 §3; Keller Jordan's reference `muon`)
        m_t = μ·m_{t-1} + (1-μ)·g_t
        Q_t = polar(m_t)                    # Newton-Schulz, 5 iterations
        R_t = ColRow(Q_t)                   # columns to unit norm, then rows
        W_t = W_{t-1}·(1 - η·λ) - η·max(1, m/n)^½·R_t

    ADAMW (Loshchilov & Hutter 2019, arXiv:1711.05101 Alg. 2, decoupled decay)
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        m̂_t = m_t/(1-β₁^t),  v̂_t = v_t/(1-β₂^t)
        W_t = W_{t-1}·(1 - η·λ) - η·m̂_t/(√v̂_t + ε)

§1. WHY THE 5TH-ORDER POLAR ITERATION AND NOT MUON'S `(3.4445, -4.7750, 2.0315)`.
The Jordan triple is a *minimax* polynomial for the interval the Muon reference
normalises into, and it is not the polar factor: `x ↦ 3.4445x - 4.775x³ + 2.0315x⁵`
has no stable fixed point at 1. Measured here on a seeded `[64, 64]`, Frobenius-
normalised as the reference does: singular values in `[0.68, 1.13]` and
`‖QQᵀ - I‖∞ = 3.2e-1` — a direction, not an orthogonalisation, and the anchor this
module is accepted against is 1e-5. So the chain is the 5th-order Newton–Schulz
`p(x) = x(15 - 10x² + 3x⁴)/8` (Bernstein & Newhouse; Higham 1986; the triple Muon+'s
`col_row` arm and `burn-spectral`'s retraction use), whose fixed point at 1 is
superattracting: an error of ε becomes ≈ε³, so five iterations cross from 0.1 to 1e-9.

§2. WHY A POWER-ITERATION PRESCALE AND NOT `X/‖X‖_F`. The quintic is stable on
`(0, 1.53)`, and the reference's Frobenius normalisation puts the *mean* singular value
at `1/√min(m,n)` — 0.125 at 64×64, six iterations from 1 — while at square shapes it
puts σ_max at 2.0, outside the interval, where the chain diverges. It needs σ_max
*inside* the basin, so `spectral_bound` estimates it: worst ratio 0.914, so ≤ 1.09.

§3. WHAT FIVE ITERATIONS CANNOT DO, with the number, because the 1e-5 anchor is not a
promise. The chain lifts singular values up to 1 and cannot rescue a σ_min far below
it: a random 832×832 gaussian has σ_min/σ_max ≈ 0.006 and ends at `‖QQᵀ - I‖∞ ≈ 0.09`,
while the live rectangles (`[64, 832]`, `[832, 2048]`, `[2048, 832]`) end at ≈4e-7
because their spectra are conditioned to 0.2–0.6. The tests pin both facts.

§4. WHY ColRow IS HERE AT ALL. `Q_t` is orthogonal, so `‖Q_t‖_F = √min(m,n)` varies
with shape; ColRow makes every row unit-norm, so `‖R_t‖_F = √m` exactly and the step's
magnitude is a function of the parameter's shape, not its spectrum — which is what lets
one learning rate serve a `[64, 832]` and a `[2048, 832]`. Measured on the old stack at
−0.0428 BPB against per-slice Frobenius normalisation.

§5. THE PER-HEAD ARM, AND ITS ONE HONEST PROPERTY. Kimi K3 §2.5 and DeepSeek V4.1
apply the preconditioner per head to Q and K. Splitting a `[3d, d]` packed qkv weight
into `2·n_heads` slices of `[head_dim, d]` and running the chain on each changes the
*direction* (the Gram is `[head_dim, head_dim]` instead of `[d, d]`) and, because
ColRow is applied per slice, leaves `‖R_t‖_F = √(3d)` — exactly what the whole-matrix
path gives. So it is a direction change at fixed step magnitude, which is why it needs
no learning-rate retune. V and the output projection stay whole: neither reference
splits them, and a per-head split of `proj` is a column split, a different mechanism.

§6. ROUTING. A parameter goes to Muon iff it is a 2-D matrix that is not the token
table; to AdamW iff it is 1-D (norms' gains, mixture coefficients, biases) or the token
table. Anything else — a 3-D table, a convolution — routes to neither, and that is a
refusal, not a default: a fallback is how a table is trained by the wrong rule for
3000 steps and nobody sees it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from dormouse_core.errors import LoudError
from torch import nn
from torch.optim import Optimizer

__concepts__ = ("AdamW", "MixOptimizer", "Muon")

#: Newton–Schulz iterations; five is what the architecture decision fixes, and §3
#: states what five reaches.
NS_STEPS = 5

#: `p(x) = x(15 - 10x² + 3x⁴)/8`, the 5th-order polar iteration, as `(a, b, c)` in
#: `p(x) = ax + bx³ + cx⁵`. §1 says why it is not the Jordan triple.
NS_COEFFS = (1.875, -1.25, 0.375)

#: Power iterations used to place σ_max inside the polynomial's stable interval (§2).
NS_POWER_ITERS = 3

#: Floor inside the two ColRow divisions; `norm + eps` never divides by a zero row.
COLROW_EPS = 1e-7

#: Floor under the spectral prescale. A parameter whose gradient is exactly zero (a
#: saturated gain, a zero-initialised coefficient) has σ_max = 0, and `x/0` is not a
#: direction; with the floor the chain returns zero and the step is a no-op.
NS_EPS = 1e-7

#: Muon's rate for the live 275M recipe. Shipped as a default, not a verdict.
MUON_LR = 0.02

#: AdamW's rate on the embedding table and the norms (the recipe's `×0.15`).
ADAMW_LR = 3e-3


def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`mm` for one matrix, `bmm` for a `[N, m, n]` batch — the same kernel each."""
    return torch.bmm(x, y) if x.ndim == 3 else torch.mm(x, y)


def spectral_bound(x: torch.Tensor, iters: int = NS_POWER_ITERS) -> torch.Tensor:
    """Return a lower bound on `σ_max`, by `iters` power iterations on the Gram.

    The floor inside the estimate is the dtype's `tiny`, never `1e-7`: `‖Gv‖` for a real
    momentum is ~1e-8, and an absolute floor at 1e-7 does not guard the division — it
    *rescales the iteration*, which measured a 1000× under-estimate on a qkv slice and
    ran the quintic to `inf`.

    Started from the **row of largest norm**, not a constant vector, which can be
    nearly orthogonal to the top right-singular vector. On that slice the constant
    start recovered 0.604 of σ_max after five iterations — enough to leave the quintic's
    basin — where the max-row start recovered 0.999 after three. Worst ratio over ten
    tensors of the shapes this model has: 0.914, so `σ_max ≤ 1.09`, inside §2's basin.
    """
    rows = x.norm(dim=-1).argmax(dim=-1)
    index = rows[..., None, None].expand(*rows.shape, 1, x.shape[-1])
    v = torch.gather(x, -2, index).transpose(-2, -1)
    floor = torch.finfo(x.dtype).tiny
    v = v / v.norm(dim=(-2, -1), keepdim=True).clamp_min(floor)
    for _ in range(iters):
        v = _matmul(x.transpose(-2, -1), _matmul(x, v))
        v = v / v.norm(dim=(-2, -1), keepdim=True).clamp_min(floor)
    bound: torch.Tensor = _matmul(x, v).norm(dim=(-2, -1), keepdim=True)
    return bound


def polar(x: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """Return the orthogonal factor of `x` for `[m, n]` or a batch `[N, m, n]`.

    Runs on the **short side**: a tall matrix is transposed first, so the Gram is
    `[min, min]` and a `[128, 832]` costs a `128×128` product per iteration. Exact
    orthogonality holds to the conditioning of `x` — §3 states the ceiling.
    """
    if x.ndim not in (2, 3):
        raise LoudError(
            f"polar takes a matrix or a batch of matrices, got shape {tuple(x.shape)}",
            "a 3-D optimizer state is a batch: view it as [N, m, n] before the chain",
        )
    a, b, c = NS_COEFFS
    work = x.to(torch.float32)
    work = work / spectral_bound(work).clamp_min(NS_EPS)
    transposed = work.shape[-2] > work.shape[-1]
    if transposed:
        work = work.transpose(-2, -1)
    for _ in range(steps):
        gram = _matmul(work, work.transpose(-2, -1))
        work = a * work + _matmul(b * gram + c * _matmul(gram, gram), work)
    if transposed:
        work = work.transpose(-2, -1)
    return work.contiguous()


def colrow(x: torch.Tensor, eps: float = COLROW_EPS) -> torch.Tensor:
    """Scale columns to unit norm, then rows — in place, per slice if batched.

    `ColRow` from Muon+ (§4): the output has every row at unit norm, so
    `‖ColRow(z)‖_F = √rows` whatever the input.
    """
    if x.ndim == 3:
        x.div_(x.norm(dim=-2, keepdim=True) + eps)
        return x.div_(x.norm(dim=-1, keepdim=True) + eps)
    x.div_(x.norm(dim=0, keepdim=True) + eps)
    return x.div_(x.norm(dim=1, keepdim=True) + eps)


def muon_direction(
    momentum: torch.Tensor,
    *,
    head_split: tuple[int, int] | None = None,
    steps: int = NS_STEPS,
) -> torch.Tensor:
    """`ColRow(polar(momentum))`, per head when the parameter is a packed qkv weight.

    `head_split=(groups, rows)` means the first `rows` rows are `groups` equal
    per-head slices and the rest is one matrix: Q and K of a packed `[3d, d]`
    projection, with V as the remainder (§5).
    """
    if head_split is None:
        return colrow(polar(momentum, steps))
    groups, rows = head_split
    if rows % groups or rows > momentum.shape[0] or momentum.ndim != 2:
        raise LoudError(
            f"head_split=(groups={groups}, rows={rows}) does not cut a "
            f"{tuple(momentum.shape)} parameter into equal slices",
            "pass (2·n_heads, 2·d_model) for a [3d, d] packed qkv weight",
        )
    lead = momentum[:rows].view(groups, rows // groups, momentum.shape[1])
    headwise = colrow(polar(lead, steps)).view(rows, -1)
    return torch.cat((headwise, colrow(polar(momentum[rows:], steps))), dim=0)


class Muon(Optimizer):
    """Momentum, then the polar factor, then ColRow, then one step of `η·√(m/n)`."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float = MUON_LR,
        momentum: float = 0.95,
        steps: int = NS_STEPS,
        weight_decay: float = 0.0,
    ) -> None:
        """Bind the four numbers every group inherits unless it overrides them."""
        if not 0.0 <= momentum < 1.0:
            raise LoudError(
                f"muon momentum={momentum} is not in [0, 1)",
                "0.95 is the reference's; 1.0 never forgets the first gradient",
            )
        if steps < 1:
            raise LoudError(
                f"muon ns steps={steps} runs no iteration",
                "5 is the recipe's, and §3 of the module docstring prices fewer",
            )
        super().__init__(
            params,
            {"lr": lr, "momentum": momentum, "steps": steps, "weight_decay": weight_decay},
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Advance every parameter that carries a gradient; return the closure's loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            steps = group["steps"]
            lr = group["lr"]
            decay = group["weight_decay"]
            head_split = group.get("head_split")
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(param)
                buffer = state["momentum"]
                buffer.mul_(momentum).add_(param.grad, alpha=1.0 - momentum)
                direction = muon_direction(buffer, head_split=head_split, steps=steps)
                if decay:
                    param.mul_(1.0 - lr * decay)
                rows, cols = param.shape
                param.add_(direction, alpha=-lr * max(1.0, rows / cols) ** 0.5)
        return loss


class AdamW(torch.optim.AdamW):
    """AdamW with §-6's formula in this docstring and one knob for the schedule.

    Torch's implementation is the formula: decoupled decay, bias-corrected moments.
    Its first step with `λ=0` is `-η·g/(|g|+ε)`, i.e. `η·sign(g)` to within ε, and
    `tests/test_optim.py` pins that as the anchor.
    """

    def set_lr(self, lr: float) -> None:
        """Set one rate on every group, which is what a schedule produces."""
        for group in self.param_groups:
            group["lr"] = lr


class MixOptimizer:
    """One `step()` over two arms, routed by §6 and refusing anything unrouted."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr_muon: float = MUON_LR,
        lr_adamw: float = ADAMW_LR,
        momentum: float = 0.95,
        steps: int = NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        muon_weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        """Route `model`'s parameters, then build the two optimizers that own them."""
        self.table = route_all(model)
        self.base_lr = (lr_muon, lr_adamw)
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
        self.muon = Muon(
            groups,
            lr=lr_muon,
            momentum=momentum,
            steps=steps,
            weight_decay=muon_weight_decay,
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

    def step(self) -> None:
        """Run both arms; the routing already decided which parameter is where."""
        self.muon.step()
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

    def state_dict(self) -> dict[str, Any]:
        """Both arms' state, keyed by arm, as a snapshot rather than a view.

        `torch.optim`'s `load_state_dict` copies the dict shallowly, so a state dict
        holding the live momentum buffers would be mutated by the run it resumed
        from — two optimizers sharing one momentum, and a resume that diverges.
        """
        return {
            "muon": deepcopy(self.muon.state_dict()),
            "adamw": deepcopy(self.adamw.state_dict()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore both arms from `state_dict`'s layout."""
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
    """`(2·n_heads, 2·d_model)` for a packed qkv weight, else None (§5).

    The head count comes from the module that owns the projection; a qkv outside an
    attention block is refused rather than quietly given the whole-matrix chain.
    """
    if not name.endswith(".attn.qkv.weight"):
        return None
    n_heads = getattr(model.get_submodule(name.rsplit(".", 2)[0]), "n_heads", None)
    if not isinstance(n_heads, int):
        raise LoudError(
            f"{name} looks like a packed qkv weight but its owner has no n_heads",
            "either pass a real attention block, or rename the parameter so the "
            "router stops reading it as q/k/v",
        )
    return 2 * n_heads, 2 * param.shape[0] // 3
