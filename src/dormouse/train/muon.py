"""Muon: momentum, the polar factor of the momentum, and ColRow, on matrices.

The formulas, as implemented:

    MUON (Jordan 2024, arXiv:2409.16986 §3; Keller Jordan's reference `muon`)
        m_t = μ·m_{t-1} + (1-μ)·g_t
        Q_t = polar(m_t)                    # Newton–Schulz, 5 iterations
        R_t = ColRow(Q_t)                   # columns to unit norm, then rows
        W_t = W_{t-1}·(1 - η·λ) - η·max(1, m/n)^½·R_t

    POLAR (Higham 1986; the orthogonal factor of the SVD, U V^T)
        X_0 = m/‖m‖_σ,  X_{k+1} = X_k(15 - 10X_k² + 3X_k⁴)/8

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

`schedule="sf"` (see `sf.py`) does NOT use ColRow, and runs the same channel instead
through the row-wise second moment of `polar` that NorMuon adds. The two are the same
invariance — a step whose magnitude does not depend on the spectrum — and applying
ColRow to a NorMuon direction divides out exactly the per-row scale the second moment
exists to set, so the normalization would be a no-op there.

§5. THE PER-HEAD ARM, AND ITS ONE HONEST PROPERTY. Kimi K3 §2.5 and DeepSeek V4.1
apply the preconditioner per head to Q and K. Splitting a `[3d, d]` packed qkv weight
into `2·n_heads` slices of `[head_dim, d]` and running the chain on each changes the
*direction* (the Gram is `[head_dim, head_dim]` instead of `[d, d]`) and, because
ColRow is applied per slice, leaves `‖R_t‖_F = √(3d)` — exactly what the whole-matrix
path gives. So it is a direction change at fixed step magnitude, which is why it needs
no learning-rate retune. V and the output projection stay whole: neither reference
splits them, and a per-head split of `proj` is a column split, a different mechanism.

The compressed layers have no packed matrix: their Q is written by `q_up`
`[n_heads·head_dim, c_lat]` and their K/V by `kv_up` `[2·n_kv_heads·head_dim, c_lat]`.
`q_up` splits into `n_heads` slices and `kv_up`'s K half into `n_kv_heads`, with V — the
remainder — whole, so the arm is the same one applied to the same tensors, and the
router reads it off the modules rather than off a shape heuristic.

§6. THE MOMENTUM AT 8 BITS, MEASURED, AND WHY THE DEFAULT IS 32. Muon orthogonalises, so
an error in the buffer is not an error in a scale — it is an error in the *direction*, and
§1 makes the direction the whole update. The gate is therefore the cosine between
`polar(m)` at 32 bits and at 8, on the two live shapes, with the momentum of a run 20
steps in (`tools/lowbit_gates.py --only direction`):

    [832, 2048]   block 256: 0.954    block 64: 0.964    block 16: 0.976
    [832, 832]    block 256: 0.985    block 64: 0.990    block 16: 0.993

All six are below the 0.999 a direction has to hold to be worth quantising, and no block
size reaches it: a young buffer is close to noise, its spectrum is nearly flat, and the
small singular values sit on the quantisation floor — the polar factor turns with them,
which is exactly the sensitivity §3 records for a badly conditioned matrix. So
`optim_momentum_bits` defaults to 32 and is a separate knob from `optim_state_bits`, which
is AdamW's second moment: a relative error in `v` is halved by the square root before it
reaches a step, and a relative error in `m` is the step.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from dormouse_core.errors import LoudError
from torch.optim import Optimizer

from .lowbit import BLOCK, LowBitState, adopt, check, plain_state_dict, revive_state_dict

__all__ = [
    "COLROW_EPS",
    "MUON_LR",
    "NS_COEFFS",
    "NS_EPS",
    "NS_POWER_ITERS",
    "NS_STEPS",
    "Muon",
    "colrow",
    "muon_direction",
    "orthogonalise",
    "polar",
    "spectral_bound",
]

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


def _regions(momentum: torch.Tensor, head_split: tuple[int, int] | None) -> list[torch.Tensor]:
    """Return the views one momentum is preconditioned in: one, or heads-then-rest (§5)."""
    if head_split is None:
        return [momentum]
    groups, rows = head_split
    if rows % groups or rows > momentum.shape[0] or momentum.ndim != 2:
        raise LoudError(
            f"head_split=(groups={groups}, rows={rows}) does not cut a "
            f"{tuple(momentum.shape)} parameter into equal slices",
            "pass (2·n_heads, 2·d_model) for a [3d, d] packed qkv weight",
        )
    lead = momentum[:rows].view(groups, rows // groups, momentum.shape[1])
    rest = momentum[rows:]
    return [lead] if rest.shape[0] == 0 else [lead, rest]


def _assemble(parts: list[torch.Tensor]) -> torch.Tensor:
    """Stack the preconditioned regions back into one `[rows, cols]` matrix."""
    if len(parts) == 1:
        single = parts[0]
        return single.reshape(-1, single.shape[-1])
    return torch.cat([part.reshape(-1, part.shape[-1]) for part in parts], dim=0)


def orthogonalise(
    momentum: torch.Tensor,
    *,
    head_split: tuple[int, int] | None = None,
    steps: int = NS_STEPS,
) -> torch.Tensor:
    """`polar(momentum)` per head when the parameter is a packed qkv weight (§5).

    The direction without ColRow: what `sf.py` needs, because its row-wise second
    moment divides each row by its own scale and ColRow's unit row norm would leave
    nothing to divide.
    """
    return _assemble([polar(part, steps) for part in _regions(momentum, head_split)])


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
    return _assemble([colrow(polar(part, steps)) for part in _regions(momentum, head_split)])


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
        state_bits: int = 32,
        block: int = BLOCK,
    ) -> None:
        """Bind the four numbers every group inherits unless it overrides them.

        `state_bits=32` is the momentum buffer `torch.empty_like` always allocated;
        `8` is the same buffer in blocks of `block` values (see §6).
        """
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
        check(state_bits, block)
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "steps": steps,
                "weight_decay": weight_decay,
                "state_bits": state_bits,
                "block": block,
            },
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
            #: `get`, not `[]`: a checkpoint written before this knob existed restores
            #: its own `param_groups`, and a 32-bit state is what that checkpoint holds.
            state_bits = group.get("state_bits", 32)
            block = group.get("block", BLOCK)
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum" not in state:
                    state["momentum"] = LowBitState(param, bits=state_bits, block=block)
                buffer = state["momentum"]
                buffer.apply(partial(_ema, momentum=momentum), param.grad)
                direction = muon_direction(buffer.dense(), head_split=head_split, steps=steps)
                if decay:
                    param.mul_(1.0 - lr * decay)
                rows, cols = param.shape
                param.add_(direction, alpha=-lr * max(1.0, rows / cols) ** 0.5)
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Write the momentum as plain tensors, so `weights_only=True` can read it back (§5)."""
        return plain_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a checkpoint into the container, on each parameter's own device."""
        super().load_state_dict(revive_state_dict(state_dict))
        adopt(self.state)


def _ema(buffer: torch.Tensor, grad: torch.Tensor, *, momentum: float) -> None:
    """`m ← μ·m + (1-μ)·g`, the one in-place pair a window of the buffer sees."""
    buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)
