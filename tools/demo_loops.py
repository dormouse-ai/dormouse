"""F1 inference-time loop graft for the Friday demo (S6, arXiv 2605.23872 equations).

`graft` patches ONLY the model's forward: the contiguous mid-block window is applied
as a damped Euler fixed point ``x <- (1-beta) x + beta g(x)`` K times (S6, default).
``gated=True`` replaces the scalar beta with the G1 per-token write-back

    beta_p(h) = clamp(sigma(W_g^p h + b_p), 0.05, 1)
    x_{p+1}   = x_p + beta_p(h_p) (g_p(x_p) - x_p)

with ``W_g = 0`` and complementary biases ``b = (-2.2, +2.2)`` (identity start,
GLM-5.4 style). The gate is a standalone module held outside ``model._modules``, so
``state_dict()`` stays bit-for-bit the original in both modes. No kernel, no
optimizer and no backbone weight is changed.

    model = graft(model, window=(7, 10), K=2, beta=0.5, mode="layer")
    model = graft(model, window=(7, 10), K=2, gated=True)   # opt-in

The implementation is from the update equation, not copied from any CC BY-NC-ND code.
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

#: logit(0.1) / logit(0.9): conservative pass 1, full pass 2, complementary start.
GATE_BIAS = (-2.2, 2.2)
#: A closed gate leaves no gradient path; a gate above 1 inflates the residual norm.
GATE_FLOOR = 0.05
GATE_CEIL = 1.0


class LoopGates(nn.Module):
    """Per-token, per-pass write-back gate for the G1 variant.

    ``beta_p(h) = clamp(sigma(W_g^p h + b_p), 0.05, 1)``. ``W_g`` starts at zero so
    the learned part is neutral; ``difficulty`` adds the training-free G1 proxy
    (normalised ``||h||`` deviation) to pass 1, which is what makes the gate vary
    across tokens before any continued training. Harder tokens write more.
    """

    def __init__(self, d_model: int, K: int = 2, *, difficulty: float = 0.5) -> None:  # noqa: N803
        """Initialise zero weights and complementary pass biases."""
        super().__init__()
        if d_model < 1 or K < 1:
            raise LoudError(f"LoopGates(d_model={d_model}, K={K})", "positive widths and passes")
        self.K = K
        self.difficulty = difficulty
        self.W = nn.Parameter(torch.zeros(K, d_model, 1))
        self.b = nn.Parameter(torch.tensor(GATE_BIAS[:K], dtype=torch.float32))

    def forward(self, h: torch.Tensor, p: int) -> torch.Tensor:
        """One [batch, seq, 1] gate per token for pass ``p``."""
        if not 0 <= p < self.K:
            raise LoudError(f"pass p={p} outside K={self.K}", "p indexes the loop passes")
        score = torch.matmul(h, self.W[p]) + self.b[p]
        if self.difficulty and p == 0:
            scale = h.norm(dim=-1, keepdim=True)
            spread = scale.std(dim=1, keepdim=True).clamp_min(1e-6)
            proxy = ((scale - scale.mean(dim=1, keepdim=True)) / spread).clamp(-2.0, 2.0)
            score = score + self.difficulty * proxy
        return torch.sigmoid(score).clamp(GATE_FLOOR, GATE_CEIL)


def _check(
    model: nn.Module,
    window: tuple[int, int],
    K: int,  # noqa: N803
    beta: float,
    mode: str,
    *,
    gated: bool,
) -> None:
    """Refuse a loop the stack cannot express, by name."""
    if mode != "layer":
        raise LoudError(f"mode={mode!r} is not the layer loop", "use mode='layer'")
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise LoudError(
            "graft needs model.blocks", "the loop is over contiguous Transformer blocks"
        )
    lo, hi = window
    if not (0 <= lo < hi < len(blocks)):
        raise LoudError(
            f"window={window} is outside {len(blocks)} blocks",
            "the loop needs at least one interior layer after a prefix and before a suffix",
        )
    if K < 1 or not isinstance(K, int):
        raise LoudError(f"K={K} is not a positive iteration count", "K counts fixed-point steps")
    if not gated and not 0.0 < beta <= 1.0:
        raise LoudError(f"beta={beta} is outside (0, 1]", "the damping coefficient must shrink x")


def loop_gates(model: nn.Module) -> LoopGates | None:
    """Return the gated wrapper's gates, or None for a plain/S6 model.

    The gate lives in ``model.__dict__`` on purpose: ``nn.Module.__setattr__`` would
    register it in ``_modules`` and break the bitwise ``state_dict`` contract.
    """
    return model.__dict__.get("_loop_gates")


def gate_stats(model: nn.Module) -> dict[str, Any]:
    """Return the last forward's gate means, token entropy and pass norms."""
    return dict(model.__dict__.get("_loop_gate_stats", {}))


def fp64_gate_anchor(h: torch.Tensor, gates: LoopGates, p: int) -> torch.Tensor:
    """Independent fp64 evaluation of the gate formula (an external anchor)."""
    with torch.no_grad():
        x = h.detach().double()
        score = x @ gates.W[p].detach().double() + gates.b[p].detach().double()
        if gates.difficulty and p == 0:
            scale = x.norm(dim=-1, keepdim=True)
            spread = scale.std(dim=1, keepdim=True).clamp_min(1e-6)
            proxy = ((scale - scale.mean(dim=1, keepdim=True)) / spread).clamp(-2.0, 2.0)
            score = score + gates.difficulty * proxy
        return torch.sigmoid(score).clamp(GATE_FLOOR, GATE_CEIL)


def graft(
    model: nn.Module,
    *,
    window: tuple[int, int] = (7, 10),
    K: int = 2,  # noqa: N803
    beta: float = 0.5,
    mode: str = "layer",
    gated: bool = False,
    gates: LoopGates | None = None,
    difficulty: float = 0.5,
) -> nn.Module:
    """Wrap forward with K damped applications of blocks[window[0]:window[1]+1].

    S6 (``gated=False``): ``h = (1-beta) h + beta g(h)``, zero new parameters.
    G1 (``gated=True``): ``h = h + beta_p(h) (g(h) - h)`` with a per-token gate.
    ``K=1`` is the identity case: forward is left untouched, so logits and
    ``state_dict`` are bitwise identical to the ungrafted model (S6 contract).

    With counters, each forward increments ``loop_fwd`` (one per pass), records
    ``gate_write_mean_p1/p2`` and ``gate_entropy`` (>0 when the gate varies across
    tokens), and registers a hook that increments ``loop_bwd`` on every backward
    through a gate. ``check()`` at the end of forward proves the forward counters;
    a trainer's post-backward ``check()`` additionally proves ``loop_bwd``.
    """
    _check(model, window, K, beta, mode, gated=gated)
    if K == 1:
        return model
    blocks = model.blocks
    lo, hi = window
    tail = range(hi + 1, len(blocks))
    if gated:
        gate = gates or LoopGates(model.embed.embedding_dim, K, difficulty=difficulty)
        if gate.K != K:
            raise LoudError(f"gates.K={gate.K} != K={K}", "one gate per loop pass")
        # Plain dict entry: NOT a submodule, NOT a buffer, NOT a parameter of model.
        model.__dict__["_loop_gates"] = gate
    else:
        gate = None

    def looped_forward(self: nn.Module, ids: torch.Tensor, counters: Any = None) -> torch.Tensor:
        """Run the stack's forward with the mid-block window iterated K times."""
        if ids.dim() != 2 or ids.is_floating_point():
            raise LoudError("loop expects token ids [batch, seq]", "cast to a 2-D long tensor")
        if counters is not None:
            self._enable(counters)
            counters.count("embed_fwd", ids.numel())
        h = self.embed(ids)
        for index in range(lo):
            h = blocks[index](h, counters)
        if gate is None:
            for _ in range(K):
                updated = h
                for index in range(lo, hi + 1):
                    updated = blocks[index](updated, counters)
                h = (1.0 - beta) * h + beta * updated
        else:
            means: list[float] = []
            norms: list[float] = []
            entropy = 0.0
            for p in range(K):
                beta_p = gate(h, p)
                if counters is not None and beta_p.requires_grad:

                    def _bwd(grad: torch.Tensor, c: Counters = counters) -> torch.Tensor:
                        c.enable("loop_bwd", counter="loop_bwd", why="gradient reached the gate")
                        c.count("loop_bwd", 1)
                        return grad

                    beta_p.register_hook(_bwd)
                updated = h
                for index in range(lo, hi + 1):
                    updated = blocks[index](updated, counters)
                h = h + beta_p * (updated - h)
                value = beta_p.detach().clamp(1e-6, 1.0 - 1e-6)
                means.append(float(value.mean()))
                norms.append(float(h.detach().norm()))
                entropy += float(
                    -(value * value.log() + (1.0 - value) * (1.0 - value).log()).mean()
                )
                if counters is not None:
                    counters.count("loop_fwd", 1)
            if counters is not None:
                for p, mean in enumerate(means, start=1):
                    # Counter values are ints by convention; these are measured stats,
                    # set directly so they survive Counters.line()/snapshot().
                    counters.counts[f"gate_write_mean_p{p}"] = mean
                counters.counts["gate_entropy"] = entropy / max(K, 1)
            model.__dict__["_loop_gate_stats"] = {
                "gate_write_mean": tuple(means),
                "gate_entropy": entropy / max(K, 1),
                "activation_norm_pass": tuple(norms),
            }
        for index in tail:
            h = blocks[index](h, counters)
        logits = self.head(self.norm(h, counters), counters)
        if counters is not None:
            counters.check()
        return logits

    model.forward = types.MethodType(looped_forward, model)
    return model


def polynomial_weights(K: int = 3) -> tuple[float, ...]:  # noqa: N803
    """F4: binomial/C(K-1,k)/2^(K-1) weights for a K-1 loop polynomial."""
    if K < 2:
        raise LoudError(f"K={K} has no polynomial ablation", "K>=2 gives at least two states")
    denom = 2.0 ** (K - 1)
    return tuple(math.comb(K - 1, k) / denom for k in range(K))


def _demo_model() -> tuple[nn.Module, Any]:
    """Build a tiny assembled stack for the CPU self-test."""
    from dormouse.model.assemble import assemble
    from dormouse.model.spec import StackSpec

    spec = StackSpec(
        d_model=64,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        c_lat=32,
        d_ffn=128,
        n_experts=4,
        topk=1,
        expert_tsct_rank=8,
        attn_k=64,
        attn_window=32,
        seq_len=128,
        batch=2,
        torch_compile=False,
    )
    return assemble(spec, device="cpu", smoke=True).model, spec


def self_test() -> str:
    """Check on CPU that both wrappers change logits and never touch weights."""
    from dormouse_core.counters import Counters as _Counters

    plain, spec = _demo_model()
    plain.eval()
    before = {key: value.clone() for key, value in plain.state_dict().items()}
    ids = torch.randint(0, spec.vocab, (2, 128))
    with torch.no_grad():
        base = plain(ids)
    graft(plain, window=(3, 5), K=2, beta=0.5, mode="layer")
    after = plain.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    with torch.no_grad():
        looped = plain(ids)
    assert looped.shape == base.shape
    assert not torch.equal(looped, base)

    gated, spec = _demo_model()
    gated.eval()
    before = {key: value.clone() for key, value in gated.state_dict().items()}
    graft(gated, window=(3, 5), K=2, gated=True)
    gates = loop_gates(gated)
    assert gates is not None
    assert torch.equal(gates.W, torch.zeros_like(gates.W))
    assert torch.allclose(gates.b, torch.tensor(GATE_BIAS))
    after = gated.state_dict()
    assert before.keys() == after.keys()  # gates are NOT registered on the model
    assert all(torch.equal(before[key], after[key]) for key in before)
    counters = _Counters()
    with torch.no_grad():
        gated(ids, counters)
    assert counters.counts["loop_fwd"] == 2
    stats = gate_stats(gated)
    assert stats["gate_write_mean"][0] < stats["gate_write_mean"][1]
    assert stats["gate_entropy"] > 0
    counters = _Counters()
    out = gated(ids, counters)
    out.sum().backward()
    assert counters.counts["loop_bwd"] == 2
    assert gates.W.grad is not None and gates.b.grad is not None
    assert polynomial_weights(3) == (0.25, 0.5, 0.25)
    return (
        f"graft self-test PASS: {len(before)} tensors bitwise in both modes, "
        "loop_fwd=2 loop_bwd=2, gate_entropy>0"
    )


if __name__ == "__main__":
    print(self_test())
