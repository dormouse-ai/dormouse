"""D01 fixes: the log-decay is fp32 before autocast, QB bias is not saved, alpha has a floor.

B4 computed `a_up(a_down(x))` under the step's bf16 autocast and only then called
`.float()`: the value the recurrence exponentiates was already rounded. B5 stored a
checkpointed QB bias against router.py's own contract; moe.py owns a Router, so the
bank's state_dict is the place the regression would show up. N2/N6 drove alpha to
zero at high lr, turning the Gamma division into 1e12 — now a named refusal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dormouse_core.errors import LoudError
from torch import nn

from dormouse.model.attn_variants.kda import KDA
from dormouse.model.moe import MoEFFN
from dormouse.model.router import Router

D_MODEL, HEADS, WIDTH = 32, 2, 8


def _fp32_log_decay(kda: KDA, x: torch.Tensor) -> torch.Tensor:
    """The exponentiated quantity with autocast off: what B4 must produce."""
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        work = (
            kda.a_up(kda.activation(kda.a_down(x))).view(
                x.shape[0], x.shape[1], kda.n_heads, kda.head_dim
            )
            + kda.a_bias
        )
        return torch.exp(-nn.functional.softplus(work))


def test_log_decay_is_not_bf16_rounded_before_the_fp32_guard() -> None:
    """Under bf16 autocast, alpha must equal the fp32 log-decay, not its rounding."""
    torch.manual_seed(11)
    kda = KDA(D_MODEL, HEADS, WIDTH, chunk=8)
    x = torch.randn(2, 8, D_MODEL)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, _, alpha, _ = kda._project(x)
    want = _fp32_log_decay(kda, x)
    assert alpha.dtype == torch.float32
    #: At init the rounded logits are small, so the bf16 gap is ~8e-7 — far above
    #: the exactness the fp32 path has. A tolerance of 1e-6 would hide B4.
    assert torch.allclose(alpha, want, atol=1e-9, rtol=0), (
        float((alpha.detach() - want.detach()).abs().max()),
    )


def test_log_decay_keeps_an_external_fp64_anchor() -> None:
    """A fp64 module projects through fp64; numpy computes the reference independently."""
    torch.manual_seed(12)
    kda = KDA(D_MODEL, HEADS, WIDTH, chunk=8).double()
    x = torch.randn(1, 8, D_MODEL, dtype=torch.float64)
    _, _, _, alpha, _ = kda._project(x)

    h = x.detach().numpy() @ kda.a_down.weight.detach().numpy().T
    h = h / (1.0 + np.exp(-h))  # SiLU
    work = (h @ kda.a_up.weight.detach().numpy().T).reshape(1, 8, HEADS, WIDTH)
    work += kda.a_bias.detach().numpy()
    want = np.exp(-np.logaddexp(0.0, work))

    assert alpha.dtype == torch.float64
    assert np.allclose(alpha.detach().numpy(), want, atol=1e-12, rtol=0)


def test_qb_bias_is_checkpointed_but_finite_and_centered() -> None:
    """B1: QB bias MUST be in the checkpoint for bit-exact resume/eval.

    N2's fix is the fp32 softplus+clamp that keeps it finite; removing the bias
    from state_dict broke resume (8/20 losses off by 7.5e-4). The guard here is
    that the persisted load controller is finite and zero-mean, not absent.
    """
    bank = MoEFFN(
        d_model=16,
        d_ffn=12,
        n_experts=4,
        topk=2,
        act="silu",
        expert_tsct_rank=0,
        min_expert_params=0,
    )
    assert isinstance(bank.router, Router)
    assert "bias" in bank.router.state_dict()
    assert any(key.startswith("router.bias") for key in bank.state_dict())
    bias = bank.router.state_dict()["bias"]
    assert torch.isfinite(bias).all()
    assert bias.abs().max() < 1.0, "the load controller must not drift"

    # A checkpoint's live load controller IS restored: bit-exact resume depends
    # on it. The fp32 softplus+clamp gate (N2) is what keeps it safe.
    snapshot = bank.state_dict()
    snapshot["router.bias"] = torch.full((4,), 0.37)
    bank.load_state_dict(snapshot, strict=False)
    assert torch.equal(bank.router.bias, torch.full((4,), 0.37))


def test_a_collapsed_decay_is_a_named_refusal() -> None:
    """alpha below the floor breaks the Gamma division; refuse instead of solving 1e12."""
    kda = KDA(D_MODEL, HEADS, WIDTH, chunk=8)
    with torch.no_grad():
        kda.a_bias.fill_(100.0)
    with pytest.raises(LoudError, match="alpha"):
        kda(torch.randn(1, 8, D_MODEL))


def test_a_healthy_decay_does_not_trip_the_floor() -> None:
    """The guard is a collapse detector, not a clamp on the init's 0.982."""
    kda = KDA(D_MODEL, HEADS, WIDTH, chunk=8)
    assert float(kda(torch.randn(1, 8, D_MODEL)).detach().abs().max()) < 1e3
