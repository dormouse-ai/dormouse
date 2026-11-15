"""The fused ops: bit-identity with the flag off, parity and a counter with it on.

The two gates here are the two ways a fused kernel is a lie. The first is a *different*
result: the kernel is faster because it is not computing what it replaced, and the
quality baseline moves with a green test suite. That is what the fp32 parity bound is
for, and its derivation is in `kernels/fused.py` — it is a bound on the order of the
`d`-term sum, not a number picked to fit the observed difference. The second is a kernel
that never ran: the flag is on, the forward passes, and the eager path executed under a
fused name. `fused_ops` is zero exactly in that case, so the flag-on/zero-counter
combination is an assertion rather than a log line.

The bit-identity tests run on CPU on purpose: they are about *which statements* run, and
running them through the GPU would test the wrong thing on a machine that has no card.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.errors import LoudError
from stacks import reference_stack

from dormouse.kernels import fused
from dormouse.model.blocks import RMSNorm

#: A shape with a live stack's width (832) and a row count a step actually has.
SHAPE = (256, 832)

REQUIRES_CARD = torch.cuda.is_available()
NO_CARD = "the fused kernels need a CUDA device; the eager path is what CI runs"


def _inputs(rows: int = SHAPE[0], width: int = SHAPE[1]) -> tuple[torch.Tensor, torch.Tensor]:
    """Two seeded residual/delta tensors, fp32, on the card when there is one."""
    device = "cuda" if REQUIRES_CARD else "cpu"
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(rows, width, generator=generator, device=device)
    delta = torch.randn(rows, width, generator=generator, device=device) * 0.1
    return x, delta


# ------------------------------------------------------------ the flag off is today


def test_flag_off_is_the_two_statements_it_replaces() -> None:
    """`residual_norm` with the flag off is `x + delta` then `norm(...)`, bit for bit."""
    fused.set_fused(False)
    x, delta = _inputs()
    norm = RMSNorm(x.shape[-1], device=x.device)
    residual, normed = fused.residual_norm(x, delta, norm)
    expected_residual = x + delta
    assert torch.equal(residual, expected_residual)
    assert torch.equal(normed, norm(expected_residual))


def test_flag_off_leaves_a_block_bit_identical() -> None:
    """A whole block's forward with the flag off does not move a bit.

    `Block.forward` now calls `residual_norm` where it used to write the add and the
    pre-norm as two statements. With the flag off those are the same two statements, so
    the quality baseline cannot move when this change lands — which is the property that
    lets the kernel be adopted before it is measured.
    """
    fused.set_fused(False)
    torch.manual_seed(5)
    model = reference_stack(vocab=16, d_model=32, n_layers=1, n_heads=2, head_dim=16, d_ffn=64)
    x = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(3))
    block = model.blocks[0]
    got = block(x)
    # The pre-change expression, written out here so the comparison is against the code
    # that used to be in the file rather than against the new code calling itself.
    expected = x + block.attn(block.norm1(x))
    expected = expected + block.ffn(block.norm2(expected))
    assert torch.equal(got, expected)


def test_flag_off_counts_nothing() -> None:
    """The counter is zero while the flag is off — the fused path did not run."""
    fused.set_fused(False)
    fused.reset_fused_ops()
    x, delta = _inputs(rows=8, width=64)
    fused.residual_norm(x, delta, RMSNorm(64, device=x.device))
    fused.glu(x, delta)
    assert fused.fused_ops() == 0


def test_flag_on_without_a_card_refuses_instead_of_going_eager() -> None:
    """A flag that is on and cannot fuse raises; it must not run the eager path."""
    if REQUIRES_CARD:
        return
    fused.set_fused(True)
    try:
        x, delta = _inputs(rows=8, width=64)
        with pytest.raises(LoudError, match="cannot run here"):
            fused.residual_norm(x, delta, RMSNorm(64, device=x.device))
    finally:
        fused.set_fused(False)


# ------------------------------------------------------------- the flag on is a kernel


def test_fused_flag_produces_launches_and_a_parity_match() -> None:
    """With the flag on: the counter moves, and fp32 output matches eager within 2e-6."""
    if not REQUIRES_CARD:
        return
    fused.reset_fused_ops()
    fused.set_fused(True)
    try:
        x, delta = _inputs()
        norm = RMSNorm(x.shape[-1], device=x.device)
        residual, normed = fused.residual_norm(x, delta, norm)
        assert fused.fused_ops() > 0
        eager_residual = x + delta
        eager_normed = norm(eager_residual)
        assert torch.equal(residual, eager_residual)
        worst = float((normed.detach() - eager_normed).abs().max())
        assert worst <= fused.FP32_PARITY, f"max|delta| {worst:.3e} > {fused.FP32_PARITY:.0e}"
    finally:
        fused.set_fused(False)


def test_bf16_path_matches_eager_within_two_ulp() -> None:
    """A bf16 stream is where the relative gate is measured, not the fp32 one.

    The eager path adds in bf16 and normalises in fp32, exactly as the kernel does, so
    the only difference is the reduction's order — and the output is then quantised to
    `2⁻⁸` relative, which is why the gate is two ulp times the row's largest entry and
    not the fp32 absolute bound.
    """
    if not REQUIRES_CARD:
        return
    fused.set_fused(True)
    try:
        x, delta = _inputs()
        x, delta = x.bfloat16(), delta.bfloat16()
        norm = RMSNorm(x.shape[-1], device=x.device)
        _, normed = fused.residual_norm(x, delta, norm)
        eager = norm(x + delta)
        worst = float((normed.detach() - eager).abs().max())
        scale = float(eager.abs().max())
        assert worst <= fused.BF16_PARITY * scale, (
            f"max|delta| {worst:.3e} > {fused.BF16_PARITY * scale:.3e} "
            f"({fused.BF16_PARITY:.0e} x {scale:.2f})"
        )
    finally:
        fused.set_fused(False)


def test_a_fused_flag_with_a_zero_counter_fails() -> None:
    """The defect the counter exists for: flag on, nothing fused, every assertion green."""
    if not REQUIRES_CARD:
        return
    fused.reset_fused_ops()
    fused.set_fused(True)
    try:
        x, delta = _inputs(rows=8, width=64)
        fused.glu(x, delta)
        assert fused.fused_ops() > 0, (
            "the flag is on and fused_ops is 0: the kernel did not run, and an eager step "
            "under a fused flag is the failure this counter exists to catch"
        )
    finally:
        fused.set_fused(False)


def test_glu_matches_the_eager_product_and_its_gradient() -> None:
    """`silu(g)*u` and its two gradients agree with torch's own path."""
    if not REQUIRES_CARD:
        return
    fused.set_fused(True)
    try:
        gate, up = _inputs(rows=64, width=512)
        gate = gate.detach().requires_grad_(True)
        up = up.detach().requires_grad_(True)
        out = fused.glu(gate, up)
        out.square().sum().backward()
        fast_gate, fast_up = gate.grad.clone(), up.grad.clone()

        eager_gate = gate.detach().clone().requires_grad_(True)
        eager_up = up.detach().clone().requires_grad_(True)
        torch.nn.functional.silu(eager_gate).mul(eager_up).square().sum().backward()

        assert float((out - torch.nn.functional.silu(gate.detach()) * up.detach()).abs().max()) <= (
            fused.FP32_PARITY
        )
        assert float((fast_gate - eager_gate.grad).abs().max()) <= 4 * fused.FP32_PARITY
        assert float((fast_up - eager_up.grad).abs().max()) <= 4 * fused.FP32_PARITY
    finally:
        fused.set_fused(False)


def test_residual_norm_gradient_matches_the_eager_norm() -> None:
    """The norm's gradient reaches the gain, the stream and the delta, exactly as eager."""
    if not REQUIRES_CARD:
        return
    fused.set_fused(True)
    try:
        x, delta = _inputs(rows=32, width=256)
        fast_norm = RMSNorm(256, device=x.device)
        torch.nn.init.normal_(fast_norm.weight)
        slow_norm = RMSNorm(256, device=x.device)
        slow_norm.weight.data.copy_(fast_norm.weight.data)

        fast_x, fast_delta = x.clone().requires_grad_(True), delta.clone().requires_grad_(True)
        weight = torch.linspace(0.1, 1.0, 256, device=x.device)
        _, fast_out = fused.residual_norm(fast_x, fast_delta, fast_norm)
        (fast_out * weight).sum().backward()

        slow_x = x.clone().requires_grad_(True)
        slow_delta = delta.clone().requires_grad_(True)
        slow_out = slow_norm(slow_x + slow_delta)
        (slow_out * weight).sum().backward()

        assert float((fast_out - slow_out).abs().max()) <= fused.FP32_PARITY
        assert float((fast_x.grad - slow_x.grad).abs().max()) <= 8 * fused.FP32_PARITY
        assert float((fast_delta.grad - slow_delta.grad).abs().max()) <= 8 * fused.FP32_PARITY
        assert float((fast_norm.weight.grad - slow_norm.weight.grad).abs().max()) <= 8 * (
            fused.FP32_PARITY
        )
    finally:
        fused.set_fused(False)
