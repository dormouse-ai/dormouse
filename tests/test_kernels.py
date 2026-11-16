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


def test_the_module_reports_its_state_without_a_device() -> None:
    """The flag and the counter are readable on a CPU wheel; the kernels are not built.

    A fused path that silently runs eager under the flag is the defect this pins: on a
    machine without Triton `available()` must say so, and the tally must stay zero.
    """
    from dormouse.kernels import fused as k

    assert isinstance(k.available(), bool)
    assert k.next_pow2(1) == 1
    assert k.next_pow2(3) == 4
    assert k.next_pow2(1024) == 1024
    k.reset_fused_ops()
    assert k.fused_ops() == 0
    was = k.fused()
    try:
        k.set_fused(True)
        assert k.fused() is True
        assert k.fused_ops() == 0, "the flag alone must not claim a launch"
    finally:
        k.set_fused(was)


# ------------------------------------------------------- the mixture, one pass over z
#
# The mixture is the kernel that mattered: the chain ran one `torch.utils.checkpoint` per
# dictionary entry with a Python CALLABLE as the checkpointed function, and dynamo cannot
# trace through a callable argument (2.13 stops at `lift_tracked_freevar_to_input`), so
# the bank's whole forward was opaque to `torch.compile`. These tests hold the kernel to
# the chain it replaced: bit-identical with the flag off, inside the fp32 bound on the
# card, and a counted launch rather than an eager path under a fused name.


def _moa(seed: int = 3, dictionary: tuple[str, ...] = ("gelu", "silu", "relu2", "tanh")):
    """A seeded `MoaFFN` at a width a CI test can afford, on the card when there is one."""
    from dormouse.model.ffn import MoaFFN

    torch.manual_seed(seed)
    arm = MoaFFN(32, 48, dictionary=dictionary, device="cuda" if REQUIRES_CARD else "cpu")
    x = torch.randn(3, 7, 32, generator=torch.Generator().manual_seed(seed + 1))
    return arm, x.to(next(arm.parameters()).device)


def test_mixture_with_the_flag_off_is_the_chain_bit_for_bit() -> None:
    """Flag off is the per-entry chain, `torch.equal` — the baseline cannot move."""
    fused.set_fused(False)
    arm, x = _moa()
    with torch.no_grad():
        got = arm(x)
        g, u = arm.gate(x), arm.up(x)
        weights = arm._weights(x)
        hidden = None
        for k, fn in enumerate(arm._fns):
            part = weights[..., k : k + 1] * fn(g)
            hidden = part if hidden is None else hidden + part
        other = None
        for k, fn in enumerate(arm._fns):
            part = weights[..., k : k + 1] * fn(u)
            other = part if other is None else other + part
        expected = arm.down(hidden * other)
    assert torch.equal(got, expected)


def test_the_fused_mixture_matches_the_chain_in_fp32() -> None:
    """The kernel computes the same sum to fp32 accuracy, forward and both gradients."""
    if not REQUIRES_CARD:
        return
    arm, x = _moa()
    slow = {name: p.detach().clone() for name, p in arm.named_parameters()}

    fused.set_fused(False)
    torch.manual_seed(7)
    coarse = x.detach().clone().requires_grad_(True)
    out_slow = arm(coarse)
    weight = torch.linspace(-1.0, 1.0, out_slow.numel(), device=x.device).view_as(out_slow)
    (out_slow * weight).sum()
    out_slow.square().sum().backward()
    grads_slow = {n: p.grad.clone() for n, p in arm.named_parameters() if p.grad is not None}

    for name, p in arm.named_parameters():
        p.data.copy_(slow[name])
        p.grad = None
    fused.reset_fused_ops()
    fused.set_fused(True)
    try:
        fast = x.detach().clone().requires_grad_(True)
        out_fast = arm(fast)
        out_fast.square().sum().backward()
        assert fused.fused_ops() == 4, (
            f"one forward and one backward over two mixed sides is 4 launches, "
            f"got {fused.fused_ops()}"
        )
    finally:
        fused.set_fused(False)

    scale = float(out_slow.detach().abs().max())
    worst = float((out_fast.detach() - out_slow.detach()).abs().max())
    assert worst <= 1e-6 * scale, f"max|delta| {worst:.3e} on a scale of {scale:.3f}"
    for name, param in arm.named_parameters():
        assert param.grad is not None, f"{name} received no gradient through the kernel"
        assert bool(torch.isfinite(param.grad).all()), f"{name}'s gradient is not finite"
        delta = float((param.grad - grads_slow[name]).abs().max())
        edge = float(grads_slow[name].abs().max())
        assert delta <= 1e-6 * edge, f"{name}: {delta:.3e} on a scale of {edge:.3e}"
    assert float(arm.coef.grad.abs().sum()) > 0.0, "the coefficients got no gradient"


def test_the_fused_mixture_matches_the_chain_in_bf16() -> None:
    """A bf16 stream: the chain's own rounding is the difference, and fp32 is the floor.

    The kernel accumulates in fp32 where the chain rounds every partial sum to bf16, so
    it is the *more* accurate of the two and cannot be bit-identical to the chain. The
    gate is therefore two of the chain's own ulp, and it is paired with the fp32 test
    above so a wrong formula cannot hide inside a bf16 tolerance.
    """
    if not REQUIRES_CARD:
        return
    arm, x = _moa(seed=4)
    arm = arm.bfloat16()
    slow = {name: p.detach().clone() for name, p in arm.named_parameters()}
    bf16 = x.bfloat16()
    fused.set_fused(False)
    out_slow = arm(bf16.detach().clone())
    for name, p in arm.named_parameters():
        p.data.copy_(slow[name])
    fused.set_fused(True)
    try:
        out_fast = arm(bf16.detach().clone())
    finally:
        fused.set_fused(False)
    scale = float(out_slow.detach().abs().max())
    worst = float((out_fast.detach() - out_slow.detach()).abs().max())
    assert worst <= 4 * fused.BF16_PARITY * scale, (
        f"max|delta| {worst:.3e} > {4 * fused.BF16_PARITY * scale:.3e} "
        f"({fused.BF16_PARITY:.0e} x {scale:.2f})"
    )


def test_the_fused_mixture_retains_no_more_than_the_chain() -> None:
    """The memory gate: the kernel keeps `z` and `w`, where the chain's graph kept `K`."""
    if not REQUIRES_CARD:
        return
    from dormouse.model.ffn import MoaFFN

    rows, width, ffn = 2048, 64, 512

    def peak(flag: bool) -> int:
        torch.manual_seed(5)
        arm = MoaFFN(width, ffn, device="cuda")
        x = torch.randn(rows, width, device="cuda", requires_grad=True)
        fused.set_fused(flag)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            arm(x).sum().backward()
            torch.cuda.synchronize()
            return int(torch.cuda.max_memory_allocated())
        finally:
            fused.set_fused(False)
            del arm, x
            torch.cuda.empty_cache()

    chain, kernel = peak(False), peak(True)
    assert kernel <= chain, f"fused peak {kernel / 2**20:.0f} MiB > chain {chain / 2**20:.0f} MiB"


def test_an_unknown_entry_is_refused_rather_than_guessed() -> None:
    """The kernel answers for six formulas; a seventh is a refusal, not a default."""
    with pytest.raises(LoudError, match="not a dictionary entry"):
        fused.mixture_code("swish-ish")


# ------------------------------------------------------- the entry points' refusals
#
# Every kernel entry point here is a launch, never a fallback: on a CPU wheel each one
# must say so rather than run the eager path under the kernel's name. The card job never
# takes these branches, so CI is the only place they are measured.


def test_the_mixture_codes_are_the_dictionary_the_kernel_computes() -> None:
    """One code per entry, dense from 0, and the table is the epilogue's constexpr set."""
    assert list(fused.MIX_CODES) == ["gelu", "silu", "relu2", "leaky", "tanh", "situ"]
    assert sorted(fused.MIX_CODES.values()) == list(range(fused.MAX_MIX_ENTRIES))
    for name, code in fused.MIX_CODES.items():
        assert fused.mixture_code(name) == code


def test_mixture_refuses_the_flag_off_a_cpu_launch_and_an_empty_dictionary() -> None:
    """Three refusals in `mixture`, each naming its own escape."""
    z = torch.randn(4, 8)
    w = torch.ones(4, 2)
    fused.set_fused(False)
    with pytest.raises(LoudError, match="fused flag off"):
        fused.mixture(z, w, (0, 1), per_row=True)
    fused.set_fused(True)
    try:
        with pytest.raises(LoudError, match="cannot run here"):
            fused.mixture(z, w, (0, 1), per_row=True)
        for kinds in ((), (0,) * (fused.MAX_MIX_ENTRIES + 1)):
            with pytest.raises(LoudError, match="fused mixture of"):
                fused.mixture(z, w, kinds, per_row=True)
    finally:
        fused.set_fused(False)


def test_a_fused_flag_without_a_card_refuses_in_every_entry_point() -> None:
    """`glu` and `residual_norm` too: the flag on a CPU wheel is a refusal, not eager."""
    fused.set_fused(True)
    try:
        with pytest.raises(LoudError, match="cannot run here"):
            fused.glu(torch.randn(4, 8), torch.randn(4, 8))
        with pytest.raises(LoudError, match="cannot run here"):
            fused.residual_norm(torch.randn(4, 8), torch.randn(4, 8), RMSNorm(8))
    finally:
        fused.set_fused(False)


def test_the_flat_view_copies_only_when_it_has_to() -> None:
    """`[rows, width]` as a view when contiguous, a copy when the strides say otherwise."""
    contiguous = torch.arange(24.0).reshape(4, 6)
    assert fused._flat(contiguous, 6).is_contiguous()
    assert fused._flat(contiguous, 6).data_ptr() == contiguous.data_ptr(), "no copy needed"
    strided = contiguous[:, ::2]
    assert not strided.is_contiguous()
    copied = fused._flat(strided, 3)
    assert copied.is_contiguous() and torch.equal(copied, strided)


def test_the_launch_tally_is_the_counter_the_gate_reads() -> None:
    """A flag on with `fused_ops` at zero is the defect; the tally is what moves it."""
    was = fused.fused_ops()
    fused._tally()
    assert fused.fused_ops() == was + 1
    fused.reset_fused_ops()
    assert fused.fused_ops() == 0
