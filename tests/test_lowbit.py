"""8-bit optimizer state: the code's bound, what is exact, and where the momentum stays.

Every number here is derived from the module's own docstring — the codebook in the
header, the `N/71` and 55% lines of §3, the fp32 range that §2's tail covers — or run
through the real arms. The two measurements are the ones the design is accepted or
rejected against: the AdamW drift under a quantised `v`, and the cosine between Muon's
direction at 32 bits and at 8 on the shapes the stack actually holds.
"""

from __future__ import annotations

import io

import pytest
import torch
from dormouse_core.errors import LoudError
from stacks import reference_stack

from dormouse.train.adamw import AdamW
from dormouse.train.graph import state_bytes
from dormouse.train.lowbit import (
    BLOCK,
    MAGNITUDE,
    LowBitState,
    check,
    plain_state_dict,
    revive_state_dict,
)
from dormouse.train.muon import Muon, muon_direction
from dormouse.train.optim import MixOptimizer

#: The cosine a quantised momentum would have to hold to be used: below it, the buffer
#: stays 32 bits. `muon.py` §6 is where the decision is written.
DIRECTION_BAR = 0.999

#: `[832, 2048]` is a down projection, `[832, 832]` an attention output: the two live
#: shapes the gate names.
LIVE = {"vocab": 64, "d_model": 832, "d_ffn": 2048, "n_heads": 13, "head_dim": 64}
SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}
LIVE_NAMES = ("blocks.0.ffn.down.weight", "blocks.0.attn.proj.weight")


def round_trip(values: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Quantise and dequantise through the container: one window, every block (§2)."""
    state = LowBitState(values, bits=8, block=block)
    state.apply(lambda window, source: window.copy_(source), values)
    return state.dense()


def block_maxima(values: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """`N_j` per element: the largest magnitude in the block that element sits in (§1)."""
    flat = values.reshape(-1)
    pad = -(-flat.numel() // block) * block
    padded = torch.cat([flat, flat.new_zeros(pad - flat.numel())])
    return padded.view(-1, block).abs().amax(dim=1).repeat_interleave(block)[: flat.numel()]


# --------------------------------------------------------------- the code and its bound


def test_the_codebook_is_ascending_and_spends_eight_bits() -> None:
    """128 codes: a zero, 126 magnitudes and `1` — the layout the header states."""
    assert MAGNITUDE.numel() == 128
    assert MAGNITUDE[0] == 0.0
    assert MAGNITUDE[-1] == 1.0
    assert bool((MAGNITUDE[1:] > MAGNITUDE[:-1]).all()), "a non-monotone code is unsearchable"


def test_the_tail_covers_the_range_adams_second_moment_actually_spans() -> None:
    """§2: `g²` spans `10^-9` of its block's maximum, and a `10^-6` floor zeroed it.

    The measured failure the tail exists for: an element whose `v` is floored to zero
    steps by `m/ε`, which at the first step of AdamW is 500× the correct `sign(g)·η`.
    """
    values = torch.cat([torch.full((BLOCK,), 1.0), torch.full((BLOCK,), 1e-9)])
    got = round_trip(values)
    assert bool((got[BLOCK:] > 0).all()), "a nonzero moment quantised to zero"
    relative = ((got[BLOCK:] - 1e-9).abs() / 1e-9).max()
    assert relative.item() <= 0.55, relative.item()


def test_the_round_trip_error_is_the_codebook_grid_and_nothing_else() -> None:
    """§3: `≤ N_j/71` above `N_j/10`, `≤ 55%` of the value below it, for every element."""
    torch.manual_seed(0)
    values = torch.cat(
        [
            torch.randn(4096),
            torch.randn(512) * 1e-3,
            torch.randn(512) * 1e3,
            torch.full((512,), 40.0),
        ]
    )
    error = (round_trip(values) - values).abs()
    scale = block_maxima(values)
    top = values.abs() >= scale / 10
    assert top.any() and (~top).any(), "the fixture must cover both sides of the bound"
    assert bool((error[top] <= scale[top] / 71).all()), error[top].div(scale[top]).max().item()
    below = error[~top] / values[~top].abs()
    assert below.max().item() <= 0.55, below.max().item()


def test_every_code_is_the_nearest_one_to_its_value() -> None:
    """The quantiser is `argmin |M - |x|/N_j|`; a bucketize off by one doubles the error."""
    values = torch.logspace(0.0, -37.0, 4096) * torch.tensor([1.0, -1.0]).repeat_interleave(2048)
    state = LowBitState(values, bits=8)
    state.apply(lambda window, source: window.copy_(source), values)
    scale = block_maxima(values)
    ratio = values.abs() / scale
    nearest = (MAGNITUDE.unsqueeze(0) - ratio.unsqueeze(1)).abs().argmin(dim=1)
    assert torch.equal(state.data["codes"].abs().to(torch.long), nearest), "not the nearest"
    assert torch.equal(state.dense().abs(), MAGNITUDE[nearest] * scale)


def test_a_constant_magnitude_block_round_trips_bit_for_bit() -> None:
    """§3: `M[126] = 1`, so `sgn·1·N = ±N` exactly — and the scale is the whole block."""
    for magnitude in (1.0, 3.7, 1 / 3, 1e-8, 1234.5678, 6.02e23):
        for sign in (1.0, -1.0):
            values = torch.full((600,), sign * magnitude)
            assert torch.equal(round_trip(values), values), (sign, magnitude)


def test_a_zero_block_quantises_to_its_own_code_and_dequantises_to_zero() -> None:
    """The smallest nonzero code is `1e-37·N`, not `0`, so a zero must have code 0 (§3)."""
    values = torch.cat([torch.zeros(300), torch.ones(300)])
    state = LowBitState(values, bits=8)
    state.apply(lambda window, source: window.copy_(source), values)
    codes = state.data["codes"][:300]
    assert bool((codes == 0).all()), "zeros found a nonzero code"
    assert torch.equal(state.dense(), values)


def test_the_block_is_the_unit_the_scale_is_computed_over() -> None:
    """One block of `1` and one of `3.7e-3` must not share a scale (§1)."""
    values = torch.cat([torch.full((BLOCK,), 1.0), torch.full((BLOCK,), 3.7e-3)])
    assert torch.equal(round_trip(values), values), "a block's own maximum is its scale"
    shared = round_trip(values, block=2 * BLOCK)[BLOCK:]
    assert (shared - 3.7e-3).abs().max().item() > 0.0, "a shared scale reproduced the block"


# ------------------------------------------------------------- the width is a real choice


def test_a_32_bit_state_is_the_tensor_torch_kept_and_a_chunked_pass_moves_nothing() -> None:
    """`bits=32` is the optimiser that was here, chunks included: `torch.equal`, not close."""
    torch.manual_seed(1)
    values = torch.randn(1_500_000)
    state = LowBitState(values, bits=32)
    assert torch.equal(state.dense(), torch.zeros_like(values))
    state.apply(lambda window, source: window.copy_(source), values)
    assert torch.equal(state.dense(), values), "the 32-bit window is not a view of the state"
    want = values.clone().mul_(0.95).add_(values, alpha=0.05)
    state.apply(lambda window, source: window.mul_(0.95).add_(source, alpha=0.05), values)
    assert torch.equal(state.dense(), want), "the chunked walk is not the tensor walk"


def test_muon_at_32_bits_runs_the_in_place_call_it_always_ran() -> None:
    """The same `mul_`/`add_` on the same values: the container adds no rounding (gate 9)."""
    torch.manual_seed(2)
    param = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([param], lr=0.0, momentum=0.95)
    want = torch.zeros(16, 8)
    for _ in range(3):
        param.grad = torch.randn(16, 8)
        want.mul_(0.95).add_(param.grad, alpha=0.05)
        opt.step()
        assert torch.equal(opt.state[param]["momentum"].dense(), want)


def test_adamw_at_32_bits_is_torch_adamw_bit_for_bit() -> None:
    """The inherited kernel, untouched: four steps with decoupled decay, `torch.equal`."""
    torch.manual_seed(3)
    ours = torch.nn.Parameter(torch.randn(64))
    theirs = torch.nn.Parameter(ours.detach().clone())
    ours_opt = AdamW([ours], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    theirs_opt = torch.optim.AdamW([theirs], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(4):
        ours.grad = torch.randn(64)
        theirs.grad = ours.grad.clone()
        ours_opt.step()
        theirs_opt.step()
    assert torch.equal(ours.detach(), theirs.detach())
    assert torch.equal(ours_opt.state[ours]["exp_avg_sq"], theirs_opt.state[theirs]["exp_avg_sq"])


def test_an_eight_bit_second_moment_keeps_adamw_on_the_same_path() -> None:
    """The measurement the codebook is accepted on: 20 steps, drift `≤ 5e-3` (§2)."""
    for seed in (0, 3, 7):
        torch.manual_seed(seed)
        exact = torch.nn.Parameter(torch.randn(4096))
        quantised = torch.nn.Parameter(exact.detach().clone())
        first = AdamW([exact], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        second = AdamW(
            [quantised], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, state_bits=8
        )
        for step in range(20):
            grad = torch.randn(4096) * (0.01 + 0.5 * step / 20)
            exact.grad = grad.clone()
            quantised.grad = grad.clone()
            first.step()
            second.step()
            assert bool(torch.isfinite(quantised.detach()).all()), f"seed {seed} step {step}"
        drift = (exact.detach() - quantised.detach()).norm() / exact.detach().norm()
        assert drift.item() <= 5e-3, f"seed {seed}: {drift.item():.2e}"


def test_the_momentum_does_not_hold_muons_direction_at_eight_bits() -> None:
    """§6 of `muon.py`, measured: the polar factor of a quantised momentum is not close.

    This is the gate the default `optim_momentum_bits=32` exists for. A pass here would
    be a reason to quantise the buffer; a failure is the reason not to.
    """
    torch.manual_seed(4)
    model = reference_stack(n_layers=1, max_seq_len=8, **LIVE)
    optimizer = MixOptimizer(model)
    ids = torch.randint(0, LIVE["vocab"], (2, 9))
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(ids[:, :-1])
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        ).backward()
        optimizer.step()
    params = dict(model.named_parameters())
    measured = {}
    for name in LIVE_NAMES:
        buffer = optimizer.muon.state[params[name]]["momentum"].dense()
        exact = muon_direction(buffer)
        for block in (256, 64):
            quantised = muon_direction(round_trip(buffer, block))
            measured[name, block] = torch.nn.functional.cosine_similarity(
                exact.reshape(1, -1), quantised.reshape(1, -1)
            ).item()
    for (name, block), cosine in measured.items():
        assert cosine < DIRECTION_BAR, f"{name} block={block}: {cosine:.4f} held the direction"
    #: A smaller block is a smaller scale, so the direction is monotonically better — the
    #: property that says the number is a quantisation effect and not the chain's own.
    for name in LIVE_NAMES:
        assert measured[name, 64] > measured[name, 256], name


# ------------------------------------------------------------------------ the wiring


def test_the_token_table_keeps_32_bit_state_whatever_is_asked() -> None:
    """`lowbit.py` §4: the table is the one parameter with dense moments, by name."""
    mix = MixOptimizer(
        reference_stack(**SMALL),
        optim_state_bits=8,
        optim_first_moment_bits=8,
        optim_momentum_bits=8,
    )
    table = dict(mix.model.named_parameters())["embed.weight"]
    for group in mix.adamw.param_groups:
        if any(param is table for param in group["params"]):
            assert (group["state_bits"], group["first_bits"]) == (32, 32)
        else:
            assert group["state_bits"] == 8, group["params"]
    assert all(group["state_bits"] == 8 for group in mix.muon.param_groups)


def test_a_step_at_eight_bits_moves_both_arms_and_stores_codes_and_scales() -> None:
    torch.manual_seed(5)
    model = reference_stack(**SMALL)
    mix = MixOptimizer(model, optim_state_bits=8)
    ids = torch.randint(0, SMALL["vocab"], (2, 9))
    _backward(model, ids)
    mix.step()
    params = dict(model.named_parameters())
    momentum = mix.muon.state[params["blocks.0.attn.proj.weight"]]["momentum"]
    assert momentum.bits == 8
    assert set(momentum.data) == {"codes", "scales"}
    assert momentum.data["codes"].dtype == torch.int8
    assert mix.adamw.state[params["norm.weight"]]["exp_avg_sq"].bits == 8
    assert mix.adamw.state[params["norm.weight"]]["exp_avg"].bits == 32


def test_a_checkpoint_carries_the_widths_and_refuses_a_mismatch() -> None:
    """A run cannot resume 8-bit state into 32-bit buffers, and the refusal says which."""
    low = MixOptimizer(reference_stack(**SMALL), optim_state_bits=8)
    high = MixOptimizer(reference_stack(**SMALL))
    with pytest.raises(LoudError, match="8"):
        high.load_state_dict(low.state_dict())


def test_a_block_is_part_of_the_mode_only_when_a_moment_is_quantised() -> None:
    """At 32 bits the block changes no stored value, so a resume must not be refused."""
    saved = MixOptimizer(reference_stack(**SMALL), optim_block=512).state_dict()
    MixOptimizer(reference_stack(**SMALL), optim_block=128).load_state_dict(saved)
    quantised = MixOptimizer(reference_stack(**SMALL), optim_state_bits=8, optim_block=512)
    other = MixOptimizer(reference_stack(**SMALL), optim_state_bits=8, optim_block=128).state_dict()
    with pytest.raises(LoudError, match="block"):
        quantised.load_state_dict(other)


def test_a_saved_state_is_tensors_and_ints_and_resumes_bit_for_bit() -> None:
    """The checkpoint is read with `weights_only=True`; a container in it would not load."""
    torch.manual_seed(6)
    model = reference_stack(**SMALL)
    twin = reference_stack(**SMALL)
    twin.load_state_dict(model.state_dict())
    first = MixOptimizer(model, optim_state_bits=8)
    second = MixOptimizer(twin, optim_state_bits=8)
    ids = torch.randint(0, SMALL["vocab"], (2, 9))
    _backward(model, ids)
    first.step()
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    entries = [entry for arm in ("muon", "adamw") for entry in saved[arm]["state"].values()]
    stored = [value for entry in entries for value in entry.values()]
    assert not any(isinstance(value, LowBitState) for value in stored), "a class in the file"
    twin.load_state_dict(model.state_dict())
    second.load_state_dict(saved)
    model.zero_grad(set_to_none=True)
    twin.zero_grad(set_to_none=True)
    _backward(model, ids)
    _backward(twin, ids)
    first.step()
    second.step()
    twins = dict(twin.named_parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param, twins[name]), name


def test_plain_and_revive_are_inverse_without_the_container() -> None:
    entry = {"momentum": LowBitState(torch.zeros(600), bits=8)}
    plain = plain_state_dict({"state": {0: entry}, "param_groups": []})
    assert set(plain["state"][0]["momentum"]) == {"bits", "block", "shape", "codes", "scales"}
    revived = revive_state_dict(plain)["state"][0]["momentum"]
    assert isinstance(revived, LowBitState)
    assert revived.bits == 8
    assert torch.equal(revived.dense(), torch.zeros(600))


def test_a_width_the_module_cannot_keep_is_refused() -> None:
    for bits in (16, 4, 0):
        with pytest.raises(LoudError, match="bits"):
            check(bits, BLOCK)
    with pytest.raises(LoudError, match="block"):
        check(8, 0)
    with pytest.raises(LoudError, match="bits"):
        MixOptimizer(reference_stack(**SMALL), optim_state_bits=16)


def test_a_schedule_free_run_refuses_the_knob_it_does_not_read() -> None:
    """The sf arms keep three buffers and no quantiser; a silent ignore is the defect."""
    with pytest.raises(LoudError, match="sf"):
        MixOptimizer(reference_stack(**SMALL), schedule="sf", optim_state_bits=8)


def test_an_eight_bit_arm_refuses_the_torch_branches_it_does_not_run() -> None:
    param = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(LoudError, match="amsgrad"):
        AdamW([param], state_bits=8, amsgrad=True)
    with pytest.raises(LoudError, match="maximize"):
        AdamW([param], state_bits=8, maximize=True)


def test_a_moment_shaped_like_another_parameter_is_refused() -> None:
    state = LowBitState(torch.zeros(64), bits=8)
    with pytest.raises(LoudError, match="shaped like"):
        state.apply(lambda window, other: None, torch.zeros(32))


def test_the_memory_price_of_a_step_sees_the_codes_and_not_only_tensors() -> None:
    """`graph.state_bytes` walks the state; a walk that skips the container prices it at 0."""
    torch.manual_seed(7)
    ids = torch.randint(0, SMALL["vocab"], (2, 9))
    dense_model = reference_stack(**SMALL)
    dense = MixOptimizer(dense_model)
    _backward(dense_model, ids)
    dense.step()
    quantised_model = reference_stack(**SMALL)
    quantised = MixOptimizer(quantised_model, optim_state_bits=8)
    _backward(quantised_model, ids)
    quantised.step()
    weights = sum(p.numel() * p.element_size() for p in quantised_model.parameters())
    gradients = sum(p.grad.numel() * p.grad.element_size() for p in quantised_model.parameters())
    counted = state_bytes(quantised_model, quantised)
    assert counted > weights + gradients, "the 8-bit moment was not counted at all"
    assert counted < state_bytes(dense_model, dense), "8-bit state is not cheaper than 32"


def _backward(model: torch.nn.Module, ids: torch.Tensor) -> None:
    """The step's loss, without the step: CE against the one-token shift."""
    logits = model(ids[:, :-1])
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    ).backward()
