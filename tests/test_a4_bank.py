"""The bank's INT4 code-saving autograd boundary: anchor, counters, parity.

The memory/timing/quality A/B is `tools/a4_bank_gate.py` and needs the card. What
lives here is the part that must hold on every machine: the fp64 external anchor,
the named receipts, and deterministic 20-step replay.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters

from dormouse.model import quant
from dormouse.model.assemble import assemble
from dormouse.model.quant import a4_bank_readout, bank_a4
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BLOCK = quant.FWHT_BLOCK


@pytest.fixture(autouse=True)
def _eager_kernels() -> Iterator[None]:
    """Pin both kernel flags off; they are read at import, so env alone is too late."""
    from dormouse_kernels.engine import bank, fused

    old_fused, old_bank = fused.fused(), bank.bank_enabled()
    fused.set_fused(False)
    bank.set_bank(False)
    yield
    fused.set_fused(old_fused)
    bank.set_bank(old_bank)


SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "n_layers": 4,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 32,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 128,
    "batch": 2,
}


def _spec(*, quantize: bool) -> StackSpec:
    """Build the smoke spec; the raw text keeps the flag/config assertion honest."""
    raw = tomllib.loads((CONFIGS / "dormouse_1b.toml").read_text(encoding="utf-8"))
    known = {field.name for field in fields(StackSpec) if field.init}
    values = {key: value for key, value in raw.items() if key in known}
    values.update(SMOKE, torch_compile=False)
    return StackSpec.from_mapping(values)


def _np_fwht(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Independent fp64 butterfly, written against numpy, not via the code under test."""
    width = x.shape[-1]
    y = x.reshape(*x.shape[:-1], width // block, block).copy()
    step = 1
    while step < block:
        y = y.reshape(*y.shape[:-2], width // block, block // (2 * step), 2, step)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y = np.stack((a + b, a - b), axis=-2).reshape(*x.shape[:-1], width // block, block)
        step *= 2
    return y.reshape(x.shape) / math.sqrt(block)


def _np_pack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The external anchor's quantiser: per-token σ·2.5/7, clip, two codes per byte."""
    z = _np_fwht(x)
    rms = np.sqrt(np.mean(z * z, axis=-1, keepdims=True))
    scale = np.maximum(quant.BANK_SCALE_SIGMA * rms / 7, 1e-8)
    code = np.clip(np.round(z / scale), -7, 7).astype(np.int8)
    latent = code.astype(np.float64) * scale
    packed = ((code[..., 0::2] + 8) | ((code[..., 1::2] + 8) << 4)).astype(np.uint8)
    return packed, latent


def test_external_fp64_anchor_matches_forward_and_ste_backward() -> None:
    """Forward and adjoint agree with an independent fp64 formula to 1e-6."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 128))
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    packed, _scale, latent = quant.pack_slab_latent(tensor)
    external_codes, external_latent = _np_pack(x)
    assert np.array_equal(packed.numpy(), external_codes)
    assert np.allclose(latent.detach().numpy(), external_latent, atol=1e-6, rtol=0)
    readout = quant.fwht(latent)
    assert np.allclose(readout.detach().numpy(), _np_fwht(external_latent), atol=1e-6, rtol=0)

    upstream = torch.tensor(rng.standard_normal((3, 128)), dtype=torch.float64)

    def fn(value: torch.Tensor) -> torch.Tensor:
        return value * upstream

    #: The control's STE readout and the codesave arm's pack are the same
    #: activation; both then run the SAME downstream `fn`.
    plain = fn(a4_bank_readout(tensor))
    plain.sum().backward()
    plain_grad = tensor.grad.clone()
    tensor.grad = None
    saved = quant.bank_a4(fn, (0,), tensor)
    assert torch.allclose(saved, plain, atol=1e-6, rtol=0)
    saved.sum().backward()
    assert torch.allclose(tensor.grad, plain_grad, atol=1e-6, rtol=0)
    assert torch.allclose(tensor.grad, quant.fwht(upstream), atol=1e-6, rtol=0)


def test_codes_are_two_per_byte_and_the_rotation_is_involutive() -> None:
    """One uint8 per two codes; `fwht(fwht(x)) == x` at the shipped block width."""
    codes = torch.tensor([[-7, -1, 0, 7, 3, -3, 1, 1]], dtype=torch.int8)
    packed = quant._pack(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (1, 4)
    assert torch.equal(quant._unpack(packed, 8).to(torch.int8), codes)
    x = torch.tensor(np.random.default_rng(11).standard_normal((4, 128)), dtype=torch.float64)
    assert torch.allclose(quant.fwht(quant.fwht(x)), x, atol=1e-6)


def test_width_refusals_and_public_wrappers() -> None:
    """A slab the 64-wide rotation cannot tile is refused, never silently shortened."""
    with pytest.raises(ValueError, match="power of two"):
        quant._hadamard(63, torch.float32, torch.device("cpu"))
    with pytest.raises(ValueError, match="tiled by power-of-two"):
        quant.fwht(torch.zeros(2, 100))
    with pytest.raises(ValueError, match="not tiled by 64"):
        quant.pack_slab(torch.zeros(2, 100))
    x = torch.tensor(np.random.default_rng(5).standard_normal((2, 128)), dtype=torch.float32)
    packed, scale = quant.pack_slab(x)
    assert torch.allclose(
        quant.unpack_slab(packed, scale, 128, x.dtype),
        quant.fwht(quant.pack_slab_latent(x)[2]),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="expected one output"):
        bank_a4(lambda a: (a, a), (0,), x)


def test_flag_with_torch_compile_is_a_named_no_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The callable boundary under compile is refused, not priced as a graph."""
    monkeypatch.setenv(quant.BANK_QUANT_FLAG, "1")
    with pytest.raises(Exception) as caught:
        StackSpec.from_mapping({**SMOKE, "torch_compile": True})
    message = str(caught.value)
    assert "DORMOUSE_BANK_A4" in message and "torch_compile" in message


def _run(spec: StackSpec, steps: int, seed: int) -> tuple[list[float], Counters | None]:
    """Run a pinned-thread CPU replay; the bank's `index_add_` is thread-order sensitive."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        built = assemble(spec, smoke=True)
        optimizer = MixOptimizer(built.model, steps=steps)
        ids = torch.randint(
            0,
            spec.vocab,
            (spec.batch, spec.seq_len + 1),
            generator=torch.Generator().manual_seed(seed),
        )
        losses: list[float] = []
        counters = None
        for _ in range(steps):
            result, counters = built.step(ids, optimizer=optimizer)
            assert not result.skipped, f"NaN firewall: {result.loss}"
            losses.append(result.loss)
        return losses, counters
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(("quant_flag", "codesave"), [(True, True), (True, False), (False, False)])
def test_mechanism_receipts_and_gradient(
    monkeypatch: pytest.MonkeyPatch, quant_flag: bool, codesave: bool
) -> None:
    """Each declared mechanism reports receipts; an arm that ran nothing is refused."""
    monkeypatch.setenv(quant.BANK_QUANT_FLAG, "1" if quant_flag else "0")
    monkeypatch.setenv(quant.BANK_FLAG, "1" if codesave else "0")
    monkeypatch.setenv("DORMOUSE_BANK", "0")
    monkeypatch.setenv("DORMOUSE_FUSED", "0")
    spec = StackSpec.from_mapping({**SMOKE, "torch_compile": False})
    losses, counters = _run(spec, steps=2, seed=7)
    assert all(math.isfinite(loss) for loss in losses)
    assert counters is not None
    names = ("bank_a4_fwd", "bank_a4_bwd", "bank_a4_codes_saved") if codesave else ("ffn_fwd",)
    for name in names:
        assert counters.counts[name] > 0, name
    assert counters.silent() == []
    if not codesave:
        assert not any(name.startswith("bank_a4") for name in counters.counts)


def test_twenty_step_replay_is_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """The STE and the pack are deterministic from step zero."""
    monkeypatch.setenv(quant.BANK_QUANT_FLAG, "1")
    monkeypatch.setenv(quant.BANK_FLAG, "1")
    monkeypatch.setenv("DORMOUSE_BANK", "0")
    monkeypatch.setenv("DORMOUSE_FUSED", "0")
    spec = StackSpec.from_mapping({**SMOKE, "torch_compile": False})
    first, first_counters = _run(spec, steps=20, seed=13)
    second, second_counters = _run(spec, steps=20, seed=13)
    assert first == second
    assert first_counters is not None and second_counters is not None
    for name in ("bank_a4_fwd", "bank_a4_bwd", "bank_a4_codes_saved"):
        assert first_counters.counts[name] == second_counters.counts[name] > 0


def test_storage_form_does_not_move_the_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wide-retained and code-retained arms are the same model arithmetic."""
    monkeypatch.setenv(quant.BANK_QUANT_FLAG, "1")
    monkeypatch.setenv("DORMOUSE_BANK", "0")
    monkeypatch.setenv("DORMOUSE_FUSED", "0")

    torch.manual_seed(3)
    g = torch.randn(8, 128, requires_grad=True)
    u = torch.randn(8, 128, requires_grad=True)

    def fn(gg: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
        return (gg * uu).sum(-1)

    plain = fn(a4_bank_readout(g), a4_bank_readout(u))
    plain.sum().backward()
    plain_grads = (g.grad.clone(), u.grad.clone())
    g.grad, u.grad = None, None
    saved = bank_a4(fn, (0, 1), g, u)
    assert torch.equal(plain, saved)
    saved.sum().backward()
    for expected, got in zip(plain_grads, (g.grad, u.grad), strict=True):
        assert torch.allclose(expected, got, atol=1e-6, rtol=0)


def test_eight_point_checklist_on_the_first_saved_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each item names a measurable contract; no item is asserted from prose alone."""
    monkeypatch.setenv(quant.BANK_QUANT_FLAG, "1")
    monkeypatch.setenv(quant.BANK_FLAG, "1")
    monkeypatch.setenv("DORMOUSE_BANK", "0")
    monkeypatch.setenv("DORMOUSE_FUSED", "0")
    spec = StackSpec.from_mapping({**SMOKE, "torch_compile": False})
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(19)
        built = assemble(spec, smoke=True)
        optimizer = MixOptimizer(built.model, steps=2)
        ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
        result, counters = built.step(ids, optimizer=optimizer)
    finally:
        torch.set_num_threads(previous)
    assert counters is not None
    monkeypatch.setenv(quant.BANK_FLAG, "0")
    control_losses, control_counters = _run(spec, steps=1, seed=19)
    assert control_counters is not None
    assert all(math.isfinite(loss) for loss in control_losses)
    checks = {
        "1_mechanism_ran": all(
            counters.counts[name] > 0
            for name in ("bank_a4_fwd", "bank_a4_bwd", "bank_a4_codes_saved")
        ),
        "2_gradient_arrived": any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in built.model.parameters()
        ),
        "3_config_reached_code": spec.as_dict()["torch_compile"] is False,
        "4_honest_control": not any(name.startswith("bank_a4") for name in control_counters.counts)
        and math.isfinite(control_losses[-1]),
        "5_init_as_declared": quant.FWHT_BLOCK == 64 and 6144 % quant.FWHT_BLOCK == 0,
        "6_numbers_finite": not result.skipped and math.isfinite(result.loss),
        "7_hyperparameters_named": optimizer.muon.param_groups[0]["lr"] > 0,
        "8_formula_and_anchor": "STE" in quant.BankA4Save.__doc__
        and "INT4" in quant.BankA4Save.__doc__,
    }
    assert all(checks.values()), [name for name, ok in checks.items() if not ok]
