"""The numeric policy: what bf16 changes, what it must not, and the counters that prove both.

Two halves. The CPU half pins the *shape* of the policy — which dtype the parameters are,
which moments are fp32, that the cross-entropy upcasts, that an fp32 spec enters no region
at all. The CUDA half is the gate: thirty steps of the same seed and the same data, fp32
against bf16, where the losses have to track inside bf16's own mantissa.

KDA's chunk path solves a triangular system, and bf16 `solve_triangular` is CUDA-only
(`triangular_solve_cpu` refuses BFloat16), so the CPU tests use the DSA arm: it is the same
numeric policy over a mechanism whose kernels exist on both devices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble
from dormouse.model.spec import DTYPES, RUN_DEFAULT, StackSpec
from dormouse.train import muon as muon_module
from dormouse.train import step as step_module
from dormouse.train.optim import MixOptimizer
from dormouse.train.precision import COUNTER, Precision

#: bf16's mantissa is 8 bits, so any value it holds is within 2⁻⁸ of the fp32 one. One
#: step from the same weights is inside that (measured 5.6e-4 relative on the smoke
#: stack); a 30-step trajectory is not, and the number that bounds it is measured rather
#: than assumed: at the recipe's rate this stack drops its loss 10x in 30 steps and
#: amplifies the rounding to 21%, and the same comparison at 1/100 the rate separates by
#: 1.6·2⁻⁸. Both numbers are printed by the parity test.
BF16_EPS = 2.0**-8

#: The 30-step band at the recipe's rate: the measured 21% (0.214) with room for another
#: GPU's kernel selection. A policy defect — a bf16 cross-entropy, a bf16 moment — breaks
#: the one-step gate and the finiteness/curve checks long before it reaches this band.
CURVE_BAND = 0.25

#: The one config at widths a CPU test can afford. Every slot is kept.
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
    "tokens": 1_000_000,
}

#: The repo's configs, so a test does not depend on the directory pytest was run from.
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _spec(dtype: str, *, attn: str = "dsa") -> StackSpec:
    """Return the smoke stack at one policy, on the arm whose kernels that device has."""
    return StackSpec(attn=attn, dtype=dtype, **SMOKE)


# ------------------------------------------------------------------- the literal


def test_a_dtype_the_policy_has_no_path_for_is_refused_with_both_that_it_has() -> None:
    with pytest.raises(LoudError) as caught:
        Precision("fp16")
    message = str(caught.value)
    assert "fp16" in message
    for option in DTYPES:
        assert option in message


def test_a_config_that_omits_the_dtype_is_a_run_and_a_spec_in_code_is_a_test() -> None:
    """The one field whose default depends on where the spec came from (model/spec.py)."""
    assert StackSpec().dtype == "fp32"
    assert StackSpec.from_mapping({}).dtype == RUN_DEFAULT
    assert StackSpec.from_mapping({"dtype": "fp32"}).dtype == "fp32"


def test_a_smoke_run_takes_the_reference_policy_back(tmp_path: Path) -> None:
    """`--smoke` is the CPU seam test, not a measurement: it runs fp32 whatever the file says."""
    from dormouse.train import entry

    assert StackSpec.from_toml(CONFIGS / "dormouse_1b.toml").dtype == "bf16"
    receipt = entry.train(CONFIGS / "smoke.toml", runs=tmp_path, smoke=True, until=1, echo=_noop)
    stored = json.loads((Path(receipt["directory"]) / "config.json").read_text(encoding="utf-8"))
    assert stored["stack"]["dtype"] == "fp32"


def test_the_plan_line_prints_the_policy() -> None:
    from dormouse.model.assemble import plan_lines

    text = "\n".join(plan_lines(StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")))
    assert "dtype=bf16" in text


# ------------------------------------------------------------------ the tree, built


def test_the_parameters_are_the_policy_and_a_bf16_step_counts_its_region() -> None:
    assembled = assemble(_spec("bf16"), smoke=True)
    assert {param.dtype for param in assembled.model.parameters()} == {torch.bfloat16}
    optimizer = MixOptimizer(assembled.model, steps=2)
    ids = torch.randint(0, assembled.spec.vocab, (assembled.spec.batch, 129))
    result, counters = assembled.step(ids, optimizer=optimizer)
    assert not result.skipped
    assert counters.counts[COUNTER] == 1, "one forward, one region"
    assert {param.dtype for param in assembled.model.parameters()} == {torch.bfloat16}
    grads = {param.grad.dtype for param in assembled.model.parameters() if param.grad is not None}
    assert grads == {torch.bfloat16}, "a gradient follows the parameter it belongs to"


def test_fp32_takes_no_region_and_no_cast_so_the_reference_run_is_the_old_one() -> None:
    assembled = assemble(_spec("fp32"), smoke=True)
    assert {param.dtype for param in assembled.model.parameters()} == {torch.float32}
    assert Precision("fp32").cast(assembled.model) is assembled.model
    optimizer = MixOptimizer(assembled.model, steps=2)
    ids = torch.randint(0, assembled.spec.vocab, (assembled.spec.batch, 129))
    result, counters = assembled.step(ids, optimizer=optimizer)
    assert not result.skipped
    assert COUNTER not in counters.counts
    assert all(state["momentum"].dtype == torch.float32 for state in optimizer.muon.state.values())


def test_the_moments_and_the_update_arithmetic_stay_fp32_under_bf16(monkeypatch) -> None:
    """The leak this policy refuses: a bf16 moment is a bf16 update of a bf16 weight."""
    assembled = assemble(_spec("bf16"), smoke=True)
    optimizer = MixOptimizer(assembled.model, steps=2)
    seen: list[torch.dtype] = []
    real = muon_module.muon_direction

    def spy(momentum: torch.Tensor, **kwargs: object) -> torch.Tensor:
        seen.append(momentum.dtype)
        return real(momentum, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(muon_module, "muon_direction", spy)
    ids = torch.randint(0, assembled.spec.vocab, (assembled.spec.batch, 129))
    result, _ = assembled.step(ids, optimizer=optimizer)
    assert not result.skipped
    assert set(seen) == {torch.float32}, "the preconditioner saw a half-precision momentum"
    moments = {state["momentum"].dtype for state in optimizer.muon.state.values()}
    assert moments == {torch.float32}
    moments = {
        value.dtype
        for state in optimizer.adamw.state.values()
        for key, value in state.items()
        if key.startswith("exp_avg")
    }
    assert moments == {torch.float32}


def test_the_cross_entropy_upcasts_before_it_reduces(monkeypatch) -> None:
    """A bf16 log-softmax over the vocabulary is the silent quality leak; assert the dtype."""
    seen: list[torch.dtype] = []
    real = step_module.functional.cross_entropy

    def spy(tensor: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        seen.append(tensor.dtype)
        return real(tensor, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(step_module.functional, "cross_entropy", spy)
    logits = torch.randn(6, 8, dtype=torch.bfloat16)
    loss = step_module.chunked_cross_entropy(logits, torch.randint(0, 8, (6,)), chunk=4)
    assert loss.dtype == torch.float32
    assert seen and set(seen) == {torch.float32}


def test_the_schedule_free_arms_refuse_half_precision_by_name() -> None:
    assembled = assemble(_spec("bf16"), smoke=True)
    with pytest.raises(LoudError) as caught:
        MixOptimizer(assembled.model, schedule="sf")
    message = str(caught.value)
    assert "schedule=sf" in message and ("fp32" in message or "cosine" in message)


# ------------------------------------------------------------------------ the gate

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bf16 KDA's triangular solve is CUDA-only"
)


@CUDA
def test_thirty_steps_of_bf16_track_fp32_inside_the_mantissa() -> None:
    """The parity gate: same seed, same data, 30 steps, both dtypes.

    Two numbers, because the claim is two claims. ONE STEP from the same weights is the
    policy's own error, and it is inside bf16's 2⁻⁸ — that is where "bf16 epsilon" is a
    measurement. THIRTY STEPS is a trajectory, and this stack at the recipe's rate drops
    its loss 10x in them, so an 8-bit rounding compounds into a different optimisation
    path: measured 21% by step 29, and 1.6·2⁻⁸ with the same comparison run at 1/100 the
    rate (the amplifer is the step size, not the dtype). The gate is both of those; a
    policy defect shows up in the first one.
    """
    recipe = _gaps(_curves(0.02))
    slow = _gaps(_curves(0.0002))
    print(
        f"\nparity: one step rel diff {recipe[0]:.2e} (bf16 eps {BF16_EPS:.2e}); "
        f"30 steps at lr=0.02 max {max(recipe):.2e}; 30 steps at lr=2e-4 max {max(slow):.2e}"
    )
    assert recipe[0] <= BF16_EPS, "one step from the same weights is not one ulp off"
    assert max(recipe) <= CURVE_BAND
    assert max(slow) <= 2 * BF16_EPS, "a rate the loss can follow must stay inside two ulps"


def _curves(lr: float, steps: int = 30) -> dict[str, list[float]]:
    """Run both policies from one seed and one batch; refuse a skipped step or a NaN."""
    torch.manual_seed(7)
    ids = torch.randint(0, 8, (SMOKE["batch"], SMOKE["seq_len"] + 1)).cuda()
    out: dict[str, list[float]] = {}
    for dtype in ("fp32", "bf16"):
        torch.manual_seed(7)
        assembled = assemble(_spec(dtype, attn="hybrid"), smoke=True, device="cuda")
        optimizer = MixOptimizer(assembled.model, steps=2, lr_muon=lr, lr_adamw=lr * 0.15)
        curve: list[float] = []
        for _ in range(steps):
            result, _ = assembled.step(ids, optimizer=optimizer)
            assert not result.skipped, f"{dtype}: the NaN firewall fired at loss={result.loss}"
            assert torch.isfinite(torch.tensor(result.loss)), f"{dtype}: loss={result.loss}"
            curve.append(result.loss)
        finite = all(bool(torch.isfinite(param).all()) for param in assembled.model.parameters())
        assert finite, f"{dtype}: a parameter stopped being finite"
        out[dtype] = curve
    return out


def _gaps(curves: dict[str, list[float]]) -> list[float]:
    """The per-step relative difference of bf16 against fp32."""
    return [
        abs(bf16 - fp32) / max(abs(fp32), 1e-12)
        for fp32, bf16 in zip(curves["fp32"], curves["bf16"], strict=True)
    ]


def _noop(_line: str) -> None:
    """Swallow a run's transcript; the journal is what these tests read."""
