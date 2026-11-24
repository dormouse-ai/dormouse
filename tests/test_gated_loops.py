"""CPU gates for the opt-in G1 gated loop wrapper.

Eight checks, each a named falsifier for a plausible wrong implementation:
K=1 identity, gate init/clamp, fp64 anchor, forward counters, backward counter,
bitwise state_dict, activation-norm finiteness, and refusal of bad configs.
"""

from __future__ import annotations

import math

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from tools.demo_loops import (
    GATE_BIAS,
    GATE_CEIL,
    GATE_FLOOR,
    LoopGates,
    _demo_model,
    fp64_gate_anchor,
    gate_stats,
    graft,
    loop_gates,
)


def _ids(spec: object, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, spec.vocab, (2, 128), generator=generator)  # type: ignore[attr-defined]


def test_k1_is_bitwise_identity() -> None:
    """A single pass is the untouched model: same logits, same state_dict."""
    model, spec = _demo_model()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = _ids(spec)
    with torch.no_grad():
        base = model(ids)
    out = graft(model, window=(3, 5), K=1, gated=True)
    assert out is model
    assert loop_gates(model) is None
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
    with torch.no_grad():
        after = model(ids)
    assert torch.equal(base, after)


def test_gate_init_and_clamp() -> None:
    """W_g starts at zero, biases are complementary, beta stays in [0.05, 1]."""
    gates = LoopGates(16, K=2)
    assert torch.equal(gates.W, torch.zeros_like(gates.W))
    assert torch.allclose(gates.b, torch.tensor(GATE_BIAS))
    assert GATE_BIAS[0] == pytest.approx(math.log(0.1 / 0.9), abs=0.05)
    assert GATE_BIAS[1] == pytest.approx(math.log(0.9 / 0.1), abs=0.05)
    h = torch.randn(2, 8, 16)
    for p in (0, 1):
        beta = gates(h, p)
        assert beta.shape == (2, 8, 1)
        assert bool((beta >= GATE_FLOOR).all()) and bool((beta <= GATE_CEIL).all())
        assert float(beta.detach().mean()) == pytest.approx((0.1, 0.9)[p], abs=0.12)


def test_fp64_anchor_matches_fp32_to_one_e_minus_six() -> None:
    """The write-back gate equals an independently evaluated fp64 formula."""
    torch.manual_seed(11)
    gates = LoopGates(16, K=2)
    with torch.no_grad():
        gates.W[0].normal_(std=0.05)
        gates.b[1].add_(0.1)
    h = torch.randn(3, 5, 16)
    for p in (0, 1):
        got = gates(h, p)
        want = fp64_gate_anchor(h, gates, p)
        assert torch.allclose(got, want.float(), atol=1e-6, rtol=0.0), (p, (got - want).abs().max())


def test_forward_counters_and_stats() -> None:
    """loop_fwd=2 per call, p1 writes less than p2, token entropy > 0."""
    model, spec = _demo_model()
    model.eval()
    graft(model, window=(3, 5), K=2, gated=True)
    counters = Counters()
    with torch.no_grad():
        model(_ids(spec), counters)
    assert counters.counts["loop_fwd"] == 2
    stats = gate_stats(model)
    assert len(stats["gate_write_mean"]) == 2
    assert stats["gate_write_mean"][0] < stats["gate_write_mean"][1]
    assert counters.counts["gate_entropy"] == stats["gate_entropy"] > 0
    assert counters.silent() == []


def test_backward_counter_and_gate_gradients() -> None:
    """The gate's backward hook fires twice and both gate tensors receive grad."""
    model, spec = _demo_model()
    graft(model, window=(3, 5), K=2, gated=True)
    gates = loop_gates(model)
    assert gates is not None
    counters = Counters()
    out = model(_ids(spec), counters)
    out.sum().backward()
    assert counters.counts["loop_bwd"] == 2
    assert counters.counts["loop_fwd"] == 2
    assert "loop_bwd" in counters.enabled
    assert gates.W.grad is not None and bool(gates.W.grad.abs().sum() > 0)
    assert gates.b.grad is not None and bool(gates.b.grad.abs().sum() > 0)


def test_state_dict_is_bitwise_after_forward_and_backward() -> None:
    """Gates live outside the module tree: no backbone weight moves, ever."""
    model, spec = _demo_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    graft(model, window=(3, 5), K=2, gated=True)
    gates = loop_gates(model)
    assert gates is not None and gates not in list(model.modules())
    out = model(_ids(spec))
    out.sum().backward()
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    # Gradients exist after backward, but no optimizer step has moved any weight;
    # a state_dict comparison (above) is the actual zero-update contract.


def test_activation_norms_are_finite() -> None:
    """The per-pass write-back never produces NaN/Inf (GLM's blowup gate)."""
    model, spec = _demo_model()
    graft(model, window=(3, 5), K=2, gated=True)
    with torch.no_grad():
        logits = model(_ids(spec))
    assert bool(torch.isfinite(logits).all())
    norms = gate_stats(model)["activation_norm_pass"]
    assert len(norms) == 2 and all(math.isfinite(n) for n in norms)


def test_bad_configs_refuse_by_name() -> None:
    """K<1 and a gate with the wrong pass count fail loudly, not silently."""
    model, _ = _demo_model()
    with pytest.raises(LoudError, match="K=0"):
        graft(model, window=(3, 5), K=0, gated=True)
    with pytest.raises(LoudError, match=r"gates\.K"):
        graft(model, window=(3, 5), K=2, gated=True, gates=LoopGates(64, K=3))
