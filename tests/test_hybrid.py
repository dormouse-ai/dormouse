"""The hybrid stack: the pattern tiles or it is refused, and the CSA layer is used.

Three things are checked here that a shape test would miss:

(a) `[KDA, KDA, KDA, CSA]` and `[KDA] x 4` produce different outputs, so the one
    sparse layer is really in the path — a pattern that silently built four KDA
    layers would pass every "does it run" test.
(b) The bill per layer type is counted from the modules, not quoted: a hybrid
    whose CSA layer costs more than the three KDA layers it replaces is a
    different arm from the one that was measured.
(c) An impossible combination dies before allocation, with the conflict named:
    a pattern that does not tile, a mechanism that does not exist, a context that
    contradicts a coeffect, and two mechanisms claiming one layer.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.attn_variants.hybrid import HYBRID_3_TO_1, HybridStack, check_compatible

D_MODEL = 32
N_HEADS = 2
HEAD_DIM = 8
SEQ = 32


def _stack(pattern: tuple[str, ...], **kwargs: int) -> HybridStack:
    """A small stack with the same widths as the tests' other modules."""
    return HybridStack(
        D_MODEL,
        N_HEADS,
        HEAD_DIM,
        n_layers=4,
        pattern=pattern,
        chunk=kwargs.get("chunk", 8),
        csa_chunk=kwargs.get("csa_chunk", 8),
    )


def test_the_default_pattern_is_the_three_to_one_recipe() -> None:
    stack = _stack(HYBRID_3_TO_1)
    assert stack.layer_types() == ("KDA", "KDA", "KDA", "CSA")
    assert stack.coeffects().provides["attn.stack"] == "hybrid"


def test_the_csa_layer_is_really_in_the_path() -> None:
    """Parity against the two control arms: same shapes, different functions."""
    torch.manual_seed(21)
    x = torch.randn(1, SEQ, D_MODEL)
    hybrid = _stack(HYBRID_3_TO_1)(x)
    all_kda = _stack(("KDA",) * 4)(x)
    all_csa = _stack(("CSA",) * 4)(x)
    assert hybrid.shape == all_kda.shape == all_csa.shape
    assert not torch.allclose(hybrid, all_kda, atol=1e-6)
    assert not torch.allclose(hybrid, all_csa, atol=1e-6)


def test_the_bill_is_reported_per_layer_type() -> None:
    stack = _stack(HYBRID_3_TO_1)
    report = stack.param_report()
    assert report["per_kda"] > 0 and report["per_csa"] > 0
    assert report["kda"] == 3 * report["per_kda"]
    assert report["csa"] == report["per_csa"]
    assert report["total"] == sum(p.numel() for p in stack.parameters())
    assert report["total"] == report["kda"] + report["csa"]


def test_the_layer_counters_name_every_mechanism() -> None:
    counters = Counters()
    _stack(HYBRID_3_TO_1)(torch.zeros(1, SEQ, D_MODEL), counters)
    assert counters.counts["hybrid_kda_layer_fwd"] == 3
    assert counters.counts["hybrid_csa_layer_fwd"] == 1
    assert counters.counts["hybrid_layer_fwd"] == 4


def test_a_later_token_cannot_change_an_earlier_output_through_the_stack() -> None:
    torch.manual_seed(22)
    stack = _stack(HYBRID_3_TO_1)
    x = torch.randn(1, SEQ, D_MODEL)
    y = stack(x)
    moved = x.clone()
    moved[:, 24] += 1.0
    assert torch.equal(y[:, :24], stack(moved)[:, :24])
    assert not torch.equal(y[:, 24], stack(moved)[:, 24])


def test_a_pattern_that_does_not_tile_is_refused_with_the_remainder() -> None:
    with pytest.raises(LoudError) as e:
        HybridStack(D_MODEL, N_HEADS, HEAD_DIM, n_layers=10, pattern=HYBRID_3_TO_1)
    assert "2 layers are left over" in str(e.value)


def test_an_unknown_mechanism_is_refused_by_name() -> None:
    with pytest.raises(LoudError) as e:
        _stack(("KDA", "HCA", "KDA", "CSA"))
    assert "HCA" in str(e.value)


def test_an_empty_pattern_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        _stack(())
    assert "names no mechanism" in str(e.value)


def test_a_context_that_contradicts_a_coeffect_is_refused_before_allocation() -> None:
    """KDA has no rotary table: asking for RoPE must fail at construction."""
    with pytest.raises(LoudError) as e:
        HybridStack(
            D_MODEL,
            N_HEADS,
            HEAD_DIM,
            n_layers=4,
            pattern=HYBRID_3_TO_1,
            context={"attn.pos": "rope"},
        )
    assert "attn.pos" in str(e.value) and "rope" in str(e.value)


def test_two_mechanisms_on_one_layer_conflict_by_name() -> None:
    """The first row of the conflict matrix, refused with both names in the message."""
    with pytest.raises(LoudError) as e:
        check_compatible(
            ["KDA", "CSA"],
            {"attn.pos": "none", "attn.rule": "delta", "attn.selector": "true"},
        )
    assert "KDA and CSA" in str(e.value)
    assert "attn.state" in str(e.value)


def test_one_mechanism_on_its_own_is_compatible() -> None:
    """The check is not a blanket refusal: each layer alone is fine."""
    check_compatible(["KDA"], {"attn.pos": "none", "attn.rule": "delta"})
    check_compatible(["CSA"], {"attn.pos": "none", "attn.selector": "true"})


def test_a_stack_with_no_layers_is_refused() -> None:
    with pytest.raises(LoudError, match="assembles no attention"):
        HybridStack(D_MODEL, N_HEADS, HEAD_DIM, n_layers=0)


def test_the_repr_prints_the_pattern() -> None:
    assert "pattern=['KDA', 'KDA', 'KDA', 'CSA']" in repr(_stack(HYBRID_3_TO_1))


def _mixed(mix: str) -> HybridStack:
    """A small stack built from `attn_mix`, not from an explicit pattern."""
    return HybridStack(
        D_MODEL,
        N_HEADS,
        HEAD_DIM,
        n_layers=4,
        attn_mix=mix,
        chunk=8,
        csa_chunk=8,
        dsa_k=8,
        dsa_window=4,
    )


def test_the_default_mix_is_three_kda_per_dsa() -> None:
    """DSA is the exact-attention partner: `kda:dsa` is what a config gets unasked."""
    stack = _mixed("kda:dsa")
    assert stack.layer_types() == ("KDA", "KDA", "KDA", "DSA")
    assert (
        HybridStack(
            D_MODEL, N_HEADS, HEAD_DIM, n_layers=4, chunk=8, csa_chunk=8, dsa_k=8, dsa_window=4
        ).layer_types()
        == stack.layer_types()
    )


def test_the_csa_mix_is_still_built() -> None:
    """Every number measured before DSA existed was measured on this arm."""
    stack = _mixed("kda:csa")
    assert stack.layer_types() == HYBRID_3_TO_1
    assert stack.param_report()["csa"] > 0


def test_the_default_stack_really_contains_a_dsa_layer() -> None:
    """The DSA layer changes the output: a pattern that silently built four KDA
    layers would pass every shape test."""
    torch.manual_seed(23)
    x = torch.randn(1, SEQ, D_MODEL)
    dsa = _mixed("kda:dsa")(x)
    all_kda = _stack(("KDA",) * 4)(x)
    assert dsa.shape == all_kda.shape
    assert not torch.allclose(dsa, all_kda, atol=1e-6)


def test_the_dsa_bill_is_counted_per_layer_type() -> None:
    report = _mixed("kda:dsa").param_report()
    assert report["per_dsa"] > 0
    assert report["dsa"] == report["per_dsa"]
    assert report["total"] == report["kda"] + report["dsa"]


def test_an_unknown_mix_is_refused_with_the_alternatives() -> None:
    with pytest.raises(LoudError) as caught:
        _mixed("kda:mla")
    message = str(caught.value)
    assert "attn_mix='kda:mla'" in message and "kda:dsa" in message and "kda:csa" in message


def test_kda_and_dsa_conflict_on_one_layer_by_name() -> None:
    with pytest.raises(LoudError) as caught:
        check_compatible(
            ["KDA", "DSA"],
            {"attn.pos": "none", "attn.rule": "delta", "attn.selector": "true"},
        )
    assert "KDA and DSA" in str(caught.value)
    assert "attn.state" in str(caught.value)


def test_the_dsa_layer_reports_its_own_counter_through_the_stack() -> None:
    counters = Counters()
    _mixed("kda:dsa")(torch.zeros(1, SEQ, D_MODEL), counters)
    assert counters.counts["hybrid_dsa_layer_fwd"] == 1
    assert counters.counts["dsa_fwd"] == SEQ
