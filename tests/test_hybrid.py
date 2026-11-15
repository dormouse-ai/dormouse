"""The hybrid stack: one pattern, two mechanisms, and the refusals around them.

Four things are checked here that a shape test would miss:

(a) the pattern is `[kda, kda, kda, dsa]` and nothing else — there is no argument that
    renames, reorders or extends it, because the previous version of this file offered
    exactly that menu;
(b) the DSA layer is really in the path: the stack's output is the residual
    composition of its own layers, so a layer that was built and never called shows
    up as a different tensor;
(c) the bill per layer type is counted from the modules, not quoted;
(d) an impossible combination dies before allocation, with the conflict named: a
    layer count the period cannot tile, a mechanism that does not exist, a context
    that contradicts a coeffect, and two mechanisms claiming one state.
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

#: The DSA layer's budget, shrunk to the test window (`k >= seq` is refused).
K, WINDOW = 8, 4


def _stack(**kwargs: int) -> HybridStack:
    """A small stack with the same widths as the tests' other modules."""
    return HybridStack(
        D_MODEL,
        N_HEADS,
        HEAD_DIM,
        n_layers=int(kwargs.get("n_layers", 4)),
        chunk=int(kwargs.get("chunk", 8)),
        k=int(kwargs.get("k", K)),
        window=int(kwargs.get("window", WINDOW)),
    )


def test_the_pattern_is_the_three_to_one_recipe() -> None:
    stack = _stack()
    assert stack.layer_types() == HYBRID_3_TO_1 == ("kda", "kda", "kda", "dsa")
    assert stack.coeffects().provides["attn.stack"] == "hybrid"


def test_the_dsa_layer_is_really_in_the_path() -> None:
    """The stack IS the residual composition of the layers it says it built."""
    torch.manual_seed(21)
    stack = _stack()
    x = torch.randn(1, SEQ, D_MODEL)
    manual = x
    for layer in stack.layers:
        manual = manual + layer(manual)
    assert torch.equal(stack(x), manual)
    assert not torch.equal(manual, x)


def test_the_bill_is_reported_per_layer_type() -> None:
    stack = _stack()
    report = stack.param_report()
    assert report["per_kda"] > 0 and report["per_dsa"] > 0
    assert report["kda"] == 3 * report["per_kda"]
    assert report["dsa"] == report["per_dsa"]
    assert report["total"] == sum(p.numel() for p in stack.parameters())
    assert report["total"] == report["kda"] + report["dsa"]


def test_the_layer_counters_name_every_mechanism() -> None:
    counters = Counters()
    _stack()(torch.zeros(1, SEQ, D_MODEL), counters)
    assert counters.counts["hybrid_kda_layer_fwd"] == 3
    assert counters.counts["hybrid_dsa_layer_fwd"] == 1
    assert counters.counts["hybrid_layer_fwd"] == 4
    assert counters.counts["dsa_fwd"] == SEQ


def test_a_later_token_cannot_change_an_earlier_output_through_the_stack() -> None:
    torch.manual_seed(22)
    stack = _stack()
    x = torch.randn(1, SEQ, D_MODEL)
    y = stack(x)
    moved = x.clone()
    moved[:, 24] += 1.0
    assert torch.equal(y[:, :24], stack(moved)[:, :24])
    assert not torch.equal(y[:, 24], stack(moved)[:, 24])


def test_a_layer_count_the_period_cannot_tile_is_refused_with_the_remainder() -> None:
    with pytest.raises(LoudError) as e:
        _stack(n_layers=10)
    assert "2 layers are left over" in str(e.value)


def test_an_unknown_mechanism_is_refused_by_name() -> None:
    with pytest.raises(LoudError) as e:
        check_compatible(["kda", "hca"], {"attn.pos": "none", "attn.rule": "delta"})
    assert "kda" in str(e.value) and "dsa" in str(e.value)


def test_a_context_that_contradicts_a_coeffect_is_refused_before_allocation() -> None:
    """Neither mechanism has a rotary table: asking for RoPE must fail at construction."""
    with pytest.raises(LoudError) as e:
        HybridStack(
            D_MODEL,
            N_HEADS,
            HEAD_DIM,
            n_layers=4,
            chunk=8,
            k=K,
            window=WINDOW,
            context={"attn.pos": "rope"},
        )
    assert "attn.pos" in str(e.value) and "rope" in str(e.value)


def test_two_mechanisms_on_one_layer_conflict_by_name() -> None:
    """The first row of the conflict matrix, refused with both names in the message."""
    with pytest.raises(LoudError) as e:
        check_compatible(
            ["kda", "dsa"],
            {"attn.pos": "none", "attn.rule": "delta", "attn.selector": "true"},
        )
    assert "kda and dsa" in str(e.value)
    assert "attn.state" in str(e.value)


def test_one_mechanism_on_its_own_is_compatible() -> None:
    """The check is not a blanket refusal: each layer alone is fine."""
    check_compatible(["kda"], {"attn.pos": "none", "attn.rule": "delta"})
    check_compatible(["dsa"], {"attn.pos": "none", "attn.selector": "true"})


def test_a_stack_with_no_layers_is_refused() -> None:
    with pytest.raises(LoudError, match="assembles no attention"):
        HybridStack(D_MODEL, N_HEADS, HEAD_DIM, n_layers=0)


def test_the_repr_prints_the_period() -> None:
    assert "period=['kda', 'kda', 'kda', 'dsa']" in repr(_stack())
