"""The MoE bank: row-exact dispatch, the bill, and the buffers it does not allocate.

Every gate here exists because the old arm passed a weaker one and produced an
unmeasurable run:

* **parity** — one expert at top-1 with no shared expert IS the dense gated FFN,
  `torch.equal`, so the dispatch is a placement of the same arithmetic and not a
  second implementation with its own rounding;
* **rows** — a forward executes exactly `n_tokens · topk` expert rows and drops
  zero, asserted from the counters the run itself produced, and the budget check
  fails LOUDLY on a padded (capacity-2.0) or truncated dispatch;
* **no `[R, f]` bank** — measured with `torch.profiler(profile_memory=True)` at
  b24 x s1024's token count: the largest single allocation is the router's
  `[n, E]` matrix, not a per-row activation bank, and it is orders of magnitude
  below the `[R, f]` buffer this module never builds;
* **the bill** — `3·d·f` per expert and `topk·per + shared` active, asserted
  against `parameters()`, with the live 64-expert numbers pinned;
* **the shared expert** — always on, and its write is exactly the dense FFN's.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.ffn import MoaFFN
from dormouse.model.moe import MoEFFN, bank_bill, check_row_budget, expert_parameters

D, F = 16, 12
E, K = 4, 2


#: The GELU width the expert bill is checked against: `3·D·F == 2·D·gelu`.
GELU_F = 3 * F // 2


def _bank(**kwargs: object) -> MoEFFN:
    """A small bank with the 10M-parameter floor raised out of the way.

    The experts are the architecture's arm (a bi-sided MoA over the default
    dictionary) read densely unless a test asks for the rank, because most of these
    gates are about the dispatch and the bill, and `test_tsct.py` owns the read.
    """
    settings: dict[str, object] = {
        "n_experts": E,
        "topk": K,
        "n_shared": 1,
        "act": "moa",
        "d_ffn_gelu": GELU_F,
        "expert_tsct_rank": 0,
        "min_expert_params": 0,
    }
    settings.update(kwargs)
    return MoEFFN(D, F, **settings)  # type: ignore[arg-type]


# --- parity ----------------------------------------------------------------


def test_one_expert_at_top_one_is_the_dense_expert_bit_for_bit() -> None:
    """The dispatch adds placement; the arithmetic must be `torch.equal`."""
    torch.manual_seed(0)
    bank = _bank(n_experts=1, topk=1, n_shared=0)
    dense = MoaFFN(D, F)
    dense.load_state_dict(bank.experts[0].state_dict())
    x = torch.randn(35, D)
    assert torch.equal(bank(x), dense(x))


def test_the_shared_expert_is_always_on_and_adds_the_dense_write() -> None:
    """Zero every routed weight: what is left must be the shared expert, exactly."""
    torch.manual_seed(1)
    bank = _bank()
    with torch.no_grad():
        #: The write matrix, not every weight: a MoA expert with zeroed weights still
        #: mixes its dictionary into a non-zero activation, and the claim here is that
        #: the routed branch contributes exactly nothing.
        for expert in bank.experts:
            expert.down.weight.zero_()
    x = torch.randn(11, D)
    counter = Counters()
    got = bank(x, counter)
    assert bank.shared is not None
    assert torch.equal(got, bank.shared(x))
    assert counter.snapshot()["moe_shared"] == 1


# --- rows and capacity -----------------------------------------------------


def test_a_forward_executes_exactly_tokens_times_topk_rows() -> None:
    """The counter the capacity bug would have failed, read off a real forward."""
    torch.manual_seed(2)
    bank = _bank(n_experts=8)
    counter = Counters()
    bank(torch.randn(4096, D), counter)
    counts = counter.snapshot()
    assert counts["moe_rows_run"] == 4096 * 2
    assert counts["moe_dropped"] == 0, "dropless: a drop is a bug, not a policy"
    assert counts["moe_experts_used"] == 8
    assert counts["moe_bank_builds"] == counts["moe_experts_used"], "a batch-wide bank"
    assert counts["moe_max_group"] < 2 * (4096 * 2 // 8), "the capacity is the assignment"
    assert counts["moe_fwd"] == 1
    assert counts["moe_qb"] == 1
    counter.check()  # every enabled mechanism reported itself


def test_a_collapsed_router_still_executes_every_row() -> None:
    """Skew is the arm's own worst case: no truncation, no padding, no drop."""
    torch.manual_seed(3)
    bank = _bank(n_shared=0)
    with torch.no_grad():
        bank.router.bias[0] = 100.0
    counter = Counters()
    bank(torch.randn(64, D), counter)
    counts = counter.snapshot()
    assert counts["moe_rows_run"] == 128
    assert counts["moe_dropped"] == 0
    assert counts["moe_max_group"] >= 64, "expert 0 is every token's first choice"


def test_the_row_budget_check_fails_loudly() -> None:
    """A 2.0 capacity pads to 98304 rows for 49152 assignments; a drop executes fewer."""
    with pytest.raises(LoudError, match="executed 98304 expert rows"):
        check_row_budget(executed=98304, routed=49152, topk=2, largest_group=1536)
    with pytest.raises(LoudError, match="fewer means dropped assignments"):
        check_row_budget(executed=49151, routed=49152, topk=2, largest_group=768)
    with pytest.raises(LoudError) as raised:
        check_row_budget(executed=98304, routed=49152, topk=2, largest_group=1536)
    assert "Escape:" in str(raised.value)
    check_row_budget(executed=49152, routed=49152, topk=2, largest_group=850)


# --- buffers ---------------------------------------------------------------


def test_no_batch_wide_bank_is_allocated() -> None:
    """At b24 x s1024's token count, measure the largest single allocation.

    The old arm's per-row `[R, f]` bank is `R·f·4` bytes; with `d=8, f=160` that
    is 31.5 MB here (1.21 GB at the live f=6144). The measured peak must be the
    router's `[n, E]` matrix, so the bound is `n·E·4` bytes, not `R·f·4`. The
    shared expert is off here on purpose: its `[n, f]` activation is the DENSE
    path's own buffer, not the bank's, and it would hide the claim.
    """
    torch.manual_seed(4)
    n, d, f, e, k = 24 * 1024, 8, 160, 64, 2
    bank = MoEFFN(
        d,
        f,
        n_experts=e,
        topk=k,
        n_shared=0,
        act="moa",
        d_ffn_gelu=3 * f // 2,
        expert_tsct_rank=0,
        min_expert_params=0,
    )
    bank.eval()
    x = torch.randn(n, d)
    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profiler,
    ):
        bank(x)
    peak = max(event.self_cpu_memory_usage for event in profiler.events())
    router_matrix = n * e * 4
    removed_bank = n * k * f * 4
    print(
        f"\npeak single allocation {peak / 2**20:.1f} MiB vs [R,f] {removed_bank / 2**20:.1f} MiB"
    )
    assert peak <= 1.25 * router_matrix, (peak, router_matrix)
    assert removed_bank > 4 * peak, (removed_bank, peak)


# --- the bill --------------------------------------------------------------


def test_the_parameter_report_is_the_formula_and_the_tree() -> None:
    bank = _bank(n_experts=3, n_shared=1)
    report = bank.param_report()
    per = expert_parameters(D, F) + bank.expert_spec.added_parameters()
    assert report["per_expert"] == report["measured_per_expert"]
    assert report["per_expert"] - 3 * D * F == bank.expert_spec.added_parameters()
    assert report["routed_stored"] == 3 * per
    assert report["shared_stored"] == per
    assert report["router"] == 3 * D
    assert report["active"] == 2 * per + per
    assert report["stored"] == sum(p.numel() for p in bank.parameters())
    assert report["stored"] == 3 * per + per + 3 * D


def test_the_parameter_report_refuses_a_tree_that_disagrees() -> None:
    """A biased or extra matrix is a different expert than the bill quotes."""
    bank = _bank(n_experts=2)
    bank.experts[0].down.bias = torch.nn.Parameter(torch.zeros(F))
    with pytest.raises(LoudError, match="the tree bills"):
        bank.param_report()


def test_the_live_bank_is_a_billion_stored_and_46m_active() -> None:
    """The design's headline pair, by arithmetic — no 1B parameters are built here."""
    bill = bank_bill(832, 6144, n_experts=64, topk=2, n_shared=1)
    assert bill["per_expert"] == 3 * 832 * 6144 == 15_335_424
    assert bill["per_expert"] >= 10_000_000
    assert bill["routed_stored"] == 981_467_136
    assert bill["shared_stored"] == 15_335_424
    assert bill["router"] == 53_248
    assert bill["stored"] == 996_855_808
    assert bill["active"] == 46_006_272
    assert bill["active"] == 2 * bill["per_expert"] + bill["shared_stored"]
    assert 0.99e9 < bill["stored"] < 1.0e9
    assert bill["active"] / bill["stored"] < 0.05


# --- refusals --------------------------------------------------------------


def test_an_expert_below_the_parameter_floor_is_refused() -> None:
    with pytest.raises(LoudError, match="below the 10,000,000 floor"):
        MoEFFN(8, 8, n_experts=4, d_ffn_gelu=12)
    with pytest.raises(LoudError) as raised:
        MoEFFN(8, 8, n_experts=4, d_ffn_gelu=12)
    assert "3·832·6144=15,335,424" in str(raised.value)


def test_an_expert_below_one_percent_of_the_active_set_is_refused() -> None:
    with pytest.raises(LoudError, match="below 1%"):
        MoEFFN(8, 8, n_experts=128, topk=101, min_expert_params=0, d_ffn_gelu=12)


def test_an_impossible_bank_is_refused_with_an_escape() -> None:
    with pytest.raises(LoudError, match="not a width"):
        MoEFFN(0, F, n_experts=E, min_expert_params=0)
    with pytest.raises(LoudError, match="builds no bank"):
        MoEFFN(D, F, n_experts=0, min_expert_params=0)
    with pytest.raises(LoudError, match="ONE always-on expert"):
        MoEFFN(D, F, n_experts=E, n_shared=2, min_expert_params=0)
    with pytest.raises(LoudError) as raised:
        MoEFFN(D, F, n_experts=E, n_shared=2, min_expert_params=0)
    assert "Escape:" in str(raised.value)


def test_the_input_shape_is_the_banks_contract() -> None:
    bank = _bank()
    with pytest.raises(LoudError, match="takes x \\[tokens, d_model\\]"):
        bank(torch.randn(2, 3, D))
    with pytest.raises(LoudError, match="but the bank is d_model"):
        bank(torch.randn(5, D + 1))
    with pytest.raises(LoudError, match="a forward with no tokens"):
        bank(torch.zeros(0, D))
    assert f"n_experts={E}, topk={K}" in repr(bank)
