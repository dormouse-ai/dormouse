"""TSCT inside the experts: exact at full rank, cheaper below it, counted either way.

The gate that matters is (a): at `r = min(in, out)` the factorised read has nothing to
truncate, and the expert must be `torch.equal` to the dense one — not close. A
factorisation that is merely close is a different expert with the same name, and every
number the bank reports would be about a model nobody chose.

(b) is the FLOP claim, printed: `in·r + r + r·out` multiplies against `in·out` at the
live width. (c) is the counter: the read is a compute path, so it reports itself like
any other mechanism, and an expert bank that says `tsct` without a `tsct_fwd` would
be the flag nobody read.

(d) the parameters do NOT move: the masters stay dense, so `state_dict()` and the
`3·d·f` bill are the same on both sides. That is what makes this a compute path and
not a second parameterisation.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.ffn import GatedFFN, MoaFFN
from dormouse.model.ffn_spec import FFNSpec
from dormouse.model.moe import MoEFFN
from dormouse.model.tsct import TSCT_RANK, factors, low_rank

D, F = 16, 12


def test_full_rank_is_the_dense_expert_bit_for_bit() -> None:
    """`r = min(d, f)`: the two-matmul path must return the dense expert's tensor."""
    torch.manual_seed(0)
    dense = MoaFFN(D, F)
    exact = MoaFFN(D, F, tsct_rank=min(D, F))
    exact.load_state_dict(dense.state_dict())
    x = torch.randn(35, D)
    assert torch.equal(exact(x), dense(x))


def test_full_rank_is_exact_in_both_orientations() -> None:
    """`in < out` (gate/up) and `in > out` (down) take different identity bases."""
    torch.manual_seed(1)
    for rank in (min(D, F), F + 10):
        up = torch.nn.Linear(D, F, bias=False)
        wide = torch.nn.Linear(F, D, bias=False)
        x_in = torch.randn(7, D)
        x_out = torch.randn(7, F)
        assert torch.equal(low_rank(up, x_in, rank), up(x_in))
        assert torch.equal(low_rank(wide, x_out, rank), wide(x_out))


def test_below_full_rank_the_read_is_the_truncated_spectrum() -> None:
    """The rank-`r` weight `u·diag(s)·v` is the rank-`r` SVD of the master, not noise."""
    torch.manual_seed(2)
    weight = torch.randn(F, D)
    u, s, v = factors(weight, 4)
    assert u.shape == (D, 4) and s.shape == (4,) and v.shape == (4, F)
    left, spectrum, right = torch.linalg.svd(weight.T, full_matrices=False)
    truncated = left[:, :4] @ torch.diag(spectrum[:4]) @ right[:4]
    assert torch.allclose(u @ torch.diag(s) @ v, truncated, atol=1e-5)


def test_the_rank_buys_flops_and_no_parameters() -> None:
    """`in·r + r + r·out` against `in·out` at the live expert width."""
    d, f, r = 832, 6144, TSCT_RANK
    factored, dense = d * r + r + r * f, d * f
    print(
        f"\nexpert matrix: dense={dense:,} mac/token, rank-{r}={factored:,} "
        f"({factored / dense:.1%}, {dense / factored:.2f}x fewer)"
    )
    assert factored / dense == pytest.approx(0.3494, abs=1e-3)
    for rank in (0, TSCT_RANK):
        spec = FFNSpec(d, f, act="moa", d_ffn_gelu=3 * f // 2, tsct_rank=rank)
        expert = spec.build()
        assert sum(p.numel() for p in expert.parameters()) == 3 * d * f + d * 6 + 2 * 6


def test_the_read_reports_itself_and_the_dense_path_does_not() -> None:
    counters = Counters()
    expert = MoaFFN(D, F, tsct_rank=4)
    expert(torch.zeros(3, D), counters)
    assert counters.counts["tsct_fwd"] == 9  # three matrices x three rows
    dense = Counters()
    MoaFFN(D, F)(torch.zeros(3, D), dense)
    assert dense.counts["tsct_fwd"] == 0


def test_the_bank_reads_its_experts_through_the_factors() -> None:
    """The bank passes the rank down: a forward counts the rows it factorised."""
    torch.manual_seed(3)
    bank = MoEFFN(
        D,
        F,
        n_experts=4,
        topk=2,
        act="moa",
        d_ffn_gelu=18,
        expert_tsct_rank=4,
        min_expert_params=0,
    )
    counters = Counters()
    bank(torch.randn(64, D), counters)
    #: 64 tokens x 2 routed + 64 shared, each through three matrices.
    assert counters.counts["tsct_fwd"] == (64 * 2 + 64) * 3
    assert bank.param_report()["per_expert"] == 3 * D * F + bank.expert_spec.added_parameters()


def test_the_bank_without_the_rank_is_the_dense_expert_bank() -> None:
    """`r = 0` is the dense read: the arm exists so the factorisation can be off."""
    torch.manual_seed(4)
    bank = MoEFFN(
        D,
        F,
        n_experts=2,
        topk=1,
        act="moa",
        d_ffn_gelu=18,
        expert_tsct_rank=0,
        min_expert_params=0,
    )
    counters = Counters()
    bank(torch.randn(8, D), counters)
    assert counters.counts["tsct_fwd"] == 0


def test_a_rank_of_zero_or_a_non_matrix_is_refused() -> None:
    with pytest.raises(LoudError, match="keeps no direction"):
        factors(torch.randn(F, D), 0)
    with pytest.raises(LoudError, match="factorises a matrix"):
        factors(torch.randn(3, 4, 5), 2)


def test_a_gated_expert_can_be_read_through_the_factors_too() -> None:
    """The path is on `_Glu`, so both arms share it (the parity anchor depends on it)."""
    torch.manual_seed(5)
    dense = GatedFFN(D, F, act="silu")
    gated = GatedFFN(D, F, act="silu", tsct_rank=min(D, F))
    gated.load_state_dict(dense.state_dict())
    x = torch.randn(5, D)
    assert torch.equal(gated(x), dense(x))
