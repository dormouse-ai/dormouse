"""TSCT stores the factors: no decomposition runs, and the bill and the MACs are theirs.

Five numbers, each one a defect this module has actually shipped or could ship:

* no `torch.linalg.svd` in a forward or a backward. Measured on this module's own
  four-block stack BEFORE the factors became parameters: **60** SVD calls in one
  forward (4 bank passes x 5 groups x 3 matrices), and each `u, s, v` triple stayed
  alive for the backward. On the live 12-layer bank that is 12 x 65 x 3 = 2340 per
  forward, and one shared-bank call allocated 5 265 MiB on the card because of it;
* the bill is the factors: `3·(832·256 + 256·6144)` stored per expert, and no
  `[832, 6144]` tensor is allocated by an expert's forward at all;
* the MACs are the two matmuls: `2·rows·(d·r + r·f)` per matrix, measured with the
  profiler's FLOP counter and computed independently in the test — 2.862x fewer than
  the dense draw at r=256;
* `r = min(d, f)` is the dense expert `torch.equal`, not close. A factorisation that
  cannot express the dense layer is a different model wearing the same name;
* at r=256 the fp32 read is the fp64 product to within float error (1e-6), so the two
  matmuls are the same arithmetic as one.

The dense arm (`rank=0`) is the control, and it is still there: `project` builds an
`nn.Linear` and the dense bank reports zero `tsct_fwd`.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble
from dormouse.model.ffn import GatedFFN, MoaFFN
from dormouse.model.ffn_spec import FFNSpec
from dormouse.model.moe import MoEFFN, expert_parameters
from dormouse.model.spec import StackSpec
from dormouse.model.tsct import FactorLinear, project


@pytest.fixture(autouse=True)
def _cpu_eager_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU tests exercise the eager factors; fused kernels need a GPU."""
    monkeypatch.setenv("DORMOUSE_FUSED", "0")


D, F = 16, 12

#: The live expert width and the recipe's rank; the gates about the bill quote these.
LIVE_D, LIVE_F, LIVE_R = 832, 6144, 256

#: A stack a CPU test can afford that keeps every slot: the smoke widths of
#: `test_assemble.py`, four layers so the shared bank is entered four times.
SMOKE = {
    "unit": "byte",
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


def _expert(rank: int) -> MoaFFN:
    """The live expert's arm at the live widths, at `rank`."""
    return MoaFFN(LIVE_D, LIVE_F, tsct_rank=rank)


# --- gate 4: nothing is decomposed ---------------------------------------------


def test_a_full_forward_and_backward_decomposes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero SVDs in one forward+backward; the same probe saw 60 before this change."""
    seen = {"svd": 0, "svdvals": 0}
    real = (torch.linalg.svd, torch.linalg.svdvals)

    def counter(kind: str, fn: object) -> object:
        def counted(*args: object, **kwargs: object) -> object:
            seen[kind] += 1
            return fn(*args, **kwargs)  # type: ignore[operator]

        return counted

    monkeypatch.setattr(torch.linalg, "svd", counter("svd", real[0]))
    monkeypatch.setattr(torch.linalg, "svdvals", counter("svdvals", real[1]))

    spec = StackSpec(**SMOKE)  # type: ignore[arg-type]
    model = assemble(spec, smoke=True).model
    model.train()
    ids = torch.randint(0, 256, (spec.batch, spec.seq_len + 1))
    logits = model(ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1))
    loss.backward()
    print(f"\ndecompositions in one forward+backward: {seen}")
    assert seen == {"svd": 0, "svdvals": 0}


# --- gate 5: the parameters are the factors ------------------------------------


def test_the_bill_is_the_factors_not_the_dense_master() -> None:
    """`param_report` bills `3·(832·256 + 256·6144)` for the expert's three matrices."""
    bank = MoEFFN(
        LIVE_D,
        LIVE_F,
        n_experts=1,
        topk=1,
        n_shared=0,
        act="moa",
        d_ffn_gelu=3 * LIVE_F // 2,
        expert_tsct_rank=LIVE_R,
        min_expert_params=0,
    )
    report = bank.param_report()
    factored = 3 * (LIVE_D * LIVE_R + LIVE_R * LIVE_F)
    assert factored == 5_357_568
    assert report["per_expert"] - factored == bank.expert_spec.added_parameters()
    assert report["per_expert"] == report["measured_per_expert"]
    dense = expert_parameters(LIVE_D, LIVE_F) + bank.expert_spec.added_parameters()
    assert report["stored"] < dense


def test_no_dense_expert_matrix_is_ever_allocated() -> None:
    """No single allocation in an expert's forward is a `[832, 6144]` slab.

    The batched read stacks a group's rank-256 factors into 3-D operands, so the largest
    block is ONE factor (`r·f`), still 3.2x below the dense master the factors replaced —
    and the group is chosen from the slab budget, so it does not grow with the bank.
    """
    bank = MoEFFN(
        LIVE_D,
        LIVE_F,
        n_experts=1,
        topk=1,
        n_shared=0,
        act="moa",
        d_ffn_gelu=3 * LIVE_F // 2,
        expert_tsct_rank=LIVE_R,
        min_expert_params=0,
    )
    bank.eval()
    dense = LIVE_D * LIVE_F * 4
    factored = LIVE_R * LIVE_F * 4
    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profiler,
    ):
        bank(torch.randn(8, LIVE_D))
    peak = max(event.self_cpu_memory_usage for event in profiler.events())
    print(
        f"\npeak single allocation {peak / 2**20:.2f} MiB vs one factor "
        f"{factored / 2**20:.2f} MiB and a dense master {dense / 2**20:.2f} MiB"
    )
    assert peak <= 1.25 * factored, (peak, factored)
    assert peak < dense / 3, (peak, dense)


# --- gate 6: the MACs are the two matmuls --------------------------------------


def test_the_forward_runs_the_two_matmuls_and_nothing_else() -> None:
    """FLOPs are `2·rows·(d·r + r·f)` per matrix, computed here from the widths."""
    d, f, rank, rows = LIVE_D, LIVE_F, LIVE_R, 8
    expert = GatedFFN(d, f, act="silu", tsct_rank=rank)
    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
        ) as profiler,
    ):
        expert(torch.randn(rows, d))
    flops = sum(
        event.flops
        for event in profiler.events()
        if event.key in ("aten::mm", "aten::addmm", "aten::bmm")
    )
    factored = 6 * rows * (d * rank + rank * f)
    dense = 6 * rows * d * f
    print(f"\nmatmul flops {flops:,} = {flops / factored:.4f}x the factored bill")
    assert flops == factored
    assert dense / factored == pytest.approx(2.862, abs=1e-3)


# --- gate 7: the degeneracy and the reference ----------------------------------


def _load_identity_basis(dense: MoaFFN, factored: MoaFFN) -> None:
    """Set each factor pair to `u, v = I, Wᵀ` (or `Wᵀ, I`): the exact dense map."""
    with torch.no_grad():
        #: The mixture's own parameters are not factors; they must still be the same.
        factored.mix.weight.copy_(dense.mix.weight)
        factored.mix.bias.copy_(dense.mix.bias)
        factored.coef.copy_(dense.coef)
        for name in ("gate", "up", "down"):
            weight = getattr(dense, name).weight.T  # [in, out] = Wᵀ
            target = getattr(factored, name)
            assert isinstance(target, FactorLinear)
            short = min(weight.shape)
            assert target.rank == short, "the degeneracy is at r = min(in, out)"
            eye = torch.eye(short)
            if weight.shape[0] == short:
                target.u.copy_(eye)
                target.v.copy_(weight)
            else:
                target.u.copy_(weight)
                target.v.copy_(eye)


def test_the_full_rank_read_is_the_dense_expert_bit_for_bit() -> None:
    """`r = min(d, f)`: nothing is truncated, and both factors are exact."""
    torch.manual_seed(0)
    dense = MoaFFN(D, F)
    factored = MoaFFN(D, F, tsct_rank=min(D, F))
    _load_identity_basis(dense, factored)
    x = torch.randn(35, D)
    assert torch.equal(factored(x), dense(x))


def test_the_live_rank_read_is_the_fp64_product_to_float_error() -> None:
    """At r=256 the fp32 two-matmul read matches an independent fp64 evaluation."""
    torch.manual_seed(1)
    layer = _expert(LIVE_R).gate
    x = torch.randn(4, LIVE_D)
    with torch.no_grad():
        got = layer(x)
        want = torch.einsum("...i,ir,ro->...o", x.double(), layer.u.double(), layer.v.double())
    error = (got.double() - want).abs().max().item()
    scale = want.abs().max().item()
    print(
        f"\nat r={LIVE_R}: max |fp32 - fp64| {error:.2e} on |y|max {scale:.3f} "
        f"= {error / scale:.1e} relative"
    )
    assert error / scale < 1e-6


def test_the_draw_has_the_dense_matrixs_variance() -> None:
    """`σ_u = σ_v = √std·r^{-1/4}`, so `Var((x·u)·v) == std²·Σx²` like the dense draw."""
    torch.manual_seed(2)
    std, rank, d, f = 0.02, 64, 512, 256
    drawn = project(d, f, rank, std=std)
    assert isinstance(drawn, FactorLinear)
    want = float(std) ** 0.5 * rank**-0.25
    assert drawn.u.std().item() == pytest.approx(want, rel=0.02)
    assert drawn.v.std().item() == pytest.approx(want, rel=0.02)
    x = torch.randn(4096, d)
    with torch.no_grad():
        dense_out, factored_out = project(d, f, 0, std=std)(x), drawn(x)
    ratio = factored_out.std().item() / dense_out.std().item()
    print(f"\noutput std ratio factored/dense: {ratio:.4f}")
    assert ratio == pytest.approx(1.0, rel=0.05)


# --- the wiring, the counter and the refusals ----------------------------------


def test_the_rank_buys_macs_and_the_bill_is_the_factors() -> None:
    """`d·r + r·f` against `d·f` at the live width, and the stored bill beside it."""
    factored = LIVE_D * LIVE_R + LIVE_R * LIVE_F
    dense = LIVE_D * LIVE_F
    print(
        f"\nexpert matrix: dense={dense:,} mac/token, rank-{LIVE_R}={factored:,} "
        f"({factored / dense:.1%}, {dense / factored:.2f}x fewer)"
    )
    assert factored / dense == pytest.approx(0.3494, abs=1e-3)
    for rank in (0, LIVE_R):
        spec = FFNSpec(LIVE_D, LIVE_F, act="moa", d_ffn_gelu=3 * LIVE_F // 2, tsct_rank=rank)
        expert = spec.build()
        matrices = expert_parameters(LIVE_D, LIVE_F, rank)
        assert sum(p.numel() for p in expert.parameters()) == (matrices + LIVE_D * 6 + 2 * 6)


def test_the_read_reports_itself_and_the_dense_arm_does_not() -> None:
    counters = Counters()
    expert = MoaFFN(D, F, tsct_rank=4)
    expert(torch.zeros(3, D), counters)
    assert counters.counts["tsct_fwd"] == 9  # three matrices x three rows
    dense = Counters()
    MoaFFN(D, F)(torch.zeros(3, D), dense)
    assert dense.counts["tsct_fwd"] == 0


def test_the_bank_reads_its_experts_through_the_factors() -> None:
    """The bank passes the rank down: a forward counts the rows it read that way."""
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
    assert bank.param_report()["per_expert"] == expert_parameters(D, F, 4) + (
        bank.expert_spec.added_parameters()
    )


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
    assert isinstance(bank.experts[0].gate, torch.nn.Linear)
    counters = Counters()
    bank(torch.randn(8, D), counters)
    assert counters.counts["tsct_fwd"] == 0
    assert bank.param_report()["per_expert"] == 3 * D * F + bank.expert_spec.added_parameters()


def test_a_zero_rank_or_a_rank_past_the_short_side_is_refused() -> None:
    with pytest.raises(LoudError, match="keeps no direction"):
        FactorLinear(D, F, 0, std=0.02)
    with pytest.raises(LoudError, match="wider than the short side"):
        FactorLinear(F, D, D + 1, std=0.02)
    with pytest.raises(LoudError, match="wider than the short side"):
        project(F, D, D + 1, std=0.02)


def test_a_gated_expert_can_be_read_through_the_factors_too() -> None:
    """The storage is on `_Glu`, so both arms share it (the parity anchor depends on it)."""
    torch.manual_seed(5)
    dense = GatedFFN(D, F, act="silu")
    gated = GatedFFN(D, F, act="silu", tsct_rank=min(D, F))
    for name in ("gate", "up", "down"):
        weight = getattr(dense, name).weight.T
        target = getattr(gated, name)
        short = min(weight.shape)
        eye = torch.eye(short)
        with torch.no_grad():
            if weight.shape[0] == short:
                target.u.copy_(eye)
                target.v.copy_(weight)
            else:
                target.u.copy_(weight)
                target.v.copy_(eye)
    x = torch.randn(5, D)
    assert torch.equal(gated(x), dense(x))
