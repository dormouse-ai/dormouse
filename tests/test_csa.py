"""CSA's gates: the selector finds a needle, and the sparsity is real.

The number that matters here is the keep-rate. A previous CSA arm in this project
measured 119/128 selections of the needed block — exactly `k/n_chunks`, i.e. a
selector that was not reading the content at all — and that was a defect, not a
result. So the gate is a synthetic retrieval: a needle in the first block, a query
at the end that needs it, 64 trials, and a rate compared against chance.

The HCA level is checked on the same module with `hca_group=8`: it must keep a
needle inside a coarse block and stay bit-identical under a later perturbation.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.model.assemble import plan_lines
from dormouse.model.attn_variants.csa import CSA
from dormouse.model.attn_variants.hybrid import (
    HCA_GROUP,
    HYBRID_6_KDA_CSA_HCA,
    HYBRID_12_KDA_2CSA_2HCA,
    HYBRID_24_KDA_4CSA_4HCA,
    HybridStack,
)
from dormouse.model.spec import StackSpec

CHUNK = 16
SEQ = 128
HEAD_DIM = 8
TOPK = 2
TRIALS = 64


def _needle(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    """One query and one key per position: block 0's keys are the query's needle."""
    q = torch.zeros(1, seq, 1, HEAD_DIM)
    q[0, :, 0, 0] = 1.0
    k = torch.randn(1, seq, 1, HEAD_DIM)
    k = k / k.norm(dim=-1, keepdim=True)
    k[:, :CHUNK] = 0.0
    k[:, :CHUNK, 0, 0] = 1.0
    return q, k


def _chance(seq: int = SEQ) -> float:
    """`k` kept past blocks out of the ones a last-position query may read."""
    return min(TOPK, seq // CHUNK - 1) / (seq // CHUNK - 1)


@pytest.mark.parametrize("group", [1, 2, 4, HCA_GROUP])
def test_the_selector_keeps_the_needle_far_above_chance(group: int) -> None:
    """64 trials, four coarsenesses: chance is `topk / n_eligible_blocks`."""
    torch.manual_seed(11)
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=CHUNK, topk=TOPK, hca_group=group)
    hits = 0
    for _ in range(TRIALS):
        ids, _, _ = csa.select(*_needle(256))
        hits += int((ids[0, 255, 0] == 0).any())
    rate = hits / TRIALS
    assert rate > 3.0 * _chance(256), f"keep-rate {hits}/{TRIALS}, chance {_chance(256):.3f}"
    assert rate >= 0.95


def test_no_query_reads_a_block_after_itself() -> None:
    """The selector is causal by construction, not by a mask applied afterwards."""
    torch.manual_seed(12)
    csa = CSA(CHUNK, 2, HEAD_DIM, chunk=CHUNK, topk=TOPK)
    q, k = torch.randn(2, SEQ, 2, HEAD_DIM), torch.randn(2, SEQ, 2, HEAD_DIM)
    ids, _, _ = csa.select(q, k)
    own = torch.arange(SEQ) // CHUNK
    assert bool((ids <= own.view(1, -1, 1, 1)).all())


def test_a_later_token_cannot_change_an_earlier_output() -> None:
    """Bit-identical prefix: the block means include the future, the output must not."""
    torch.manual_seed(13)
    csa = CSA(CHUNK, 2, HEAD_DIM, chunk=CHUNK, topk=TOPK)
    x = torch.randn(1, SEQ, CHUNK)
    y = csa(x)
    moved = x.clone()
    moved[:, 96] += 1.0
    assert torch.equal(y[:, :96], csa(moved)[:, :96])
    assert not torch.equal(y[:, 96], csa(moved)[:, 96])


def test_hca_group_eight_selects_a_coarse_block_and_stays_causal() -> None:
    """`hca_group=8` is the same selector one level up, over 128-token blocks.

    The query's own coarse block is forced rather than scored: its mean sees the
    future, so a token at 200 may not move the output at 199.
    """
    torch.manual_seed(19)
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=CHUNK, topk=TOPK, hca_group=HCA_GROUP)
    assert csa.hca_group == HCA_GROUP
    ids, _, _ = csa.select(*_needle(256))
    assert 0 in ids[0, 255, 0].tolist()

    counters = Counters()
    csa(torch.zeros(1, 256, CHUNK), counters)
    assert 0 < counters.counts["csa_kept"] <= (TOPK + 1) * CHUNK * 256

    x = torch.randn(1, 256, CHUNK)
    y = csa(x)
    moved = x.clone()
    moved[:, 200] += 1.0
    assert torch.equal(y[:, :200], csa(moved)[:, :200])


def test_the_selector_score_reaches_the_logits_and_its_gradient() -> None:
    """Without this path top-k is a constant mask and the gain never learns."""
    torch.manual_seed(14)
    csa = CSA(CHUNK, 2, HEAD_DIM, chunk=CHUNK, topk=TOPK)
    x = torch.randn(2, SEQ, CHUNK)
    csa(x).pow(2).mean().backward()
    assert csa.selector_gain.grad is not None
    assert csa.selector_gain.grad.abs().max() > 0
    before = csa(x).detach()
    with torch.no_grad():
        csa.selector_gain.fill_(1.0)
    assert not torch.allclose(before, csa(x))


def test_the_attention_reads_only_the_kept_blocks() -> None:
    """`(topk+1)·chunk` positions per query, not `seq`: the sparsity is counted."""
    counters = Counters()
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=8, topk=1)
    csa(torch.zeros(1, 32, CHUNK), counters)
    kept = min(32 // 8, 2) * 8
    assert counters.counts["csa_fwd"] == 32
    assert counters.counts["csa_selector"] == 32
    assert 0 < counters.counts["csa_kept"] <= kept * 32


def test_a_forward_without_a_backward_is_a_missing_report() -> None:
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=8)
    counters = Counters()
    counters.enable("csa", counter="csa_bwd")
    csa(torch.zeros(1, 16, CHUNK), counters)
    with pytest.raises(MissingReport) as e:
        counters.check()
    assert "csa" in str(e.value)


def test_block_attention_without_a_selector_is_refused() -> None:
    """Kept-every-block is the dense control arm, not CSA with a flag flipped."""
    with pytest.raises(LoudError) as e:
        CSA(CHUNK, 1, HEAD_DIM, selector=False)
    assert "dense block attention" in str(e.value)


def test_a_window_that_does_not_tile_the_blocks_is_refused() -> None:
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=8, hca_group=2)
    with pytest.raises(LoudError) as e:
        csa(torch.zeros(1, 20, CHUNK))
    assert "16" in str(e.value)


def test_the_coeffects_declare_the_selector() -> None:
    declared = CSA(CHUNK, 1, HEAD_DIM).coeffects()
    assert declared.requires["attn.selector"] == "true"
    assert declared.provides["attn.state"] == "sparse"


def test_the_sparsity_parameters_are_validated() -> None:
    """A block of zero tokens, a selector of zero blocks, a group of zero blocks."""
    with pytest.raises(LoudError, match="positive widths"):
        CSA(0, 1, HEAD_DIM)
    with pytest.raises(LoudError, match="holds no tokens"):
        CSA(CHUNK, 1, HEAD_DIM, chunk=0)
    with pytest.raises(LoudError, match="keeps no past block"):
        CSA(CHUNK, 1, HEAD_DIM, topk=0)
    with pytest.raises(LoudError, match="groups no blocks"):
        CSA(CHUNK, 1, HEAD_DIM, hca_group=0)


def test_the_repr_names_the_sparsity() -> None:
    assert "topk=2" in repr(CSA(CHUNK, 1, HEAD_DIM, topk=2))


def test_hca_keeps_the_query_s_own_fine_block() -> None:
    """The forced coarse block must carry the fine block the query sits in."""
    torch.manual_seed(23)
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=CHUNK, topk=TOPK, hca_group=HCA_GROUP)
    q, k = _needle(256)
    ids, _, _ = csa.select(q, k)
    own = (torch.arange(256) // CHUNK).view(1, -1, 1, 1)
    assert bool((ids == own).any(-1).all())
    assert 15 in ids[0, 255, 0].tolist()


def test_select_refuses_a_partial_coarse_block() -> None:
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=8, hca_group=2)
    with pytest.raises(LoudError, match="leaves 4 tokens"):
        csa.select(torch.zeros(1, 20, 1, HEAD_DIM), torch.zeros(1, 20, 1, HEAD_DIM))


def test_an_unselected_coarse_block_receives_gradient() -> None:
    """ST top-k: a block the forward did not keep still gets the "should have" gradient."""
    torch.manual_seed(31)
    csa = CSA(CHUNK, 1, HEAD_DIM, chunk=CHUNK, topk=TOPK, hca_group=4)
    q = torch.randn(1, 512, 1, HEAD_DIM, requires_grad=True)
    k = torch.randn(1, 512, 1, HEAD_DIM)
    ids, _, _, bias, cscore = csa.select(q, k, return_aux=True)
    assert bias is not None and cscore is not None
    cscore.retain_grad()
    bias.sum().backward()
    assert cscore.grad is not None
    eligible = csa._coarse_eligibility(512, q.device)[0, -1, 0]
    picked_ids = ids[0, -1, 0] // 4
    picked_coarse = torch.zeros(eligible.numel(), dtype=torch.bool, device=q.device)
    picked_coarse[picked_ids] = True
    unselected = eligible & ~picked_coarse
    assert bool(unselected.any()), "the window has no unselected eligible coarse block"
    assert cscore.grad[0, -1, 0][unselected].abs().max() > 0


def test_the_8_12_and_32_layer_patterns_tile() -> None:
    small = {
        "d_model": 64,
        "n_heads": 2,
        "c_lat": 32,
        "head_dim": 8,
        "d_ffn": 128,
        "attn_k": 64,
        "attn_window": 32,
        "seq_len": 128,
        "ffn": "dense",
        "n_experts": 0,
        "topk": 0,
        "n_shared": 0,
        "expert_tsct_rank": 0,
    }
    sixteen = StackSpec(**small, n_layers=16, attn_pattern=list(HYBRID_12_KDA_2CSA_2HCA))
    assert sixteen.pattern == HYBRID_12_KDA_2CSA_2HCA
    assert len(sixteen.pattern) == 16
    assert sixteen.pattern.count("csa") == 2 and sixteen.pattern.count("hca") == 2
    text = "\n".join(plan_lines(sixteen))
    assert "hybrid pattern=" in text and "csa, hca" in text


def test_the_8_and_32_layer_patterns_tile_and_the_plan_prints_them() -> None:
    """One period, tiled: 6 KDA + CSA + HCA at 8 layers, 24 + 4 + 4 at 32."""
    small = {
        "d_model": 64,
        "n_heads": 2,
        "c_lat": 32,
        "head_dim": 8,
        "d_ffn": 128,
        "attn_k": 64,
        "attn_window": 32,
        "seq_len": 128,
        "ffn": "dense",
        "n_experts": 0,
        "topk": 0,
        "n_shared": 0,
        "expert_tsct_rank": 0,
    }
    eight = StackSpec(**small, n_layers=8, attn_pattern=list(HYBRID_6_KDA_CSA_HCA))
    thirty_two = StackSpec(**small, n_layers=32, attn_pattern=list(HYBRID_24_KDA_4CSA_4HCA))
    assert eight.pattern == HYBRID_6_KDA_CSA_HCA
    assert thirty_two.pattern == HYBRID_24_KDA_4CSA_4HCA
    text = "\n".join(plan_lines(eight))
    assert "hybrid pattern=" in text and "csa, hca" in text

    stack = HybridStack(
        32,
        1,
        HEAD_DIM,
        n_layers=8,
        pattern=HYBRID_6_KDA_CSA_HCA,
        csa_chunk=CHUNK,
        chunk=8,
        k=8,
        window=4,
    )
    assert stack.layer_types() == HYBRID_6_KDA_CSA_HCA
    assert stack.layers[-1].hca_group == HCA_GROUP
    assert stack.layers[-2].hca_group == 1

    with pytest.raises(LoudError, match="1 layers are left over"):
        StackSpec(**small, n_layers=9, attn_pattern=list(HYBRID_6_KDA_CSA_HCA))
