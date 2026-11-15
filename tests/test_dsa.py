"""DSA's gates: the selection is causal, the indexer is linear, and reuse is measured.

Four things are checked here that "it runs and has the right shape" would miss:

(a) the indexer is LINEAR in the window — `T·d_idx` MACs, so 8k tokens cost 8x 1k
    and not 64x; that number is the reason this layer replaces a block selector at
    1M (see `dsa.py`'s bill);
(b) the selection is causal by construction — a query's read set holds no position
    after itself, so a perturbed later token leaves the prefix bit-identical;
(c) the degenerate configuration IS dense attention, bit for bit: with the window
    covering the sequence the indexer's selection is masked away and the sparse
    path must produce the same tensor as the module's own dense anchor;
(d) MLI's premise is measured, not assumed: an overlap under the floor is refused
    rather than silently degrading the layer it is meant to speed up.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.attn_variants.dsa import DSA
from dormouse.model.attn_variants.dsa_indexer import MLI_MIN_OVERLAP

D_MODEL = 16
N_HEADS = 2
HEAD_DIM = 8
SEQ = 32
K = 8
WINDOW = 4


def _dsa(**kwargs: int | bool) -> DSA:
    """A small layer with the widths the other attention tests use."""
    return DSA(
        D_MODEL,
        N_HEADS,
        HEAD_DIM,
        k=int(kwargs.get("k", K)),
        window=int(kwargs.get("window", WINDOW)),
        d_idx=int(kwargs.get("d_idx", 16)),
        chunk=4,
        mli=bool(kwargs.get("mli", False)),
    )


def test_the_indexer_cost_scales_with_the_window_not_its_square() -> None:
    """`T·d_idx`: 8k tokens cost 8x 1k, not 64x. The block scan would be 64x."""
    d_idx = 16
    macs = {}
    for seq in (1024, 8192):
        counters = Counters()
        layer = DSA(8, 1, 4, k=K, window=WINDOW, d_idx=d_idx, chunk=256)
        layer(torch.zeros(1, seq, 8), counters)
        macs[seq] = counters.counts["dsa_index_macs"]
        assert macs[seq] == seq * d_idx, "the count is the scoring pass, from the pass"
    ratio = macs[8192] / macs[1024]
    print(f"\nindexer MACs: 1k={macs[1024]:,} 8k={macs[8192]:,} ratio={ratio:.1f}")
    assert ratio == pytest.approx(8.0, rel=1e-6), "linear: the ratio is T's ratio"


def test_the_counters_name_the_selection_the_scan_and_the_reuse() -> None:
    counters = Counters()
    _dsa()(torch.zeros(1, SEQ, D_MODEL), counters)
    assert counters.counts["dsa_fwd"] == SEQ
    assert counters.counts["dsa_selected"] == SEQ * K
    assert counters.counts["dsa_index_macs"] == SEQ * 16
    assert counters.counts["dsa_mli_reuse"] == 0


def test_no_query_reads_a_position_after_itself() -> None:
    """The read set — window and selected alike — is causal by construction."""
    torch.manual_seed(31)
    layer = _dsa()
    at, visible = layer.attended(torch.randn(2, SEQ, D_MODEL))
    limit = torch.arange(SEQ).view(1, SEQ, 1).expand_as(at)
    assert bool((at[visible] <= limit[visible]).all())
    assert bool((at <= limit).all())


def test_a_later_token_cannot_change_an_earlier_output() -> None:
    """Bit-identical prefix through the whole layer, not only through the softmax."""
    torch.manual_seed(32)
    layer = _dsa()
    x = torch.randn(1, SEQ, D_MODEL)
    y = layer(x)
    moved = x.clone()
    moved[:, 24] += 1.0
    assert torch.equal(y[:, :24], layer(moved)[:, :24])
    assert not torch.equal(y[:, 24], layer(moved)[:, 24])


def test_the_window_covering_the_sequence_is_dense_attention_bit_for_bit() -> None:
    """`window >= T` reads every visible position, so the sparse path must equal it.

    With k=window=T the indexer still selects k positions, and every one of them is
    masked out because the window already holds it: the equality is the proof that
    the selection branch adds an index list and no arithmetic.
    """
    torch.manual_seed(33)
    layer = _dsa(k=SEQ, window=SEQ)
    x = torch.randn(1, SEQ, D_MODEL)
    sparse = layer(x)
    assert torch.equal(sparse, layer.dense(x))
    q, key, value = layer._project(x)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
        scale=layer.scale,
    ).transpose(1, 2)
    dense = layer.proj(reference.reshape(1, SEQ, layer.d_kv))
    print(f"\ndegeneracy: max|sparse - sdpa| = {(sparse - dense).abs().max().item():.3e}")
    assert torch.allclose(sparse, dense, atol=1e-6)


def test_a_budget_that_is_the_whole_window_is_refused() -> None:
    """`k >= T` with a window that does not already cover T is dense in disguise."""
    layer = _dsa(k=SEQ, window=WINDOW)
    with pytest.raises(LoudError) as caught:
        layer(torch.zeros(1, SEQ, D_MODEL))
    assert "dense attention with extra steps" in str(caught.value)


def test_a_budget_under_the_window_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        _dsa(k=4, window=8)
    assert "k=4 < window=8" in str(caught.value)


def test_empty_widths_and_budgets_are_refused() -> None:
    with pytest.raises(LoudError, match="positive widths"):
        DSA(0, 1, HEAD_DIM)
    with pytest.raises(LoudError, match="selects nothing"):
        _dsa(k=0)
    with pytest.raises(LoudError, match="keeps no recent token"):
        _dsa(window=0)


def test_the_indexer_refuses_a_bottleneck_of_no_channels() -> None:
    with pytest.raises(LoudError) as caught:
        _dsa(d_idx=0)
    assert "projects to nothing" in str(caught.value)


def test_mli_reuses_the_layer_below_without_scanning() -> None:
    """The same answer with no indexer pass: `dsa_index_macs` stays at zero."""
    torch.manual_seed(34)
    scanning = _dsa()
    x = torch.randn(1, SEQ, D_MODEL)
    prev = scanning.indexer.select(x, K)
    reusing = _dsa(mli=True)
    reusing.load_state_dict(scanning.state_dict())
    counters = Counters()
    reused = reusing(x, counters, prev=prev)
    assert torch.equal(reused, scanning(x))
    assert counters.counts["dsa_mli_reuse"] == SEQ
    assert counters.counts["dsa_index_macs"] == 0


def test_mli_without_a_previous_selection_is_refused() -> None:
    with pytest.raises(LoudError) as caught:
        _dsa(mli=True)(torch.zeros(1, SEQ, D_MODEL))
    assert "no previous selection" in str(caught.value)


def test_mli_refuses_a_selection_of_the_wrong_budget_or_from_the_future() -> None:
    layer = _dsa()
    with pytest.raises(LoudError, match="keeps 4 tokens"):
        layer.indexer.reuse(torch.zeros(1, SEQ, 4, dtype=torch.long), k=K)
    future = torch.zeros(1, SEQ, K, dtype=torch.long)
    future[:, 4, 0] = 9
    with pytest.raises(LoudError, match="pointing after their query"):
        layer.indexer.reuse(future, k=K)


def test_mli_refuses_an_overlap_under_the_floor() -> None:
    """The paper's premise is a measurement: disagreeing layers may not share."""
    torch.manual_seed(35)
    layer = _dsa()
    mine = layer.indexer.select(torch.randn(1, SEQ, D_MODEL), K)
    theirs = layer.indexer.select(torch.randn(1, SEQ, D_MODEL), K)
    rate = layer.indexer.overlap(mine, theirs)
    print(f"\noverlap of two independent selections at seq={SEQ}: {rate:.2f}")
    assert rate < MLI_MIN_OVERLAP
    with pytest.raises(LoudError) as caught:
        layer.indexer.reuse(theirs, k=K, own=mine)
    assert "overlap" in str(caught.value)


def test_the_coeffects_declare_the_sparse_state() -> None:
    declared = _dsa().coeffects()
    assert declared.requires["attn.selector"] == "true"
    assert declared.provides["attn.state"] == "sparse"


def test_the_repr_names_the_sparsity_and_the_indexer() -> None:
    text = repr(_dsa())
    assert "k=8" in text and "window=4" in text and "d_idx=16" in text
