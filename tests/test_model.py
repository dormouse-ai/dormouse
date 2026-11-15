"""The stack: what it computes, what it reports, and that every weight learns.

Three invariants, each one a defect class from this project's history:

* the counters — a mechanism that is enabled and silent is an error, not a log line;
* the gradient — every parameter group takes a non-zero gradient on step 1, so no
  matrix is decoration;
* the parameter report — computed from the module tree and cross-checked against
  `sum(p.numel())`, so the number quoted for a preset cannot drift from the model.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.model import Transformer

SMALL = {
    "vocab": 16,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 16,
    "d_ffn": 64,
    "max_seq_len": 32,
}


def test_logits_have_one_row_per_token_and_one_column_per_symbol() -> None:
    model = Transformer(**SMALL)
    logits = model(torch.randint(0, SMALL["vocab"], (2, 5)))
    assert logits.shape == (2, 5, SMALL["vocab"])
    assert logits.dtype == torch.float32


def test_the_head_is_the_token_table_transposed() -> None:
    """Tying is the parameterisation: the logits' matrix is `embed.weight` itself."""
    model = Transformer(**SMALL)
    h = torch.randn(7, SMALL["d_model"])
    assert torch.allclose(model.head(h), h @ model.embed.weight.T, atol=1e-6)


def test_every_enabled_mechanism_reports_itself() -> None:
    counters = Counters()
    model = Transformer(**SMALL)
    model(torch.zeros(1, 4, dtype=torch.long), counters)
    assert set(counters.enabled) == {
        "embedding",
        "blocks",
        "attention",
        "ffn",
        "norm",
        "head",
        "tied_head",
        "rope",
    }
    for mechanism in counters.enabled:
        assert counters.counts[counters.required[mechanism]] > 0, mechanism


def test_an_enabled_mechanism_that_never_runs_is_an_error() -> None:
    """The wiring test: turn RoPE on for a model built without it — the run must die."""
    counters = Counters()
    counters.enable("rope", counter="rope_fwd", why="this is the wiring defect under test")
    model = Transformer(**SMALL, rope=False)
    with pytest.raises(MissingReport) as e:
        model(torch.zeros(1, 4, dtype=torch.long), counters)
    assert "rope" in str(e.value)


def test_a_model_without_rope_does_not_declare_it() -> None:
    counters = Counters()
    Transformer(**SMALL, rope=False)(torch.zeros(1, 4, dtype=torch.long), counters)
    assert "rope" not in counters.enabled


def test_every_parameter_group_gets_a_non_zero_gradient() -> None:
    """A matrix that never learns is a matrix that should not have been built."""
    model = Transformer(**SMALL)
    model(torch.randint(0, SMALL["vocab"], (2, 6))).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().max() > 0.0, name


def test_the_report_is_the_module_tree() -> None:
    model = Transformer(**SMALL)
    report = model.param_report()
    assert report["total"] == sum(p.numel() for p in model.parameters())
    assert sum(n for k, n in report.items() if k != "total") == report["total"]
    assert report["head"] == 0  # tied


def test_the_live_recipe_costs_what_the_closed_form_says() -> None:
    """d=832, L=12, f=2048: built on `meta`, so the count is free of storage."""
    model = Transformer(device="meta")
    assert model.param_report() == {
        "embedding": 256 * 832,
        "attention": 12 * (3 * 832 * 832 + 832 * 832),
        "ffn": 12 * 3 * 832 * 2048,
        "norms": 12 * 2 * 832 + 832,
        "head": 0,
        "total": 94_802_240,
    }


def test_two_forwards_of_the_same_weights_are_identical() -> None:
    model = Transformer(**SMALL)
    ids = torch.randint(0, SMALL["vocab"], (1, 5))
    assert torch.equal(model(ids), model(ids))


def test_the_same_seed_builds_the_same_model() -> None:
    torch.manual_seed(7)
    a = Transformer(**SMALL)
    torch.manual_seed(7)
    b = Transformer(**SMALL)
    ids = torch.randint(0, SMALL["vocab"], (1, 3))
    assert torch.equal(a(ids), b(ids))


def test_the_embedding_starts_at_unit_rms_and_projections_at_their_std() -> None:
    """The init contract the variance argument is written against."""
    model = Transformer(**SMALL)
    assert model.embed.weight.std().item() == pytest.approx(1.0, rel=0.2)
    qkv = model.blocks[0].attn.qkv.weight
    assert qkv.std().item() == pytest.approx(0.02, rel=0.2)


def test_a_head_count_that_does_not_tile_the_width_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Transformer(**{**SMALL, "n_heads": 3})
    assert "tile" in str(e.value)


def test_a_one_symbol_vocabulary_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Transformer(**{**SMALL, "vocab": 1})
    assert "predicts nothing" in str(e.value)


def test_a_one_dimensional_batch_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Transformer(**SMALL)(torch.zeros(4, dtype=torch.long))
    assert "ids [batch, seq]" in str(e.value)


def test_float_ids_are_refused() -> None:
    with pytest.raises(LoudError) as e:
        Transformer(**SMALL)(torch.zeros(1, 4))
    assert "integers" in str(e.value)
