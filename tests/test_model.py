"""The stack: what it computes, what it reports, and that every weight learns.

Three invariants, each one a defect class from this project's history:

* the counters — a mechanism that is enabled and silent is an error, not a log line;
* the gradient — every parameter group takes a non-zero gradient on step 1, so no
  matrix is decoration;
* the parameter report — computed from the module tree and cross-checked against
  `sum(p.numel())`, so the number quoted for a preset cannot drift from the model.

The model is a SHELL over prebuilt blocks (`assemble.assemble` is the only thing that
knows which attention mechanism a layer holds), so most of these build a two-block
stack through `stacks.reference_stack` — the dense anchor — and the assembled shape
is covered where the assembler is (`tests/test_assemble.py`).
"""

from __future__ import annotations

import math

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport
from stacks import reference_stack

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
    model = reference_stack()
    logits = model(torch.randint(0, SMALL["vocab"], (2, 5)))
    assert logits.shape == (2, 5, SMALL["vocab"])
    assert logits.dtype == torch.float32


def test_the_head_is_the_token_table_transposed() -> None:
    """Tying is the parameterisation: the logits' matrix is `embed.weight` itself."""
    model = reference_stack()
    h = torch.randn(7, SMALL["d_model"])
    assert torch.allclose(model.head(h), h @ model.embed.weight.T, atol=1e-6)


def test_every_enabled_mechanism_reports_itself() -> None:
    counters = Counters()
    model = reference_stack()
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
        "lns",
    }
    for mechanism in counters.enabled:
        assert counters.counts[counters.required[mechanism]] > 0, mechanism


def test_an_enabled_mechanism_that_never_runs_is_an_error() -> None:
    """The wiring test: enable a counter nothing increments and the run must die."""
    counters = Counters()
    counters.enable("tsct", counter="tsct_fwd", why="this is the wiring defect under test")
    model = reference_stack()
    with pytest.raises(MissingReport) as e:
        model(torch.zeros(1, 4, dtype=torch.long), counters)
    assert "tsct" in str(e.value)


def test_a_stack_without_declarations_declares_nothing_of_its_own() -> None:
    """The shell cannot infer which mechanisms ran; a caller that says nothing gets none."""
    counters = Counters()
    reference_stack(declarations=())(torch.zeros(1, 4, dtype=torch.long), counters)
    assert not {"attention", "ffn", "rope"} & set(counters.enabled)


def test_every_parameter_group_gets_a_non_zero_gradient() -> None:
    """A matrix that never learns is a matrix that should not have been built."""
    model = reference_stack()
    model(torch.randint(0, SMALL["vocab"], (2, 6))).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().max() > 0.0, name


def test_the_report_is_the_module_tree() -> None:
    model = reference_stack()
    report = model.param_report()
    assert report["total"] == sum(p.numel() for p in model.parameters())
    assert sum(n for k, n in report.items() if k != "total") == report["total"]
    assert report["head"] == 0  # tied


def test_the_report_puts_every_matrix_in_the_group_that_owns_it() -> None:
    """`attention` and `ffn` are named from the parameter paths, not from a table."""
    model = reference_stack()
    report = model.param_report()
    attention = {
        "blocks.0.attn.qkv.weight": (3 * 32 * 32),
        "blocks.0.attn.proj.weight": (32 * 32),
        "blocks.1.attn.qkv.weight": (3 * 32 * 32),
        "blocks.1.attn.proj.weight": (32 * 32),
    }
    assert report["attention"] == sum(attention.values())
    assert report["ffn"] == 2 * 3 * 32 * 64
    assert report["norms"] == (2 * 2 + 1) * 32
    assert report["embedding"] == 16 * 32


def test_two_forwards_of_the_same_weights_are_identical() -> None:
    model = reference_stack()
    ids = torch.randint(0, SMALL["vocab"], (1, 5))
    assert torch.equal(model(ids), model(ids))


def test_the_same_seed_builds_the_same_model() -> None:
    torch.manual_seed(7)
    a = reference_stack()
    torch.manual_seed(7)
    b = reference_stack()
    ids = torch.randint(0, SMALL["vocab"], (1, 3))
    assert torch.equal(a(ids), b(ids))


def test_the_embedding_starts_at_scaled_rms_and_projections_at_their_std() -> None:
    """The init contract the variance argument is written against."""
    model = reference_stack()
    expected = 1.0 / math.sqrt(model.d_model)
    assert model.embed.weight.std().item() == pytest.approx(expected, rel=0.2)
    qkv = model.blocks[0].attn.qkv.weight
    assert qkv.std().item() == pytest.approx(0.02, rel=0.2)


def test_a_one_symbol_vocabulary_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        reference_stack(vocab=1)
    assert "predicts nothing" in str(e.value)


def test_a_stack_with_no_blocks_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        Transformer(
            vocab=16,
            d_model=32,
            d_ffn=64,
            max_seq_len=32,
            blocks=[],
            declarations=(),
        )
    assert "at least one block" in str(e.value)


def test_a_one_dimensional_batch_is_refused() -> None:
    with pytest.raises(LoudError) as e:
        reference_stack()(torch.zeros(4, dtype=torch.long))
    assert "ids [batch, seq]" in str(e.value)


def test_float_ids_are_refused() -> None:
    with pytest.raises(LoudError) as e:
        reference_stack()(torch.zeros(1, 4))
    assert "integers" in str(e.value)
