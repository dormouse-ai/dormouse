"""The spec's refusals, one per conflict, each naming both members of the pair.

The point of these tests is not that a bad config raises — it is that the message
contains the two values that cannot coexist, because a refusal that says "invalid
configuration" costs the reader an hour and this project's defect history is exactly
those hours. Every assertion here looks for both names in the same message.

The spec is ONE architecture. What is checked here is that the menu around it is
gone: `attn` names the stack or one of its two mechanisms and nothing else, and a
config that still spells a deleted mechanism is told the two names that exist.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

from dormouse.data.tokenizer import DEFAULT_VOCAB
from dormouse.model.registry import pieces
from dormouse.model.spec import (
    ACT_QUANT,
    ATTN,
    OBJECTIVES,
    OPTIMIZERS,
    UNITS,
    WEIGHT_QAT,
    StackSpec,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _refused(**slots: object) -> str:
    with pytest.raises(LoudError) as caught:
        StackSpec.from_mapping(dict(slots))
    return str(caught.value)


@pytest.mark.parametrize(
    ("key", "value", "options"),
    [
        ("unit", "char", UNITS),
        ("attn", "mla", ATTN),
        ("optimizer", "sgd", OPTIMIZERS),
        ("objective", "mlm", OBJECTIVES),
        ("weight_qat", "int2", WEIGHT_QAT),
        ("act_quant", "int8", ACT_QUANT),
    ],
)
def test_an_unknown_literal_names_the_alternatives(
    key: str, value: str, options: tuple[str, ...]
) -> None:
    message = _refused(**{key: value})
    assert f"{key}={value!r}" in message
    for option in options:
        assert option in message


@pytest.mark.parametrize("deleted", ["csa", "hca", "full", "dense"])
def test_a_deleted_attention_mechanism_is_refused_with_the_two_that_exist(deleted: str) -> None:
    message = _refused(attn=deleted)
    assert f"attn={deleted!r}" in message
    assert "kda" in message and "dsa" in message


def test_the_ffn_menu_is_gone_and_naming_a_dense_arm_is_an_unknown_key() -> None:
    for key in ("ffn", "activation", "attn_pattern", "attn_pos"):
        message = _refused(**{key: "gated"})
        assert f"unknown key(s) {key}" in message


def test_fp8_is_known_and_refused_because_a_weight_is_quantized_once() -> None:
    message = _refused(weight_qat="fp8")
    assert "fp8" in message and "ternary" in message and "none" in message


def test_a_dense_expert_is_refused_and_the_rank_is_named() -> None:
    message = _refused(expert_tsct_rank=0)
    assert "expert_tsct_rank=0" in message and "256" in message


def test_the_objective_with_a_second_loss_is_refused_with_its_reason() -> None:
    for objective in ("ntp+tst", "ntp+jepa"):
        message = _refused(objective=objective)
        assert objective in message and "no loss module computes it" in message


def test_tst_over_a_multi_source_mix_names_both_the_objective_and_the_stream() -> None:
    message = _refused(objective="ntp+tst", sources="corpus,code")
    assert "ntp+tst" in message and "corpus,code" in message
    assert "one contiguous stream" in message


def test_int4_with_compile_names_the_measured_graph_break() -> None:
    message = _refused(act_quant="int4", torch_compile=True)
    assert "act_quant=int4" in message and "torch_compile=true" in message
    assert "11 breaks with int4 against 10 without" in message


def test_a_layer_count_the_period_cannot_tile_is_refused_with_the_remainder() -> None:
    with pytest.raises(LoudError) as caught:
        StackSpec(attn="hybrid", n_layers=6)
    assert "2 layers are left over" in str(caught.value)


def test_a_key_value_head_count_that_does_not_tile_the_queries_is_refused() -> None:
    message = _refused(n_heads=8, n_kv_heads=3)
    assert "n_kv_heads=3" in message and "tile" in message


def test_a_bottleneck_wider_than_the_stream_is_refused() -> None:
    message = _refused(c_lat=1024, d_model=832)
    assert "c_lat=1024" in message and "d_model=832" in message


def test_an_attention_budget_inside_the_window_is_refused() -> None:
    message = _refused(attn_k=64, attn_window=128)
    assert "attn_k=64" in message and "attn_window=128" in message


def test_an_unknown_key_is_an_error_not_a_no_op() -> None:
    message = _refused(ffn_act="moa")
    assert "unknown key(s) ffn_act" in message


def test_a_wrong_type_is_refused_instead_of_coerced() -> None:
    assert "expects int" in _refused(n_layers="12")
    assert "got bool" in _refused(n_layers=True)


def test_a_uniform_stack_is_one_mechanism_and_a_hybrid_is_three_to_one() -> None:
    hybrid = StackSpec()
    assert hybrid.pattern == ("kda", "kda", "kda", "dsa") * 3
    assert StackSpec(attn="kda").pattern == ("kda",) * 12
    assert StackSpec(attn="dsa").pattern == ("dsa",) * 12


def test_the_mechanisms_own_coeffects_refuse_a_rope_context() -> None:
    spec = StackSpec(n_layers=4)
    with pytest.raises(LoudError) as caught:
        pieces(spec, ctx={"attn.pos": "rope", "attn.rule": "delta", "attn.selector": "true"})
    message = str(caught.value)
    assert "attn.pos='none'" in message and "'rope'" in message
    assert "no attention mechanism here rotates positions" in message


def test_the_unit_prices_the_token_table() -> None:
    assert StackSpec(unit="byte").vocab == 256
    assert StackSpec(unit="bpe").vocab == DEFAULT_VOCAB


def test_the_one_config_is_the_architecture() -> None:
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    assert (spec.unit, spec.attn, spec.optimizer) == ("bpe", "hybrid", "muon")
    assert (spec.objective, spec.weight_qat, spec.act_quant) == ("ntp", "none", "none")
    assert spec.pattern.count("kda") == 9 and spec.pattern.count("dsa") == 3
    assert (spec.n_heads, spec.n_kv_heads, spec.c_lat) == (8, 2, 512)
    assert (spec.n_experts, spec.topk, spec.n_shared) == (64, 2, 1)
    assert (spec.expert_tsct_rank, spec.expert_arm) == (256, "moa")
    assert (spec.attn_k, spec.attn_window) == (512, 128)


def test_the_other_config_files_are_gone() -> None:
    """One model: a second architecture would wear a filename, and there is none.

    `smoke.toml` is not a second architecture — it is the CI seam test's stack, at the
    smallest widths the arms accept, and `train/entry.py` prices its step from the bill
    because it sits below the expert floor `dorm plan` refuses to price. The A/B arm is
    this same file with two values changed; `test_ab_configs.py` holds that line.
    """
    assert sorted(p.name for p in CONFIGS.glob("*.toml")) == [
        "dormouse_1b.toml",
        "dormouse_1b_top1_r192.toml",
        "smoke.toml",
    ]


def test_a_config_is_a_value_and_a_copy_can_be_scaled() -> None:
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    small = replace(
        spec,
        d_model=64,
        n_heads=4,
        head_dim=16,
        c_lat=32,
        n_layers=4,
        d_ffn=128,
        n_experts=4,
        attn_k=64,
        attn_window=32,
    )
    assert small.as_dict()["d_ffn"] == 128
    assert small.pattern == ("kda", "kda", "kda", "dsa")
    assert small.vocab == DEFAULT_VOCAB
