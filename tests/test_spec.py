"""The spec's refusals, one per conflict, each naming both members of the pair.

The point of these tests is not that a bad config raises — it is that the message
contains the two values that cannot coexist, because a refusal that says "invalid
configuration" costs the reader an hour and this project's defect history is exactly
those hours. Every assertion here looks for both names in the same message.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

from dormouse.model.registry import pieces
from dormouse.model.spec import (
    ACT_QUANT,
    ACTIVATIONS,
    ATTN,
    AUXES,
    FFNS,
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
        ("ffn", "glu", FFNS),
        ("activation", "gelu", ACTIVATIONS),
        ("optimizer", "sgd", OPTIMIZERS),
        ("objective", "mlm", OBJECTIVES),
        ("aux", "mtp", AUXES),
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


def test_fp8_is_known_and_refused_because_a_weight_is_quantized_once() -> None:
    message = _refused(weight_qat="fp8")
    assert "fp8" in message and "ternary" in message and "none" in message


def test_two_arms_for_the_ffn_slot_are_refused_by_name() -> None:
    message = _refused(ffn="moa", activation="moa")
    assert "ffn='moa'" in message and "activation='moa'" in message
    assert "one FFN slot" in message


def test_a_factorised_expert_is_refused_and_zero_is_what_builds() -> None:
    message = _refused(expert_tsct_rank=8)
    assert "expert_tsct_rank=8" in message and "use 0" in message
    assert StackSpec(expert_tsct_rank=0).is_moe is False


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


def test_a_pattern_without_a_hybrid_is_refused() -> None:
    message = _refused(attn="kda", attn_pattern="3:1")
    assert "attn_pattern='3:1'" in message and "attn='kda'" in message


@pytest.mark.parametrize(
    ("pattern", "n_layers", "expected"),
    [("3x1", 4, "is not a ratio"), ("0:0", 4, "names no layer"), ("3:1", 6, "left over")],
)
def test_a_pattern_that_does_not_tile_is_refused(
    pattern: str, n_layers: int, expected: str
) -> None:
    with pytest.raises(LoudError) as caught:
        StackSpec(attn="hybrid", attn_pattern=pattern, n_layers=n_layers)
    assert expected in str(caught.value) or "is not a ratio" in str(caught.value)


def test_an_unknown_key_is_an_error_not_a_no_op() -> None:
    message = _refused(ffn_act="moa")
    assert "unknown key(s) ffn_act" in message
    assert "ffn" in message


def test_a_wrong_type_is_refused_instead_of_coerced() -> None:
    assert "expects int" in _refused(n_layers="12")
    assert "got bool" in _refused(n_layers=True)


def test_kda_with_rope_is_refused_by_the_mechanisms_own_coeffects() -> None:
    spec = StackSpec(attn="kda", attn_pos="rope", n_layers=4)
    with pytest.raises(LoudError) as caught:
        pieces(spec)
    message = str(caught.value)
    assert "attn.pos='none'" in message and "'rope'" in message
    assert "set attn_pos='none'" in message


def test_full_attention_with_pos_none_is_refused_by_the_same_check() -> None:
    spec = StackSpec(attn="full", attn_pos="none")
    with pytest.raises(LoudError) as caught:
        pieces(spec)
    message = str(caught.value)
    assert "attn.pos='rope'" in message and "set attn_pos='rope'" in message


def test_the_pattern_is_what_makes_a_hybrid_hybrid() -> None:
    spec = StackSpec(attn="hybrid", n_layers=8)
    assert spec.pattern == ("kda", "kda", "kda", "csa") * 2
    assert spec.pos == "none"


def test_the_unit_prices_the_token_table() -> None:
    assert StackSpec(unit="byte").vocab == 256
    assert StackSpec(unit="bpe").vocab == 32768


def test_the_two_configs_differ_in_exactly_the_stack() -> None:
    control = StackSpec.from_toml(CONFIGS / "control.toml")
    sink = StackSpec.from_toml(CONFIGS / "kitchen_sink.toml")
    assert (control.attn, control.ffn, control.optimizer) == ("full", "gated", "muon")
    assert (control.objective, control.weight_qat, control.act_quant) == ("ntp", "none", "none")
    assert (sink.attn, sink.attn_pattern, sink.ffn, sink.activation) == (
        "hybrid",
        "3:1",
        "moe",
        "moa",
    )
    assert (sink.weight_qat, sink.act_quant, sink.optimizer) == ("ternary", "int4", "muon")
    shared = {"unit", "optimizer", "objective", "sources", "d_model", "n_heads", "head_dim"}
    for key in shared:
        assert getattr(control, key) == getattr(sink, key), key


def test_a_config_value_no_head_reads_is_refused_by_name() -> None:
    """`aux_k=4` with `aux='none'` is a number nothing reads: the four are checked."""
    message = _refused(aux_k=4)
    assert "aux='none'" in message and "aux_k=4" in message
    assert "no head is built" in message


def test_an_aux_horizon_with_no_target_is_refused() -> None:
    message = _refused(aux="lightmtp", aux_k=0)
    assert "aux_k=0" in message and "zero positions ahead" in message
    message = _refused(aux="lightmtp", aux_k=12, seq_len=8)
    assert "aux_k=12" in message and "seq_len=8" in message


def test_an_aux_weight_that_trains_nothing_is_refused() -> None:
    message = _refused(aux="lightmtp", aux_weight=0.0)
    assert "aux_weight=0.0" in message and "0.1 is the paper's λ_MTP" in message


def test_a_target_layer_outside_the_stack_is_refused() -> None:
    message = _refused(aux="lightmtp", aux_layer=13)
    assert "aux_layer=13" in message and "12-layer stack" in message
    assert "1..n_layers-1" in message


def test_the_lightmtp_config_is_the_control_plus_one_slot() -> None:
    """The first A/B is one variable, and this is the assertion that keeps it one."""
    control = StackSpec.from_toml(CONFIGS / "control.toml")
    auxed = StackSpec.from_toml(CONFIGS / "lightmtp.toml")
    assert (auxed.aux, auxed.aux_k, auxed.aux_layer, auxed.aux_varreg) == ("lightmtp", 8, 0, False)
    assert control.aux == "none"
    differing = {key for key in control.as_dict() if getattr(control, key) != getattr(auxed, key)}
    assert differing == {"aux"}, differing


def test_a_config_is_a_value_and_a_copy_can_be_scaled() -> None:
    spec = StackSpec.from_toml(CONFIGS / "control.toml")
    small = replace(spec, d_model=64, n_heads=4, head_dim=16, n_layers=2, d_ffn=128)
    assert small.as_dict()["d_ffn"] == 128
    assert small.pattern == ("full", "full")
