"""The FFN degeneracy gate: one dictionary entry IS the plain activation, bit for bit.

`LaFFN`/`MoaFFN` are only admissible because `K=1` collapses exactly: a mixture that
is merely close to the plain FFN is a mixture wired wrong, and every later number it
produces would be measured against a different model than the control. `torch.equal`,
not `allclose`, at the module and at the logits.
"""

from __future__ import annotations

import pytest
import torch
from stacks import REFERENCE, arm_declarations, reference_stack

from dormouse.model.ffn import PROJ_STD, GatedFFN, LaFFN, MoaFFN
from dormouse.model.ffn_spec import FFNSpec

D, F = 12, 8
#: 3·12·8 = 288 = 2·12·12: the neutral pair the `la`/`moa` width rule wants.
GELU_F = 12
SMALL = {
    "vocab": 16,
    "d_model": D,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 6,
    "d_ffn": F,
    "max_seq_len": 8,
}

#: The stack's widths, as `reference_stack` spells them.
STACK = {
    "d_model": D,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 6,
    "d_ffn": F,
    "max_seq_len": 8,
}


def _copy(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Load `source`'s weights into `target`; the extra mixture parameters stay put."""
    target.load_state_dict(source.state_dict(), strict=False)


@pytest.mark.parametrize("name", ["silu", "situ"])
def test_a_one_entry_la_is_the_plain_ffn_bit_for_bit(name: str) -> None:
    """`["silu"]` and `["situ"]` at `c₀ = 1` reproduce `down(f(gate(x))·up(x))`."""
    torch.manual_seed(0)
    plain = GatedFFN(D, F, act=name)
    la = LaFFN(D, F, dictionary=(name,))
    _copy(la, plain)
    x = torch.randn(2, 5, D)
    assert torch.equal(la(x), plain(x))


@pytest.mark.parametrize("sides", ["one", "bi"])
def test_a_one_entry_moa_is_the_plain_ffn_on_either_topology(sides: str) -> None:
    """The normalised sigmoid leaves `sigmoid/sigmoid = 1` exactly, so `K=1` collapses."""
    torch.manual_seed(1)
    plain = GatedFFN(D, F, sides=sides)
    moa = MoaFFN(D, F, dictionary=("silu",), sides=sides)
    _copy(moa, plain)
    x = torch.randn(2, 5, D)
    assert torch.equal(moa(x), plain(x))


@pytest.mark.parametrize("act", ["la", "moa"])
def test_a_one_entry_model_produces_the_plain_logits_bit_for_bit(act: str) -> None:
    """The gate the paper's claim is judged on: same weights, same logits."""
    torch.manual_seed(2)
    plain = reference_stack(**STACK)
    arm = reference_stack(
        **STACK,
        ffn=FFNSpec(
            D,
            F,
            act=act,
            dictionary=["silu"],
            sides="one" if act == "moa" else None,
            d_ffn_gelu=GELU_F,
        ),
        declarations=REFERENCE + arm_declarations(act),
    )
    result = arm.load_state_dict(plain.state_dict(), strict=False)
    extra = {"blocks.0.ffn.coef", "blocks.1.ffn.coef"}
    if act == "moa":
        extra |= {f"blocks.{i}.ffn.{p}" for i in range(2) for p in ("mix.weight", "mix.bias")}
    assert set(result.missing_keys) == extra
    assert result.unexpected_keys == []
    ids = torch.randint(0, SMALL["vocab"], (2, 6))
    assert torch.equal(arm(ids), plain(ids))


def test_a_bi_moa_model_produces_the_two_sided_plain_logits_bit_for_bit() -> None:
    """The default topology has no plain config arm, so its control is built here."""
    torch.manual_seed(3)
    moa = reference_stack(
        **STACK,
        ffn=FFNSpec(D, F, act="moa", dictionary=["silu"], d_ffn_gelu=GELU_F),
        declarations=REFERENCE + arm_declarations("moa"),
    )
    plain = reference_stack(**STACK)
    for block in plain.blocks:
        write_std = PROJ_STD / (2.0 * SMALL["n_layers"]) ** 0.5
        block.ffn = GatedFFN(D, F, sides="bi", write_std=write_std)
    plain.load_state_dict(moa.state_dict(), strict=False)
    ids = torch.randint(0, SMALL["vocab"], (1, 5))
    assert torch.equal(moa(ids), plain(ids))
