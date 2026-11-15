"""The stack, built from the two configs, priced, trained for three steps, and gated.

Three contracts live here, and each one is a number rather than an opinion:

* `control.toml` builds a parameter count that equals the arithmetic written out by hand
  (`4·d²` for attention, `3·d·d_ffn` for the FFN, `2·d` per block for the norms) — so a
  builder that forgets a matrix disagrees with the formula instead of quietly costing
  less;
* `kitchen_sink.toml` stores about a billion parameters and runs at most 60M per token,
  which is the contract that makes the stack trainable on one card;
* both stacks take three CPU steps with a finite loss, and every enabled mechanism
  moved its counter — no `MissingReport`, and no mechanism enabled and silent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

#: The stack a config names, at widths a CPU test can afford. Every slot is kept; only
#: the widths shrink, so the wiring and the refusals are the ones the config asks for.
SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "head_dim": 16,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}


def _smoke(name: str) -> StackSpec:
    return replace(StackSpec.from_toml(CONFIGS / name), **SMOKE)


def _control_params(spec: StackSpec) -> int:
    """The control stack's bill, written from the module widths and nothing else."""
    embed = spec.vocab * spec.d_model
    norms = (2 * spec.n_layers + 1) * spec.d_model
    attention = spec.n_layers * 4 * spec.d_model**2
    ffn = spec.n_layers * 3 * spec.d_model * spec.d_ffn
    return embed + norms + attention + ffn


def test_the_control_builds_the_parameter_count_the_arithmetic_promises() -> None:
    spec = StackSpec.from_toml(CONFIGS / "control.toml")
    report = assemble(spec).report()
    expected = _control_params(spec)
    print(f"\ncontrol: {report}")
    print(f"control: formula={expected:,} tree={report['stored']:,}")
    assert report["stored"] == expected
    assert report["active"] == expected  # a dense stack runs every parameter
    assert report["head"] == 0  # the head is tied; a nonzero here is a second matrix
    assert report["ffn"] == spec.n_layers * 3 * spec.d_model * spec.d_ffn


def test_the_kitchen_sink_stores_a_billion_and_runs_under_sixty_million() -> None:
    spec = StackSpec.from_toml(CONFIGS / "kitchen_sink.toml")
    report = assemble(spec, device="meta").report()
    print(f"\nkitchen_sink: stored={report['stored']:,} active={report['active']:,}")
    print(f"kitchen_sink: groups={report}")
    assert 0.95e9 <= report["stored"] <= 1.10e9
    assert 40e6 <= report["active"] <= 60e6
    # The number the card cares about, as a hard gate rather than a printout.
    assert report["active"] < 60_000_000
    assert report["active"] < report["stored"] / 10


def test_the_bank_is_one_module_shared_by_every_layer() -> None:
    spec = _smoke("kitchen_sink.toml")
    assembled = assemble(spec, smoke=True)
    banks = [module for module in assembled.model.modules() if type(module).__name__ == "FlatBank"]
    inner = {id(module.bank) for module in banks}
    assert len(banks) == spec.n_layers
    assert len(inner) == 1, "a bank per layer would store and run n_layers times as much"


def test_the_kitchen_sink_plan_names_the_stack_and_its_mechanisms() -> None:
    lines = plan_lines(StackSpec.from_toml(CONFIGS / "kitchen_sink.toml"))
    text = "\n".join(lines)
    print("\n" + text)
    for expected in (
        "hybrid 3:1",
        "kda, kda, kda, csa",
        "experts: moa",
        "weight_qat=ternary",
        "act_quant=int4",
        "stored=1,010,891,710",
        "741.6h  train",
        "moe→moe_fwd",
        "ntp→ntp_loss",
    ):
        assert expected in text, expected


@pytest.mark.parametrize("name", ["control.toml", "kitchen_sink.toml"])
def test_three_cpu_steps_on_a_config_leave_every_enabled_mechanism_reported(name: str) -> None:
    spec = _smoke(name)
    assembled = assemble(spec, smoke=True)
    optimizer = MixOptimizer(assembled.model, steps=2)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    for step in range(3):
        result, counters = assembled.step(ids, optimizer=optimizer)
        assert not result.skipped, f"step {step} tripped the NaN firewall"
        assert torch.isfinite(torch.tensor(result.loss)), f"step {step}: loss={result.loss}"
    print(f"\n{name}: {counters.line()}")
    print(f"{name}: enabled={sorted(counters.enabled)}")
    assert counters.silent() == []
    declared = {piece.mechanism for piece in assembled.pieces}
    assert declared <= counters.enabled, declared - counters.enabled
    if name == "kitchen_sink.toml":
        assert {"kda", "csa", "moe", "ffn_moa", "weight_qat", "act_quant"} <= counters.enabled
    else:
        assert {"attention", "rope", "ffn"} <= counters.enabled


def test_a_mechanism_that_is_enabled_and_silent_still_raises() -> None:
    """The gate above is only worth anything if it can fail."""
    from dormouse_core.counters import Counters, MissingReport

    counters = Counters()
    counters.enable("ternary", counter="weight_ternary")
    with pytest.raises(MissingReport):
        counters.check()


def test_the_assemble_side_refuses_the_same_conflicts_before_allocating() -> None:
    with pytest.raises(LoudError) as caught:
        assemble(StackSpec(attn="kda", attn_pos="rope", n_layers=4))
    assert "attn.pos" in str(caught.value) and "kda" in str(caught.value)


def test_dorm_plan_is_wired_and_train_is_still_refused(capsys: pytest.CaptureFixture[str]) -> None:
    from dormouse.cli import main

    assert main(["plan", str(CONFIGS / "kitchen_sink.toml")]) == 0
    printed = capsys.readouterr().out
    assert "params: stored=1,010,891,710" in printed
    assert "mechanisms:" in printed
    assert main(["train", "configs/control.toml"]) == 2
    assert "not wired yet" in capsys.readouterr().err


def test_plan_needs_no_allocation_so_a_billion_parameter_stack_is_cheap() -> None:
    """The plan builds on `meta`: the same numbers, no 4 GiB of weights."""
    spec = StackSpec.from_toml(CONFIGS / "kitchen_sink.toml")
    measured = assemble(spec, device="meta").report()
    assert measured["stored"] > 1e9
    assert all(isinstance(value, int) for value in measured.values())
