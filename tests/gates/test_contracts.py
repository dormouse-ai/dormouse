"""The mechanism-contract gate: a technology cannot be enabled without a proof.

Three refusals are the point of this file:

* a mechanism with no anchor cannot be constructed;
* an anchor that disagrees with the mechanism, or that is a golden captured from
  our own run, fails the gate;
* an enabled mechanism whose counter stays zero is a `MissingReport` before a
  step can be called successful.

The three real mechanisms run their proofs here too, so these lines are under
the coverage floor rather than living in an unmeasured script.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.cli import main as dorm_main
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.verify import (
    REGISTRY,
    Anchor,
    AnchorSource,
    Contract,
    Fixture,
    Invariant,
    ProofError,
    bounded,
    checks,
    finite,
    shape,
)
from dormouse.verify.anchors import kda_paper
from dormouse.verify.contracts import (
    DSA_CONTRACT,
    KDA_CONTRACT,
    TSCT_CONTRACT,
    active_names,
)
from dormouse.verify.degenerations import kda_degeneration
from dormouse.verify.fixtures import kda_fixture
from dormouse.verify.registry import MechanismRegistry, registry_table

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


def test_every_registered_mechanism_passes_its_proofs() -> None:
    results = checks.check_all()
    assert {result.name for result in results} == {"kda", "dsa", "tsct"}
    assert all(result.ok for result in results), checks.render(results)
    printed = checks.render(results)
    print(f"\n{printed}")
    assert printed.count("PASS") == 3


def test_the_gate_is_green_with_proofs_and_red_without_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert checks.main([]) == 0
    assert "kda" in capsys.readouterr().out
    monkeypatch.setattr(checks, "REGISTRY", MechanismRegistry())
    assert checks.main([]) == 1  # nothing registered is not a green gate
    empty = Contract(
        "never",
        "never_fwd",
        lambda fx: (torch.ones(1), torch.ones(1)),
        Anchor(lambda fx: torch.ones(1), source=AnchorSource.PAPER, formula="x"),
        (finite(),),
    )
    checks.REGISTRY.register(empty)
    assert checks.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_a_mechanism_with_no_anchor_cannot_be_constructed() -> None:
    with pytest.raises(ProofError, match="no external anchor"):
        Contract(
            "fake",
            "fake_fwd",
            lambda fx: (torch.ones(1), torch.ones(1)),
            None,  # type: ignore[arg-type]
            (finite(),),
        )
    with pytest.raises(ProofError, match="must be callable"):
        Anchor(None, source=AnchorSource.PAPER, formula="x")  # type: ignore[arg-type]
    with pytest.raises(ProofError, match="not an external reference"):
        Anchor(lambda fx: torch.ones(1), source="my run", formula="x")  # type: ignore[arg-type]
    with pytest.raises(LoudError, match="without a formula"):
        Anchor(lambda fx: torch.ones(1), source=AnchorSource.PAPER, formula="  ")


def test_capturing_a_golden_from_our_own_run_is_refused() -> None:
    with pytest.raises(LoudError, match="cannot be an anchor"):
        Anchor.from_golden(torch.ones(3))
    captured = Anchor(
        lambda fx: torch.zeros(1), source=AnchorSource.PAPER, formula="saved from our run"
    )
    with pytest.raises(ProofError, match="captured golden"):
        checks.check_contract(replace(KDA_CONTRACT, anchor=captured))
    with pytest.raises(ProofError, match="not a tensor"):
        Anchor(lambda fx: 1.0, source=AnchorSource.NUMPY, formula="x").evaluate(Fixture(0))


def test_an_anchor_that_disagrees_fails_the_gate() -> None:
    wrong = replace(
        KDA_CONTRACT,
        anchor=Anchor(
            lambda fx: 2.0 * kda_paper(fx),
            source=AnchorSource.PAPER,
            formula="wrong on purpose: twice the paper formula",
        ),
    )
    result = checks.check_contract(wrong)
    assert not result.ok
    assert "anchor off by" in result.reason

    def drifts(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
        on, off = kda_degeneration(fixture)
        return on, off + 1.0

    result = checks.check_contract(replace(KDA_CONTRACT, degeneration=drifts))
    assert not result.ok and "degeneration off by" in result.reason


def test_a_wrong_shape_or_a_false_invariant_fails_the_gate() -> None:
    misshaped = Anchor(
        lambda fx: torch.ones(1) * fx.tensor("q").sum(),
        source=AnchorSource.FP64,
        formula="a different shape",
    )
    result = checks.check_contract(replace(KDA_CONTRACT, anchor=misshaped))
    assert not result.ok and "anchor off by inf" in result.reason
    broke = replace(
        TSCT_CONTRACT,
        invariants=(finite(), Invariant("always false", lambda out, fx: False)),
    )
    result = checks.check_contract(broke)
    assert not result.ok and "always false" in result.reason


def test_an_enabled_mechanism_with_a_zero_counter_refuses() -> None:
    counters = Counters()
    with pytest.raises(MissingReport, match="never reported: kda"):
        checks.check_ran(KDA_CONTRACT, counters)
    counters.count("kda_fwd", 1)
    checks.check_ran(KDA_CONTRACT, counters)
    assert counters.silent() == []


def test_an_active_mechanism_without_a_contract_refuses_before_allocation() -> None:
    with pytest.raises(LoudError, match="no contract"):
        REGISTRY.require("never_registered")
    saved = REGISTRY.all()
    try:
        REGISTRY.clear()
        with pytest.raises(LoudError, match="has no contract"):
            assemble(StackSpec(**SMOKE))  # type: ignore[arg-type]
    finally:
        for contract in saved:
            REGISTRY.register(contract)


def test_active_names_are_what_a_config_turns_on() -> None:
    assert active_names(["kda", "dsa"], 256) == ("kda", "dsa", "tsct")
    assert active_names(["kda"], 0) == ("kda",)
    assert active_names([], 0) == ()


def test_registry_reports_proofs_and_refuses_a_duplicate_name() -> None:
    registry = MechanismRegistry()
    registry.register(KDA_CONTRACT)
    registry.register(KDA_CONTRACT)  # importing a module twice is idempotent
    assert len(registry.all()) == 1
    assert "kda" in registry.table() and "kda_fwd" in registry.table()
    assert "dsa" in registry_table()
    with pytest.raises(LoudError, match="two contracts claim"):
        registry.register(replace(KDA_CONTRACT))
    registry.clear()
    assert registry.all() == ()


def test_fixtures_and_invariants_name_what_is_missing_or_broken() -> None:
    fixture = Fixture(7)
    with pytest.raises(ProofError, match="no array"):
        fixture.tensor("x")
    with pytest.raises(ProofError, match="no scalar"):
        fixture.number("rank")
    out = torch.ones(3)
    finite().check(out, fixture)
    shape(3).check(out, fixture)
    bounded(2.0).check(out, fixture)
    with pytest.raises(ProofError, match="invariant 'shape'"):
        shape(2).check(out, fixture)
    with pytest.raises(ProofError, match="invariant 'bounded"):
        bounded(0.5).check(out, fixture)


def test_contract_validation_refuses_an_unfalsifiable_proof() -> None:
    with pytest.raises(LoudError, match="needs a name"):
        replace(KDA_CONTRACT, name=" ")
    with pytest.raises(LoudError, match="no counter"):
        replace(KDA_CONTRACT, counter="")
    with pytest.raises(ProofError, match="no degeneration"):
        replace(KDA_CONTRACT, degeneration=None)  # type: ignore[arg-type]
    with pytest.raises(ProofError, match="no invariants"):
        replace(KDA_CONTRACT, invariants=())
    with pytest.raises(ProofError, match="repeats an invariant name"):
        replace(KDA_CONTRACT, invariants=(finite(), finite()))
    with pytest.raises(LoudError, match="unusable epsilon"):
        replace(KDA_CONTRACT, epsilon=math.inf)
    assert "kda" in repr(KDA_CONTRACT) and "paper formula" in repr(KDA_CONTRACT.anchor)


def test_no_fixture_or_no_contract_is_a_failed_check() -> None:
    with pytest.raises(ProofError, match="no CPU fixture"):
        checks.check_contract(
            Contract(
                "unfixtured",
                "unfixtured_fwd",
                lambda fx: (torch.ones(1), torch.ones(1)),
                Anchor(lambda fx: torch.ones(1), source=AnchorSource.PAPER, formula="x"),
                (finite(),),
            )
        )
    with pytest.raises(ProofError, match="no fixtures to check"):
        checks.check_contract(KDA_CONTRACT, fixtures=())


def test_doctor_and_plan_print_the_proof_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert dorm_main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "kda" in report["contracts"] and "dsa" in report["contracts"]
    config = Path(__file__).parents[2] / "configs" / "dormouse_1b.toml"
    assert dorm_main(["plan", str(config)]) == 0
    printed = capsys.readouterr().out
    assert "contracts (every enabled mechanism" in printed
    assert "kda_fwd" in printed


def test_every_proof_uses_a_different_seed_pair() -> None:
    """A second seed is what refutes a golden: the three anchors actually move."""
    for name, maker in checks.FIXTURES.items():
        contract = {"kda": KDA_CONTRACT, "dsa": DSA_CONTRACT, "tsct": TSCT_CONTRACT}[name]
        first = contract.anchor.evaluate(maker(0))
        second = contract.anchor.evaluate(maker(1))
        assert not torch.equal(first, second), name
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert kda_fixture(0).number("chunk") == 8
