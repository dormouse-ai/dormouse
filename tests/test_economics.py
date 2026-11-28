"""The trade rule: time saved is only worth the quality that data buys back.

The three anchors are the owner's decisions, and they must come out of the formula rather
than out of a table: 1.30x of speed pays for 2.2 % of loss, 2.00x for 5.9 %, 2.50x for
7.7 %. The last test proves the plan actually reads a journal instead of always using the
constant — a rule nobody measures against is a comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dormouse_core.journal import Journal

from dormouse.cli import main
from dormouse.economics import (
    ELASTICITY,
    Lever,
    break_even_loss,
    measured_elasticity,
    recovery_multiple,
    verdict,
)
from dormouse.model.assemble import levers, plan_lines
from dormouse.model.spec import StackSpec

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_the_three_break_even_points_come_out_of_the_formula() -> None:
    assert recovery_multiple(2.2) == pytest.approx(1.30, abs=0.02)
    assert recovery_multiple(5.9) == pytest.approx(2.00, abs=0.02)
    assert recovery_multiple(7.7) == pytest.approx(2.50, abs=0.02)
    assert break_even_loss(1.30) == pytest.approx(2.2, abs=0.05)
    assert break_even_loss(2.00) == pytest.approx(5.9, abs=0.05)
    assert break_even_loss(2.50) == pytest.approx(7.7, abs=0.05)


def test_a_lever_with_no_loss_is_always_taken() -> None:
    for speedup in (1.0, 1.01, 1.3, 4.0):
        assert verdict(speedup, 0.0) == "take"


def test_the_rank_ladder_is_cost_neutral() -> None:
    """192 buys 1.30x for 3.0 %, and 3.0 % costs 1.43x of data: it refuses."""
    assert recovery_multiple(3.0) == pytest.approx(1.43, abs=0.01)
    assert Lever("expert_tsct_rank=192", 1.30, 3.0).verdict() == "refuse"


def test_an_unmeasured_lever_is_never_taken() -> None:
    assert Lever("weight_qat=ternary").verdict() == "unmeasured"
    assert "unmeasured" in Lever("weight_qat=ternary").row()


def test_the_plan_prices_every_lever_the_shipped_config_sets() -> None:
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    rows = {lever.name: lever.row() for lever in levers(spec)}
    assert rows["weight_qat=ternary_native"].endswith("unmeasured")
    assert rows["act_quant=a4"].endswith("unmeasured")
    assert rows["torch_compile=False"].endswith("take")
    assert "expert_tsct_rank=256 | 1.00\u00d7 | 0.0 % | take" in rows.values()
    text = "\n".join(plan_lines(spec))
    assert "lever | speedup | loss | verdict" in text
    assert f"elasticity {ELASTICITY:.4f} (constant)" in text


def test_a_proposed_rung_prints_refuse() -> None:
    rows = [lever.row() for lever in levers(StackSpec(expert_tsct_rank=192))]
    assert "expert_tsct_rank=192 | 1.30\u00d7 | 3.0 % | refuse" in rows


def _journal_with_two_evals(path: Path) -> Path:
    """16x the tokens halving the loss: elasticity `ln 2 / ln 16 = 0.25`."""
    journal = Journal.open(path)
    journal.write("eval", step=1, tokens=1_000_000_000, loss=2.0)
    journal.write("eval", step=8, tokens=16_000_000_000, loss=1.0)
    return path


def test_a_measured_elasticity_changes_a_borderline_verdict(tmp_path: Path) -> None:
    measured = measured_elasticity(Journal.open(_journal_with_two_evals(tmp_path / "run.jsonl")))
    assert measured is not None
    assert measured == pytest.approx(0.25, abs=1e-9)
    spec = StackSpec(expert_tsct_rank=192)
    constant = next(line for line in plan_lines(spec) if "expert_tsct_rank=192" in line)
    assert constant.endswith("refuse")
    measured_row = next(
        line
        for line in plan_lines(spec, elasticity=measured, elasticity_from="measured")
        if "expert_tsct_rank=192" in line
    )
    assert measured_row.endswith("take")


def test_dorm_plan_reads_the_journal_and_says_which_elasticity_it_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = _journal_with_two_evals(tmp_path / "run.jsonl")
    config = tmp_path / "rank192.toml"
    config.write_text("expert_tsct_rank = 192\n", encoding="utf-8")
    assert main(["plan", str(config)]) == 0
    without = capsys.readouterr().out
    assert "(constant)" in without
    assert next(line for line in without.splitlines() if "expert_tsct_rank=192" in line).endswith(
        "refuse"
    )

    assert main(["plan", str(config), "--journal", str(journal)]) == 0
    with_journal = capsys.readouterr().out
    assert f"(measured from {journal})" in with_journal
    assert next(
        line for line in with_journal.splitlines() if "expert_tsct_rank=192" in line
    ).endswith("take")


def test_a_journal_without_two_evals_refuses_instead_of_guessing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = Journal.open(tmp_path / "run.jsonl")
    journal.write("eval", step=1, tokens=1_000_000_000, loss=2.0)
    config = tmp_path / "rank192.toml"
    config.write_text("expert_tsct_rank = 192\n", encoding="utf-8")
    assert main(["plan", str(config), "--journal", str(journal.path)]) == 2
    assert "Escape" in capsys.readouterr().err
