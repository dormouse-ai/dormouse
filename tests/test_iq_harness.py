"""The IQ harness has to prove its own floor, ceiling and refusal gates.

A deterministic known-answer mock scores one on every suite; a seeded random
floor scores zero with a reason on every task; an unexplained zero is refused
at construction, and an enabled judge that never ran is a MissingReport.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.eval.iq import (
    SCHEMA_VERSION,
    BlindJudge,
    MockJudge,
    Score,
    Task,
    build_suite,
    evaluate,
    from_fixture,
    score,
)
from dormouse.eval.model_adapter import EchoAdapter, KnownAdapter, RandomAdapter

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "iq_eval.py"
_spec = importlib.util.spec_from_file_location("iq_eval", _TOOL)
assert _spec is not None and _spec.loader is not None
iq_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iq_eval)

SUITES = ("smoke", "full")


@pytest.mark.parametrize("name", SUITES)
def test_known_mock_scores_one_on_every_suite(name: str) -> None:
    suite = build_suite(name)
    rows = evaluate(suite.tasks, KnownAdapter(suite.tasks), MockJudge())
    assert rows
    assert all(row.score == 1.0 for row in rows)


@pytest.mark.parametrize("name", SUITES)
def test_random_floor_scores_zero_with_a_reason(name: str) -> None:
    suite = build_suite(name)
    rows = evaluate(suite.tasks, RandomAdapter(0), MockJudge())
    assert rows
    assert all(row.score == 0.0 and row.reason for row in rows)


def test_zero_without_reason_is_refuted() -> None:
    with pytest.raises(LoudError, match="score 0 without reason"):
        Score(0.0, "   ")
    assert Score(1.0, "").value == 1.0


def test_missing_counter_gate() -> None:
    """An enabled judge whose count stayed zero is a MissingReport, not a clean run."""
    suite = build_suite("smoke")
    counters = Counters()
    with pytest.raises(MissingReport, match="blind_judge"):
        evaluate((), KnownAdapter(suite.tasks), MockJudge(), counters=counters)
    rows = evaluate(suite.tasks, KnownAdapter(suite.tasks), MockJudge(), counters=counters)
    assert counters.snapshot()["iq_judge_calls"] == len(rows) == len(suite)


def test_json_schema_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert iq_eval.main(["--model", "known", "--suite", "smoke", "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows
    for row in rows:
        assert set(row) == {"schema_version", "task", "score", "reason"}
        assert row["schema_version"] == SCHEMA_VERSION == 1
        assert row["score"] == 1.0


def test_json_zero_rows_carry_a_reason(capsys: pytest.CaptureFixture[str]) -> None:
    assert iq_eval.main(["--model", "mock", "--suite", "smoke", "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows and all(row["score"] == 0.0 and row["reason"] for row in rows)


def test_cli_text_mode_and_missing_checkpoint(capsys: pytest.CaptureFixture[str]) -> None:
    assert iq_eval.main(["--model", "known", "--suite", "smoke"]) == 0
    assert "suite smoke: tasks=" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        iq_eval.main(["--model", "checkpoint"])


def test_suites_are_deterministic() -> None:
    for name in SUITES:
        first, second = build_suite(name, seed=7), build_suite(name, seed=7)
        assert first == second
        assert [(task.prompt, task.expected, task.kind) for task in first] == [
            (task.prompt, task.expected, task.kind) for task in second
        ]
        assert len(list(first)) == len(first) == len(second)


def test_unknown_suite_is_refused() -> None:
    with pytest.raises(LoudError, match="unknown suite"):
        build_suite("nope")


def test_a_fixture_without_usable_sentences_is_refuted(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("Short.\ngo go go go go go go go go go.\n", encoding="utf-8")
    with pytest.raises(LoudError, match="usable sentences"):
        from_fixture(path, n=1)


def test_exact_scoring_normalizes_case_space_and_punctuation() -> None:
    task = Task("q", "42", "exact")
    assert score(task, " 42. ").value == 1.0
    failed = score(task, "43")
    assert failed.value == 0.0 and "43" in failed.reason


def test_choice_scoring_accepts_a_trailing_paren() -> None:
    task = Task("q", "B", "choice", ("A", "B", "C"))
    assert score(task, "B)").value == 1.0
    assert score(task, "C").value == 0.0


def test_code_scoring_compiles_and_looks_for_the_snippet() -> None:
    task = Task("q", "x * 2", "code")
    assert score(task, "```python\nx * 2\n```").value == 1.0
    assert "not valid Python" in score(task, "return x * 2").reason
    assert "missing" in score(task, "y * 3").reason


def test_an_unknown_kind_is_refuted() -> None:
    with pytest.raises(LoudError, match="unknown task kind"):
        score(Task("q", "a", "bogus"), "a")  # type: ignore[arg-type]


def test_a_pluggable_judge_is_used_verbatim() -> None:
    class StubJudge:
        def judge(self, task: Task, answer: str) -> Score:
            return Score(1.0, "stub verdict")

    judge: BlindJudge = StubJudge()
    rows = evaluate([Task("q", "a", "exact")], EchoAdapter(), judge)
    assert rows[0].reason == "stub verdict"


def test_known_adapter_miss_is_empty() -> None:
    assert KnownAdapter(()).generate("never seen") == ""


def test_random_adapter_is_replayable() -> None:
    first, second = RandomAdapter(3), RandomAdapter(3)
    assert [first.generate("x") for _ in range(3)] == [second.generate("x") for _ in range(3)]
