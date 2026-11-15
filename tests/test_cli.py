"""The CLI contract: a real command does something, a planned one refuses.

The test that matters is the last one — a command that is not wired must exit
non-zero with an escape, because "it ran and did nothing" is precisely the failure
mode this whole rewrite exists to remove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dormouse_core.journal import Journal

from dormouse import cli
from dormouse.cli import PLANNED, main


def test_doctor_reports_what_the_machine_has(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "python" in report and "disk_free_gb" in report
    assert isinstance(report["disk_free_gb"], float)


def test_status_reads_a_journal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.jsonl"
    j = Journal.open(path)
    j.write("start", config="c.toml")
    j.write("step", step=7)
    assert main(["status", str(path)]) == 0
    out = capsys.readouterr().out
    assert "running" in out
    assert "7" in out


def test_status_accepts_an_absolute_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "a3f2" / "run.jsonl"
    Journal.open(path).write("step", step=7)
    assert main(["status", str(tmp_path / "*" / "run.jsonl")]) == 0
    assert "a3f2" in capsys.readouterr().out


def test_status_with_nothing_to_read_names_the_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status", str(tmp_path / "none.jsonl")]) == 2
    assert "Escape" in capsys.readouterr().err


@pytest.mark.parametrize("command", sorted(PLANNED))
def test_a_planned_command_refuses_with_an_escape(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    err = capsys.readouterr().err
    assert "not wired yet" in err
    assert "Escape" in err


def test_measured_rate_comes_from_the_journal(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "run.jsonl"
    journal = Journal.open(path)
    journal.write("start", config="c.toml")
    journal.write("step", step=1, loss=1.0, sustained_tflops=61.5)
    assert cli._measured_tflops(str(path)) == 61.5
    assert cli._measured_tflops(str(tmp_path / "none.jsonl")) is None


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: dorm" in capsys.readouterr().out
