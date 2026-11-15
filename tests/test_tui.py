"""The CLI's screen: one rich renderer, and the machine interface that mirrors it.

The two `--json` snapshots are literals on purpose: that document is a contract for
scripts, so changing it should be a deliberate edit here. The rich screens are
asserted for content, because their box drawing is the library's business.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dormouse import cli, tui
from dormouse.cli import PLANNED, journal_row, main


def _doctor_report() -> dict[str, Any]:
    """A machine's worth of facts, fixed so the snapshot is the renderer's and not the host's."""
    return {
        "python": "3.12.7",
        "platform": "Linux-6.9.1-x86_64",
        "git_sha": "68c0c33",
        "torch": "2.5.0+cu124",
        "cuda": "12.4",
        "gpu": "NVIDIA GeForce RTX 4090",
        "disk_free_gb": 412.3,
        "nvidia_smi": "NVIDIA GeForce RTX 4090, 24564 MiB, 1024 MiB",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "vram_total_mib": 24564,
        "vram_used_mib": 1024,
    }


def _journal(tmp_path: Path) -> Path:
    """A two-step run that swapped a segment, with timestamps a test can age."""
    path = tmp_path / "a3f2" / "run.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                '{"config": "configs/dormouse_1b.toml", "kind": "start", "t": 1000.0}',
                '{"kind": "step", "loss": 1.25, "step": 7, "t": 1004.0}',
                '{"kind": "swap", "step": 5, "t": 1002.0}',
            ]
        )
        + "\n"
    )
    return path


DOCTOR_JSON = """{
  "cuda": "12.4",
  "disk_free_gb": 412.3,
  "git_sha": "68c0c33",
  "gpu": "NVIDIA GeForce RTX 4090",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "nvidia_smi": "NVIDIA GeForce RTX 4090, 24564 MiB, 1024 MiB",
  "platform": "Linux-6.9.1-x86_64",
  "python": "3.12.7",
  "torch": "2.5.0+cu124",
  "vram_total_mib": 24564,
  "vram_used_mib": 1024
}
"""

STATUS_JSON = """{
  "runs": [
    {
      "age_seconds": 98.0,
      "config": "configs/dormouse_1b.toml",
      "id": "a3f2",
      "metric": "loss=1.25",
      "phase": "running",
      "segments": 1,
      "step": 7
    }
  ]
}
"""


def test_doctor_json_is_a_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "doctor", _doctor_report)
    assert main(["doctor", "--json"]) == 0
    assert capsys.readouterr().out == DOCTOR_JSON


def test_status_json_is_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _journal(tmp_path)
    monkeypatch.setattr(cli, "_now", lambda: 1100.0)
    assert main(["status", str(path), "--json"]) == 0
    assert capsys.readouterr().out == STATUS_JSON


@pytest.mark.parametrize("command", sorted(PLANNED))
def test_every_planned_command_renders_its_escape_on_its_own_line(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    err = capsys.readouterr().err
    escapes = [line.strip().removeprefix("│").strip() for line in err.splitlines()]
    assert any(line.startswith("Escape:") for line in escapes), err


def test_a_failing_check_is_visible_without_reading_prose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {**_doctor_report(), "torch": None, "cuda": None}
    tui.UI().doctor(report)
    out = capsys.readouterr().out
    assert "✗" in out
    assert "▲" in out
    assert "✓" in out


def test_no_color_emits_no_ansi(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "dorm doctor" in out


def test_no_color_console_is_unstyled_not_unrendered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    rendered = tui.console()
    assert rendered.no_color
    assert not rendered.is_terminal


def test_progress_without_a_tty_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    with tui.Progress(2, "pricing") as bar:
        bar.advance("kda")
        bar.advance("dsa")
    assert capsys.readouterr().err == ""


def test_progress_on_a_tty_is_a_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _Tty()
    monkeypatch.setattr(sys, "stderr", stream)
    with tui.Progress(2, "pricing") as bar:
        bar.advance("kda")
        bar.advance("dsa")
    assert bar.done == 2
    assert "pricing" in stream.getvalue()


def test_doctor_json_is_the_table_numbers(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _doctor_report()
    monkeypatch.setattr(cli, "doctor", lambda: report)
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert main(["doctor"]) == 0
    table = capsys.readouterr().out
    assert str(report["python"]) in table
    assert f"{report['disk_free_gb']} GB" in table
    assert str(report["git_sha"]) in table


def test_status_json_is_the_table_numbers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _journal(tmp_path)
    assert main(["status", str(path), "--json"]) == 0
    row = json.loads(capsys.readouterr().out)["runs"][0]
    assert main(["status", str(path)]) == 0
    table = capsys.readouterr().out
    assert str(row["id"]) in table
    assert f"{row['step']}" in table
    assert str(row["metric"]) in table
    assert tui.age(float(row["age_seconds"])) in table


def test_plan_json_is_the_panel_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    config = str(Path(__file__).resolve().parents[1] / "configs" / "dormouse_1b.toml")
    assert main(["plan", config, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert main(["plan", config]) == 0
    panel = capsys.readouterr().out
    assert f"{data['params']['stored']:,}" in panel
    assert f"{data['params']['active']:,}" in panel
    assert str(data["plan"]["human"]) in panel
    assert all(
        f"{name}" in panel and f"{counter}" in panel for name, counter in data["mechanisms"].items()
    )


def test_the_status_row_is_read_from_the_journal(tmp_path: Path) -> None:
    row = journal_row(_journal(tmp_path), 1100.0)
    assert row == {
        "id": "a3f2",
        "config": "configs/dormouse_1b.toml",
        "phase": "running",
        "step": 7,
        "metric": "loss=1.25",
        "segments": 1,
        "age_seconds": 98.0,
    }
