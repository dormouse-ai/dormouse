"""The decisive bench's unit gates: conversions, card guard and compile gate, on CPU.

The earlier confusion was a unit error — tokens/s multiplied by the wrong constant and an
MFU quoted against the wrong peak. These tests pin the arithmetic with numbers that do not
come from the script's own constants, so a changed BPT or peak fails here.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
from dormouse_core.errors import LoudError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
_SPEC = importlib.util.spec_from_file_location(
    "decisive_bench", Path(__file__).resolve().parents[1] / "tools" / "decisive_bench.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

from dormouse.train import compiled  # noqa: E402

#: The old engine's measured step, written as literals rather than read from the constants.
OLD_MS = 5.06 * 1000.0
OLD_TOKENS = 24_576
BPT = 4.718


def test_known_token_rate_gives_the_expected_source_bytes() -> None:
    """1,000 tok/s at BPT 4.718 is exactly 4,718 B/s of source text."""
    assert bench.SOURCE_BYTES_PER_TOKEN == BPT
    assert bench.bytes_per_s(1_000.0, BPT) == 4_718.0
    assert bench.bytes_per_s(1_000.0) == 4_718.0  # the default really is the manifest BPT


def test_a_failed_compile_is_never_printed_as_a_compiled_speed() -> None:
    """The label, not the caller's intent, names the mode; a failed compile is eager."""
    assert bench.speed_label(compiled_measured=True) == "compiled"
    for reason in ("compile+graph buffers OOM at b24", "inductor timeout", "0 graphs"):
        assert bench.speed_label(compiled_measured=False, compile_reason=reason) == "eager"
        # Even a caller that wrongly claims a graph must not print a failed compile as
        # compiled: the reason is authoritative.
        assert bench.speed_label(compiled_measured=True, compile_reason=reason) == "eager"

    measured = {
        "median_ms": 12.5,
        "measure_steps": 30,
        "repetitions": 3,
        "tokens_per_s": 100.0,
        "bytes_per_s": 471.8,
    }
    assert bench.speed_line("eager", measured).startswith("eager median 12.5 ms/step")
    assert "compiled" not in bench.speed_line("eager", measured)
    assert bench.speed_line("compiled", measured).startswith("compiled median 12.5 ms/step")
    assert bench.compile_refusal_line("out of memory").startswith("compiled not measured:")
    assert bench.compile_refusal_line("no compile attempted", skipped=True).startswith(
        "compiled skipped by --eager-only:"
    )
    with pytest.raises(LoudError, match="unknown speed label"):
        bench.speed_line("maybe", measured)


def test_old_engine_conversion_matches_its_published_byte_rate() -> None:
    """24,576 tokens in 5.06 s at BPT 4.718 gives 22,914.93 B/s, not tok/s/1000."""
    rate = OLD_TOKENS / (OLD_MS / 1000.0)
    expected = OLD_TOKENS * BPT / (OLD_MS / 1000.0)
    assert expected == pytest.approx(22_914.93, rel=1e-5)
    assert bench.bytes_per_s(rate, BPT) == pytest.approx(expected, rel=1e-12)
    assert pytest.approx(expected, rel=1e-12) == bench.OLD_ENGINE_BYTES_PER_S


def test_mfu_is_against_the_measured_92_tflop_peak() -> None:
    """The same rate must report a smaller MFU at 92 TFLOP/s than at the old 52 constant."""
    rate = OLD_TOKENS / (OLD_MS / 1000.0)
    expected = 100.0 * 407.0 * 1e6 * rate / (92.0 * 1e12)
    assert expected == pytest.approx(2.148, rel=1e-3)
    assert bench.mfu_percent(rate, 92.0) == pytest.approx(expected, rel=1e-12)
    assert bench.mfu_percent(rate, 92.0) < bench.mfu_percent(rate, 52.0)


def test_old_engine_threshold_scales_with_tokens() -> None:
    """A smaller batch is compared to the old engine at the same token count."""
    assert bench.old_engine_ms(OLD_TOKENS) == OLD_MS
    assert bench.old_engine_ms(OLD_TOKENS // 2) == OLD_MS / 2


def test_card_guard_only_allows_the_desktop_pid() -> None:
    """A compute app outside the allow-list is refused; an empty table is not busy."""
    assert bench.busy_pids("1234\n5678\n", {1234}) == [5678]
    assert bench.busy_pids("No running processes found\n", {1234}) == []
    with pytest.raises(LoudError, match="unreadable PID"):
        bench.busy_pids("not-a-pid\n", set())


def test_compile_budget_restores_the_signal_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is installed for one call and removed on both paths."""
    before = signal.getsignal(signal.SIGALRM)
    with bench.compiled.compile_budget(60):
        assert signal.getsignal(signal.SIGALRM) is not before
    assert signal.getsignal(signal.SIGALRM) is before
    with pytest.raises(LoudError, match="budget"), bench.compiled.compile_budget(0):
        pass
    assert signal.getsignal(signal.SIGALRM) is before


def test_require_graph_refuses_zero_graphs_and_names_the_break() -> None:
    """A zero-graph compile is a named refusal that quotes dynamo's break reason."""
    broken: dict[str, Any] = {
        "unique_graphs": 0,
        "graph_break": {"tensor.item": 2},
        "frames": {},
    }
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(compiled, "proof", lambda: broken)
        with pytest.raises(LoudError, match="0 inductor graphs") as error:
            compiled.require_graph()
        assert "tensor.item" in str(error.value)
        good = dict(broken, unique_graphs=1)
        monkeypatch.setattr(compiled, "proof", lambda: good)
        assert compiled.require_graph()["unique_graphs"] == 1
    finally:
        monkeypatch.undo()
