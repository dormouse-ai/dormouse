"""RED repros for the N9/D05 IQ-harness bugs; every test here fails at the audited HEAD.

Each test drives the shipped path (solver, stub, scorer, loop gate) rather than a helper
that only asserts a string, so a silent 0.300 or a footer-as-answer cannot hide again.
"""

from __future__ import annotations

import importlib.util
import json
import math
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from dormouse_core.errors import LoudError

from dormouse.eval import iq
from dormouse.eval.iq import MockJudge, build_suite, evaluate
from dormouse.eval.model_adapter import FunctionAdapter

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solver = _load_tool("opencode_solver")
stub_tool = _load_tool("stub_server")
loop_tool = _load_tool("demo_loop_eval")


@contextmanager
def _stub(mode: str, seed: int, suite: str) -> Iterator[int]:
    handler = stub_tool.StubHandler
    old = (handler.mode, handler.seed, handler.suite, handler.answers)
    handler.mode = mode
    handler.seed = seed
    handler.suite = suite
    handler.answers = stub_tool.oracle_answers(seed, suite=suite) if mode == "ORACLE" else {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        handler.mode, handler.seed, handler.suite, handler.answers = old


def _answer(port: int, prompt: str) -> str:
    payload = {"model": "dormouse", "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return str(json.load(response)["choices"][0]["message"]["content"])


# --- B-01 / B-02: a bare "429" in the answer is arithmetic, not a rate limit ----------


def test_only_structured_429_is_a_rate_limit() -> None:
    assert solver._is_rate_limit("429") is False
    assert solver._is_rate_limit("cost 1429 tokens") is False
    assert solver._is_rate_limit("What is 13 * 33? answer 429") is False
    assert solver._is_rate_limit('{"detail":{"code":429,"message":"slow down"}}') is True
    assert solver._is_rate_limit('{"error":{"code":429}}') is True
    assert solver._is_rate_limit("Rate limit exceeded") is True


def test_seed_170_oracle_answer_is_scored_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = next(t.prompt for t in build_suite("smoke", seed=170) if t.expected == "429")
    calls: list[object] = []
    stdout = "> solver · dormouse\n429\n[tokens: 5]\n"

    def fake_run(cmd: object, **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(solver.subprocess, "run", fake_run)
    monkeypatch.setattr(solver, "preflight_stub", lambda *a, **k: None)
    assert solver.solve(prompt, port=1) == "429"
    assert len(calls) == 1


# --- B-05 / B-06: the text-mode fallback must find the answer, not the footer ---------


def test_text_mode_fallback_returns_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "> solver · dormouse\nThe answer is\nParis\n[tokens: 5]\n"
    monkeypatch.setattr(
        solver.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=stdout, stderr="", returncode=0),
    )
    monkeypatch.setattr(solver, "preflight_stub", lambda *a, **k: None)
    assert solver.solve("capital of France", port=1) == "Paris"


# --- B-08: a judge cannot smuggle nan/inf/out-of-range into a row ---------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.5, -0.5, 2.0])
def test_score_refuses_nonfinite_and_out_of_range(bad: float) -> None:
    with pytest.raises(LoudError, match="finite"):
        iq.Score(bad, "judge verdict")


def test_score_still_accepts_a_reasoned_zero_and_one() -> None:
    assert iq.Score(0.0, "wrong answer").value == 0.0
    assert iq.Score(1.0, "exact").value == 1.0


# --- B-03 / B-11: stub suite and seed are bound, mismatch is a refusal ----------------


def test_full_suite_oracle_scores_every_task() -> None:
    suite = build_suite("full", seed=0)
    assert len(suite) == 18
    with _stub("ORACLE", 0, "full") as port:
        rows = evaluate(suite.tasks, FunctionAdapter(lambda p: _answer(port, p)), MockJudge())
    assert [row.score for row in rows] == [1.0] * 18


def test_stub_seed_mismatch_is_refused_not_scored() -> None:
    with _stub("ORACLE", 0, "smoke") as port:
        generate = solver.make_generate(port=port, seed=1, suite="smoke", preflight=False)
        with pytest.raises(LoudError, match="seed"):
            generate("ping", 96, ())


def test_stub_suite_mismatch_is_refused_not_scored() -> None:
    with _stub("ORACLE", 0, "smoke") as port:
        generate = solver.make_generate(port=port, seed=0, suite="full", preflight=False)
        with pytest.raises(LoudError, match="suite"):
            generate("ping", 96, ())


def test_stub_reports_its_suite_and_seed() -> None:
    with (
        _stub("ORACLE", 7, "full") as port,
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/config", timeout=10) as resp,
    ):
        config = json.load(resp)
    assert config == {"mode": "ORACLE", "seed": 7, "suite": "full"}


# --- B-12 / B-16: n=10 is a smoke-signal, never a GREEN quality/IQ claim ---------------


def test_wilson_lower_bound_is_finite_and_ordered() -> None:
    lo, hi = iq.wilson_95(5, 10)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_loop_gate_never_labels_smoke_a_quality_win() -> None:
    small = loop_tool.loop_gate(n=30, wins=30, regressions=0, latency_ratio=1.0)
    assert small["verdict"] == "smoke-signal"
    assert small["iq_verdict"] == iq.IQ_REFUTED
    high = loop_tool.loop_gate(n=50, wins=50, regressions=0, latency_ratio=1.0)
    low = loop_tool.loop_gate(n=50, wins=20, regressions=0, latency_ratio=1.0)
    assert high["verdict"] == "PASS"
    assert low["verdict"] == "REFUTE"


def test_iq_verdict_refuses_without_blind_kappa_and_named_anchor() -> None:
    assert iq.iq_verdict() == iq.IQ_REFUTED
    assert iq.iq_verdict(blind_kappa=0.9, anchor="") == iq.IQ_REFUTED
    assert iq.iq_verdict(blind_kappa=None, anchor="human raters") == iq.IQ_REFUTED
    assert iq.iq_verdict(blind_kappa=float("nan"), anchor="human raters") == iq.IQ_REFUTED
    assert iq.iq_verdict(blind_kappa=0.9, anchor="human raters") != iq.IQ_REFUTED
