"""Contract of the OpenCode bridge: stub floor/ceiling, NDJSON parser, loud failure.

The score tests drive the stub over HTTP (fast and deterministic). A separate test
drives one real `opencode run` so config, agent and parser are checked together.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import threading
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dormouse.eval.harness import run
from dormouse.eval.tasks import suite

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stub_tool = _load_tool("stub_server")
solver = _load_tool("opencode_solver")


@contextmanager
def stub(mode: str, seed: int = 0) -> Iterator[int]:
    """Serve one stub mode on an ephemeral port for the duration of the block."""
    handler = stub_tool.StubHandler
    old = (handler.mode, handler.seed, handler.answers)
    handler.mode = mode
    handler.seed = seed
    handler.answers = stub_tool.oracle_answers(seed) if mode == "ORACLE" else {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        handler.mode, handler.seed, handler.answers = old


def http_generate(port: int) -> Callable[[str, int, object], str]:
    """Turn the stub's Chat Completions endpoint into a harness Generate."""

    def generate(prompt: str, max_tokens: int, stop: object) -> str:
        payload = {"model": "dormouse", "messages": [{"role": "user", "content": prompt}]}
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.load(response)
        text: str = body["choices"][0]["message"]["content"]
        return text

    return generate


def test_stub_random_fails_and_oracle_passes_every_task() -> None:
    tasks = suite(0)
    with stub("RANDOM", 0) as port:
        floor = [row.score for row in run(tasks, http_generate(port))]
    with stub("ORACLE", 0) as port:
        ceiling = [row.score for row in run(tasks, http_generate(port))]
    assert floor == [0.0] * len(tasks), floor
    assert ceiling == [1.0] * len(tasks), ceiling


def test_stub_is_deterministic_for_a_seed() -> None:
    prompt = suite(0)[1].examples[0].prompt
    with stub("RANDOM", 3) as port:
        first = http_generate(port)(prompt, 16, ())
    with stub("RANDOM", 3) as port:
        second = http_generate(port)(prompt, 16, ())
    with stub("RANDOM", 4) as port:
        other = http_generate(port)(prompt, 16, ())
    assert first == second
    assert first != other


def test_ndjson_parser_takes_the_last_assistant_text() -> None:
    lines = [
        '{"type":"step_start","part":{"type":"step-start"}}',
        '{"type":"text","part":{"type":"text","text":"first"}}',
        "not json at all",
        '{"type":"text","part":{"type":"text","text":"second"}}',
        '{"type":"step_finish","part":{"type":"step-finish"}}',
    ]
    assert solver.last_assistant_text(lines) == "second"
    assert solver.last_assistant_text(["{}", "", "  "]) == ""


def test_a_stub_that_is_off_is_a_named_refusal() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(solver.StubUnavailable, match="stub not reachable"):
        solver.solve("ping", port=port, preflight=True)


def test_rate_limit_is_retried_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_run(cmd: object, **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(stdout="Rate limit exceeded", stderr="", returncode=0)
        return SimpleNamespace(
            stdout='{"type":"text","part":{"type":"text","text":"ok"}}', stderr="", returncode=0
        )

    monkeypatch.setattr(solver.subprocess, "run", fake_run)
    monkeypatch.setattr(solver, "preflight_stub", lambda *a, **k: None)
    monkeypatch.setattr(solver.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    assert solver.solve("ping", port=1, max_retries=1) == "ok"
    assert ("sleep", 1) in calls


@pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode not installed")
def test_opencode_lists_the_local_provider() -> None:
    env = os.environ.copy()
    env["DORMOUSE_STUB_PORT"] = "8765"
    proc = subprocess.run(
        ["opencode", "models"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "local/dormouse" in proc.stdout


@pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode not installed")
def test_opencode_solver_returns_the_oracle_answer() -> None:
    example = suite(0)[1].examples[0]
    with stub("ORACLE", 0) as port:
        assert solver.solve(example.prompt, port=port, timeout=60) == example.expected
