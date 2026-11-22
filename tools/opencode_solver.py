"""Headless OpenCode solver: one prompt in, one assistant text out.

Wraps `opencode run --model local/dormouse --agent solver --format json
--archived --standalone`. A stub that is down raises `StubUnavailable` instead of
looking like a model that scored zero, so the harness cannot report a silent pass.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
Generate = Callable[[str, int, Sequence[str]], str]


class SolverError(RuntimeError):
    """Base class for a solver failure that must not be scored as zero."""


class StubUnavailable(SolverError):
    """The local stub is not listening; the harness must abort, not score."""


class ProtocolError(SolverError):
    """OpenCode returned no parseable assistant text."""


class RateLimitError(SolverError):
    """The provider kept rate limiting after backoff."""


def last_assistant_text(lines: Iterable[str]) -> str:
    """Return the assistant text of the last `text` event in an NDJSON stream."""
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "text":
            continue
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text = part["text"]
    return text


def preflight_stub(port: int, *, host: str = "127.0.0.1", timeout: float = 2.0) -> None:
    """Fail loudly if nothing listens on the stub port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise StubUnavailable(f"stub not reachable at {host}:{port}: {exc}") from exc


def _is_rate_limit(output: str) -> bool:
    """Detect the common rate-limit phrasings in provider output."""
    lowered = output.lower()
    return any(mark in lowered for mark in ("rate limit", "too many requests", "429"))


def solve(
    prompt: str,
    *,
    port: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    preflight: bool = True,
) -> str:
    """Run one isolated OpenCode session and return its final assistant text."""
    port = port if port is not None else int(os.environ.get("DORMOUSE_STUB_PORT", DEFAULT_PORT))
    if preflight:
        preflight_stub(port)
    env = os.environ.copy()
    env["DORMOUSE_STUB_PORT"] = str(port)
    # OpenCode v2.2.7 `--format json` emits only step_start/step_finish events,
    # NOT the assistant text (verified on-card 22.11). Default text mode prints
    # a ``> solver · dormouse`` header then the reply, so parse that instead.
    cmd = [
        "opencode",
        "run",
        "--model",
        "local/dormouse",
        "--agent",
        "solver",
        "--standalone",
        prompt,
    ]
    for attempt in range(max_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SolverError("opencode executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SolverError(f"opencode timed out after {timeout}s") from exc
        output = f"{proc.stdout}\n{proc.stderr}"
        if _is_rate_limit(output) and attempt < max_retries:
            time.sleep(2**attempt)
            continue
        if _is_rate_limit(output):
            raise RateLimitError(f"rate limited after {max_retries} retries: {output[-400:]}")
        if proc.returncode != 0:
            raise SolverError(f"opencode exit {proc.returncode}: {proc.stderr[-400:]}")
        text = last_assistant_text(proc.stdout.splitlines())
        if not text.strip():
            # Fallback: default text mode prints '> solver · dormouse' plus the
            # reply. The reply may appear before OR after the header depending
            # on the build, so take the last non-empty, non-header line.
            for line in reversed(proc.stdout.splitlines()):
                stripped = line.strip()
                if not stripped or stripped.startswith(">") and "·" in stripped:
                    continue
                text = stripped
                break
        if not text.strip():
            raise ProtocolError(f"opencode returned no assistant text: {output[-400:]}")
        return text
    raise RateLimitError("unreachable")


def make_generate(**kwargs: object) -> Generate:
    """Adapt `solve` to the harness's `Generate` interface."""

    def generate(prompt: str, max_tokens: int, stop: Sequence[str]) -> str:
        return solve(prompt, **kwargs)  # type: ignore[arg-type]

    return generate


def main() -> None:
    """Print the answer to one prompt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    print(solve(args.prompt, port=args.port), end="")


if __name__ == "__main__":
    main()
