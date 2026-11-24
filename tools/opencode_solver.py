"""Headless OpenCode solver: one prompt in, one assistant text out.

Wraps `opencode run --model local/dormouse --agent solver --format json
--archived --standalone`. A stub that is down raises `StubUnavailable` instead of
looking like a model that scored zero, so the harness cannot report a silent pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from dormouse_core.errors import LoudError

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
Generate = Callable[[str, int, Sequence[str]], str]

_RATE_PHRASE = re.compile(r"rate\s*limit|too\s+many\s+requests", re.IGNORECASE)
_HTTP_429 = re.compile(r"(?:http|status|error|code)\s*[:=]?\s*429\b", re.IGNORECASE)
_STATUS_KEYS = ("code", "status", "status_code")
_ANSWER_LABELS = ("the answer is", "answer:")
_FOOTER_PREFIXES = ("[tokens:", "[cost:", "[time:", "[duration:", "[steps:")


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


def _error_status_429(event: object) -> bool:
    """Read an explicit 429 from a provider error object, never from answer text."""
    if not isinstance(event, dict):
        return False
    for key in ("detail", "error"):
        if _error_status_429(event.get(key)):
            return True
    for key in _STATUS_KEYS:
        value = event.get(key)
        if (isinstance(value, int) and value == 429) or (
            isinstance(value, str) and value.strip() == "429"
        ):
            return True
    return False


def _is_rate_limit(output: str) -> bool:
    """Detect a rate limit without treating an answer containing 429 as one."""
    if _RATE_PHRASE.search(output) or _HTTP_429.search(output):
        return True
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _error_status_429(event):
            return True
    return False


def _is_json_object(line: str) -> bool:
    """Parse a line only when it is a JSON object, skipping plain text."""
    try:
        return isinstance(json.loads(line), dict)
    except json.JSONDecodeError:
        return False


def text_mode_answer(stdout: str) -> str:
    """Return the assistant text of a text-mode reply, not its header or footer."""
    kept: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        if stripped.startswith(_FOOTER_PREFIXES):
            continue
        if stripped.lower().rstrip(":") in _ANSWER_LABELS:
            continue
        if _is_json_object(stripped):
            continue  # NDJSON tail carries step events, not the answer
        kept.append(stripped)
    return "\n".join(kept)


def check_stub_binding(port: int, *, seed: int | None = None, suite: str | None = None) -> None:
    """Refuse a stub whose reported seed/suite differs from the harness's."""
    if seed is None and suite is None:
        return
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/config", timeout=5) as resp:
            config = json.load(resp)
    except (OSError, ValueError) as exc:
        raise StubUnavailable(f"stub {port} cannot report its suite/seed: {exc}") from exc
    if seed is not None and config.get("seed") != seed:
        raise LoudError(
            f"stub seed {config.get('seed')} != harness seed {seed}",
            "start the stub with --seed equal to the harness seed",
        )
    if suite is not None and config.get("suite") != suite:
        raise LoudError(
            f"stub suite {config.get('suite')} != harness suite {suite}",
            "start the stub with --suite equal to the harness suite",
        )


def solve(
    prompt: str,
    *,
    port: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    preflight: bool = True,
    seed: int | None = None,
    suite: str | None = None,
) -> str:
    """Run one isolated OpenCode session and return its final assistant text."""
    port = port if port is not None else int(os.environ.get("DORMOUSE_STUB_PORT", DEFAULT_PORT))
    if preflight:
        preflight_stub(port)
    check_stub_binding(port, seed=seed, suite=suite)
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
            text = text_mode_answer(proc.stdout)
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
