"""OpenAI-compatible stub — the floor and ceiling of the OpenCode harness.

Run from the repository root (src on the path):

    DORMOUSE_STUB_PORT=8765 STUB_MODE=ORACLE STUB_SEED=0 \
        PYTHONPATH=src python tools/stub_server.py

Modes: RANDOM (seeded noise, must score 0), ORACLE (the harness's own reference
answers, must score 1), ECHO (last user message, wiring probe). No dependencies
outside the stdlib and the repo's own task suite.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

DEFAULT_PORT = 8765
MODES = ("RANDOM", "ORACLE", "ECHO")
SUITES = ("smoke", "full")
CHAT_PATH = "/v1/chat/completions"
CONFIG_PATH = "/v1/config"


def oracle_answers(seed: int, suite: str = "smoke") -> dict[str, str]:
    """Return the reference map for the same suite and seed the harness scores."""
    from dormouse.eval.iq import build_suite

    # The stub must answer exactly the prompts the scorer will ask, or ORACLE
    # silently scores zero. Keep this import against the real module (grill/70).
    return {task.prompt: task.expected for task in build_suite(suite, seed=seed)}


def _last_user_text(messages: object) -> str:
    """Extract the last user message, tolerating string and part-list content."""
    if not isinstance(messages, list):
        return ""
    text = ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "".join(parts)
    return text


def respond(mode: str, seed: int, prompt: str, answers: dict[str, str]) -> str:
    """Produce one stub completion. Deterministic for a given mode and seed."""
    if mode == "ECHO":
        return prompt
    if mode == "ORACLE":
        if prompt in answers:
            return answers[prompt]
        # OpenCode may wrap the message; still answer the longest prompt we can see.
        matches = [key for key in answers if key in prompt]
        if matches:
            return answers[max(matches, key=len)]
        return ""
    rng = random.Random(f"{seed}|{prompt}")
    # 32 chars cannot equal any expected answer, so RANDOM cannot pass by luck.
    return "".join(rng.choice(string.printable[:94]) for _ in range(32))


class StubHandler(BaseHTTPRequestHandler):
    """Serve one Chat Completions endpoint in streaming or non-streaming form."""

    mode: str = "RANDOM"
    seed: int = 0
    suite: str = "smoke"
    answers: ClassVar[dict[str, str]] = {}

    def do_GET(self) -> None:
        """Report the suite/seed the harness must bind to, or 404."""
        if self.path.split("?", 1)[0].rstrip("/") != CONFIG_PATH:
            self.send_error(404)
            return
        self._send_json({"mode": self.mode, "seed": self.seed, "suite": self.suite})

    def do_POST(self) -> None:
        """Answer a Chat Completions request."""
        if self.path.split("?", 1)[0].rstrip("/") != CHAT_PATH:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        if not isinstance(request, dict):
            self.send_error(400)
            return
        prompt = _last_user_text(request.get("messages", []))
        content = respond(self.mode, self.seed, prompt, self.answers)
        message = {"role": "assistant", "content": content}
        if request.get("stream"):
            self._send_stream(message, str(request.get("model", "dormouse")))
            return
        payload = {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", "dormouse"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(payload)

    def _send_json(self, payload: dict[str, object]) -> None:
        """Write one JSON response."""
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, message: dict[str, str], model: str) -> None:
        """Write an SSE stream with one content chunk and a finish reason."""
        chunk = {
            "id": "stub",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": message, "finish_reason": None}],
        }
        done = dict(chunk)
        done["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        raw = (
            "".join(f"data: {json.dumps(part)}\n\n" for part in (chunk, done)).encode("utf-8")
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        """Keep CI logs quiet; the harness prints its own table."""

    def log_error(self, fmt: str, *args: object) -> None:
        """Keep CI logs quiet; failures surface as named solver errors."""


def main() -> None:
    """Parse flags and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DORMOUSE_STUB_PORT", DEFAULT_PORT))
    )
    parser.add_argument("--mode", choices=MODES, default=os.environ.get("STUB_MODE", "RANDOM"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("STUB_SEED", "0")))
    parser.add_argument("--suite", choices=SUITES, default=os.environ.get("STUB_SUITE", "smoke"))
    args = parser.parse_args()
    StubHandler.mode = args.mode
    StubHandler.seed = args.seed
    StubHandler.suite = args.suite
    StubHandler.answers = (
        oracle_answers(args.seed, suite=args.suite) if args.mode == "ORACLE" else {}
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), StubHandler)
    print(
        f"stub {args.mode} suite={args.suite} seed={args.seed} on http://127.0.0.1:{args.port}/v1",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
