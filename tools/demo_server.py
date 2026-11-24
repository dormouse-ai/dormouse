"""Serve the demo checkpoint as an OpenAI-compatible chat endpoint for OpenCode.

    PYTHONPATH=src DORMOUSE_STUB_PORT=18766 \
      python tools/demo_server.py --checkpoint runs/demo_200m/export/model.pt
    DORMOUSE_STUB_PORT=18766 python tools/opencode_solver.py "Как тебя зовут?"

This is a test seam, not `dorm serve` (which is intentionally unwired): one model,
one greedy Talker, one chat-completions route. Nothing is scored by this file.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]


class DemoHandler(BaseHTTPRequestHandler):
    """Answer chat completions from the loaded checkpoint's greedy generator."""

    model: ClassVar[object] = None

    def do_POST(self) -> None:
        """Generate one chat completion for the last user message."""
        if self.path.split("?", 1)[0].rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        text = ""
        for message in request.get("messages", []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
        answer = self.model.generate(text)  # type: ignore[attr-defined]
        payload = {
            "id": "demo",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", "dormouse"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        """Keep the demo transcript clean."""
        return


def main() -> int:
    """Load one checkpoint and serve it until interrupted."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from dormouse.eval.model_adapter import CheckpointAdapter

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--max-tokens", type=int, default=96)
    args = p.parse_args()
    DemoHandler.model = CheckpointAdapter(args.checkpoint, max_tokens=args.max_tokens)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"demo checkpoint on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
