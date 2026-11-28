"""A local-only demo for the trained checkpoint.

This is deliberately not `dorm serve`: one model, one GPU, one greedy/sampled
continuation, and a small page that shows the latency, the tokens per second and
the model's next-token distribution. No network calls, no cloud.

    PYTHONPATH=src:../dormouse-kernels/src .venv-cuda/bin/python \
      tools/demo_local.py --checkpoint runs_fp32_fresh/train_404m_fp32ns/ckpt.pt

The checkpoint loaded here is a base model, not an instruction-tuned assistant:
the demo shows continuation and next-token probabilities, not a chat product.
"""
# ruff: noqa: E501, D101, D102, D103, D107, RUF001

from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dormouse · local</title>
<style>
  :root { color-scheme: light; --ink:#202124; --muted:#5f6368; --line:#e5e7eb;
          --paper:#ffffff; --blue:#1a73e8; --soft:#e8f0fe; --bg:#f8f9fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
  header { display:flex; justify-content:space-between; align-items:center;
           max-width:1180px; margin:0 auto; padding:28px 28px 0; }
  .brand { font-size:18px; font-weight:700; letter-spacing:.04em; }
  .pill { border:1px solid var(--line); background:#fff; border-radius:999px;
          padding:7px 12px; color:var(--muted); font-size:12px; }
  .dot { display:inline-block; width:7px; height:7px; border-radius:50%;
         background:#34a853; margin-right:7px; vertical-align:1px; }
  main { max-width:1180px; margin:0 auto; padding:28px; }
  h1 { font-size:46px; line-height:1.05; letter-spacing:-.03em; margin:34px 0 10px; }
  .lead { color:var(--muted); max-width:700px; margin:0 0 30px; font-size:16px; }
  .grid { display:grid; grid-template-columns: 1.35fr .65fr; gap:18px; align-items:start; }
  .card { background:var(--paper); border:1px solid var(--line); border-radius:18px; padding:20px; }
  label { display:block; font-size:11px; letter-spacing:.08em; text-transform:uppercase;
          color:var(--muted); margin-bottom:9px; }
  textarea { width:100%; min-height:112px; resize:vertical; border:1px solid var(--line);
             border-radius:12px; padding:14px; font:15px/1.5 inherit; color:var(--ink);
             outline:none; background:#fff; }
  textarea:focus { border-color:var(--blue); box-shadow:0 0 0 3px var(--soft); }
  button { border:0; border-radius:999px; background:var(--blue); color:#fff;
           padding:11px 18px; font:600 14px inherit; cursor:pointer; }
  button:disabled { opacity:.45; cursor:wait; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:14px; }
  .preset { background:#fff; color:var(--ink); border:1px solid var(--line); font-weight:400;
            padding:8px 12px; font-size:12px; }
  .preset:hover { border-color:#c9cdd2; }
  .out { min-height:170px; white-space:pre-wrap; font:16px/1.65 inherit; }
  .out.empty { color:#9aa0a6; }
  .muted { color:var(--muted); font-size:12px; }
  .metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:16px; }
  .metric { border:1px solid var(--line); border-radius:12px; padding:10px 12px; }
  .metric strong { display:block; font-size:20px; letter-spacing:-.02em; }
  .bar { display:flex; align-items:center; gap:8px; margin-top:8px; font-size:11px; color:var(--muted); }
  .bar .name { width:82px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bar .track { flex:1; height:6px; background:#f1f3f4; border-radius:99px; overflow:hidden; }
  .bar .fill { height:100%; background:var(--blue); border-radius:99px; }
  .bar .pct { width:42px; text-align:right; }
  .badge { display:inline-block; background:var(--soft); color:#0b57d0; border-radius:999px;
           padding:3px 9px; font-size:10px; letter-spacing:.06em; }
  .foot { color:#9aa0a6; font-size:11px; margin-top:26px; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } h1 { font-size:34px; } }
</style>
</head>
<body>
<header>
  <div class="brand">DORMOUSE / LOCAL</div>
  <div class="pill"><span class="dot"></span> NO CLOUD</div>
</header>
<main>
  <div class="badge">BASE MODEL · CONTINUATION</div>
  <h1>Интеллект<br>без аренды.</h1>
  <p class="lead">Модель работает на этой машине. Выключи интернет — она продолжит отвечать:
     запрос не уходит никуда. Это не чат-ассистент, а локальный инференс и его вероятности.</p>
  <div class="grid">
    <section class="card">
      <label>Начало текста</label>
      <textarea id="prompt">The quick brown fox</textarea>
      <div class="row">
        <button id="run" onclick="run()">Продолжить</button>
        <button class="preset" onclick="preset('Once upon a time')">Once upon a time</button>
        <button class="preset" onclick="preset('def add(a, b):')">def add(a, b):</button>
        <button class="preset" onclick="preset('The quick brown fox')">The quick brown fox</button>
      </div>
      <div class="row"><span class="muted" id="status">готово · нажми «Продолжить»</span></div>
      <div class="out empty" id="out">Здесь появится продолжение.</div>
    </section>
    <aside class="card">
      <label>Следующий токен</label>
      <div id="top" class="muted">—</div>
      <div class="metrics">
        <div class="metric"><strong id="ms">—</strong><span class="muted">мс</span></div>
        <div class="metric"><strong id="tps">—</strong><span class="muted">ток/с</span></div>
        <div class="metric"><strong id="tokens">—</strong><span class="muted">токенов</span></div>
        <div class="metric"><strong id="vram">—</strong><span class="muted">VRAM, MiB</span></div>
      </div>
      <p class="muted" id="model">—</p>
    </aside>
  </div>
  <p class="foot">Dormouse · one GPU · no external requests. Base checkpoint, not an SFT assistant.</p>
</main>
<script>
const $ = id => document.getElementById(id);
function preset(t){ $('prompt').value = t; $('prompt').focus(); }
function esc(t){ return t.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
async function run(){
  const prompt = $('prompt').value.trim();
  if(!prompt) return;
  $('run').disabled = true; $('status').textContent = 'считает на GPU…';
  $('out').classList.add('empty'); $('out').textContent = '…';
  const t0 = performance.now();
  try {
    const r = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, max_tokens: 28, temperature: 0.0})});
    const d = await r.json();
    if(d.error){ throw new Error(d.error); }
    $('out').classList.remove('empty'); $('out').textContent = d.text || '∅';
    $('ms').textContent = d.ms.toFixed(0);
    $('tps').textContent = d.tok_s.toFixed(1);
    $('tokens').textContent = d.tokens;
    $('vram').textContent = d.vram_mb;
    $('model').textContent = d.device + ' · ' + d.model;
    $('top').innerHTML = d.top.map(x =>
      `<div class="bar"><span class="name">${esc(x.token)}</span><span class="track"><span class="fill" style="width:${Math.max(1,x.pct)}%"></span></span><span class="pct">${x.pct.toFixed(0)}%</span></div>`
    ).join('');
    $('status').textContent = 'локально · без сети';
  } catch(e) {
    $('status').textContent = 'ошибка: ' + e.message;
  } finally { $('run').disabled = false; }
}
</script>
</body>
</html>
"""


class LocalModel:
    """One checkpoint on one CUDA device, greedy/sampled continuation only."""

    def __init__(self, checkpoint: Path, config: Path) -> None:
        import torch

        from dormouse.data.tokenizer import END_ID
        from dormouse.model.assemble import assemble
        from dormouse.model.spec import StackSpec
        from dormouse.train.talker import vocabulary

        self.torch = torch
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
        keys = {f.name for f in fields(StackSpec) if f.init}
        self.spec = StackSpec.from_mapping(
            {k: v for k, v in raw.items() if k in keys}, source=str(config)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        started = time.time()
        self.model = assemble(self.spec, device=str(self.device)).model
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = artifact["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.encode, self.decode = vocabulary(self.spec)
        self.vram_mb = (
            torch.cuda.memory_allocated(self.device) // (1024 * 1024)
            if self.device.type == "cuda"
            else 0
        )
        self.device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        self.load_s = time.time() - started
        self.model_name = f"dormouse · {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M"
        self.end_id = END_ID

    def _width(self, used: int) -> int:
        """KDA tiles at 64, and 576 covers prompt+28 on the demo path."""
        return 576 if used <= 575 else 1024

    def generate(
        self, prompt: str, max_tokens: int = 28, temperature: float = 0.0
    ) -> dict[str, Any]:
        torch = self.torch
        if not prompt:
            return {"error": "empty prompt"}
        ids = list(self.encode(prompt))[-510:] or [0]
        produced: list[int] = []
        top: list[dict[str, Any]] = []
        started = time.time()
        with torch.inference_mode():
            for step in range(max_tokens):
                used = len(ids) + 1
                width = self._width(used)
                padded = ids + [0] * (width - len(ids))
                logits = self.model(torch.tensor([padded], dtype=torch.long, device=self.device))[
                    0, len(ids) - 1
                ]
                probs = torch.softmax(logits.float(), dim=-1)
                if step == 0:
                    values, indices = torch.topk(probs, 5)
                    top = [
                        {"token": self.decode([int(i)]), "pct": float(v) * 100}
                        for v, i in zip(values.tolist(), indices.tolist(), strict=False)
                    ]
                if temperature <= 0:
                    next_id = int(probs.argmax())
                else:
                    next_id = int(torch.multinomial(probs, 1).item())
                if next_id == self.end_id:
                    break
                ids.append(next_id)
                produced.append(next_id)
        elapsed = time.time() - started
        return {
            "text": self.decode(produced),
            "ms": elapsed * 1000,
            "tok_s": len(produced) / elapsed if elapsed else 0.0,
            "tokens": len(produced),
            "vram_mb": self.vram_mb,
            "device": self.device_name,
            "model": self.model_name,
            "top": top,
        }


class Handler(BaseHTTPRequestHandler):
    model: ClassVar[LocalModel]

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            out = self.model.generate(
                str(req.get("prompt", "")),
                max_tokens=min(int(req.get("max_tokens", 28)), 48),
                temperature=float(req.get("temperature", 0.0)),
            )
            raw = json.dumps(out, ensure_ascii=False).encode("utf-8")
            self._send(200, raw, "application/json; charset=utf-8")
        except Exception as exc:  # the demo reports the error on the page
            raw = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(500, raw, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs_fp32_fresh/train_404m_fp32ns/ckpt.pt"
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs/train_404m_b6.toml")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    Handler.model = LocalModel(args.checkpoint, args.config)
    print(
        f"loaded {Handler.model.model_name} on {Handler.model.device_name} "
        f"in {Handler.model.load_s:.1f}s; http://127.0.0.1:{args.port}/",
        flush=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
