"""A one-turn local chat page for the SFT checkpoint.

    PYTHONPATH=src:../dormouse-kernels/src .venv-cuda/bin/python \
      tools/demo_chat.py --checkpoint runs/chat_50m_sft/ckpt.pt \
      --config configs/chat_50m_sft.toml --port 8001

One user turn, one greedy answer, no cloud. This is deliberately a small SFT
model; if the probe table is not green, do not present it as an assistant.
"""
# ruff: noqa: E501, D101, D102, D103, RUF001

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from chat_probe import ChatModel, load

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dormouse · local chat</title>
<style>
:root{color-scheme:light;--ink:#202124;--muted:#5f6368;--line:#e5e7eb;--blue:#1a73e8;--soft:#e8f0fe;--bg:#f8f9fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
header{max-width:980px;margin:0 auto;padding:28px 28px 0;display:flex;justify-content:space-between;align-items:center}
.brand{font-weight:700;letter-spacing:.04em}.pill{border:1px solid var(--line);background:#fff;border-radius:99px;padding:7px 12px;color:var(--muted);font-size:12px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#34a853;margin-right:7px}
main{max-width:980px;margin:0 auto;padding:28px}h1{font-size:44px;letter-spacing:-.03em;line-height:1.05;margin:30px 0 8px}
.lead{color:var(--muted);max-width:650px;margin:0 0 26px}.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:20px}
.log{min-height:260px;display:flex;flex-direction:column;gap:12px}.msg{max-width:78%;padding:10px 14px;border-radius:14px;white-space:pre-wrap}
.user{align-self:flex-end;background:var(--blue);color:#fff;border-bottom-right-radius:4px}.bot{align-self:flex-start;background:#f1f3f4;border-bottom-left-radius:4px}
.empty{color:#9aa0a6}.row{display:flex;gap:8px;margin-top:14px}input{flex:1;border:1px solid var(--line);border-radius:12px;padding:13px 14px;font:15px inherit;outline:none}
input:focus{border-color:var(--blue);box-shadow:0 0 0 3px var(--soft)}button{border:0;border-radius:12px;background:var(--blue);color:#fff;padding:0 18px;font-weight:600;cursor:pointer}
button:disabled{opacity:.45}.meta{color:#9aa0a6;font-size:11px;margin-top:12px}.foot{color:#9aa0a6;font-size:11px;margin-top:24px}
</style></head><body>
<header><div class="brand">DORMOUSE / CHAT</div><div class="pill"><span class="dot"></span> LOCAL · NO CLOUD</div></header>
<main><h1>Минимальный<br>локальный чат.</h1><p class="lead">Один ход, одна видеокарта, запрос не уходит в сеть. Это маленькая SFT-модель, а не продукт.</p>
<section class="card"><div class="log" id="log"><div class="msg bot empty" id="empty">Спроси что-нибудь.</div></div>
<div class="row"><input id="q" placeholder="Привет! Как тебя зовут?" autocomplete="off"><button id="send" onclick="send()">→</button></div>
<div class="meta" id="meta">—</div></section><p class="foot">Dormouse · 50M · SFT · local only</p></main>
<script>
const $=id=>document.getElementById(id);
function esc(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function send(){
 const q=$('q').value.trim(); if(!q) return;
 $('empty')?.remove(); $('q').value=''; $('send').disabled=true; $('meta').textContent='считает…';
 $('log').insertAdjacentHTML('beforeend','<div class="msg user">'+esc(q)+'</div>');
 try{
  const t=performance.now();
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q})});
  const d=await r.json(); if(d.error) throw new Error(d.error);
  $('log').insertAdjacentHTML('beforeend','<div class="msg bot">'+esc(d.a||'∅')+'</div>');
  $('meta').textContent=d.device+' · '+d.ms.toFixed(0)+' ms · '+d.tok_s.toFixed(1)+' tok/s · '+d.vram_mb+' MiB';
 }catch(e){$('log').insertAdjacentHTML('beforeend','<div class="msg bot">ошибка: '+esc(e.message)+'</div>');}
 finally{$('send').disabled=false;$('q').focus();}
}
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    model: ClassVar[ChatModel]

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
        if self.path.rstrip("/") != "/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            started = time.time()
            answer = self.model.answer(str(req.get("q", "")), max_tokens=48)
            elapsed = (time.time() - started) * 1000
            out = {
                "a": answer,
                "ms": elapsed,
                "tok_s": len(answer) / max(elapsed / 1000, 1e-9),
                "vram_mb": self.model.vram_mb,
                "device": self.model.device_name,
            }
            self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
        except Exception as exc:
            self._send(
                500,
                json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                "application/json",
            )

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    model = load(args.checkpoint, args.config)
    Handler.model = model
    print(f"chat on http://127.0.0.1:{args.port}/", flush=True)
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
