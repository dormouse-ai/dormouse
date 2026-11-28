"""Run a fixed chat probe against one checkpoint and print the answers.

    PYTHONPATH=src:../dormouse-kernels/src .venv-cuda/bin/python \
      tools/chat_probe.py --checkpoint runs/chat_50m_sft/ckpt.pt \
      --config configs/chat_50m_sft.toml

The probe is deliberately small and fixed: greetings, identity, two arithmetic
facts, one translation and one code request. It is a smoke test, not a benchmark.
"""
# ruff: noqa: D101, D102, D103

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PROBES: tuple[tuple[str, str], ...] = (
    ("Привет!", "привет"),
    ("Как тебя зовут?", "dormouse"),
    ("Кто ты?", "dormouse"),
    ("Сколько будет 2+2?", "4"),
    ("What is 7 times 8?", "56"),
    ("Переведи hello на русский.", "привет"),
    ("Напиши функцию Python, которая складывает два числа.", "def"),
)


@dataclass
class ChatModel:
    model: Any
    spec: Any
    encode: Any
    decode: Any
    device: Any
    vram_mb: int = 0
    device_name: str = "CPU"

    def answer(self, text: str, max_tokens: int = 48) -> str:
        import torch

        from dormouse.data.tokenizer import END_ID, format_turn

        prompt = format_turn("user", text) + "<|assistant|>"
        ids = list(self.encode(prompt))[-(self.spec.seq_len - 8) :]
        produced: list[int] = []
        with torch.inference_mode():
            for _ in range(max_tokens):
                width = self.spec.seq_len
                padded = ids + [0] * (width - len(ids))
                logits = self.model(torch.tensor([padded], dtype=torch.long, device=self.device))[
                    0, len(ids) - 1
                ]
                next_id = int(logits.argmax())
                if next_id == END_ID:
                    break
                ids.append(next_id)
                produced.append(next_id)
                text = self.decode(produced)
                if "<|end|>" in text or "<|user|>" in text:
                    break
        text = self.decode(produced).split("<|end|>")[0].split("<|user|>")[0]
        return text.lstrip(" ,.:-").strip()


def load(checkpoint: Path, config: Path) -> ChatModel:
    import torch

    from dormouse.model.assemble import assemble
    from dormouse.model.spec import StackSpec
    from dormouse.train.talker import vocabulary

    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = assemble(spec, device=str(device)).model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model") or ckpt["state"])
    model.eval()
    encode, decode = vocabulary(spec)
    if device.type == "cuda":
        vram = torch.cuda.memory_allocated(device) // (1024 * 1024)
        name = torch.cuda.get_device_name(device)
    else:
        vram, name = 0, "CPU"
    return ChatModel(model, spec, encode, decode, next(model.parameters()).device, vram, name)


def score(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    chat = load(args.checkpoint, args.config)
    rows = []
    for question, needle in PROBES:
        answer = chat.answer(question)
        rows.append({"q": question, "a": answer, "expect": needle, "hit": score(answer, needle)})
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        for row in rows:
            print(f"{'✓' if row['hit'] else '·'} {row['q']}\n  {row['a']}")
    print(f"\n{sum(r['hit'] for r in rows)}/{len(rows)} probes hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
