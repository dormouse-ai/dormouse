# The two-hour chat recipe

One config, one card, one number: `configs/fastchat_400m.toml` spends **≤2.0 h** on
1x RTX 5070 Ti at the measured whole-step **39.92% MFU** (92 TFLOP/s peak,
commit `6e210b6`). Everything below is a gate or a command; nothing here is a
target that gets rounded up when it is missed.

| config | P (stored / active) | D (tokens) | hours @39.92% MFU | gate |
|---|---:|---:|---:|---|
| `configs/fastchat_400m.toml` | 400,985,812 / 68,506,348 | 108,000,000 | **1.965** (headline 6·P_stored·D = 2.598e17 FLOP) | `tests/test_fastchat_config.py` |
| same, engine bill 6·P_active·D | — | 108,000,000 | 0.336 (4.437e16 FLOP, 407 MFLOP/token) | reported, not the gate |

`uv run --no-sync python tools/fastchat_bench.py` prints the same table from the
meta-built parameter report. **Why stored, not 400M active:** this stack shares one
bank across layers, so 68.5M active is what 401M stored buys. Reaching 400M active
means ~3.8B stored (measured: d_model=2048, 16 layers, d_ffn=16384, rank=1024,
topk=4) and ~46 GB of bf16 optimizer state — it does not fit a 5070 Ti. The headline
charges **every stored bank parameter**,
so it is the conservative price: if the routed bank stops saving FLOPs, the run
still fits. The smaller engine bill is why 108M tokens is enough data, not why the
2-hour gate is allowed to pass.

## What the config does and does not carry

At this worktree's HEAD (`6abf49b`) `StackSpec` has no `config_version` or `amp`
field, and `RunRecipe.read` refuses unknown keys. So the file carries only widths
the spec prices and builds. The run identity is the **401M topology**
(`config_version = 1` once the v2 spec lands); precision is **bf16 autocast**;
**fp16 Newton–Schulz** in Muon and the **fused epilogue** are engine defaults at
this HEAD; the captured step graph is `torch_compile = true` plus `dorm train
--graph`. Adding `config_version`, `amp`, `fp16_ns`, `graph` or `epilogue` as TOML
keys today would make `dorm plan` refuse the file by name — do not do it.

## Phase (a) — 90M tokens of SYNTH-enriched pretrain

Source: the filtered corpus enriched with SYNTH-generated tasks (arithmetic,
filling, factual QA, code completion; the same families the IQ smoke suite scores).
SYNTH at the 20B scale was measured at **×5.5 effective, not ×10** (grill/65), from
3.6B physical tokens. At this scale we spend **90M physical tokens**, not 0.6B:
scaling the physical budget down does NOT carry the ×5.5 with it as free FLOPs.
Treat that multiplier as a hypothesis — the gate below measures what, if anything,
the enrichment bought. The remaining 18M tokens of the 108M budget are phase (b).

## Phase (b) — SFT on 30–70k multi-turn pairs

Use ready pairs, not generated ones: **PiKa** and **UltraChat** (CC BY / MIT),
converted through the same `tokenizer32k.json` the token table answers to.
70k pairs × ≈256 tokens ≈ 18M tokens, which is the rest of the budget. The check
is behavioural, not a loss curve: the model stops at the end marker, answers in
first person, knows its name, and answers in Russian and English.

## Phase (c) — gates, in order

1. **BPB window.** `src/dormouse/eval/bpb.py` on a held-out shard; report the
   window, not a point. Phase (a) passes only if the SYNTH-enriched mix beats the
   same-token clean mix outside the overlapping window. A `0.000` BPB with an empty
   window is an error, not a score.
2. **IQ harness smoke.** The same core the run uses:
   `uv run --no-sync python tools/iq_eval.py --model mock --suite smoke --json`
   must average **0.0** (random floor) and
   `uv run --no-sync python tools/iq_eval.py --model known --suite smoke --json`
   must average **1.0** (oracle ceiling). `tests/test_fastchat_config.py`
   asserts both in-process, so a smoke wiring that cannot fail is a red build.
   This is a plumbing falsifier, not an IQ number; see `docs/iq-harness.md`.
3. **Cost gate.** `make check` green, then
   `uv run --no-sync python tools/fastchat_bench.py` — headline hours ≤ 2.0.

## Run

```
dorm plan configs/fastchat_400m.toml --graph
dorm train configs/fastchat_400m.toml --graph --yes
dorm status runs/fastchat_400m/run.jsonl
```

`--graph` on both commands: a plan priced eager and a train run replaying a graph
are two different recipes. `--yes` is required for a run longer than an hour.
The journal is the run; read it with `dorm status`, never with `grep`.

## Quality boundary, stated honestly

From grill/64 at this scale: 360M total params → MT-Bench 3.66, 500M → 4.16.
401M stored sits between them, so the expected band is **≈3.7–4.2**, not "good
chatbot". This config buys a measurable format-following model inside two hours;
it does not claim a quality the arithmetic cannot support.
