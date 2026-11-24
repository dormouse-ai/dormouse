# Friday demo — 200M-active chat model, honest numbers

Worktree `wt/demo-fri` (branch `demo/fri`, HEAD `cf3607a`). Nothing committed.
One 5070 Ti (16 GB), CUDA torch via `app/.venv-cuda`, all card runs under
`flock /tmp/dormouse-gpu.lock`. bf16 autocast + fp16 Newton–Schulz Muon are engine
defaults at this HEAD; `device = "auto"` selects the card.

## Root cause of the first (rejected) journal — and the fix

The first run was stopped: loss 900→150 looked like a data-unit bug, but the data was
valid. Measuring the forward at init on CPU, before any CE, showed the logits scale
with `sqrt(d_model)` because the head is tied to an `N(0,1)` embedding while RMSNorm
makes the final hidden state unit-RMS:

| config | d_model | logit min/max/std at init | loss0 | bpb0 | root | fix |
|---|---:|---|---:|---:|---|---|
| `configs/smoke.toml` | 32 | -19.8 / 42.1 / 6.0 | 31.75 | 11.20 | tied head `N(0,1)` | — |
| `configs/fastchat_400m.toml` | 832 | -160 / 803 / 29.1 | 721.8 | 254.7 | same, ×√832 | — |
| `configs/demo_200m.toml` | 2816 | -291 / 1201 / 53.3 | 901.6 | 317.6 | same, ×√2816 | `embed std = 1/√d_model` |
| same, after fix | 2816 | — / — / 1.00 | 10.91 | 3.84 | — | applied in `model/model.py` |

The bad journal is preserved at `runs/demo_200m_badinit/` and is NOT a demo result.
Post-fix evidence: 2-step GPU smoke `bpb 3.4045`; `scripts/stream_check.py` on the
demo stream 200 steps: `loss0 10.9345 → 8.1551`, `heldout_bpb 2.5342`,
`learned=true, bpb_in_band=true`.

## What actually fits in the budget

A smoke at the target widths (b2 s1024, d_model 2816, 12 layers, rank-256 MoA,
8 experts) measured **8.35 s/step** (`runs/../demo_timing2` journal: 20 steps,
step_s 8.16–8.83 after warm-up), i.e. **~246 tokens/s**, not the 45 ms/step in the
brief. CUDA-graph replay was tried and did NOT improve it (8.15 s/step at step 1).

Consequences, stated before any table:

* 5B tokens = 2.44M steps ≈ **5.6 years** at 246 tok/s. 150M tokens ≈ **7 days**.
  The delivered demo trains **1.23M tokens / 600 steps ≈ 83 min**, which is what
  the card and the Friday deadline allow.
* 10B stored parameters alone are ~20 GB in bf16, before optimizer state or
  activations; the 16 GB card cannot hold them. The 200M-active stack here stores
  269.8M (ratio 74%), the 150M-active variant 838M (18%). Engram LUT in host DRAM
  and RAG are NOT implemented — they are new mechanisms, not surviving levers.
* Assistant-only loss is NOT implemented in `train/step.py`: the SFT stream is
  chat-formatted with `format_turn`, but loss covers every token.

## Configs

| variant | config | stored | active | ratio | tokens | steps | hours | ms/step | G0 FLOP/tok | IQ smoke |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 200M | `configs/demo_200m.toml` | 269.8M | 200.1M | 74.2% | 1,228,800 | 600 | 1.36 | 8185 | 1.20e9 PASS | PASS (falsifier) |
| 150M | `configs/demo_extreme_150m.toml` | 838.1M | 154.7M | 18.5% | 1,228,800 | 600 | — | — | 1.20e9 PASS | config-only, untrained |
| 500M/10B | `configs/demo_500m_10b.toml` | 10.056B | 513.2M | 5.1% | — | — | — | — | 6·active=3.08e9 PASS (arithmetic) | config-only, cannot fit card |

G0 gate: executed FLOP/token = `2.4583e12 / 2048 = 1.20e9`, under the 5e9 ceiling.
Both configs are the one architecture; widths were chosen on the meta device and
checked with `assemble(..., device="meta").report()`.

## Data

| file | tokens | bytes/token | documents | tokenizer |
|---|---:|---:|---:|---|
| `data-prep-2h/demo_pretrain.bin` | 61,359,631 | 4.097287 | 72,974 | `50235922e0bacb3a` |
| ├ DCLM | 46,976,161 | 4.013881 | 35,834 | NEW 14-special |
| └ SYNTH | 14,383,470 | 4.369688 | 37,140 | NEW 14-special (23.4%) |
| `data-prep-2h/demo_heldout.bin` | 1,992,244 | 4.206775 | 1,562 | NEW, 0 doc overlap |
| `data-prep-2h/demo_sft.bin` | 1,369,594 | 2.758746 | 30,000 | NEW, synthetic RU/EN chat |

SYNTH is Q/A pairs from the 9 local parquet files (pyarrow installed to
`/tmp/opencode/parquet-libs`, not added to the project). Every manifest carries a
measured BPT; the merged pretrain stream is interleaved 3 DCLM : 1 SYNTH by token
window so one BPT serves the whole run. Ids are uint16, min 14, max 32767.

## Stage 1 / Stage 2

| stage | config | tokens | steps | hours/days | BPB heldout | IQ smoke |
|---|---|---:|---:|---:|---:|---|
| pretrain | `demo_200m.toml` | 1,228,800 | 600 | 1.36 h | 2.614 | PASS (falsifier) |
| SFT continuation | `demo_200m_sft.toml` | 102,400 | 50 | partial | chat loss 5.25→2.36 (BPB 2.75→1.23) | no artifact |
| total | — | 1,331,200 | 650 | ~1.5 h | pretrain 2.614 | pretrain checkpoint used |

The SFT continuation ran chat-format steps from the pretrain weights (loss fell
5.25→2.36 in 50 steps), but each resume process was terminated by the shared
environment before its final artifact write; `runs/demo_200m_sft/` holds the
journal and the seeded checkpoint, not a finished SFT model. The IQ and loop
checkpoint below is therefore `runs/demo_200m/export/model.pt` (pretrain).

The 5B/10B-token continuations in the brief are arithmetic targets only; they are
not runnable on this card inside the demo window (see first section).

## What the 48-hour brief cannot get on this card

At the measured 8.18 s/step (2048 tokens/step = 250 tok/s), 1.3B tokens need
~60 days of GPU time, not 48 hours. The 500M/10B config is therefore meta-priced
only; the trained artifact is the 200M/270M stack. PiKa is not on disk, so the
SFT stage uses a locally generated 30k-example RU/EN chat set; no assistant-only
loss mask exists in `train/step.py`. LARE, Engram LUT, RAG and DCLM top-10%
reweighting are not implemented mechanisms in this engine and were not guessed at.

## F0 falsifier (before any scored run)

`tools/demo_falsifier.py` drives `tools/stub_server.py` → `opencode_solver.py`
(provider `local/dormouse`) → the same IQ scorer `tools/iq_eval.py` uses:

| mode | score | verdict |
|---|---:|---|
| RANDOM seed 0 | 0.000 | PASS floor |
| ORACLE seed 0 | 1.000 | PASS ceiling |
| iq_eval mock / known | 0.000 / 1.000 | PASS |

## F1/F2 inference loop (S6)

`tools/demo_loops.py::graft(model, window=(7,10), K=2, beta=0.5, mode="layer")`
applies the damped Euler update `x ← (1−β)x + β·blocks[7..10](x)` K times. No kernel,
weight or optimizer edit; the CPU self-test proves 492 `state_dict` tensors bitwise
identical and changed logits. `tools/demo_loop_eval.py` runs the paired A/B.

| baseline | looped | Δscore | wins | regressions | latency ratio | anchor | verdict |
|---:|---:|---:|---:|---:|---:|---|---|
| not scored | not scored | — | — | — | — | F3 not run | BLOCKED: GPU lock held 4 h by sibling; tool validated on a tiny CPU checkpoint (`demo_loop_dry.json`) |

`tools/demo_loop_eval.py --checkpoint runs/demo_200m/export/model.pt --window 7 10 --K 2 --beta 0.5`
is the exact command to run when the card lock is free; it emits
`/tmp/opencode/demo_loop_results.json` and refuses green unless wins>=3, regressions=0
and latency<=2x. F1's own CPU self-test passes: 492 state_dict tensors bitwise, logits changed.

Scored names are **IQ_smoke_base / IQ_smoke_loop**. No MMLU-Pro number is claimed:
the +2.64 figure belongs to a different model and protocol. F3 (LoopCD-Hidden
ω sweep) and F4 (binomial K=3 ablation) run only if F2 is green; otherwise the
tool records an honest REFUTE and leaves them not run.

## Gates

| gate | status |
|---|---|
| real `run.jsonl` with step events, finite BPB (not 1030) | PASS: 600 step events, heldout BPB 2.614 |
| IQ smoke falsifier RANDOM=0 / ORACLE=1 | PASS |
| demo answers 2–3 RU/EN questions via OpenCode adapter | BLOCKED: CPU generation >300 s/question; GPU lock held by sibling |
| `make check` green | 651 passed, 28 skipped, coverage 93.85%; 2 pre-existing env failures remain (`test_cubin_gate` needs a free GPU/cubin build; `test_muon_foreach` launch-count 80 vs 75 reproduced on clean `app/` checkout). All new files ruff/mypy clean; `make regression` baseline re-recorded 2.356925 after the init fix |
| F2 loop A/B, latency ≤2×, 0 train FLOP | BLOCKED: GPU lock (tool dry-run REFUTE on tiny stack) |
