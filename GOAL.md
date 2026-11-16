# The goal: a chatbot that is actually good

Set by the owner on 2026-11-15. Not an agent that edits repositories — a chat model
that is pleasant to talk to, knows what it is, answers in Russian and English, and
runs from one file on one card.

## Why this is the right target at our size

Agentic tool use has a measured floor around 0.6–1B **active** parameters. We have
46M active. Chat does not have that floor: a 46M-active model with a good
tokenizer, a clean 25 GB corpus, distillation and SFT lands in the "repeats format,
answers simple questions, holds a conversation" band, and every one of those steps
is measured rather than hoped for.

## The four stages, in order, each with the number that says it worked

| # | stage | data | hours | the gate |
|---|---|---|---|---|
| 1 | **pretrain** the 1B/46M hybrid on the filtered mix | 25 GB | ~50 | held-out BPB below the 2.5 GB control at equal tokens |
| 2 | **distill** from Ternary-Bonsai-2-27B (a ternary Qwen3.8-27B) | 2–5B generated tokens, verifier-filtered | 40–85 | beats the same model without distillation on our 5-task harness |
| 3 | **SFT** on chat pairs: ours + distilled traces | 200–400M | 4–8 | the model stops at the end marker, answers in first person, knows its name |
| 4 | **light RLVR** only if stage 3 passes tests sometimes | 50–100M | 10–20 | pass rate rises from >0 %, never from zero |

## What "good" is measured by

Not vibes and not a benchmark we cannot run. Five things, all local:

1. **Ten fixed prompts** with a written rubric (identity, Russian grammar, one
   factual answer, one refusal, one code snippet, one multi-turn follow-up).
2. **The eval harness's five tasks** — reverse/copy, arithmetic, needle at 4k,
   format-following (stops at the marker), code-executes.
3. **Bits per byte on a held-out shard** — the only number comparable across runs.
4. **Decode speed** on the real card, because a chatbot that takes 30 s to answer
   is not a chatbot.
5. **A human check by the owner**: ten turns, does it feel like a conversation.

## What must exist before stage 1 starts

- `dorm train <config>` runs the whole chain from one command (no bash scripts).
- The 1B config fits 16 GB: 8-bit/bf16 optimizer state is landed; **activations are
  the open blocker** (int4 storage or block checkpointing).
- The mixture enforces its declared shares and refuses a tokenizer mismatch
  (landed, `data/mix.py`).
- The eval harness runs on a checkpoint (landed, `eval/`).
- A journal that explains its own byte count (landed, `train/report.py`).

## What we deliberately do not do

- No agentic tool use at 46M active — the floor is 13× above us.
- No RL before expert iteration shows a non-zero pass rate.
- No architectural work: the stack is settled (KDA+DSA 3:1, GQA+latent, MoE with
  TSCT rank-256 experts, MoA, ternary QAT). Every remaining lever is data,
  distillation and post-training.
