# FLOPs-per-byte A/B — fewer active experts / lower rank (2026-11-16)

Base: `origin/master` = `c8f1900` (the speed stack, ~9.8 kB/s at b8×s1024).
Branch: `ab/top1-r192-flops`. All card work ran under
`flock /tmp/dormouse-gpu.lock`; GPU load before and after every chain was
992 MiB / 0–21 % (desktop context only). No model or kernel source was changed.

## What was built

- `configs/dormouse_1b_top1_r192.toml` — the control with exactly two value changes
  (`topk 2→1`, `expert_tsct_rank 256→192`). `tests/test_ab_configs.py` asserts the
  parsed configs differ in exactly those two keys; the extra header comment is a
  verdict, not an assignment. `tests/test_spec.py` was updated to name the new file.
- `scripts/ab_make_configs.py`, `scripts/ab_make_data.py`, `scripts/ab_train.py`,
  `scripts/ab_chain.sh`, `scripts/ab_speed.sh`, `scripts/ab_report.py` — the paired
  harness. Every number below is read back from a file under `runs/ab_top1_r192/`.

## Speed — b8 × s1024, bf16, three fresh runs per arm

`tools/profile_step.py --batch 8 --seq 1024 --steps 3 --dtype bf16`.

| arm | ms/step (3 runs) | median ms | tok/s | B/s (4.72 B/tok) | launches/step |
|---|---|---:|---:|---:|---:|
| control topk2/r256 | 4028.0 / 3872.9 / 3918.7 | 3918.7 | 2,090 | 9,860 | 25,530 |
| topk1/r256 | 3404.1 / 3422.4 / 3393.9 | 3404.1 | 2,407 | 11,356 | 24,562 |

Phase table (median-shaped run, ms/step): control forward 682.0, backward 2287.3,
Muon 997.0, other ~9.4; topk1 forward 600.9, backward 1870.3, Muon 998.5. The
saving is in the bank's forward/backward, not in the optimizer — which is why it is
1.15× and not the 1.30× the 12 % FLOP cut would suggest if the step were compute-bound.

Evidence: `runs/ab_top1_r192/speed.jsonl` and `speed_{ctrl,topk1}_{1,2,3}/profile.txt`.

## Quality — 3 seeds × 4,000 steps on the 200 MB slice: BLOCKED, no BPB exists

| arm | seed 0 | seed 1 | seed 2 | mean delta |
|---|---:|---:|---:|---:|
| control topk2/r256 | — blocked — | — blocked — | — blocked — | — |
| arm topk1/r192 | — refused — | — refused — | — refused — | — |

Three independent blockers, each from a guard that was left intact:

1. **r192 is refused by the fused kernel.**
   `bank.py:675`: *"the fused bank's rank is 192, not a power of two >= 16 … use
   16, 32, 64, 128 or 256 … or run with DORMOUSE_BANK=0"*. The escape switches
   kernels, which would make the speed number a different stack, so it was not used.
   Verbatim capture: `runs/ab_top1_r192/speed_arm_refused/profile.txt`.
2. **The 1B config cannot train on this corpus with the shipped guard.**
   `moe.py:575`: *"the routing needs 5 dispatch passes: one expert took 1113 rows
   against a 256-row slot … past 4 … the router has collapsed"*. It fires within
   ~20 steps at the shipped Muon rate 0.02 (loss starts at ~720 nats because the
   832-wide table is untrained), and within ~66 steps even at 0.002. fp32 is not an
   alternative: the bank's backward needs 151 KiB of shared memory against a 101 KiB
   hardware limit. Full trace: `runs/ab_top1_r192/chain.log` (attempts 1–2).
3. **The 200 MB bpe slice is a 32768-id tokenizer but the engine's bpe table has
   32000 rows.** A derived, id-clamped copy (0.29 % of ids, `→ 0`) was built for both
   arms so a run could at least start; it is data prep for the A/B, not a default.

No BPB was produced, so no per-seed delta and no mean delta can be quoted. Reporting
one from the 5–66 steps that ran before the guard fired would be a number, but not
the requested measurement.

## Pre-registered decision, evaluated explicitly

- **(a) speed ≥ 1.30×: FAIL** — measured 1.151× (3918.7 / 3404.1 ms).
- **(b) BPB loss ≤ 3 % on all three seeds: NOT EVALUABLE** — no completed run.
- **Verdict: REFUSE.** The arm stays in `configs/` with its measured verdict in the
  header; no default changed. Because (a) fails on its own, no follow-up attribution
  run could change the verdict: `r192` is refused outright and `topk1` alone is the
  1.151× already measured.

## Projected hours for 4 B tokens

At the measured live shape and 4.717789 B/token from the slice manifest:

| arm | tok/s | hours for 4 B tokens |
|---|---:|---:|
| control | 2,090 | 531.8 h |
| topk1/r256 | 2,407 | 461.6 h (−70.2 h, −13.2 %) |

`hours = 4e9 / tok_s / 3600`. This is the speed projection only; it does not
discount the unmeasured quality cost.
