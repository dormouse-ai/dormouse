# LNS — LayerNorm Scaling (arXiv 2502.05795, NeurIPS 2025)

**State: implemented behind `DORMOUSE_LNS=1`, default OFF. Measured quality signal is a
large, consistent improvement; the literal `|ΔBPB|≤0.01` gate REFUSEs because the
improvement exceeds the band. Do not enable by default on this evidence alone.**

## What the paper actually says (primary, v6 22.08.2026)

- Eq. 12 / Fig. 1: scale the output of **both** LayerNorms in block `l` by `1/sqrt(l)`,
  `l ∈ 1..L`. Zero parameters, zero FLOP beyond one multiply per norm.
- Table 1: Pre-LN → LNS PPL 17.02 → 15.71 at 1B, 26.73 → 25.76 at 130M.
  Table 3: OLMo-1B/20B tok loss 2.96 → 2.85. Table 4: Qwen2.5-0.5B 20.62 → 19.57.
- Footnote p.1 / Table 6: LNS **must replace** scaled initialization — combining it
  with `write_std ∝ 1/sqrt(2L)` is worse than LNS alone (250M: 20.35 → 20.79).
- Table 8: placement matters. Scaling **after the residual** diverges (PPL 1358.11);
  "after LN" is the best variant (25.76). The final norm after the stack is not scaled.

`docs/smart-fast.md:30` claimed `lns_replace_init` was already shipped; at HEAD
`93b8206` it was not — `grep lns src/` was empty and `blocks.py:120` still had the
depth init. This work closes that discrepancy with code, not with a doc edit.

## Implementation (paths)

| path | change |
|---|---|
| `src/dormouse/model/blocks.py` | `DORMOUSE_LNS` flag (`lns_enabled()`), `residual_write_std()` (plain `0.02` under LNS), `lns_scale(l)=1/sqrt(l)`, `Block(layer_idx=…)` scales **both** pre-norm outputs before their sublayers and counts `lns_layers`; layer 1's factor is exactly 1 |
| `src/dormouse/model/assemble.py` | one `residual_write_std(n_layers)` for bank + attention writes; blocks numbered 1..L |
| `src/dormouse/model/model.py` | declares mechanism `lns` ↔ counter `lns_layers` in `Transformer._enable`, so a silent arm raises `MissingReport` |
| `tests/test_lns.py` | 8 tests: flag semantics, init swap, assembler numbering, layer index refusal, counter, numpy-fp64 residual anchor, final norm unscaled, wrong-placement rejection |
| `tools/ab_gate.py` | own honest paired protocol (no `ab_gate.py` at this HEAD): 3 seeds × 200 steps, CPU 1 thread + deterministic algorithms, bitwise control-vs-control, heldout BPB, t(df=2) upper-95, per-arm and isolated ms |

The residual math is anchored to **external numpy fp64** with fixed norm outputs and
nonlinear (`tanh`) sublayers: `x + tanh((n1/√l)·A)` then `+ tanh((n2/√l)·B)`, matched
to `rtol=atol=1e-12`. A reference with the scale applied after the residual is
asserted *not* to match — the anchor distinguishes the divergent placement.

## Paired A/B (200 steps, seeds 1234/1235/1236)

Protocol: `tools/regression.py`'s exact 190,200-param stack (4 layers, 3 KDA + 1 DSA,
rank-32 MoA bank), committed 2 MB byte slice, fp32 CPU, `set_num_threads(1)`,
`use_deterministic_algorithms(True)`, AdamW lr `3e-3`, identical batches and init per
seed; arms differ only in `DORMOUSE_LNS` (which also removes the depth init). Heldout
is 8 non-training windows of the same stream, CE in bits per byte (`BPT=1`).
Raw: `/tmp/opencode/lns_ab.json`, tool exit 1 (literal gate).

| variant | dBPB | 95% CI | ms/step | verdict |
|---|---|---|---|---|
| control (`DORMOUSE_LNS` off, scaled init) | — | — | 775.8 (median) | baseline |
| LNS on, scaled init removed | **−0.0866** | [−0.1370, −0.0363] | 775.5 (−0.04%) | **REFUSE** on literal `|dBPB|≤0.01`; `upper95=−0.036≤0.01` PASS, ms PASS |

Per seed: −0.076859 / −0.062887 / −0.120173. Control-vs-control (same seed, same
env, run twice) = **0.000000000 BPB exactly** — the prerequisite from R3-quality §"3
вещи" holds, so the 0.01 gate is measurable in this regime.

### How to read the REFUSE

The effect is in the *improving* direction on all three seeds and upper95 is far
below zero. The literal `|dBPB|≤0.01` check is false because the change is larger
than the band, not because it regressed. That is a reason for suspicion at this
scale, not for enabling: a 0-FLOP norm scale moving heldout BPB by 0.087 after 200
steps on a 190K-param toy stack is ~3× the paper's 1B/20B-token effect (≈0.027 BPB
equivalent). The most likely driver is the **init swap** (residual writes start
`√8×` larger before the `1/√l` scale takes hold), which the paper says must be done
but which dominates a 200-step run. This is a measurement of the configured arm,
not a verdict on the 1B recipe.

### Cost of the arithmetic

- Paired A/B CPU wall: 775.8 → 775.5 ms/step (−0.04%, within the +1% gate). Single-arm
  per-step times swung 616–817 ms under sibling load (loadavg 14), so this number is
  a median across seeds, not a precision timing.
- Interleaved min-of-7 isolated microbench on the same MoE stack, frozen weights,
  `lns_scale` flipped in place: **+17.0%** (654.7 → 766.3 ms). LNS changes the router's
  inputs, hence the top-1 assignment and the bank's grouped work; this is an
  init-transient routing effect, not the cost of the multiplies, and it vanishes in
  the trained A/B arms. Recorded rather than hidden.
- Dense-stack microbench (no bank, 4 layers, d=32): **+1.27%** (12.25 → 12.40 ms) —
  the pure 8 elementwise multiplies; the fraction shrinks with width/depth.
- **GPU ms is UNMEASURED**: `/tmp/dormouse-gpu.lock` was held by foreign jobs and
  11.7/16 GB of VRAM was resident for the whole session. No on-card claim is made.

## Verification

`tests/test_lns.py`: 8 passed. `mypy src`: no issues in 66 files. Ruff
check/format: clean on all changed files. Full `make check`: **green** — 740 passed,
30 skipped, coverage 93.18% (floor 93).

## Follow-ups (not done here)

1. Re-run the paired gate at the paper's regime (1B, 20B tokens, or at least a
   longer horizon) with the same `|dBPB|` and upper-95 checks; the 200-step toy
   effect is dominated by the init swap.
2. On-card ms/step under `flock` when the GPU is free, to close the +1% gate on the
   device that actually runs the recipe.
3. Resolve whether `lerp` of the `1/√l` factor between init and step 0 (only if a
   longer run shows the init swap is the entire Δ) keeps the paper's variance bound.
