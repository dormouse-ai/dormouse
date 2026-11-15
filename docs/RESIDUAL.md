# The two residual mechanisms, and the A/B that decides whether either stays

Both arms were written off on measurements a later audit found invalid. This file
records what they are, the init each one now has, the gates that fail a wrong one, and
the A/B that decides whether they stay — written BEFORE the A/B is run, so the bar
cannot move afterwards.

Code: `src/dormouse/model/residual.py` (`AttnResFull`, `MHC`), the `residual` slot in
`model/spec.py`, the write in `model/blocks.py`, and `tests/test_residual.py` (the
gates). Slot values: `residual = "plain" | "attnres" | "mhc"`.

## What was wrong, in one line each

1. **mHC started at the opposite corner of the Birkhoff polytope.** The port set
   `b_res = 0`, so `H_res` was exactly `J/n` — measured `‖H − J/n‖ = 0.0000`,
   `‖H − I‖ = 1.0000` at n = 2 — i.e. **uniform mixing instead of the identity**. Every
   residual stream started as the average of all of them. That single line reconciles
   the two contradictory verdicts on record (−0.078 in October, +0.0853 in November):
   both measured the average, not the mechanism.
2. **AttnRes was judged at a batch it could not run.** The full arm built a
   `[T, S, d]` score tensor — 507 MiB at b24×1024, bf16, layer 12 — and OOMed at b24,
   b20 and b16. No verdict exists for it, only "could not be measured".

## AttnRes-full, `N = 2L` (arXiv:2603.15031)

One source per **sublayer** — the paper's own layer count (§2: each self-attention and
each MLP is an individual layer), so our 12 blocks are 24 layers and `N = L` is 24
sources. The write at sublayer `i` is

    x_{i+1} = x_i + a,   a = Σ_j w_j u_j,   w = S·softmax([log n, z_1, …, z_n])
    z_j      = q_i · RMSNorm(u_j) + prior[dist(i, j)],   S = n + 1

with `u_0 = 0` the zero source that carries `log n`, `u_n` this sublayer's own output,
`q_i = 0` at init (§5 mandates it) and `prior = (0, −30, −30, …)`.

**Why the init is exactly the identity, not approximately.** At `q = 0` the weights are
`(n, e^{-30}, …, 1)`. The zero source's contribution to the denominator is written as
the integer `n`, not as `exp(log n)` — `exp(fl(log n)) ≠ n` (11.9999997 at n = 12), and
that one-ulp error is exactly what would move the write off the plain residual. Written
as `n`, the denominator is exactly `n + 1 = S`, so the current source's weight is
`S/S = 1.0` bit-exactly, and the older sources' `e^{-30}` tail (9.4e−14 relative) rounds
away in the final add. Measured: `torch.equal` against the plain arm's logits at 1, 2,
4 and 8 blocks, and against `x + u` for 24 consecutive sources.

**Cost.** `2L·(d+1) = 19,992` parameters at d = 832 — 0.002 % of the 1.05B stored,
0.020 % of the 98.4M active per token (`dorm plan` measures both deltas, not a formula:
1,049,546,452 -> 1,049,566,444). The per-BLOCK variant (`N = L = 12`, +9,996) is NOT
bit-identical at init: `x + (a + f)` is not `(x + a) + f` in fp32, so the `torch.equal`
gate forces the per-sublayer form.

**Memory.** The scores are one `[b, t]` column per source, never a `[T, S, d]` stack;
each source's fp32 RMSNorm transient lives inside a `checkpoint` segment and is
recomputed in the backward instead of retained. Measured at the live shape
(b24×1024 → T = 24576, d = 832, S = 24, fp32, forward):

| arm | peak RSS | largest single allocation |
|---|---:|---:|
| plain `x = x + u` (24 writes) | 2,641 MiB | 78.0 MiB |
| AttnRes-full, one source at a time | 2,892 MiB | 78.0 MiB |
| AttnRes-full, the old `[T, S, d]` stack | — | **1,872 MiB** (507 MiB at S = 13, bf16) |

**+251 MiB over plain** (the gate is 0.5 GiB), and no allocation above one `[T, d]`.

## mHC (arXiv:2512.24880)

`n` full-width lanes, `H_pre`/`H_post`/`H_res` from Eq. 7/8, `H_res` projected onto the
Birkhoff polytope by 20 log-domain Sinkhorn-Knopp iterations (Eq. 9). Divergences from
the port, each with its reason:

* `b_res = 12·I` (+ `log(n−1)` per lane so the leakage is `e^{-12}` at any n). The
  leakage of the identity anchor is `(n−1)·e^{-b}`; 12 gives **6.1e−6 ≤ 1e−5** at
  n = 2, 3, 4 (measured 6.14e−6, 6.20e−6, 6.08e−6). The crate's 10 leaves 4.5e−5 and
  misses the gate.
* The lanes carry the FULL state and the read-out is their **mean**; the paper's lanes
  are `d/n`-wide slices that re-sum to `d` (2409.19606 §2.1). `H_pre` is normalised to
  a convex mix for the same reason — `n` copies read as `n·x` otherwise.
* Model-level init: `max|Δ logits| = 7.7e−5` against the plain arm (a `K×` or `K`-fold
  error is 1e0).

**Cost.** `+13,323` parameters at n = 2, d = 832 (1,049,546,452 -> 1,049,559,775 — the
port's count) and `2·T·d·4` of live lanes at b24×1024 = 156.0 MiB.

## The gates (`tests/test_residual.py`)

| gate | asserted |
|---|---|
| AttnRes identity, mixer | `torch.equal(write(u), x + u)`, 1..24 sources |
| AttnRes identity, model | `torch.equal(logits_attnres, logits_plain)`, L = 1, 2, 4, 8 |
| mHC identity | `‖H_res − I‖∞ ≤ 1e-5` at n = 2, 3, 4, and `‖H − J/n‖ > 0.5` |
| mHC regression | `b_res = 0` ⇒ `H = J/n` and the identity gate fails (the audit's line) |
| plain arm | no mixer, no parameters, `param_report["residual"] == 0` |
| causality | a later write leaves the earlier residuals bit-identical |
| it learns | 200 CPU steps, three arms |
| memory | `[T,S,d]` never allocated; +0.25 GiB at the live shape |
| counters | `attnres_sources` / `mhc_mix` on `Counters.line()`, zero when off |

Counters per forward, at L = 12: `attnres_sources = 1+2+…+24 = 300`, `mhc_mix = 24`.

## Measured learning, 200 CPU steps, the `test_step.py` fixture

| arm | step 0 | step 199 |
|---|---:|---:|
| plain | 26.73 | 0.05 |
| AttnRes-full | 26.73 | 0.02 |
| mHC | 26.73 | 0.01 |

All three descend from the same value at step 0 — which is the identity-init claim
measured on the loss, not only on the weights. This is a wiring gate, not a quality
gate: 200 steps on one 3×17 batch only proves the mechanism trains.

## The A/B, pre-registered

**Arms.** `configs/dormouse_1b.toml` unchanged except `residual`, one run per value:
`plain` (control), `attnres`, `mhc`.

**Protocol.** 3 seeds per arm (seeds are the data order + the init draw), 120 MB
corpus, the same b24 schedule as the control on record, 4,880 steps, eval BPB every
610 steps on the held-out window. Report the mean and the spread per arm; a single
seed is not a measurement (the recorded single-seed spread is 0.008–0.011 BPB).

**Bar.**

* **AttnRes-full stays iff** the mean BPB is at least **0.01 BPB better than `plain`**
  on all three seeds, at a memory cost ≤ 0.5 GiB and a step-time cost ≤ 3 %.
* **mHC stays iff** the same bar is met. Its lanes cost 2× the residual residency, so
  the memory ceiling applies to the batch, not the mechanism: if b24 no longer fits at
  the measured +156 MiB, the arm is compared at the batch both arms fit and the batch
  is recorded next to the BPB.
* Either arm **fails** if any seed is worse than the control's mean — a mechanism that
  needs the best of three seeds has no effect.

**What counts as a valid measurement this time.** The card, the batch, the peak VRAM
per arm, and the init norm of the mechanism (`‖H − I‖∞` for mHC, `max|Δ logits| = 0.0`
for AttnRes) are on the same record as the BPB. A verdict without them is the one we
have already had to withdraw twice.
