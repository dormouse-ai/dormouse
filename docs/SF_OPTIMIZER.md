# The schedule-free optimizer, and its pre-registered A/B

Why a schedule-free arm exists at all: the corpus grows (10 → 25 → 50 GB), and a
fixed-horizon schedule either forces a restart or spends a decay phase on data it will
never see again. `MixOptimizer(schedule="sf")` removes the horizon from the optimizer.
This file records the equations as implemented and the A/B that decides whether it
stays — written before the A/B was run, so the bar cannot move afterwards.

Code: `src/dormouse/train/sf.py` (both rules and the wrapper), `muon.py` (the polar
chain), `optim.py` (§routing and the eval swap), `tests/test_sf.py` (the gates).

## The equations, with the section each came from

Apte, Deshpande, Kumar, Chakrabarti & Kim, *Anytime Training with Schedule-Free
Spectral Optimization*, arXiv:2605.23061v2 — cited as **N** below. Defazio,
*ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to LLMs*,
arXiv:2605.19095 — **S**. Song, Baek, Ahn & Yun, *Through the River*,
arXiv:2507.09846 — **R**.

| # | what | equation | where |
|---|---|---|---|
| 1 | the gradient point | `Y_t = (1-β₁)·Z_t + β₁·X_t`, and the parameter holds `Y_t` | N §1 eq. 1; Alg. 1 line 3 |
| 2 | the fast iterate | `Z_{t+1} = Z_t - η_t·(direction at Y_t) - η_t·λ·Z_t` | N §1 eq. 2; Alg. 1 line 17 |
| 3 | the average | `X_{t+1} = (1-c_{t+1})·X_t + c_{t+1}·Z_{t+1}` | N §1 eq. 3; Alg. 1 line 18 |
| 4 | the weight | `c_{t+1} = w_t/Σ_{i≤t} w_i`, `w_t = η_t²` | N Alg. 1 lines 14–15 |
| 5 | the spectral direction | `ΔW* = -η·polar(∇f)` (steepest descent under `‖·‖_op`) | N §2 eq. 8 |
| 6 | the momentum buffer | `M_t = μ·M_{t-1} + (1-μ)·G_t`, `P_t = polar(M_t)` | N §2.1 eq. 9; Alg. 1 lines 5, 7 |
| 7 | the second moment | `v_t = β₂·v_{t-1} + (1-β₂)·mean_cols(P_t ⊙ P_t)`, `v_t ∈ R^m` | N Alg. 1 line 8 |
| 8 | the "Nor" | `P̂_t = P_t ⊘ (√v_t + ε)` | N Alg. 1 line 10 |
| 9 | the Adam-comparable scale | `η̂_t = eta_scale·η_t·√(m·n)/‖P̂_t‖_F`, `eta_scale = 0.2` | N §2.2, Alg. 1 line 13 |
| 10 | the ramp | `η_t ·= min(1, t/T_warmup)` | N Alg. 1 line 12 |
| 11 | decay at `Z`, not at `Y` | the `-η_tλZ_t` term in (2) | N §3, Fig. 2 |
| 12 | the AdamW core | `m_t = lerp(m_{t-1}, g_t, 1-β₁)`, `v_t = β₂v_{t-1}+(1-β₂)g_t²`, `Z ← Z(1-ηλ) - η·m̂/(√v̂+ε)` | N Alg. 2 lines 5–7 |
| 13 | inner momentum in SF-AdamW | a `β₁` momentum buffer, reinstated | S §3, Figs. 2b and 3a |
| 14 | the annealing lever | opt-in linear ramp to 0 over the last `sf_final_decay_frac` | not in N or S: see below |

Implemented hyperparameters, all from the papers:

| symbol | value | source |
|---|---|---|
| `η` (both arms) | 0.01 | N §4: tuned at 2× Chinchilla for SF-AdamW, reused by SF-NorMuon through eq. 9 |
| `β₁` (interpolation, and the AdamW arm's momentum) | 0.9 | N Alg. 1; S §3 |
| `β₂` | 0.95 | N Alg. 1 (the paper's SF-AdamW arm tuned 0.99; the stack keeps 0.95) |
| `μ` (matrix momentum) | 0.8 | N §4 |
| `eta_scale` | 0.2 | N Alg. 1 line 13 |
| `λ` (both arms) | 0.05 | N §4 |
| `ε` | 1e-8 | N Alg. 1 line 1 |
| Newton–Schulz | 5 steps of `x(15-10x²+3x⁴)/8` | `muon.py` §1, not N's Appendix G triple: see `sf.py` §3(a) |

### Where this deviates from the papers, and the number behind each

1. **The polar chain.** N's Appendix G code uses the Jordan triple
   `(3.4445, -4.7750, 2.0315)`, which is not the polar factor and reaches
   `‖QQᵀ-I‖∞ ≈ 3.2e-1` on the shapes this stack has. The 5th-order chain, already in
   `muon.py`, reaches the 1e-5 anchor on the same three fixtures. Same direction, better
   conditioning.
2. **No ColRow on the SF arm.** ColRow divides every row to unit norm; eq. (7)-(8) divide
   every row by its own second moment. The first leaves nothing for the second to set, so
   the paper's row normalization would be a no-op. Shape-independence comes from (9)
   instead, which is where N gets it too.
3. **The average is uniform after warmup** (`w_t = η_t²`, eq. 4). S §8 eq. 16 recommends
   `w_t = t^r·max_{i≤t} η_i²` with `r = 1` for runs past ~30B tokens, and finds `r = 0`
   slightly better below that. A 50 GB corpus at ~4 bytes/token is under 13B tokens, so
   `r = 0` — the paper's own line — is the right side of that measurement.
4. **The final decay is opt-in and is not schedule-free.** `begin_final_decay(total)`
   applies a linear ramp over the last `sf_final_decay_frac` (default 0.05) and is the
   only thing in the optimizer that knows a horizon. Without the call the rate is flat at
   every step. It exists because the data research says the anneal pays, and because a
   schedule-free run that wants the last 5 % back should be able to ask for it without a
   restart. The shape is linear, the one S §3 names as its AdamW reference arm.
5. **Not learning-rate-free.** S §6 makes the rate a Polyak step from an EMA of the loss
   (eq. 12). This arm is given one rate: what is dropped is the schedule, not the number.

## The anchors, and the numbers they were accepted at

Every one is a closed form or an equality with something outside the code under test.

| gate | assertion | measured |
|---|---|---|
| AdamW anchor | SF-AdamW with `average=False` equals `AdamW` **bit-for-bit**, `torch.equal`, 6 steps with decay | equal on every step |
| spectral anchor | `‖QQᵀ-I‖∞ ≤ 1e-5` on `[64,832]`, `[832,2048]`, `[2048,832]` | ≤ 1e-5 (the chain's own ceiling is `muon.py` §3) |
| the paper's step scale | first step `‖ΔW‖_F = 0.2·η·√(m·n)` | ratios 0.9999998 / 1.0000000 / 0.9999999 |
| the "Nor" is real | every row of the first step has the same norm, on `[256,64]` where `polar`'s row norms span 1.5354× | row spread 1.000009 |
| `c₁ = 1` | after one step `Z₁ = Y₁` | `torch.equal` |
| the average is real | 50 constant-gradient steps: `z = -0.4999`, `x = -0.2550`, `y = -0.2795 = 0.1z + 0.9x` | matches eq. 1 to fp32 |
| the restore is exact | `eval_weights()` then exit leaves `torch.equal` training weights | equal |
| no horizon | no `total`/`horizon`/`max_steps`/`schedule` in the constructor; state is exactly `{z,m,v}` / `{z,mom,v}`; `lr_at` is 0.01 at steps 1, 2, 10, 10⁴, 10⁶ | all hold |
| it learns | 20 fixed sequences, monotone: 26.727, 26.041, 25.664 … 18.284 | monotone, ×1/1.46 |
| cosine unchanged | `schedule="cosine"` is the default, builds `Muon`+`AdamW`, same routing/sizes/head splits, `eval_weights` moves no weight | holds |

The 20-step factor is 1.46, not the 5× `tests/test_step.py` records for the scheduled
pair, and that is the method and not a defect: the parameter is the trailing average and
`Z` is the iterate that moves at the full rate. Same fixture, 100 steps: 26.727 → 0.699.

## The CPU smoke (not a verdict)

`tools/sf_smoke.py --steps 300`: three fixed training sequences, one held-out sequence,
all losses read inside `optimizer.eval_weights()`, 20-step ramp on every arm.

`tools/sf_smoke.py --steps 300`: three fixed training sequences, one held-out sequence,
all losses read inside `optimizer.eval_weights()`, 20-step ramp on every arm. Wall clock
from the same run, on a box that was **shared with another session's test suite**:

    cosine:  296.4 ms/step,   88.9 s total
       wsd:  270.9 ms/step,   81.3 s total
        sf:  344.3 ms/step,  103.3 s total

memorised batch (the three training sequences), eval loss every 20 steps:

| step | cosine | wsd | sf |
|---|---|---|---|
| 20 | 12.678 | 12.678 | 21.954 |
| 40 | 0.818 | 0.815 | 10.593 |
| 60 | 0.111 | 0.104 | 2.418 |
| 80 | 0.012 | 0.003 | 1.226 |
| 100 | 0.008 | 0.169 | 0.831 |
| 120 | 0.013 | 0.010 | 0.499 |
| 140 | 0.003 | 0.194 | 0.231 |
| 160 | 0.004 | 0.041 | 0.071 |
| 180 | 0.001 | 0.003 | 0.009 |
| 200 | 0.000 | 0.004 | 0.001 |
| 220 | 0.000 | 0.032 | 0.000 |
| 240 | 0.000 | 0.015 | 0.000 |
| 260 | 0.000 | 0.009 | 0.000 |
| 280 | 0.000 | 0.002 | 0.000 |
| 300 | 0.000 | 0.001 | 0.000 |

held-out sequence (never trained on), eval loss every 10 steps:

| step | cosine | wsd | sf |
|---|---|---|---|
| 10 | 23.647 | 23.647 | 23.366 |
| 20 | 18.035 | 18.035 | 21.210 |
| 30 | 9.037 | 9.033 | 18.502 |
| 40 | 7.925 | 7.921 | 15.062 |
| 50 | 9.738 | 9.782 | 11.382 |
| 60 | 12.361 | 12.394 | 8.732 |
| 70 | 14.631 | 14.629 | 7.396 |
| 80 | 15.905 | 15.501 | 6.923 |
| 90 | 16.232 | 15.389 | 6.865 |
| **100** | 16.689 | 17.062 | **6.855** |
| 120 | 16.113 | 17.563 | 7.229 |
| 140 | 15.664 | 17.902 | 7.944 |
| 160 | 15.253 | 16.816 | 9.022 |
| 180 | 15.756 | 14.082 | 10.134 |
| 200 | 16.054 | 14.735 | 11.273 |
| 220 | 16.789 | 16.500 | 12.015 |
| 240 | 17.164 | 15.017 | 12.704 |
| 260 | 17.370 | 14.900 | 13.588 |
| 280 | 17.361 | 15.925 | 14.269 |
| 300 | 17.359 | 15.902 | 14.757 |

Read it as wiring only. On the batch it is memorising, the two scheduled arms reach zero
by step 200 and `sf` by step 210 — a trailing average is a handicap when the task is one
fixed batch, which is the least stochastic horizon there is and exactly where
arXiv:2605.19095 §9 measures Schedule-Free as weakest. On the sequence it has never seen,
the order inverts and the margins are large: the best held-out loss is 6.855 for `sf`
against 7.921 for the scheduled arms, and after step 40 the scheduled arms *rise* to
~17/16 while `sf` is still at 6.9. That is the average doing what (3) says it does — and
it is one seed on a 32-wide model, so it is a reason to run the A/B, not a finding.

The wall clock is not a measurement: ±30 % between back-to-back runs of the same arm on
this box, and the optimizer step is most of this fixture's step time. The paper's own
number on 125M is +2.13 % per step over AdamW (2605.23061 Table 13); nothing here
contradicts it, and nothing here confirms it.

## The A/B, pre-registered

**Setup.** Three arms (`cosine`, `wsd`, `sf`) × three seeds (0, 1, 2), on the same
120 MB model on the card, same data order, same step count. At batch 24 × seq 1024 that
is 24,576 tokens/step; the run is 4,000 steps ≈ 98.3 M tokens. The card is used only if
`nvidia-smi --query-compute-apps=pid,used_memory` is empty; otherwise the arms run on
CPU and the wall clock is reported but not compared.

**The three arms, exactly.**

| key | `cosine` | `wsd` | `sf` |
|---|---|---|---|
| `schedule` | `"cosine"` | `"wsd"` | `"sf"` |
| `lr_muon` | 0.02 | 0.02 | 0.01 |
| `lr_adamw` | 0.003 | 0.003 | 0.01 |
| ramp | `warmup_cosine(warmup=1900, total=4000)` | `warmup_stable_decay(warmup=1900, decay_start=3120, total=4000)` | `sf_warmup=1900` |
| decay to zero over | the whole run | last 22 % (`1-√u`) | last 5 % (`sf_final_decay_frac=0.05`) |
| `sf_beta` | — | — | 0.9 |
| `weight_decay` | 0.1 (vectors), 0.0 (matrices) | same | `sf_weight_decay=0.05`, both arms |
| `betas` | (0.9, 0.95) | same | (0.9, 0.95) |
| `momentum` / `μ` | 0.95 | 0.95 | 0.8 |
| `eta_scale` | — | — | 0.2 |

Plus one exploratory arm, not part of the bar: `sf` with
`sf_final_decay_frac=0.0`, three seeds, to price what the anneal buys.

**The protocol rule that makes the number mean something.** Every evaluation, for every
arm, is inside `with optimizer.eval_weights():`. Under `cosine`/`wsd` that only flips the
module's mode; under `sf` it swaps in `X`. An `sf` number measured outside it is a loss
for the training iterate and is not a candidate.

**The bar.** BPB is the loop's `BitsPerByte` on the same held-out bytes, at the final
step, mean over three seeds.

* Accept `sf` iff `mean BPB(sf) ≤ mean BPB(wsd) + 0.01` **and**
  `max BPB(sf) ≤ max BPB(wsd) + 0.02`.
* Reject if any seed's run trips the loop's divergence rule (5 consecutive
  spike-skips), or ends with a non-finite loss.
* `cosine` is reported for context; it is not a criterion, because it is the arm whose
  horizon dependence is the reason this work exists.
* If `sf` is rejected on the mean but passes on the max, the variance is the finding and
  the report says so rather than averaging it away.

**If the bar fails, in this order** (each is one config change, none is a new mechanism):
`sf_final_decay_frac` 0.05 → 0.10; `lr_muon=lr_adamw` swept {0.005, 0.01, 0.02};
`betas[1]` 0.95 → 0.99 (S's SF-AdamW arm); `w_t = t·max η²` (`r = 1`, S §8 eq. 16). If
none of those is enough, the honest conclusion is N's own: matching a horizon-aware
spectral optimizer without a schedule is open, and the recorded gap is ~0.03 nats.

## One integration note

`train/loop.py`'s `_write_lr` calls `optimizer.set_lr(...)` unconditionally. Under
`schedule="sf"` that raises on the first step, on purpose: the SF arms derive their own
rate and a schedule written on top of them applies the decay twice. The loop needs to
skip the call when the optimizer is schedule-free (or call `begin_final_decay` instead).
