# Negative results (named)

## NR-2026-11-24-01 — DDP equivalence falsifier floor 1e-3 -> 2e-6

**Killed claim:** `test_per_microbatch_mean_fails_the_equivalence_gate` still
falsifies the per-micro-batch-mean bug at its old `worst > 1e-3` floor.

**Why it was suspect:** commit `c185739` scaled the tied-head init to
`1/sqrt(d_model)`. That shrinks logits and gradients, so a fixed absolute drift
floor can stop measuring the bug and start measuring the old init. The same commit
lowered the floor to `2e-6` with the comment "Still 5x the honest run's 1e-6
ceiling" — which did not match the two magnitudes it quoted (5e-6 vs 1e-3 is
200x, and 5e-6 vs 2e-6 is 2.5x).

**Measurement** (CPU fp32, `app/.venv` torch 2.15, HEAD `d5f4cb1`, repeated with
`tests/test_dist.py::run_job`; `worst` is the max absolute weight difference vs
the single-card run):

| run | worst | verdict |
|---|---|---|
| two cards vs one card, honest | 5.662441253662109e-07 | inside `TOL = 1e-6` |
| two cards, `--bug`, vs one card | 5.0067901611328125e-06 | above the new floor |
| old floor `1e-3` on the buggy run | 5.00679e-06 | **passes vacuously — gate is decoration** |

**Verdict:** the old 1e-3 floor is REFUTED as a falsifier; the new floor `2e-6`
is real but narrow: ~3.5x the measured honest drift and ~2.5x below the buggy
drift — not a "5x" margin. It is checked on every run by the existing test, which
also keeps the honest pair under 1e-6. The golden loss rewrites and the regression
baseline move in `c185739` are consequences of the same init fix, not evidence
for this threshold.

**Upgrade path:** if another BLAS build pushes the honest pair above 2e-6, replace
the absolute floor with a relative drift gate calibrated on both arms of the same
build; do not simply raise it until the test goes green. If the bug's drift falls
below 2e-6, the injected control must be strengthened instead of the floor lowered.
