# A4 activations — canonical path

One config, one mechanism, one kernel home. `act_quant="a4"` turns on A4 at the two
pre-sublayer norms of every block **and** on the bank's `[rows, d_ffn]` wide reads;
`weight_qat="ternary_native"` is paired with it from step 0. The fused Triton kernel
lives in `dormouse_kernels.engine.a4_fused`; the app imports it and keeps no copy.
Nothing here is committed, per the task.

## Wiring

* `configs/train_200m_a4.toml` — the 200M recipe, both quant slots final from step 0,
  `warmup = 0` (no warm-up, no ramp, no fp32 shadow forward). `dorm plan` prices it
  with `DORMOUSE_TSCT_NATIVE_TERN=1`: `weight_qat=ternary_native, act_quant=a4`,
  200,058,532 active parameters, 2,458,319,241,216 FLOP/step.
* `model/spec.py` — `a4` is a registry literal; `StackSpec.validate` binds the bank's
  quant+codesave to the config (`set_a4_config`), so the mechanism runs without an env
  flag. Half-quant pairings are named refusals: `a4` with `ternary`/`none` weights, and
  `ternary_native` with `int4`. `DORMOUSE_ACT_A4=1` remains a config assertion; a4 and
  the callable bank boundary both refuse `torch_compile=true`.
* `model/blocks.py` / `model/assemble.py` — both pre-sublayer norms carry A4; the step
  declares `a4_bwd`, `a4_codes_saved` and, when requested, both fused receipts as
  `MissingReport` gates after autograd has produced the gradients.
* `model/quant.py` — `A4Save` (norm boundary, butterfly `fwht_a4`, `absmax/7`, seeded
  stochastic round and an optional per-channel scale) and `BankA4Save` (slab code
  saving, `sigma·2.5/7`) share one dispatch. The bank's own packed-slab kernels stay
  in its autograd boundary; the **fused** A4 kernel does not.
* `dormouse_kernels/engine/a4_fused.py` — two Triton launches: forward FWHT(64) →
  absmax/7 → nearest-even int4 → two codes/`uint8` → inverse FWHT; backward unpack +
  STE in place. `enable_fp_fusion=False` is required: the default FMA moves ~15% of
  lanes by 1 ulp and a 20-step trace diverges. No silent eager fallback.

A fused request with SR rounding or a per-channel scale is a named `LoudError`: the
library kernel implements nearest-even absmax/7 only. The bank's fused path writes no
`[rows, d_ffn]` slab, so the canonical measurement runs with `DORMOUSE_BANK=0`; the
config still names the mechanism.

## Eight receipts

`tools/a4_integrate_gate.py --counters --steps 2 --device cuda` on this tree:

| counter | value |
|---|---:|
| `a4_fwd` | 1,048,576 |
| `a4_bwd` | 1,048,576 |
| `a4_codes_saved` | 524,288 |
| `bank_a4_fwd` | 20 |
| `bank_a4_bwd` | 5 |
| `bank_a4_codes_saved` | 5,242,880 |
| `tsct_ternary_fwd` | 24,576 |
| `tsct_ternary_bwd` | 180 |

**8/8 positive.** A flag that turned on a mechanism without running it would be a
`MissingReport`, which is why `assemble` enables these after the step rather than
inside the forward.

## Frozen-model quality

`tools/a4_integrate_gate.py --quality --seeds 3 --batches 1 --device cuda` loads one
frozen 200M checkpoint and flips only the two norms per block; weights, gains, head,
logits and the loss stay fixed. Each seed reads the same ids off→on→off (hash-checked),
with the off repeat as the determinism floor. The eval window and its 4.206775
bytes/token come from the checkpoint's own held-out manifest. Paired one-sided t95,
n=3, t=2.920.

| seed | A4 off | A4 on | dBPB | floor |
|---:|---:|---:|---:|---:|
| 0 | 2.436115 | 2.435600 | -0.000515 | 0 |
| 1 | 2.592055 | 2.591696 | -0.000359 | 0 |
| 2 | 2.665427 | 2.666052 | +0.000624 | 0 |
| **mean** | — | — | **-0.000083** | **0** |

CI95 `[-0.001125, +0.000958]`: **PASS** (upper ≤ 0.01, floor exactly 0). The
checkpoint predates the KDA alpha-floor refusal, so scoring uses a gate-only no-op on
`KDA.check_alpha_floor`; no clamp, no training step and no loss is changed.

## Correctness anchors (CUDA, 19 A4 tests)

| check | measured | gate |
|---|---:|---:|
| fused forward vs independent numpy fp64 | max abs 6.5e-7 | ≤ 2e-5 |
| fused forward vs eager `A4Save` | `torch.equal` | bitwise |
| backward vs upstream STE gradient | 0.0 | ≤ 1e-6 |
| shared memory / spills | ≤ SMEM_LIMIT, 0 spills | bounded |
| 20-step CPU replay, both arms | `losses == losses` | bitwise |

`tests/test_a4_bank.py` also pins the bank's external fp64 anchor to ≤1e-6, 2-codes-
per-byte packing, involutive rotation, width refusals and 20-step bitwise replay.

## TABLE

| variant | peak | ms | dBPB | CI | verdict |
|---|---:|---:|---:|---|---|
| bf16 bank slab, b2 (`DORMOUSE_BANK=0`) | 3617.4 MiB | 302.49 | — | — | control |
| bank INT4 codesave, b2 | 2692.1 MiB | 325.67 | +0.00102 (seed 0) | — | peak −25.6%, ms +7.7% |
| bf16 bank slab, b4 | 5936.8 MiB | 473.82 | — | — | control |
| bank INT4 codesave, b4 | 3893.5 MiB | 431.90 | −0.00050 (seed 1) | — | peak −34.4%, ms −8.8% |
| bf16 bank slab, b8 | 12313.2 MiB | 663.07 | — | — | control |
| bank INT4 codesave, b8 | 6417.7 MiB | 646.46 | −0.00054 (seed 2) | — | peak −47.9%, ms −2.5%, **b8 ≤ 0.55×base PASS** |
| frozen 200M, A4 on vs off (block norms) | shared weights | eval only | −0.000083 | [−0.001125, +0.000958] | PASS |
| eager A4 + cheap ternary (A2 regime) | 606.5 MiB | 317.5 | bitwise = fused | same as frozen | RED speed +14.1% |
| **fused A4 + cheap ternary (A2 regime)** | **606.5 MiB** | **290.6** | **bitwise = eager** | **same as frozen** | **PASS +4.5% (gate +5%)** |
| fused A4 + native ternary (canonical weights) | not measured at b8 | 391.8 (P2) | not scored | — | REFUTE +5% vs plain bf16: native itself ≈ +75% |

ms is interleaved min-of-15 CUDA events for one whole optimizer step, 5070 Ti under
the shared-card lock; the bank rows come from `tools/a4_bank_gate.py`, the fused rows
from `tools/a4_fused_ab.py`, both on the same kernel line now hosted in
`dormouse-kernels`. The plain-bf16 column is the stack with bf16 activations; for the
canonical config the honest +5% comparison is fused A4 against the **same** native
ternary weights with activations off (`--native-tern`, `ternary_control` arm), because
native ternary's own cost is a separate mechanism.

## Verification

* `make check` on this worktree: ruff, format, mypy clean; **764 passed / 36 skipped**,
  coverage **94.31%** (floor 93).
* `dormouse-kernels` suite: **65 passed / 31 skipped**, ruff and format clean. The
  library's `test_engine_port.py` pins the new module's public names and its lack of
  any app import.
* `dorm plan configs/train_200m_a4.toml` with `DORMOUSE_TSCT_NATIVE_TERN=1`: exit 0.
* Frozen quality gate: `research/solve/artifacts/a4_integrate_quality.json`.
* No commit, per the task.

## Follow-ups (named, not fixes made here)

* Re-run the b4/b8 timing table once the shared card is free; the numbers above are
  the source branches' interleaved runs on the identical arithmetic, not a new
  simultaneous measurement of all arms.
* Folding the ternary read into the fused launch is the next speed lever for the
  canonical native arm.
* Score a checkpoint trained at this HEAD to drop the alpha-floor gate shim.
