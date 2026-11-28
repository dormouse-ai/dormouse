# dense500 — the largest dense stack that fits one 5070 Ti

`configs/train_500m_dense_2b.toml` keeps the requested filename, but the 500M target
is **REFUTED on the card**, not rounded down in the config. The exact 500M shape
prices inside ±2% and passes `dorm plan`, then OOMs in Muon's shipped five-iteration
fp16 Newton–Schulz chain at b2 × s1024. The shipped shape is the largest measured
FIT: **404,379,412 stored = active parameters**.

Everything below is counted by `StackSpec` + `assemble(device="meta").report()` —
the same call `dorm plan` makes — and every VRAM/time number is from one real CUDA
step under `flock /tmp/dormouse-gpu.lock` on this 16 GB 5070 Ti.

## The config

* `ffn="dense"`: one full-rank SwiGLU per layer (no bank), so stored = active;
  `n_experts=topk=n_shared=expert_tsct_rank=0` are explicit no-bank declarations.
* `attn="hybrid"`: 3:1 KDA:DSA (12 KDA + 4 DSA over 16 layers), GQA 8/2 at
  `head_dim=64`, `c_lat=512`, tied `V=32,768` head.
* `weight_qat="ternary_native"` + `act_quant="a4"` from step 0, `warmup=0`, LNS on
  by default (`DORMOUSE_LNS=0` is the A/B arm), Muon fp16 NS, `torch_compile=false`
  (both quant boundaries refuse compile), seq 1024, batch 2.
* Corpus/source: `/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dormouse-data-2b/` —
  `train.bin` (1,929,997,803 tokens) and `heldout.bin`, same `tokenizer32k.json`.
* `DORMOUSE_TSCT_NATIVE_TERN=1` is the native-arm flag; `PYTHONPATH=src:../dormouse-kernels/src`
  relative to `app/` (absolute `/home/sehaxe/dormouse2/dormouse-kernels/src` from this worktree).

Exact bill at the shipped shape:

| group | parameters |
|---|---:|
| embedding (tied head = 0) | 41,943,040 |
| attention | 47,821,284 |
| FFN (16 × 3 × 1280 × 5120 + 48 native scales) | 314,572,848 |
| norms ((2·16+1) × 1280) | 42,240 |
| **stored = active** | **404,379,412** |

`bytes/step: state=6,470,070,592` (≤10 GB) and `checkpoint=101,094,853` (≤1 GB at
1.58 bpw) come straight from the plan. `dorm plan configs/train_500m_dense_2b.toml`
exits 0.

## Fit search (exact counts, no interpolation)

`step_s` is one timed optimizer step with the shipped 5 NS iterations and bf16
autocast; `peak` is CUDA reserved (allocated in parentheses when the run succeeded).
Every failed row OOMed inside `_polar_many`, never in the forward.

| config | active | stored | state | checkpoint | step_s | peak | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 20L d1280 f5120 — the 500M target | 494,988,185 | 494,988,185 | 7.92 GB | 0.124 GB | — | OOM | REFUTED (−1.00% was in band; card not) |
| 24L d1152 f4608 | 485,580,702 | 485,580,702 | 7.77 GB | 0.121 GB | — | OOM | REFUTED |
| 16L d1408 f5632 | 478,746,516 | 478,746,516 | 7.66 GB | 0.120 GB | — | OOM | REFUTED |
| 20L d1216 f4864 | 451,982,169 | 451,982,169 | 7.23 GB | 0.113 GB | — | OOM | REFUTED |
| 16L d1344 f5376 | 440,776,532 | 440,776,532 | 7.05 GB | 0.110 GB | — | OOM | REFUTED |
| 12L d1536 f6144 | 432,137,103 | 432,137,103 | 6.91 GB | 0.108 GB | — | OOM | REFUTED |
| 20L d1152 f4608 | 410,942,233 | 410,942,233 | 6.58 GB | 0.103 GB | — | OOM | REFUTED |
| 32L d896 f3584 | 408,692,648 | 408,692,648 | 6.54 GB | 0.102 GB | — | OOM | REFUTED |
| 28L d960 f3840 | 406,883,171 | 406,883,171 | 6.51 GB | 0.102 GB | — | OOM | REFUTED |
| **16L d1280 f5120 — shipped** | **404,379,412** | **404,379,412** | **6.47 GB** | **0.101 GB** | **4.623** | **13,566 MiB reserved / 12,496 MiB alloc** | **PASS** |
| 12L d1472 f5888 | 400,776,015 | 400,776,015 | 6.41 GB | 0.100 GB | — | not run | smaller than the FIT, no need |

Only widths with `d_model` divisible by 64 are candidates: FWHT(64) tiles the A4
norm stream. Only `n_layers` divisible by 4 tile the 3:1 attention period. The
search covered every such shape between 400M and 500M; the largest that completes
a step is the shipped row.

Things tried and **not** used to claim the fit: `DORMOUSE_A4_FUSED=1` (no peak
reduction at this shape), running the optimizer outside bf16 autocast (still OOM),
and `NS steps=1` (fits, but it is not the shipped five-iteration chain).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is needed for repeated steps:
the default caching allocator fragments at step 2 even though a single step fits.

## GPU receipts

One step, live ids from `train.bin`, bf16 autocast, `DORMOUSE_TSCT_NATIVE_TERN=1`,
`PYTHONPATH=src:/home/sehaxe/dormouse2/dormouse-kernels/src`:

```
loss=10.927597 finite=True skipped=False
step_s=4.6230
peak_alloc_mib=12495.8 peak_reserved_mib=13566.0
stored=404,379,412 active=404,379,412 state_bytes=6,470,070,592 ckpt_bytes=101,094,853
a4_fwd=83,886,080 a4_bwd=83,886,080 a4_codes_saved=41,943,040
tsct_ternary_fwd=98,304 tsct_ternary_bwd=176 ffn_fwd=32,768 opt_muon=192
```

Since 4.623 s < 5 s, the 20-step smoke was required and run (same config,
`expandable_segments:True`, one `set_step_seed(7, step)` per step, real corpus ids):

```
step  1/20 loss=10.927597 step_s=4.6584
step 10/20 loss=8.780121  step_s=1.3956
step 20/20 loss=9.323787  step_s=1.3229
20 steps: loss_first=10.927597 loss_last=9.323787 min=8.382233 max=13.728691
step_s mean=1.5391 min=1.2996 max=4.6584 total=30.78
peak_alloc_mib=12719.9 peak_reserved_mib=13642.0
```

All 20 losses finite, no firewall skips, all mechanism counters positive at the
last step. The card had another process holding ~0.72 GB throughout; the numbers
are not a best-case empty-GPU measurement.

## What changed in this worktree (uncommitted)

* `src/dormouse/model/spec.py` — `ffn` is a literal slot (`moe` | `dense`); dense
  explicitly requires rank 0 and zero bank fields, and no longer trips the MoE-only
  rank refusal. `is_moe` now reads the slot.
* `src/dormouse/model/registry.py` — the dense FFN has its own piece
  (`dense_ffn→ffn_fwd`); MoA/dictionary/TSCT pieces are only declared for the bank.
* `src/dormouse/model/assemble.py` — dense builds one full-rank `FFNSpec` arm per
  layer, no bank; `report()` already returns stored = active when no bank exists;
  bank A4 code-save counters are only enabled when a bank exists.
* `src/dormouse/model/ffn.py` — a dense `NativeQuantLinear` read moves
  `tsct_ternary_fwd`, so the native-ternary declaration is proved on this topology too.
* `tests/test_dense500.py` (new) pins the exact count, the plan bill, the no-bank
  shape and a tiny CPU step with all A4/ternary receipts; `tests/test_spec.py` and
  `tests/test_cfg500.py` were updated for the new slot and its digest.
* `docs/cfg500.md` — digest updated for the new spec field.

Verification: `ruff check`, `ruff format --check`, `mypy src` clean; full pytest
with coverage green above the 93% floor; `dorm plan` exit 0. No commit, per task.
