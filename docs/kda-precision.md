# KDA precision fix: fp32 state + real fp32 log-decay

## Question

FLA `chunk_kda` saves ~460 ms over the nine KDA layers but the full-width paired
200-step gate at `3449bc9` measured ΔBPB +0.09864 (b2 × s1024, seed 4).
grill/75 attributes that failure class to precision, not to the chunked
algorithm: FlashKDA's bf16 state gives 3.45% final-state RMSE at 8k tokens, an
fp16 cumulative-decay LUT gives 9% decay error, and GDN-2 keeps state and
log-decay in fp32. This slice applies that fix and re-runs the gates.

## What changed

- `kernels/kda_fla.py`: the FLA adapter casts q/k/v/beta to fp32, computes
  `g = log(alpha.float())` (the real log-decay, no LUT and no upcast of a bf16
  log), calls `chunk_kda` with `use_gate_in_kernel=False`, and refuses a non-fp32
  returned state with a LoudError.
- `attn_variants/kda.py`: dispatch from `_chunkwise` under `DORMOUSE_KDA_FLA`;
  softplus/exp take their input in fp32 (fp64 anchors keep fp64); the eager core
  runs under `torch.autocast("cuda", enabled=False)` so the A/B compares kernels,
  not an fp32 solve against bf16 eager matmuls.
- `configs/kda_precision.toml` / `tests/test_kda_precision.py`: the measurement
  shape and the four gates.

## Result — REFUTE

| config | dBPB | cos(fp64) | ms/layer |
|---|---|---|---|
| `kda_precision`, b2×s768, lr 0.01, seed 4 paired 200 steps | **+0.01692** | write 0.999999940, state 1.000000000 | 13.48 (eager 66.80) |

- **Precision fix is a numerical no-op here.** FLA 0.6.0 already asserts an fp32
  `initial_state`, and the previous adapter in `wt/kda-fullwidth` already passed
  a fp32 log-decay: its write cosine was 0.99999994, identical to this one to
  every printed digit. There is no bf16 state or fp16 decay LUT to remove in
  this codebase. The state is fp32 (asserted in the adapter and in the test).
- **Quality does not close.** The one paired seed that completes 200 steps is
  +0.01692, above the 0.01 gate (and 6× below the old +0.09864, but the shapes
  and LR differ, so that is not a clean improvement claim).
- **Speed gate fails.** 13.48 ms/layer > 8 in the least-contended run (eager
  66.80, −480 ms over nine layers); a later run with more foreign GPU tenants
  measured 23.22 vs 183.04 eager. Either way the 8 ms bar is missed.
- **3-seed gate is blocked, not passed.** At this HEAD (f346b5e) random ids drive
  the stack to the NaN firewall: at lr 0.1 seed 4 aborts around step 80 (the same
  on unmodified `app/`, so it is not caused by this slice), and at lr 0.01 seed 5
  aborts. b2 × s1024 additionally OOMs in Muon with the card's resident foreign
  tenants (~2.3 GiB), so the completed pair uses b2 × s768 (12 KDA chunks, the
  full 832-wide stack) and lr 0.01. Both arms share ids, init and optimizer, so
  the paired delta is valid for that setting; it is **not** the published
  b2 × s1024 / lr 0.1 number.

The default stays `DORMOUSE_KDA_FLA=0`; the quality gate is a strict xfail.
`make check` is green (630 passed, 27 skipped, 93.81%).

Follow-ups, out of scope here: a real-token (DCLM) 3-seed stream instead of
random ids to get a NaN-free 200-step gate; kernel-side fp32 decay hardening only
if a bf16/fp16-LUT backend is actually introduced; then ReWY on top of a quality-
neutral FLA path.
