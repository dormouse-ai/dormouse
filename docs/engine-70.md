# Engine to 70% MFU — measured plan

One 404M dense step, b6×1024, 6,144 tokens, bf16, RTX 5070 Ti (92 TF bf16 peak).

Measured references on 2026-11-25/26:

- 404M dense mainline: 1,945 ms, 14.7% MFU.
- 404M + symmetry-aware NS: 1,536 ms at step 20, but **NaN at step 200**; not shipped.
- 200M ideal bank + graph + epilogue + fp16 NS: 632 ms, **34.4% MFU** on the old
  `verify/mfu-locked` tree; not yet portable to mainline because that tree owns
  `forge`/`flashopt`/`fused_ce`/`forward_ce`.
- Microsoft MAI Base 2: 21% whole-step MFU on 16k GB200; Muon 2.1% of the step.

## Measured baseline (2026-11-25, `tools/profile_step.py`)

| phase | ms | share |
|---|---:|---:|
| forward | 359.7 | 18.5% |
| backward | 649.5 | 33.4% |
| Muon | 931.9 | 47.9% |
| CE + clip + AdamW | 16.6 | 0.9% |
| other | -12.8 | -0.7% |
| **total** | **1,944.9** | **14.7% MFU** |

- 6,076 kernel launches/step; 18,229 `cudaLaunchKernel` host calls.
- Muon's own table: 982 ms, 1,450 `bmm`, 775 ms of fp32 SIMT `sgemm`.
- 14.907 TFLOP/step; 70% = **0.232 s**, 40% = **0.405 s**.

## What 70% requires

Only the FLOP numerator can move the ceiling. 74% of the step's FLOPs are
attention/KDA/DSA/elementwise; FFN is ~26%. A 90%-MFU GEMM core cannot lift a step
whose FLOPs are mostly small kernels.

| lever | target | result |
|---|---|---|
| Adam-mini replaces Muon | remove 932 ms | 1.44 s, 11.2% (measured) |
| C++/CUDA step executor | remove host/launch and autograd | 0.9–1.0 s, 20–25% |
| fused KDA/A4/ternary/LNS/CE | fewer kernels | 0.6–0.7 s, 30–35% |
| persistent KDA/DSA, b8–b16 | more work per launch | 0.45–0.5 s, 40–50% |
| big-GEMM FFN ≥60% FLOP (ideal bank/MoE) | compute-shape change | 0.25–0.3 s, 60–70% |
| FP8 GEMM (if sm_120 backward exists) | +20–30% | only after the above |

70% is not a flag; it is the last row. 40–50% is the software-only target.
The first reproducible target is 34.4%: port the memory stack (`forge`, `flashopt`,
`fused_ce`, `forward_ce`) into mainline and re-run the verifier. Only after that
is 40 a credible next step.

## Order

1. Measure only: phase, op, launch tables (done).
2. Remove the extra mechanisms: one optimizer, one precision path, one global
   attention, dense + MoE as two rungs, five configs (in progress).
3. Build the C++ step executor around the existing Triton cubins; Python keeps
   orchestration only. Gate: same 20-step loss, step ≤1.0 s.
4. Fuse the elementwise/quantization boundaries; gate: step ≤0.7 s.
5. Only then change compute shape; the 70% target belongs to that branch.
