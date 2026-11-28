# AOT CUBIN GATE — grill/79, step 1

**Verdict: PASS.** A Triton kernel compiled ahead-of-time to a raw sm_120
cubin and launched from a plain CUDA driver-API C++ program is **bitwise
identical** to the same kernel launched through Triton/Python, and its GPU
kernel time is within **0.4%** (gate: ≤2%). Our Triton kernels can be reused
directly by a C++/CUDA engine — no manual port needed.

## What was tested

- Kernel: `glu_kernel` from `src/dormouse/kernels/fused.py` (`silu(gate)*up`,
  elementwise, block=1024) — the least complex jitted kernel in the tree.
- AOT target: `'cuda:120:32'` = `GPUTarget("cuda", 120, 32)` (sm_120, 32-thread
  warp), `num_warps=4`, `num_stages=3`, fp32 operands, `BLOCK=1024` constexpr.
- Card: RTX 5070 Ti, driver 620.73.04, CUDA 13.5, 16 GB.
- Data: 16M random fp32 `gate`/`up`, fixed seeds, generated once and shared by
  both arms via raw binary files.

## Exact commands

From `/home/sehaxe/dormouse2/wt/cubin`:

```bash
# 1. Build the C++ harness (CUDA driver API only, no Triton link time).
cd tools/cubin_gate
/opt/cuda/bin/nvcc -O2 -arch=sm_120 -o glu_harness glu_harness.cpp -L/usr/lib -lcuda
cd ../..

# 2. Run the gate: AOT-compile, JIT golden + time, C++ launch + time, compare.
#    app/.venv-cuda has CUDA torch 2.15 + Triton 3.9 (app/.venv is CPU-only).
flock -w 2400 /tmp/dormouse-gpu.lock \
  /home/sehaxe/dormouse2/app/.venv-cuda/bin/python tools/cubin_gate/gate.py

# 3. Minimal test (wraps step 2 at 16M elements / 30 iters, on the CUDA venv).
flock -w 2400 /tmp/dormouse-gpu.lock \
  /home/sehaxe/dormouse2/app/.venv-cuda/bin/python -m pytest tests/test_cubin_gate.py -q
```

`gate.py` writes the cubin and binary fixtures into `tools/cubin_gate/build/`.

## Results (min of 7 interleaved rounds, 50 iters each, 16M elements)

| kernel | parity | ms_triton | ms_cpp | verdict |
|---|---|---|---|---|
| `glu_kernel` | bitwise (`torch.equal`) | 0.2532 | 0.2523 | PASS |

`max_abs_vs_torch = 1.192e-07` (silu·up reference), timing delta **0.37%**.
The 7-round min is needed because GPU clocks cannot be locked without root
(`nvidia-smi -lgc` → permission denied); contention/clock ramp only add time,
so the minimum is the comparable estimator. Interleaving the arms keeps their
clock distributions aligned.

## Why it works / gotchas

- **Raw cubin has 6 kernel params, not 4.** `glu_kernel(G,U,OUT,n,BLOCK)` has
  four visible args, but Triton appends two hidden global/profile scratch
  pointers to the cubin's `.entry`. Passing only 4 to `cuLaunchKernel` makes the
  driver read stack garbage and intermittently return `CUDA_ERROR_INVALID_VALUE`.
  Always pass `{&dG, &dU, &dOut, &n, &scratch, &scratch}` (scratch pointers may be
  null — scratch size is 0 here). This was the single real bug found.
- **CUDA 13.5 changed `cuCtxCreate`** to `(CUctxCreateParams*, flags, dev)`;
  the harness uses `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` instead.
- **Timing is GPU-time via CUevents on stream 0**, not wall time. The kernel
  (~0.25 ms at 16M) is far longer than a Python launch (~15 µs), so the event
  loop measures pure kernel time in both arms. At 2M elements the kernel is
  ~8 µs and the Python arm measures ~44 µs of launch overhead — the 2% gate is
  meaningless there; both the gate and the test use 16M for this reason.
- **AOT and JIT produce the same code**: both compile the identical
  `JITFunction` object from `fused._build()`, same target/options, same process
  env (`TRITON_DEFAULT_FP_FUSION=1`) → bitwise-identical output and, modulo
  clocks, identical kernel time.

## Files

- `tools/cubin_gate/gate.py` — compile + data + JIT golden/timing + C++ launch + compare.
- `tools/cubin_gate/glu_harness.cpp` — ~120-line CUDA driver-API harness.
- `tests/test_cubin_gate.py` — minimal gate test.
- `tools/cubin_gate/build/` — generated cubin + fixtures (untracked).
