# E4 — forward/backward kernels of the 404M dense b2 step

Worktree `wt/engine-fwdbwd` (`perf/engine-fwdbwd`, base `app@e341174`). Nothing committed.

## What the profile actually says

Parent's whole-step profile (`/tmp/opencode/profile404.log`, fp32 NS chain) priced
forward 319.5 ms, CE 1.9 ms, backward 630.6 ms, Muon 962.1 ms. Its top-kernel table is
dominated by Muon's fp32 Newton-Schulz chain: `aten::bmm` 829 ms, cutlass SIMT sgemm
489+157+141 ms. Those are optimizer kernels, not fwd/bwd.

A fwd+ce+bwd-only profile (`tools/profile_fwdbwd.py`, same config, b2×s1024, bf16,
DORMOUSE_TSCT_NATIVE_TERN=1) gives, at the live shape:

| kernel | self ms | count | owner |
|---|---:|---:|---|
| `aten::mm` | 69.1 | 619 | dense FFN + projections + tied head |
| sparse `bwd_kernel` (+ autograd wrapper) | 47.1 + 47.1 | 4 + 4 | DSA, 4 layers |
| cutlass bf16 GEMM (relu/nn/nt) | 22.9+16.4+13.9 | 220 | dense SwiGLU |
| `aten::copy_`/direct-copy fp32 | 22.4+14.0 | 7,898+3,284 | casts + custom-op clones |
| `aten::mul` / `div` / `abs` | 16.4/6.9/5.9 | 1,500/554/552 | norm + A4 + LNS elementwise |
| `aten::bmm` | 7.9 | 3,888 | KDA gram/qk (small fp32 tiles) |

There is **no fp32 SIMT GEMM in fwd+bwd**. The 787 ms of `cutlass_80_simt_sgemm` in the
parent table belongs to the fp32 NS chain (MB1), which is out of scope here. The
fp32→bf16 GEMM lever is therefore REFUTED for this task: every large matmul in
fwd+bwd is already bf16 tensor-core.

The phase time (~1,400 ms measured under load) is far above the sum of kernel self
times (~250 ms): the step is launch-bound (≈20k launches in fwd+bwd). Kernel work
must remove launches, not just CPU-side microseconds.

## Shipped change

`src/dormouse/model/attn_variants/kda.py`: `_chunks_scan`, enabled by default
(`DORMOUSE_KDA_SCAN=0` restores the shipped loop).

KO1 already batches the two triangular solves over chunks but still walks the 16
chunks in Python with ~6 launches each. Substituting the pseudo-values makes the
carry affine:

```
y_c = part_v_c − part_k_c · S_{c−1}
S_c = (diag(last_c) − carried_cᵀ·part_k_c) · S_{c−1} + carried_cᵀ·part_v_c
    = M_c · S_{c−1} + P_c
```

A Hillis-Steele prefix composes `(M_c, P_c)` in `log2(16) = 4` batched bmm steps,
so every chunk's input state exists before one batched output pass. The prefix is
checkpointed on its four already-retained operands, so the four `[b,h,chunks,d,d]`
intermediates are recomputed rather than kept — peak VRAM stays **identical**
(4,939 MiB both arms).

The dispatch, the two solves (kept separate deliberately — see below), and all
per-chunk arithmetic are otherwise untouched.

## Phase table (quiet card, interleaved 7 rounds, CUDA events)

| phase | before | after | verdict |
|---|---:|---:|---|
| fwd+bwd, median | 1,056.7 ms | 795.1 ms | **1.329×**, −261.6 ms |
| fwd+bwd, min | 925.8 ms | 605.5 ms | 1.529× |
| launches (fwd+bwd) | 19,791 | 12,675 | −36.0% |
| triangular solves | 96 | 96 | unchanged |
| peak VRAM | 4,939 MiB | 4,939 MiB | PASS (flat) |
| loss finite | yes | yes | PASS |
| fp64 anchor (out / state / dq) | — | 4.2e-9 / 9.9e-10 / 1.6e-9 | PASS ≤1e-6 |
| 20-step paired params (fp32) | — | 9.26e-7 | PASS ≤1e-5 |
| fwd+bwd ≤ 400 ms (target) | 950 ms (parent) | ≈715 ms projected | **REFUTE** |

The 400 ms gate is not reachable from kernels alone on this stack. Kernel self time
is already ~250 ms; the remaining ~545 ms is host launch/lag, which a CUDA graph
removes. E1 owns the graph path — combine the two for the step target.

## Candidates examined and rejected

| candidate | result | why |
|---|---|---|
| one triangular solve with 2-wide RHS | REFUTE | fp64 probe 4e-9 but 20-step fp32 drift 9.25e-5 > 1e-5; cuSOLVER picks a different kernel for the wider RHS. Reverted. |
| `fused.glu` for dense SwiGLU | REFUTE | fp64 anchor 1.15e-6 > 1e-6 (kernel casts inputs to fp32); on-card min 1,023 ms vs 828 ms baseline. Reverted. |
| `baddbmm` in the prefix | REFUTE | parity green but launch count unchanged (12,147); reflattens to the same kernels. Reverted. |
| fp32 GEMM → bf16 | REFUTE | none in fwd+bwd (see kernel table). |

## Files

- `src/dormouse/model/attn_variants/kda.py` — scan core, prefix helper, flag, receipt.
- `tests/test_kda_scan.py` — fp64 recurrence anchor, loop equality + grads, dispatch, 20-step parity.
- `tools/profile_fwdbwd.py` — fwd+ce+bwd phase/kernel tables with no optimizer noise.
- `tools/bench_kda_scan.py` — interleaved on-card A/B + launch/solve counts.

`make check` on the final tree: ruff / format / mypy clean, 813 passed, 36 skipped,
coverage 94.25% ≥ 93%, exit 0.
