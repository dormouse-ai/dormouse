# Native ternary TSCT (A3)

The model trains ternary TSCT factors from step 1. `DORMOUSE_TSCT_NATIVE_TERN=1`
plus `weight_qat="ternary_native"` is one arm: a config without the flag, or the
flag on any other `weight_qat`, is a named `LoudError` before allocation.
A postwar conversion is not this arm.

## What the forward reads

`FactorLinear` keeps fp32 `u`/`v` only as the optimizer's shadow state. Every
forward — the eager expert (`factor_forward`), the bank's anchor and the fused
kernel's operands (`ffn.stack_matrices`) — reads `scale·round(shadow/threshold)`
with codes in `{-1,0,+1}`. `scale` is a learnable parameter sized at init to
restore the shadow's fp32 RMS (a plain `mean|x|` init shrinks a Gaussian factor
~6x and moves every activation statistic on step 0). The straight-through
estimator passes the shadow gradient unchanged and a real `Σ grad·code` to the
scale. No matmul sees the fp32 factor.

The non-TSCT `nn.Linear`s are swapped to `NativeQuantLinear` with the same
learnable/RMS-restoring code. The inherited `QuantLinear` (detached `mean|x|`
scale) is *not* used by the native arm: it cost ~1.1 nats on the 20-step A/B
before this fix. The tied head still reads `embed.weight` fp32 — follow-up.

## Counters and refusals

`tsct_ternary_fwd` is declared in the registry and counted with the bank's read
(checked inside `Transformer.forward`). `tsct_ternary_bwd` is enabled and counted
in `Assembled.step` after autograd — the forward cannot check a backward counter.
Both are MissingReport gates. A flag-on forward over a plain `FactorLinear`, or
a bank stack over fp32 factors, raises `LoudError` instead of running the fp32
read under the ternary name.

## A/B (CPU, byte unit, 2-layer KDA stack, 20 steps, seed 11)

| variant | bits | loss (step 20) | ΔBPB vs fp32 | ms/step | VRAM proxy | anchor | verdict |
|---|---|---|---|---|---|---|---|
| `weight_qat="none"` | 32 fp32 | 1.265104 | 0.00000 | UNMEASURED | 337,968 B params, 65,536 B CPU forward peak | n/a | control |
| `weight_qat="ternary_native"` | 1.58 fwd / 32 shadow | 1.242377 | **−0.03279** | UNMEASURED | 338,124 B params, 65,536 B CPU forward peak | cos 0.9999999999999973 (y), 0.999999999999996 (du/dv) | PASS quality/anchor, speed UNMEASURED |

`ms/step` is UNMEASURED: `/tmp/dormouse-gpu.lock` was held throughout this run,
so no card timing was taken. No ms number is inferred from the CPU proxy.

The VRAM proxy is honest about the design: the fp32 shadow is still stored, so
parameter bytes rise 0.05% (learnable scales) rather than falling. The saving
from 1.58-bit storage applies to a checkpoint/export, not to live optimizer
state. ΔBPB uses `loss/ln2` at one byte per token.

## External fp64 anchor

`tests/test_tsct_native_tern.py` recomputes `scale·{-1,0,+1}` and both STE
gradients in numpy fp64 for 20 SGD steps. Worst cosine across all steps:
`y=0.9999999999999973`, `du=0.999999999999996`, `dv=0.9999999999999969` (gate
`>1−1e-6`). The anchor holds the codes fixed in its gradient, so it is not a
replay of torch's autograd formula.

## Pre-existing at HEAD `adbc00f` (not introduced here)

`ruff check .` fails on `src/dormouse/cli.py:63` (E501) and
`tests/test_cubin_gate.py:18` (I001). Full pytest has 3 failures —
`test_custom_ops.py::test_capture_replays_the_compiled_step_with_fp32_parity`,
`test_entry.py::test_a_resume_reproduces_every_batch_and_every_loss`,
`test_entry.py::test_eval_prints_the_table_and_the_byte_window` — all reproduced
on a pristine HEAD worktree with identical output. My files pass ruff, format,
mypy, and the targeted suite.
