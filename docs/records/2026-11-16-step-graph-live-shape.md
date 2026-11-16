# The last lossless lever: a captured whole step, measured on the real 1B (2026-11-16)

Base: `origin/master` = `d2040bc` (PR #62, "the step captures again", landed as
`39b65aa`; the graph is **not** enabled by default). Branch:
`perf/stepgraph-live-shape-measure`. Every card command ran under
`flock /tmp/dormouse-gpu.lock`; card load before 1,222/15,899 MiB, after 992/15,899 MiB
(desktop context only). No model, kernel, precision or math source was changed — this
run only measures capture and scheduling.

## 1. Capture: refused, with the op named

| shape | result |
|---|---|
| b8 × s1024 | warm-up OOM: refused 48 MiB with 27 MiB free, 1.50 GiB reserved-but-unallocated. **b8 does not fit with the graph resident.** |
| b4 × s1024, after eager steps | warm-up OOM in the bank's recompute (126 MiB refused). |
| b4 × s1024, fresh process | reaches capture, then `LoudError`: `the step could not be captured: called a synchronizing CUDA operation`. |
| b1 × s1024, fresh process | same `LoudError` — no hook, no instrumentation, so it is not an artifact of the measurement. |

The exact op is the device-indexed slice in the bank's backward gather:

```
dormouse/kernels/bank.py:718   idx = torch.arange(e * cap, device=x.device)[live.reshape(-1)]
  called from bank.py:938 (xp = gather(...)) and bank.py:949, inside _Bank.backward
```

`StepGraph._capture` sets `set_sync_debug_mode("error")`, so this advanced indexing
raises instead of silently syncing. It is reported through `failed_on`; there is no
eager fallback and `replays()` stays 0. PR #62 fixed the *forward* pass loop's host
read; the backward gather still reads a device value to the host.

## 2. Pool headroom and the dashed line

The project rule is `max(384 MiB, 8 % VRAM)` = **1,272 MiB** on this 15,899 MiB card.
12,985 MiB was free after the model was built. No pool was ever taken: capture refuses
before `torch.cuda.graph` allocates one. At b8 the graph's own warm-up (three bodies,
`zero_grad(set_to_none=False)`, a side stream) exceeds VRAM, so **b8 does not fit with
the pool resident**; the drop to b4 was made explicitly, not by reducing the config.
At b4 a fresh process has room but capture is refused by the sync op above.

## 3. Speed ladder — bf16 autocast, frozen weights, three runs each

| shape | mode | ms/step (3 runs) | median ms | tok/s | B/s source | MFU |
|---|---|---|---:|---:|---:|---:|
| b8 × s1024 | eager | 3858.8 / 3881.9 / 4012.0 | 3,881.9 | 2,110 | 9,957 | 1.65 % |
| b8 × s1024 | replayed | — capture refused — | — | — | — | — |
| b4 × s1024 | eager | 2785.2 / 2804.7 / 2792.8 | 2,792.8 | 1,467 | 6,919 | 1.15 % |
| b4 × s1024 | replayed | — capture refused — | — | — | — | — |

The b8 eager row reproduces the standing 3,915.6 ms / 2,092 tok/s / 9,871 B/s baseline
(the 1.6 % spread is run-to-run). MFU uses the engine's own 407 MFLOP/token bill against
the 52 TFLOP/s practical peak. Peak VRAM on the b8 eager steps: 13,079 MiB allocated of
15,899 MiB. Against the old engine's 22.9 kB/s, this build still delivers **0.44×**
(2.3× slower). There is no replayed ms/step number to compare and no ≥1.25× speed-up.

## 4. Parity at the live shape — and the smoke-router question

Because a CUDA graph cannot be built, the closest honest check is the *captured body's*
arithmetic (fixed-pass dispatch, the exact body `_capture` records) against the eager
step, at b4 × s1024 with identical initial weights, ids and fresh optimizers.

| check | result |
|---|---|
| loss | 722.566406 vs 722.566406 — abs diff **0** |
| logits (all 4 M) | bit-identical (`torch.equal`) |
| gradients | 753 tensors, worst deviation 12.5 % of tensor scale (`beta.bias`), typically 3–4 %, worst per-element 23,651× on a near-zero element |
| post-step params | 759 tensors, worst deviation **53.6 %** of tensor scale (`blocks.6.attn.beta.bias`), 100 % of elements changed on many tensors |
| bf16 epsilon gate (0.0039) | **FAILS** |

The drift is concentrated in attention `beta` and `a_up` params, exactly as it is in the
smoke shape. Crucially, an **eager-vs-eager control through the same harness drifts
more** (worst 87.4 % on the same near-zero bias, logits again bit-identical), so the
difference is the stack's own non-deterministic bf16 backward (scatter/atomic
accumulation), not something the captured body adds. An fp32 control would separate the
two, but the bank kernel refuses fp32 on this card: it needs 151,552 B of shared memory
against a 101,376 B ceiling.

The tiny `[2, 32]` smoke router is not exempt: with the anchor bank
(`DORMOUSE_BANK=0`, the only way the smoke stack assembles on this torch), a captured
step drifts from eager by **17.1 %** of router.weight's scale (100 % of elements
changed), and `tests/test_graph.py::test_a_captured_step_is_the_eager_step` fails on that
assertion. With the fused bank the same test fails earlier: the rank-8 smoke expert is
below the kernel's 16-wide tile floor. CI's CPU wheel skips these card-only tests, so
the suite is green there.

Verdict on losslessness: strict bf16-epsilon parity fails at the live shape and at the
smoke shape; the graph adds no measurable drift over eager's own, but it is **not
provably lossless**, and it cannot be captured on the fused live path at all.

## 5. The third lever: torch.compile — no artifact

`tools/bench_compile_step.py` compiled the model with dynamo to let inductor take the
pointwise/norm regions, under a 540 s watchdog and an outer 600 s kill. It finished
early with no compiled artifact:

- graph break at `dormouse/model/moe.py:572` — `largest = int(counts.max())` is a
  `Tensor.item()` (the eager pass count);
- then `RuntimeError: An intermediate tensor 'idx' created inside a higher-order op
  subgraph aliases an input or output ...` from the checkpointed mixture.

No inductor graph was built, no ms/step was measured, and the 10 GB RSS timeout was
never reached. A compile that does not build a graph is not a win.

## 6. Gate and decision

`graph_replays` is **0** on every measured run. `tests/test_step_graph_bench.py` fails a
run whose declared capture reports a failure with no reason, whose eager-only speed row
claims a replay, or whose parity verdict does not follow the measured deviation.

Decision: **leave the step graph off.** Capture is refused on the live fused path (a
named host-sync op in the backward gather), no replayed step exists to be faster, and
strict bf16-epsilon parity does not hold at the live shape (53.6 % worst on a near-zero
param; the eager control itself is 87.4 %). Enabling requires a ≥1.25× speed-up and
parity; neither was observed.

Evidence: `runs/step_graph_1b/{capture_b4,speed_b8,speed_b4,body-parity_b4,control_b4}.json`.
Tooling: `tools/bench_step_graph.py`, `tools/bench_compile_step.py`.
