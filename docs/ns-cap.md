# NS workspace cap (opt/ns-cap)

Base `a8cf56a`. Problem: MEM500 b2 OOMs in Muon `_polar_many` (`projected` = 390 MiB for
65 × `[256,6144]` fp32), i.e. the Newton–Schulz workspace, not momentum
(`research/optstate/O5-muon-state.md`).

## What shipped

`src/dormouse/train/muon.py`:

- `NS_CHUNK_BYTES = 1 << 27` (128 MiB). `chunk_bytes=None` is the uncapped parity
  control; `Muon(..., chunk_bytes=...)` and `_polar_many(..., chunk_bytes=...)` take it.
  fp16 gets twice the element cap of fp32, so both hit 128 MiB per buffer.
- A stacked shape group is split into matrix slices (`_pieces`) and the prescaled
  copy is made per slice. The spectral bound is still computed on the **whole**
  group once — its power-iteration GEMM picks a different kernel at N=4 than at
  N=1 and would otherwise move an fp32 ulp.
- Slices are chained in windows whose **total** numel ≤ cap (`_polar_chain` per
  window). Capping each slice but chaining all of them in one foreach list keeps
  one `projected` alive per slice, which is no cap at all (measured: 8207 vs 8211 MiB).
- The quintic now runs in place: `bg`, `cg`, `scaled`, and the fresh `works` are
  gone. Eight live matrix temporaries → four (`works`, `gram`, `gram²`,
  `projected`); launch schedule and IEEE order unchanged.

Ceiling: the `[s,s]` Gram is 150.9 MiB fp32 at s=6144 and cannot be split —
the short side is the Gram's K dimension. 128 MiB applies to every `[s,l]` and
`[N,s,l]` intermediate; fp16 brings this Gram to 75.5 MiB.

## Gates (5070 Ti, 16 GB, `fcntl.flock /tmp/dormouse-gpu.lock`)

| config | peak | ms | cos fp64 | parity | verdict |
|---|---:|---:|---|---|---|
| 401M b2 fp32, no cap, no amp | OOM: refused 390.00 MiB at 12.54 GiB process in use | — | — | — | FAIL (control; reproduces MEM500) |
| 401M b2 fp32, 128 MiB cap, no amp | **11349.1 MiB**, step completes | 5482 ms quiet (8141 ms with a second tenant on card) | — | — | PASS on completion; no clean full-arm peak to subtract (full OOMs) |
| NS buckets fp32, no cap | 8207.5 MiB | 526.61 ms/chain | 1.0000000 (8/8, capped arm; bitwise equal to full) | reference | control |
| NS buckets fp32, 128 MiB cap | **6040.0 MiB (−2167.5 MiB)** | 541.86 ms (+2.89%) | 1.0000000 (8/8) | `torch.equal`, 20 steps bitwise | PASS (≥1 GiB peak gate) |
| NS buckets fp16, no cap | 5679.8 MiB | 191.99 ms/chain | 0.9999992–0.9999999 (8/8, capped arm) | reference | control |
| NS buckets fp16, 128 MiB cap | 5488.5 MiB (−191.3 MiB) | 195.14 ms (+1.64%) | 0.9999992–0.9999999 (8/8) | `torch.equal`, 20 steps bitwise | PASS (small peak headroom) |
| 500M b2 fp32/fp16, capped | OOM **before Muon**, in `assemble()` at a 20 MiB request with 12.57 GiB in use | — | — | — | BLOCKED, see below |

MS numbers are interleaved min-of-7 CUDA events over the four live buckets
(`65×256×6144`, `65×256×8192`, `65×256×7168`, `1×8192×6144`). Peak is
`torch.cuda.max_memory_allocated()` per fresh process. Cos is the external
numpy fp64 SVD polar on all 8 live buckets (`tests/test_muon_fp16.py`), worst
ortho 1.42e-3, worst relative 1.23e-3. The full-stack arms run with autocast
off: at this HEAD the bank backward throws `Float vs BFloat16` under amp, which
is independent of Muon (follow-up).

## 500M blocker — exact remaining buffer and next step

`train_500m_2b.toml` builds 808 experts with `d_model=7168`, `d_ffn=8192`,
`r=256`. The fp32 bank factors alone are
`808 · (7168·256 + 256·8192) · 4 B = 12,710,343,680 B = 11.84 GiB`; with shared
experts and the rest of the stack the process is at 12.57 GiB and the allocator
refuses a 20 MiB request during `assemble()`, before any optimizer step. Both
capped and full arms fail at the same point with the same refusal, so the cap
did not cause it. Next step (out of scope here): store bank u/v in fp16/bf16 or
CPU-offload the bank, then re-run `tools/ns_cap_gate.py` on the 500M config.

## Reproduce

```
PYTHONPATH=/home/sehaxe/dormouse2/dormouse-kernels/src \
  app/.venv-cuda/bin/python tools/ns_cap_shapes.py --lock --cap-bytes 0 --ns-dtype fp32
PYTHONPATH=/home/sehaxe/dormouse2/dormouse-kernels/src \
  app/.venv-cuda/bin/python tools/ns_cap_shapes.py --lock --cap-bytes 134217728 --ns-dtype fp32
app/.venv-cuda/bin/python tools/ns_cap_ms.py --lock --ns-dtype fp32 --rounds 7
PYTHONPATH=/home/sehaxe/dormouse2/dormouse-kernels/src \
  app/.venv-cuda/bin/python tools/ns_cap_gate.py --lock --config configs/dormouse_1b.toml \
    --batch 2 --steps 2 --ns-dtype fp32 --cap-bytes 134217728 --no-amp
app/.venv-cuda/bin/python -m pytest tests/test_muon_cap.py -q
```

No new counter was added: the cap is the shipped algorithm, not a declared
opt-in flag, and the golden counter dict in `tests/test_wired_graph.py` stays
valid. Parity is proved by `tests/test_muon_cap.py`.
