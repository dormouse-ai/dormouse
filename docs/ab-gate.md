# Deterministic A/B gate

`tools/ab_gate.py` (config `tools/ab_gate.toml`) is the only measurement allowed to
decide a quality A/B here. An
A/B that has not run its control twice and got **exactly zero** is not a result.

## The failure it replaces

A same-seed control-vs-control run at 200 steps reported **−2.56 dBPB** on a GPU.
Nothing was changed between the two runs: the bank's scatter lands through
`index_add_` (`src/dormouse/model/moe.py:479,493` forward,
`src/dormouse/kernels/bank.py:994` backward). CUDA atomics have no fixed reduction
order, so two identical trajectories drift. A second defect inflated the old A2
window: a 42-byte eval string divided its loss by 42 bytes while claiming 1024
tokens, which scales nats by ~35×. The A2 "quality gate" was measuring both defects,
not the arm.

## Protocol

1. **B1 — named blocker.** CUDA bank atomics are not bit-ordered. The gate refuses
   `--device cuda` with `B1 BANK_ATOMICS_NONDETERMINISTIC` and runs on CPU with
   `torch.set_num_threads(1)` (`configure_cpu()`). This changes the measurement only;
   a training run on the card is untouched.
2. One config, one parser: `RunRecipe.read()` decides the stack and the eval window.
   The eval shard's **own manifest** supplies bytes/token (4.206775 for
   `data-prep-2h/demo_heldout.bin`), via `heldout_bytes_per_token()`. The training
   ratio (4.097287) is never used to price held-out bytes.
3. BPB is `nats / ln(2) / bytes`. The tool builds a tensor BPB and cross-checks it
   against `dormouse.metrics.from_nats` over the same byte count; a mismatch is a
   `BPB_FORMULA` refusal.
4. For every seed, arm A runs **twice** with the same fixed ids, init seed and eval
   probe. `check_control()` requires `torch.equal` on all step losses, on eval nats
   and on BPB, plus equal batch digests. Any difference is `CONTROL_NOT_BITWISE`
   and exits 2 — no delta is printed.
5. Only then arm B runs, against the same ids/digests. The paired delta is
   `bpb(B) − bpb(A)`; the 95% interval uses the paired t value for n=3 (2.920) and
   the control floor is reported as exactly `0.000000`.
6. Verdict `PASS` requires `|mean ΔBPB| ≤ 0.01` and upper CI ≤ 0.01 **and** an
   exact-zero control. A NaN loss is refused by name at the step it happens.

## Run

```sh
# from the repository root; one thread is set by the tool itself
python tools/ab_gate.py --config tools/ab_gate.toml \
    --steps 200 --seeds 3 --arm-a control --arm-b ternary
python tools/ab_gate.py --config tools/ab_gate.toml \
    --steps 200 --seeds 3 --arm-a control --arm-b a4 \
    --out /path/to/g1_control_a4.json
```

Arms: `control`, `ternary` (`ternary_native`, from step 0), `int4`, `a4`
(A4 + native ternary — a half-quantised stack is refused by `StackSpec`), and the
explicit alias `a4+ternary`. An arm this source tree does not assemble is refused by
`StackSpec` rather than silently running the control.

`tools/ab_gate.toml` is deliberately small (4 layers, d=64, seq=128, batch=2): it
tests the measurement protocol cheaply, not the final 500M quality number. Point
`--config` at the real run config to gate a mechanism at production widths.

## Gates

| gate | enforcement | expected |
|---|---|---:|
| control-vs-control bitwise | `check_control()`, every seed | `torch.equal`, floor `0.000000` |
| 20-step bitwise | `tests/test_ab_gate.py::test_control_is_bitwise_equal_at_twenty_steps` | bitwise equal |
| 200-step deterministic | CLI above, `--steps 200 --seeds 3` | floor `0`, CI printed |
| eval bytes | `tests/test_ab_gate.py::test_eval_window_uses_manifest_bytes_and_the_nats_formula` | 1077 B = round(256 × 4.206775) |
| CUDA request | `test_cuda_is_a_named_refusal` | exit 2, `B1` |
| lint / types / suite | `make check` | clean |

## Upgrade path (not built)

A sorted stable segment-sum over the bank's row indices would give CUDA a fixed
reduction order without pinning a device to one thread. It is not here: the gate
needs a number today, and the CPU path already makes it reproducible. Add it when a
CUDA-side 200-step control-vs-control at production widths must pass with the card's
throughput; until then B1 stays a named refusal, not a silent CPU fallback inside a
training run.
