# smart-fast: one config, only surviving levers

`configs/smart_fast.toml` is the single "умная + быстро учится" run. It is assembled
**only** from levers that survived the `.bulba/research/iq-04-synthesis.md` report.
No new mechanism is introduced; nothing marked REFUTEd in that synthesis is enabled.

## Price

| | value | source |
|---|---|---|
| stored params | 400,985,812 | `assemble(..., device="meta").report()` |
| active params | 68,506,348 | same report, top-2 + shared expert |
| tokens `D` | 108,000,000 | 90M SYNTH-enriched pretrain + 30k-example SFT stage |
| headline bill | `6 * P_stored * D = 2.598e17` FLOP | 1.965 h @ 39.92% MFU of 92 TF |
| engine bill | `6 * P_active * D = 4.437e16` FLOP | 0.336 h, always smaller |
| envelope | `2.644e17` FLOP / 2.0 h | 92 TF x 39.92% x 7200 s |

The active size is the **honest 68.5M** of the shipped 401M stack, not the 400M-active
/ 3.8B-stored chat shape, which does not fit. `tools/smart_fast_bench.py` prints this
table and exits 1 if the headline bill exceeds the envelope.

## Lever table (lever | status | enabled | gate)

| lever | status | enabled | gate |
|---|---|---|---|
| `synth_static_25` | survived | yes | 25% static corpus, 0 FLOP, never x10; delta-test, 0 shared SHA, abstention>=0.6 |
| `fp16_ns` | survived | yes (engine) | cos>=0.9999 on 8 buckets, 3.06x chain (commit `97630f4`) |
| `graph_epilogue` | survived | yes (engine) | `torch_compile=true` + `dorm train --graph`; whole-step MFU 39.92% (commit `c553a57`) |
| `active_68.5m` | survived | yes | 401M stored / 68.5M active; not 400M active / 3.8B stored |
| `lns_replace_init` | survived | yes (engine) | **replace** `write_std=0.02/sqrt(2L)`, regression 2.714696 +/- 0.10; never combine with scaled init |
| `sft_pika_30k` | survived | yes (stage) | 30k multi-turn PiKa, assistant-only loss, re-init chat embeddings, 0 leaks |
| `verifier_1rollout` | survived | yes | 1 rollout/prompt, verifier on held-out; k>1 is red |
| `grpo_outcome_only` | survived | yes (post-train) | outcome-only reward; dense-MC and `sc_k5` are red |
| `iq_smoke` | survived | yes | random=0 / oracle=1 on smoke; **sole quality gate** |
| `tst_r_0.3` | gated | **no** | conflicts with mix_plan (`COMPATIBILITY.md:38`); single-source run only, 3 seeds |
| `xmod` | REFUTEd | no | local 0.799 FLOP, dBPB +6..+14, 1.022x slower |
| `fff` | REFUTEd | no | MoE integration unverified; conflicts with CUDA graph |
| `loop_l` | REFUTEd | no | each loop adds FLOP and worsens BPB |
| `engram` | REFUTEd | no | 0 downstream at 0.5-4B; flag unwired |
| `mtp` | REFUTEd | no | +26% FLOP for +0.05pp at 1.2B; worse at our scale |
| `jepa` | REFUTEd | no | +28% step; EBR 345x FLOP |
| `fsz` | REFUTEd | no | no surviving evidence in iq-04 synthesis |
| `sc_k5` | REFUTEd | no | +400% FLOP [N8: paper says correctness result, NOT efficiency], n_eff about 2.1; 1 rollout optimal |

## What is a config flag here, and what is not

At this HEAD `StackSpec` (and `RunRecipe.read`) refuse unknown keys by name. The lever
names above are therefore **not** config keys: adding `tst_r`, `lns`, `synth_share` or
a `[flags]` table would fail config parse, not enable a mechanism. The levers are
wired where the repo actually reads them:

- **SYNTH 25% and SFT 30k** are `[[source]]` / `[[stage]]` run-recipe entries relative
  to the config; each source needs its `<path>.json` manifest with the tokenizer SHA.
  Keep the static share at 0.25 (rewriting law R<10-32% at our FLOP), never x10.
- **fp16 NS, graph+epilogue, LNS** are properties of the binary at the cited commits.
  The config only expresses the reachable switches: `optimizer="muon"`,
  `torch_compile=true`, `act_quant="none"`, and the run uses `dorm train --graph`.
  LNS **replaces** `write_std=0.02/sqrt(2L)`; the combination is worse than LNS alone.
- **verifier / GRPO** are post-train steps, not architecture flags; GRPO is
  outcome-only with 1 rollout and no dense-MC reward.
- **TST r=0.3** is a different objective and cannot share a run with the SYNTH+SFT
  mix (`COMPATIBILITY.md:38`). It is a gated, off-by-default separate run.

## Gates

`tools/smart_fast_bench.py` checks the parse, the FLOP bill and the lever manifest in
CPU CI; `tests/test_smart_fast.py` runs those checks plus falsifier tests (an over-budget
`D` or a flipped REFUTEd lever is refused, not ignored).

The **sole quality gate** is the IQ smoke harness: `random=0` (floor) and `oracle=1`
(ceiling) on the same tasks, with `iq_judge_calls` proving the blind judge ran. An
enabled gate whose counter stayed zero is a `MissingReport`, never a clean pass.
BPB regression (2.714696 +/- 0.10) guards the LNS init swap only; it is a safety gate,
not the quality gate. Nothing claims a measured IQ delta: IQ/FLOP is not measured on
this stack ([`iq-04-synthesis.md`](../../../.bulba/research/iq-04-synthesis.md) §1).

Run it:

```sh
make install
uv run --no-sync python tools/smart_fast_bench.py
uv run --no-sync python -m pytest tests/test_smart_fast.py -q
make check
```
