# Regression baselines (canonical)

This is the **one** place for the 200-step regression numbers: the value, the date
it was recorded, and why it moved. `tools/regression.py::BASELINE` is the executable
mirror of the current row; `tests/test_regression_gate.py` refuses a copy of any
number below in another doc, config or tool, and refuses a mirror that drifts.

The gate is 200 seeded fp32 CPU steps on `tests/fixtures/corpus_2mb.bin`, band +/-0.10
in final **loss** (nats), not BPB and not IQ. Changing any seed, shape or stack in
`tools/regression.py` is a new measurement and requires a new row here.

| value | date recorded | status | reason |
|---|---|---|---|
| 2.452850 | 2026-11-15 | superseded | First one-architecture stack (commit `ef36fba`). Experts still kept a dense `d·f` master and ran a per-forward SVD; the rank-32 factors became the parameters (190,200 vs 220,920), so the number moved with the storage at commit `4a0c6de`. |
| 2.714696 | 2026-11-16 | superseded, app lineage | Recorded after that factor change. Retired 2026-11-24: tied-head init used `N(0,1)` rows, giving logit std `sqrt(d_model)` (53 at d=2816) and init CE ~900; the fix sets embedding std `1/sqrt(d_model)`, so the stack converges from a lower CE to a lower final loss (commit `c185739`). |
| 2.506861 | 2026-11-19 | alternate lineage, never canonical here | Recorded on the `engine-mainline` branch after the same tied-head scale (commit `87491b6`), declared canonical there on 2026-11-21 (`9ebaefb`). It belongs to a different fixture/measurement identity and was never the value at `app/` HEAD; treating it as interchangeable with the app number was the D07 conflict. |
| 2.356925 | 2026-11-24 | **current** | Re-recorded in the same commit as the tied-head init fix (`c185739`), 200-step gate green. This is the number `make regression` must match today. |

History debt closed here: D07 recorded three numbers in three places with no owner,
and research notes repeatedly called the conflict open. The owner is this table;
everything else names it instead of repeating a value.
