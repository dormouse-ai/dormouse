# IQ harness: what it measures

The portable core lives in `src/dormouse/eval/iq.py`; model adapters live in
`model_adapter.py`; the CLI is `tools/iq_eval.py`. Nothing here imports
OpenCode, torch at import time, or a network client. The core needs only
`dormouse_core` (for the counter gate) and the standard library.

```
uv run --no-sync python tools/iq_eval.py --model mock --suite smoke --json
```

Each line is `{"schema_version": 1, "task": ..., "score": ..., "reason": ...}`.
A score of 0 always carries a reason: `Score` refuses an unexplained zero at
construction, so a pluggable judge cannot smuggle one into the output.

## What it measures

- **Scoring plumbing**: exact/normalized match, choice, and code answers that
  compile and contain the expected snippet.
- **Suite determinism**: tasks come from a seed and local data, so two runs
  with the same seed produce the same prompts and answers. `smoke` is
  SYNTH-only (arithmetic, filling, factual QA, code completion) in RU and EN;
  `full` adds fill-in-the-blank tasks derived from the committed 2 MB corpus
  fixture (`tests/fixtures/corpus_2mb.bin`).
- **Blind judging**: a judge sees `(task, answer)` and never the model's
  identity, so swapping a model cannot change the rubric.
- **The missing-report gate**: `evaluate(..., counters=counters)` enables a
  judge counter; a suite that never reached the judge raises `MissingReport`
  instead of reporting a suspiciously clean score.

## What it does NOT measure

- **It is not an IQ test.** The name describes the harness shape, not a
  cognitive claim. It is not the AA v5 benchmark, not an approximation of it,
  and no score from it may be reported as an IQ number.
- **It does not measure general knowledge.** The factual QA set is a handful
  of fixed one-word questions; a 46M model scoring near zero there says only
  that it does not know those eight facts.
- **It does not judge open-ended answers.** Every task has one reference
  answer. There is no LLM-as-judge, no partial credit, no semantics beyond
  case-fold/whitespace normalization.
- **The code task does not execute anything.** It compiles the answer and
  checks for an expected snippet. Passing means "syntactically valid Python
  with the right fragment", not "correct program".
- **It is not comparable to the five-task `dorm eval` harness.** That one
  scores `generate(prompt, max_tokens, stop)` on reverse/copy, arithmetic,
  needle, stop-following and code execution. This one scores a different,
  one-argument interface. Keep the two tables separate.

## Plugging in an OpenCode adapter

The core is intentionally unaware of OpenCode. Write one object in your own
script that satisfies the `ModelAdapter` protocol and pass it to `evaluate`:

```python
from dormouse.eval.iq import MockJudge, build_suite, evaluate


class OpenCodeAdapter:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str) -> str:
        return self.client.complete(prompt).text  # your SDK call here


suite = build_suite("smoke", seed=0)
rows = evaluate(suite.tasks, OpenCodeAdapter(client), MockJudge())
```

Keep the adapter outside `eval/`: the harness stays portable and the
OpenCode-specific dependency stays where it belongs. A real checkpoint is
already wired through `CheckpointAdapter`, which loads an artifact via
`load_artifact` (and therefore `_build`) and greedy-decodes with `Talker`.
`--model checkpoint --checkpoint runs/<id>/final.pt` selects it. `mock` is a
seeded random floor, `echo` echoes the prompt, and `known` is the deterministic
ceiling used by the tests.
