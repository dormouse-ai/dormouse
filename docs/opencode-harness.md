# OpenCode IQ harness adapter

A day-1 bridge: OpenCode sees our checkpoint as a native OpenAI-compatible provider,
and the existing `dormouse.eval` scorer sees OpenCode as one `Generate` callable.
The model behind `local/dormouse` is a **stdlib stub**, not a trained model: the
point is to prove the wiring and the falsifier before a checkpoint exists.

## Layout

| Path | Role |
|---|---|
| `opencode.jsonc` | provider `local` → `@opencode/ai/providers/openai-compatible`, baseURL port from `{env:DORMOUSE_STUB_PORT}`, model `dormouse` with explicit capabilities/limit |
| `.opencode/agents/solver.md` | primary agent, `steps: 1`, all permissions denied: answer only, no tools |
| `tools/stub_server.py` | `POST /v1/chat/completions`, modes `RANDOM` / `ORACLE` / `ECHO`, deterministic by seed, SSE and JSON |
| `tools/opencode_solver.py` | `opencode run --model local/dormouse --agent solver --format json --archived --standalone` → last assistant text; TCP preflight, timeout, rate-limit backoff |
| `tests/test_opencode_adapter.py` | stub floor/ceiling, NDJSON parser, named refusal, rate-limit backoff, live `opencode models` and one live `opencode run` |

## Run it

From the repository root (OpenCode discovers `opencode.jsonc` and `.opencode/` there):

```bash
export DORMOUSE_STUB_PORT=8765
PYTHONPATH=src python tools/stub_server.py --mode ORACLE --seed 0 &
opencode models | grep local/dormouse
python tools/opencode_solver.py "What is 51 + 55?
Answer:"                         # 106
python -m pytest tests/test_opencode_adapter.py -q
```

`STUB_MODE`, `STUB_SEED` and `STUB_SUITE` work in place of `--mode`, `--seed` and
`--suite`. `RANDOM` emits seeded noise, `ORACLE` answers from the same suite/seed
the scorer uses, `ECHO` returns the prompt (wiring probe). The stub reports its
config at `/v1/config`; a mismatched suite or seed is a refusal, never a low score.
Stop the server after the run.

To score a suite through the real bridge:

```python
from tools.opencode_solver import make_generate
from dormouse.eval.harness import run
from dormouse.eval.tasks import suite

results = run(suite(0), make_generate(port=8765), seed=0)
```

`make_generate` adapts `solve` to the harness's `Generate` signature; each example is
one isolated archived session. A stub that is down raises `StubUnavailable` before
OpenCode starts — the harness must never report it as a zero-scoring floor.

## Gate (2026-11-22, this worktree, CPU only)

| mode | score | exit |
|---|---|---|
| `RANDOM` (seed 0) | 0.000 | 0 |
| `ORACLE` (seed 0) | 1.000 | 0 |
| `ECHO` | 0.000 | 0 |
| stub off | n/a | 3 (`StubUnavailable`) |

`opencode models` lists `local/dormouse`. `make check`: 589 passed, 23 skipped,
coverage 93.50%. Seeded determinism and the NDJSON parser are asserted in the test
file; the two live-OpenCode tests skip when `opencode` is not on `PATH`.

## What this does NOT measure

- **Not AA v5 IQ, and not an IQ number of any kind.** It measures that config, CLI,
  agent, parser and scorer are connected. The stub has no intelligence.
- **Not our checkpoint.** `dorm serve` is intentionally unwired; there is no adapter
  from our KDA/DSA/TSCT stack to `/v1/chat/completions`. The provider is a stub until
  that exists, so no answer here says anything about the model.
- **Not tool use or agentic behaviour.** `capabilities.tools = false`, `steps: 1`,
  every permission denied, and the parser reads plain `text` events only.
- **Not blind quality.** No external judge, no self-preference control, no A/B swap,
  no corpus-derived cloze, no Dyck/PPT verifier. Those are later components.
- **Not speed or device performance.** CPU-only smoke, one session per example; N>100
  needs the HTTP API instead of one process per example.

## Gotchas

- `--standalone` starts a private server; it does not ignore the project config.
- `{env:DORMOUSE_STUB_PORT}` must be set before `opencode models` too, or the baseURL
  stays literal. The solver sets it for its child processes.
- `--format json` is NDJSON of session events, **not** a `response_format` schema.
  The assistant text arrives as `{"type":"text","part":{"type":"text","text":...}}`.
- An empty stub answer produces only `step_start`/`step_finish`; the solver treats
  missing text as `ProtocolError`, never as an empty success.
