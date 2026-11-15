# dormouse

One entry point for everything the project runs:

```
dorm doctor                    what this machine can do
dorm status [journal]          read a run journal
dorm plan   <config>           what will run, and what it will cost
dorm train  <config> [--yes]   training; past the budget, --yes is required
dorm run    <config>           pretrain → sft → eval → export
dorm tokenize / eval / serve / export
```

## Why one binary

Because bash scripts failed silently. Three measurements in this project were
invalidated by a script that exited 0 after doing nothing, and two more by output
that was piped through `grep` and therefore invisible. A command that cannot do
what it was asked exits non-zero and names the way out.

Two rules the CLI enforces:

* **no silent success** — every unimplemented command refuses with an escape
  rather than pretending to work;
* **no surprise cost** — anything that predicts more than the approval budget
  requires `--yes`.

## State

`doctor` and `status` are real. The rest refuse by design until the model, data
and train layers move out of `port/` — see `dormouse-ai/plan`.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/dorm doctor
.venv/bin/python -m pytest
```
