"""The run as one command: what it writes, what it refuses, and a resume that is exact.

These are the gates a shell script cannot pass. The plan's step cost is the run's, the
resume reproduces every batch hash and every loss, a changed config refuses **by name**,
and every mechanism the run enabled left a non-zero counter in the journal. The expensive
one is first: it runs the committed smoke config through the CLI end to end, because a
seam test that does not run the seam is a unit test of a stub.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.cli import main
from dormouse.model.assemble import assemble
from dormouse.model.spec import StackSpec
from dormouse.train import entry

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"
ONE_B = ROOT / "configs" / "dormouse_1b.toml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "corpus_2mb.bin"

#: The five tasks `eval.suite` scores, in its own order.
TASKS = {"reverse-copy", "arithmetic", "needle-4k", "format-following", "code-executes"}


def _toml(value: object) -> str:
    """One TOML value: a string quoted, a bool spelled, everything else as it prints."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _tiny(directory: Path, **overrides: object) -> Path:
    """Write a 20-step DSA stack with no harness: the resume and refusal gates' config."""
    directory.mkdir(parents=True, exist_ok=True)
    fields: dict[str, object] = {
        "unit": "byte",
        "attn": "dsa",
        "optimizer": "muon",
        "objective": "ntp",
        "sources": "fixture",
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "head_dim": 8,
        "c_lat": 4,
        "d_ffn": 32,
        "n_experts": 2,
        "topk": 1,
        "expert_tsct_rank": 4,
        "attn_k": 32,
        "attn_window": 16,
        "seq_len": 128,
        "batch": 8,
        "tokens": 20 * 8 * 128,
        "schedule": "cosine",
        "seed": 3,
        "eval_batches": 1,
        "harness": False,
        "eval": str(FIXTURE),
    }
    fields.update(overrides)
    text = "\n".join(f"{key} = {_toml(value)}" for key, value in fields.items())
    text += f'\n\n[[source]]\npath = "{FIXTURE}"\nweight = 1.0\n'
    path = directory / "tiny.toml"
    path.write_text(text)
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    """Every journal line of a run, in order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quiet(argv: list[str]) -> int:
    """Run the CLI with its transcript swallowed, so a fixture is not a test of print."""
    with redirect_stdout(io.StringIO()):
        return main(argv)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run `dorm train configs/smoke.toml --smoke` once; every seam test reads its output."""
    runs = tmp_path_factory.mktemp("smoke-runs")
    assert _quiet(["train", str(SMOKE), "--smoke", "--runs", str(runs)]) == 0
    directory = runs / SMOKE.stem
    return {"directory": directory, "records": _records(directory / "run.jsonl")}


# ------------------------------------------------------------------------ the gate


def test_the_smoke_run_writes_every_seam(smoke: dict[str, Any]) -> None:
    """20 step lines, one checkpoint, one eval line, one end, and the artifact on disk."""
    directory = Path(smoke["directory"])
    kinds = Counter(record["kind"] for record in smoke["records"])
    assert kinds["step"] == 20
    assert kinds["checkpoint"] == 1
    assert kinds["eval"] == 1
    assert kinds["end"] == 1
    assert [record["step"] for record in smoke["records"] if record["kind"] == "step"] == list(
        range(20)
    )
    for name in ("run.jsonl", "ckpt.pt", "final.pt", "config.toml", "config.json"):
        assert (directory / name).is_file(), name
    assert (directory / "export" / "model.pt").is_file()
    assert (directory / "export" / "config.toml").is_file()


def test_the_artifact_loads_back_and_gives_the_same_logits(smoke: dict[str, Any]) -> None:
    """The export is the run's weights, not an init that happens to have the right shape."""
    directory = Path(smoke["directory"])
    spec, exported = entry.load_artifact(directory / "export" / "model.pt")
    _, final = entry.load_artifact(directory / "final.pt")
    ids = torch.arange(128, dtype=torch.long).reshape(1, 128) % spec.vocab
    with torch.no_grad():
        from_export = exported(ids)
        from_final = final(ids)
        from_init = assemble(spec, smoke=True).model(ids)
    assert torch.equal(from_export, from_final)
    assert torch.max(torch.abs(from_export - from_init)) > 1e-3


def test_every_enabled_mechanism_reported(smoke: dict[str, Any]) -> None:
    """The journal's eval line is the receipt: no counter the stack declared stayed zero."""
    counters = [record for record in smoke["records"] if record["kind"] == "eval"][-1]["counters"]
    expected = set(entry.STEP_COUNTERS) | {
        "dsa_fwd",
        "moe_fwd",
        "moa_fwd",
        "ffn_dict_entries_used",
        "tsct_fwd",
        "opt_muon",
        "opt_adamw",
        "clip_tensors",
        "fwd",
        "bwd",
        "ema",
    }
    silent = sorted(name for name in expected if counters.get(name, 0) == 0)
    assert silent == []
    assert all(value > 0 for value in counters.values()), counters


def test_the_start_line_carries_the_plan_and_its_hash(smoke: dict[str, Any]) -> None:
    """A run's first line is its recipe: the hash a resume is compared against, and the cost."""
    start = smoke["records"][1]
    assert start["kind"] == "start"
    assert start["plan_flops"] == start["step_flops"] > 0
    assert start["plan_source"] == "bill"  # a smoke stack is below the plan's expert floor
    assert len(start["config_hash"]) == 64


# ------------------------------------------------------------- plan versus run


def test_the_plan_prices_the_step_the_run_measures() -> None:
    """`dorm plan`'s printed flops/step is `6 x active x tokens`, at the live config.

    The 1B config is priced on the meta device, so this is the real number at no memory
    cost — and it is the number a run of that config asserts against at start.
    """
    spec = StackSpec.from_toml(ONE_B)
    assert isinstance(spec, StackSpec)
    seconds, priced, source = entry._plan(spec, smoke=False)
    assert source == "plan"
    assert seconds > 3600  # ten billion tokens is not a one-hour run
    assert (
        priced == 6 * assemble(spec, device="meta").report()["active"] * spec.batch * spec.seq_len
    )


def test_a_step_that_disagrees_with_the_plan_refuses() -> None:
    """1% is the tolerance: a 2% drift is a config pricing one shape and running another."""
    entry._assert_plan("start", 1_000_000, 1_005_000)
    with pytest.raises(LoudError, match="prices a step at"):
        entry._assert_plan("start", 1_000_000, 1_020_000)


def test_the_smoke_run_priced_its_own_step_from_the_counters(smoke: dict[str, Any]) -> None:
    """The FLOP the loop counted is the FLOP the bill priced, within the assertion's 1%."""
    start = smoke["records"][1]
    assert start["plan_flops"] == start["step_flops"]


# ---------------------------------------------------------------------- the resume


def test_a_resume_reproduces_every_batch_and_every_loss(tmp_path: Path) -> None:
    """12 steps, killed by `--steps`, then 8 more, equals one run of 20 — exactly.

    Bit-identical and not "close": the stream is a pure function of the step, so the only
    way the two differ is state the checkpoint failed to carry — the clip estimates, the
    EMA, or the optimizer's moments.
    """
    split = _tiny(tmp_path / "split")
    paused = entry.train(split, runs=tmp_path / "split" / "runs", smoke=True, until=12, echo=_noop)
    assert not (Path(paused["directory"]) / "final.pt").exists()  # a pause is not a result
    kinds = Counter(record["kind"] for record in _records(Path(paused["directory"]) / "run.jsonl"))
    assert kinds["step"] == 12 and kinds["end"] == 0

    resumed = entry.train(
        split, runs=tmp_path / "split" / "runs", smoke=True, until=20, resume=True, echo=_noop
    )
    whole = entry.train(
        _tiny(tmp_path / "whole"), runs=tmp_path / "whole" / "runs", smoke=True, echo=_noop
    )
    after = [r for r in _records(Path(resumed["directory"]) / "run.jsonl") if r["kind"] == "step"]
    straight = [r for r in _records(Path(whole["directory"]) / "run.jsonl") if r["kind"] == "step"]
    assert [record["step"] for record in after] == list(range(20))
    assert [record["batch"] for record in after] == [record["batch"] for record in straight]
    assert [record["loss"] for record in after] == pytest.approx(
        [record["loss"] for record in straight], abs=1e-12
    )


def test_a_resume_that_changes_a_rate_refuses_by_name(tmp_path: Path) -> None:
    """The recipe is the run: a new rate under the old journal is a different run."""
    runs = tmp_path / "runs"
    started = _tiny(tmp_path, lr_muon=0.02)
    entry.train(started, runs=runs, smoke=True, until=2, echo=_noop)
    with pytest.raises(LoudError, match=r"run\.lr_muon"):
        entry.train(_tiny(tmp_path, lr_muon=0.03), runs=runs, smoke=True, resume=True, echo=_noop)


def test_a_second_run_of_one_journal_refuses(tmp_path: Path) -> None:
    """Two runs may not append to one journal: the second would look like the first."""
    config = _tiny(tmp_path)
    entry.train(config, runs=tmp_path / "runs", smoke=True, until=2, echo=_noop)
    with pytest.raises(LoudError, match="already holds a run"):
        entry.train(config, runs=tmp_path / "runs", smoke=True, until=2, echo=_noop)


# ------------------------------------------------------------------------ the eval


def test_eval_prints_the_table_and_the_byte_window(smoke: dict[str, Any]) -> None:
    """`dorm eval` on the artifact reproduces the run's BPB, with its window, and five rows."""
    lines: list[str] = []
    got = entry.evaluate(Path(smoke["directory"]) / "final.pt", echo=lines.append)
    assert set(got["harness"]) == TASKS
    assert got["bytes"] == 1024
    reported = [record for record in smoke["records"] if record["kind"] == "eval"][-1]
    assert got["bpb"] == pytest.approx(reported["bpb"], rel=1e-12)
    assert "BPB over 1024 B" in lines[0]


def test_status_shows_the_run(smoke: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    """One row, from the journal, naming the config and saying the run ended."""
    assert main(["status", str(Path(smoke["directory"]) / "run.jsonl")]) == 0
    out = capsys.readouterr().out
    assert f"config={SMOKE.stem}" in out
    assert "ended" in out


def _noop(_line: str) -> None:
    """Swallow a run's transcript; the journal is what these tests read."""
