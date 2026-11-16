"""The distribution gates: two cards must be the same run as one, and cheaper per token.

Each gate is a failure mode of data parallelism, not a feature list:

* **The world is the launcher's.** A config that names `world_size` is refused, because a
  file and a launcher that disagree start a run that is not the run that was described.
* **The shard is a partition.** Every rank's rows are disjoint, and their union over the
  run is the single-card stream byte for byte — a duplicated shard trains a token twice
  and a missing one never trains it, and both look like a slightly different loss curve.
* **Two cards are one card.** 2 ranks × 2 accumulation steps equal one process over the
  same effective batch, weights within 1e-6. This is the gate the loss normalisation
  exists for, so `test_per_microbatch_mean_fails_the_equivalence_gate` runs the same
  comparison against the injected defect and requires it to fail.
* **One journal, one checkpoint.** Rank 0 writes; the numbers are global; the checkpoint
  written by two ranks loads into one, takes the same step, and reports the same loss.

Every run is a subprocess under `gloo` on CPU, started with `RANK`/`WORLD_SIZE` in the
environment. The tests skip, visibly, where torch has no process group: a two-process
gate that silently ran one process would be the defect it is here to catch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dist_job import ACCUM, ROWS, SEQ, SMALL, STEPS, ToyStream
from dormouse_core.journal import Journal
from stacks import reference_stack

from dormouse.train import dist

GLOO = torch.distributed.is_available() and torch.distributed.is_gloo_available()
pytestmark = pytest.mark.skipif(not GLOO, reason="torch.distributed/gloo is not available")

ROOT = Path(__file__).resolve().parents[1]
SRC, TESTS = ROOT / "src", ROOT / "tests"
JOB = TESTS / "dist_job.py"
TOKENS = ROWS * SEQ
#: Relative, not absolute. A cross-process reduction reorders the fp32 sums, so two
#: mathematically identical runs differ at the fp32 epsilon (~1.2e-7 relative). An
#: absolute 1e-6 fails at loss ~10 on a different BLAS build; a relative bar of 1e-6
#: is still 10x tighter than the drift it must tolerate, and it cannot hide a real bug
#: (the per-micro-batch-mean control below drifts by 1.46e-4, 280x the honest 5.2e-7;
#: both measured 2026-11-16 after the tied-head sqrt(d) scale lowered the loss level).
TOL = 1e-6


@dataclass(frozen=True)
class Job:
    """What one job left behind: the journal, each rank's weights and what it said."""

    run_dir: Path
    ranks: list[dict[str, Any]]
    weights: list[dict[str, torch.Tensor]]

    @property
    def local_losses(self) -> list[list[float]]:
        """Per rank, the loss of every step as that rank computed it."""
        return [[float(step["loss"]) for step in rank["local"]] for rank in self.ranks]

    @property
    def losses(self) -> list[float]:
        """The step's global loss: the mean over the ranks' means (equal shards)."""
        columns = list(zip(*self.local_losses, strict=True))
        return [sum(step) / len(step) for step in columns]

    @property
    def journal(self) -> list[dict[str, Any]]:
        """Every line of the run's one journal, in order."""
        lines = (self.run_dir / "run.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    @property
    def steps(self) -> list[dict[str, Any]]:
        """The journal's step lines."""
        return [record for record in self.journal if record["kind"] == "step"]


def _port() -> int:
    """Pick a free port for this job's rendezvous; two jobs run at once in CI."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_job(
    root: Path,
    *,
    world: int,
    mode: str = "train",
    bug: bool = False,
    run_dir: Path | None = None,
    seed: Path | None = None,
) -> Job:
    """Start `world` processes on one run directory and collect what they wrote.

    A job is retried once on a fresh port: the rendezvous port is picked by binding and
    closing, so a process that grabs it in between is a race in the harness, not in the
    thing under test. A seeded job (`seed=`) is re-copied for the retry, because the
    first attempt may have appended to its journal.
    """
    label = f"{mode}{'-bug' if bug else ''}-w{world}"
    run_dir = root / label / "run" if run_dir is None else run_dir
    out = root / label / "out"
    failure = ""
    for _attempt in range(2):
        for stale in (run_dir, out):
            if stale.exists():
                shutil.rmtree(stale)
        if seed is not None:
            shutil.copytree(seed, run_dir)
        failure = _start(run_dir, out, world=world, mode=mode, bug=bug)
        if not failure:
            break
    if failure:
        pytest.fail(failure)
    ranks = [json.loads((out / f"rank{rank}.json").read_text()) for rank in range(world)]
    weights = [
        torch.load(out / f"rank{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(world)
    ]
    return Job(run_dir=run_dir, ranks=ranks, weights=weights)


def _start(run_dir: Path, out: Path, *, world: int, mode: str, bug: bool) -> str:
    """Run every rank of a job; return the first rank's stderr if one failed."""
    env = os.environ | {
        "PYTHONPATH": f"{SRC}:{TESTS}",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_port()),
        "WORLD_SIZE": str(world),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(JOB),
                "--run-dir",
                str(run_dir),
                "--out",
                str(out / f"rank{rank}.json"),
                "--mode",
                mode,
                *(["--bug"] if bug else []),
            ],
            env=env | {"RANK": str(rank)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(world)
    ]
    for process in procs:
        _, err = process.communicate(timeout=600)
        if process.returncode != 0:
            for other in procs:
                if other.poll() is None:
                    other.kill()
            return f"rank failed (rc={process.returncode}):\n{err}"
    return ""


def _digest(batches: list[torch.Tensor]) -> str:
    """One hash over a run's batches, so two runs can be compared as batch sets."""
    joined = b"".join(batch.numpy().tobytes() for batch in batches)
    return hashlib.blake2b(joined, digest_size=8).hexdigest()


def _consumed(job: Job) -> list[torch.Tensor]:
    """Rebuild each step's whole batch from the ranks' recorded shards, in rank order."""
    columns: list[list[torch.Tensor]] = []
    for per_rank in zip(*(rank["seen"] for rank in job.ranks), strict=True):
        columns.append([torch.tensor(entry["ids"], dtype=torch.long) for entry in per_rank])
    return [torch.cat(shards) for shards in columns]


def _worst(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    """The largest absolute weight difference between two state dicts."""
    assert left.keys() == right.keys()
    return max(float((left[key] - right[key]).abs().max()) for key in left)


@pytest.fixture(scope="session")
def single(tmp_path_factory: pytest.TempPathFactory) -> Job:
    """One card, the reference run."""
    return run_job(tmp_path_factory.mktemp("single"), world=1)


@pytest.fixture(scope="session")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Job:
    """Two cards, the same config."""
    return run_job(tmp_path_factory.mktemp("pair"), world=2)


# ------------------------------------------------------------------- the environment


def test_one_rank_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """At world 1 nothing is special-cased: no group, no wrapper, no journal of its own."""
    monkeypatch.delenv(dist.WORLD, raising=False)
    monkeypatch.delenv(dist.RANK, raising=False)
    assert dist.init() == dist.Topology(rank=0, world=1)
    assert dist.is_main()
    dist.barrier()
    stream = ToyStream()
    assert dist.shard_stream(stream) is stream
    assert dist.wrap(torch.nn.Identity()) is not None


def test_the_world_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`RANK`/`WORLD_SIZE` are the world; a nonsense pair is refused, not rounded."""
    monkeypatch.setenv(dist.RANK, "1")
    monkeypatch.setenv(dist.WORLD, "2")
    assert dist.topology() == dist.Topology(rank=1, world=2)
    monkeypatch.setenv(dist.WORLD, "1")
    with pytest.raises(Exception, match="outside"):
        dist.topology()
    monkeypatch.setenv(dist.WORLD, "0")
    with pytest.raises(Exception, match="no process"):
        dist.topology()
    monkeypatch.setenv(dist.WORLD, "two")
    with pytest.raises(Exception, match="not a rank"):
        dist.topology()


def test_a_config_may_not_name_the_world(pair: Job) -> None:
    """A config that names `world_size` is refused, and no run writes one into a file."""
    with pytest.raises(Exception, match="world_size"):
        dist.refuse_world_size({"steps": 10, "world_size": 2})
    dist.refuse_world_size({"steps": 10})

    snapshot = torch.load(pair.run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    assert "world_size" not in snapshot["config"]


# ---------------------------------------------------------------------- the partition


def test_the_shard_is_disjoint_and_complete() -> None:
    """Over 20 steps: no sample twice, and the union is exactly the single-card stream."""
    whole = ToyStream()
    expected = [whole.batch(step, TOKENS).ids for step in range(STEPS)]
    world = 2
    shards = [dist.shard_stream(whole, rank, world) for rank in range(world)]
    seen = [[shard.batch(step, TOKENS).ids for step in range(STEPS)] for shard in shards]
    for step in range(STEPS):
        rows = [shard[step].shape[0] for shard in seen]
        assert sum(rows) == ROWS, "the shards do not cover the batch"
        rebuilt = torch.cat([shard[step] for shard in seen])
        assert torch.equal(rebuilt, expected[step]), "the union is not the single-card batch"
        assert torch.equal(torch.cat([seen[0][step], seen[1][step]]), rebuilt)
    assert _digest(
        [torch.cat([shard[step] for shard in seen]) for step in range(STEPS)]
    ) == _digest(expected)


def test_an_uneven_batch_is_refused() -> None:
    """A batch that does not divide by the world is a refusal, not a silent reweighting."""
    shard = dist.shard_stream(ToyStream(), 0, 3)
    with pytest.raises(Exception, match="does not split"):
        shard.batch(0, TOKENS)


# --------------------------------------------------------------------- the equivalence


def test_two_cards_are_one_card(single: Job, pair: Job) -> None:
    """2 ranks × 2 accumulation steps equal one process over the same effective batch.

    Weights within 1e-6 (fp32, so the difference is the order of the sums, not a
    different estimator), the same global loss per step, and the same batch *sets*: the
    ranks' shards concatenated are the single-card batch, so the hash over all 20 steps
    is one number for both runs.
    """
    assert len(pair.steps) == len(single.steps) == STEPS
    worst = _worst(pair.weights[0], single.weights[0])
    assert worst <= 1e-6, f"two cards diverged from one by {worst:.3e}"
    assert _worst(pair.weights[0], pair.weights[1]) == 0.0, "DDP let the ranks drift"
    assert _digest(_consumed(pair)) == _digest(_consumed(single))
    for step, (distributed, alone) in enumerate(zip(pair.losses, single.losses, strict=True)):
        drift = abs(distributed - alone) / max(abs(alone), 1e-6)
        assert drift <= TOL, f"step {step}: {distributed} != {alone} ({drift:.2e} relative)"


def test_per_microbatch_mean_fails_the_equivalence_gate(
    single: Job, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The bug the normalisation prevents is caught by the gate above.

    `--bug` multiplies each micro-batch's mean loss by `accum`, which is exactly "do not
    divide by `accum`": the gradient is `accum` times the global token mean on every
    rank, and the weights drift. The same comparison that passes at 1e-6 for the honest
    run must fail for this one, or the gate is decoration.
    """
    buggy = run_job(tmp_path_factory.mktemp("buggy"), world=2, bug=True)
    worst = _worst(buggy.weights[0], single.weights[0])
    # 2026-11-16, tied-head 1/sqrt(d_model) scale (PR #74): the loss level, and with it
    # the injected gradient magnitude, fell; the drift went 7.7e-3 -> 1.46e-4. The
    # honest 2-vs-1 drift is 5.2e-7, so 1e-5 keeps a 14x margin below the injected run
    # and 19x above reduction noise — the bar tracks the new scale without weakening
    # the control, which still fails on the injected bug (and the loss-ratio assert
    # below is an independent proof that the injection ran).
    assert worst > 1e-5, "the per-micro-batch mean was not caught: the gate proves nothing"
    assert buggy.losses[0] == pytest.approx(single.losses[0] * ACCUM, rel=1e-5)


# ---------------------------------------------------------------- journal and checkpoint


def test_two_ranks_write_one_journal(single: Job, pair: Job) -> None:
    """One `run.jsonl` for the run, the same step count as one card, and global numbers."""
    journals = list(pair.run_dir.rglob("run.jsonl"))
    assert len(journals) == 1, f"two ranks wrote {journals}"
    assert len(pair.steps) == len(single.steps) == STEPS

    locals_ = pair.local_losses
    for index, record in enumerate(pair.steps):
        assert record["loss"] == pytest.approx(
            sum(rank[index] for rank in locals_) / len(locals_), abs=1e-9
        )
        # The step, not the shard: the journal and `report.audit` speak about the batch.
        assert record["tokens"] == TOKENS
        assert record["bytes"] == TOKENS * 4
    assert [record["tokens"] for record in pair.steps] == [
        record["tokens"] for record in single.steps
    ]
    # Each rank carries half the step; the journal carries the step.
    assert [entry["tokens"] for entry in pair.ranks[0]["seen"]] == [TOKENS // 2] * STEPS
    assert len(pair.ranks[1]["seen"][0]["ids"]) == ROWS // 2, "a rank read the whole batch"


def test_only_rank_zero_touches_the_run_directory(pair: Job) -> None:
    """One checkpoint, no temporary, no second writer: a file per rank is a run per rank."""
    names = sorted(path.name for path in pair.run_dir.iterdir())
    assert names == ["ckpt.pt", "run.jsonl"], names
    snapshot = torch.load(pair.run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    assert not any(key.startswith("module.") for key in snapshot["model"])


def test_a_two_card_checkpoint_loads_on_one_card(
    pair: Job, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Resume the two-card checkpoint on one card: same step, same loss, same weights."""
    root = tmp_path_factory.mktemp("resume")
    two, one = root / "two" / "run", root / "one" / "run"
    resumed_two = run_job(root / "two", world=2, mode="resume", run_dir=two, seed=pair.run_dir)
    resumed_one = run_job(root / "one", world=1, mode="resume", run_dir=one, seed=pair.run_dir)
    assert resumed_two.ranks[0]["step"] == resumed_one.ranks[0]["step"] == STEPS + 1
    assert resumed_one.losses[-1] == pytest.approx(resumed_two.losses[-1], abs=TOL)
    assert _worst(resumed_one.weights[0], resumed_two.weights[0]) <= TOL


# -------------------------------------------------------------------- the reduction


def _step_record(**overrides: Any) -> dict[str, Any]:
    """One rank's version of a step line, with the fields the loop actually writes."""
    record: dict[str, Any] = {
        "batch": "aa",
        "bpb": 2.0,
        "bytes": 256,
        "clip_events": 0,
        "data_s": 1.0,
        "grad_norm": 3.0,
        "loss": 1.0,
        "lr": 0.01,
        "skipped": False,
        "stage": "bulk",
        "step": 0,
        "step_s": 1.0,
        "tokens": 64,
        "tokens_per_s": 10.0,
    }
    return record | overrides


def test_every_field_is_reduced_by_its_own_rule() -> None:
    """Loss/bpb average, work sums, the worst duration wins, and the step is the world's.

    The table is the whole claim of "one journal": a sum where a mean belongs reports a
    loss `world` times too large, and it looks like a diverging run.
    """
    left, right = (
        _step_record(),
        _step_record(
            batch="bb",
            bpb=4.0,
            loss=3.0,
            grad_norm=5.0,
            data_s=2.0,
            step_s=3.0,
            clip_events=2,
            tokens_per_s=20.0,
            skipped=True,
        ),
    )
    merged = dist._merge([left, right], "step")
    assert merged["loss"] == 2.0
    assert merged["bpb"] == 3.0
    assert merged["grad_norm"] == 5.0
    assert merged["data_s"] == 2.0
    assert merged["step_s"] == 3.0
    assert merged["clip_events"] == 2
    assert merged["tokens_per_s"] == 30.0
    assert merged["skipped"] is True
    # The step's size, not the shard's: equal shards, so a rank's number times the world.
    assert merged["tokens"] == 2 * 64
    assert merged["bytes"] == 2 * 256
    assert merged["batch"] == dist._combine(["aa", "bb"])
    # Identical fields are rank 0's, unreduced: a step index is not an average.
    assert merged["step"] == 0
    assert merged["lr"] == 0.01
    assert merged["stage"] == "bulk"


def test_an_eval_window_is_not_multiplied_by_the_world() -> None:
    """Every rank evaluates the same window; scaling it would invent bytes nobody read."""
    records = [
        {"step": 5, "bpb": 1.0, "bytes": 4096, "weights": "training"},
        {"step": 5, "bpb": 1.2, "bytes": 4096, "weights": "training"},
    ]
    merged = dist._merge(records, "eval")
    assert merged["bytes"] == 4096
    assert merged["bpb"] == pytest.approx(1.1)


def test_the_reduction_refuses_shards_that_are_not_a_partition() -> None:
    """Unequal shards and mismatched shapes are refusals, not numbers to average."""
    with pytest.raises(Exception, match="disagree on the step's tokens"):
        dist._merge([_step_record(), _step_record(tokens=32)], "step")
    with pytest.raises(Exception, match="different journal fields"):
        dist._merge([_step_record(), _step_record(extra=1)], "step")
    with pytest.raises(Exception, match="disagree on the step's bytes"):
        dist._merge([_step_record(), _step_record(bytes=128)], "step")


def test_a_single_rank_writes_the_plain_record(tmp_path: Path) -> None:
    """Without a process group the reduction is the identity and rank 0 is the journal."""
    ranked = dist.RankJournal(
        Journal.open(tmp_path / "run" / "run.jsonl"), topo=dist.Topology(0, 1)
    )
    record = ranked.write("step", **_step_record())
    assert (tmp_path / "run" / "run.jsonl").exists()
    assert ranked.path.exists()
    assert ranked.last("step") == record
    assert ranked.raw == [_step_record()]


def test_a_non_main_rank_writes_nothing(tmp_path: Path) -> None:
    """Every rank remembers the record (so `last()` branches agree); only rank 0 writes."""
    ranked = dist.RankJournal(
        Journal.open(tmp_path / "run" / "run.jsonl"), topo=dist.Topology(1, 2)
    )
    ranked.write("step", **_step_record())
    assert not (tmp_path / "run" / "run.jsonl").exists()
    assert str(ranked.path) == os.devnull
    assert ranked.last("step") is not None
    with pytest.raises(Exception, match="unknown journal event"):
        ranked.write("invented", step=0)


def test_the_wrapper_shows_the_bare_model() -> None:
    """The DDP view answers with the bare model's names: the clip and EMA are keyed by them."""
    torch.manual_seed(0)
    model = reference_stack(**SMALL)
    view = dist._Parallel(SimpleNamespace(module=model))  # type: ignore[arg-type]
    assert [name for name, _ in view.named_parameters()] == [
        name for name, _ in model.named_parameters()
    ]
    assert set(view.state_dict()) == set(model.state_dict())
    assert view.load_state_dict(model.state_dict()) is not None
