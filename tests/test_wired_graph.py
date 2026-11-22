"""The graph wiring's CPU gates: parity, replay, deferred reads and loud refusal.

The card-side parity (`tests/test_graph.py`) needs a GPU. These gates run on CPU and
pin the contract the wiring adds around it: the eager sequence is unchanged, a replay
is actually used, the hot path reads no host scalar, and a refused capture says so.
"""

from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from dormouse_core.errors import LoudError
from test_loop import build, config, lines

from dormouse.train import loop as loop_mod
from dormouse.train.step import CLIP, StepResult, train_step

# The 16-step eager sequence, captured from origin/master before the wiring:
# (step, loss, grad_norm, clip_events, skipped, batch hash, bpb).
# One float32 ulp drifts across builds (CPU reduction width / torch wheel):
# step 7 grad_norm; step 9 loss and bpb; step 10 grad_norm; step 11 loss and
# bpb; step 14 loss, grad_norm and bpb; step 15 loss and bpb. Running this exact
# test against pristine unwired master shows the same drift, so it is not a
# behaviour change from deferring host reads. Floats are compared to 2e-6
# (about two ulps at these magnitudes); clip events, skip flags, batch hashes,
# the counter set and every other field stay exact.
GOLDEN = [
    (0, 12.074398040771484, 5.131967067718506, 0, False, "45d795f783ff2f73", 4.354918543785109),
    (1, 10.487138748168945, 4.357778072357178, 1, False, "c3209df410fd9f54", 3.782435766274457),
    (2, 10.859282493591309, 5.0793280601501465, 5, False, "3264732e96a1ae44", 3.9166582502791294),
    (3, 10.772439002990723, 4.807574272155762, 7, False, "5ff4c7dcc408993a", 3.8853360819733913),
    (4, 9.983198165893555, 4.464990615844727, 6, False, "a12f987bdaeea10a", 3.6006776215366068),
    (5, 9.000713348388672, 4.40773868560791, 7, False, "c72c3647cd0a93ad", 3.2463211280458584),
    (6, 11.202688217163086, 5.471414566040039, 7, False, "ef88d55d1cd516e3", 4.040515683881602),
    (7, 7.548826217651367, 3.967719316482544, 7, False, "8c6831a96d97123f", 2.7226635371845544),
    (8, 8.86369800567627, 5.3084893226623535, 7, False, "774c5e92f22d297f", 3.1969032891816376),
    (9, 7.7037553787231445, 5.204407691955566, 7, False, "2f40a5e15addf1af", 2.77854242027639),
    (10, 8.72290325164795, 6.226996421813965, 7, False, "b1c123284c7bbd9c", 3.146122315826678),
    (11, 7.02910852432251, 5.456736087799072, 7, False, "b0e23bab55edb538", 2.535215002477606),
    (12, 6.245589256286621, 5.545576095581055, 7, False, "62552c184d0af98d", 2.252620161868524),
    (13, 4.962966442108154, 5.8926544189453125, 7, False, "7672976ebbe2b914", 1.7900117685319443),
    (14, 4.510542392730713, 4.228295803070068, 7, False, "2add7dc5cd9bb5fd", 1.6268342854280098),
    (15, 4.2043023109436035, 4.413464069366455, 7, False, "0b47a29316a0049b", 1.5163815235990865),
]
GOLDEN_COUNTS = {
    "step_heartbeat": 1,
    "fwd": 32,
    "embed_fwd": 32,
    "norm_fwd": 96,
    "attn_fwd": 32,
    "rope_fwd": 32,
    "ffn_fwd": 32,
    "block_fwd": 1,
    "head_fwd": 32,
    "tied_head": 1,
    "bwd": 32,
    "clip": 1,
    "clip_tensors": 9,
    "clip_events": 7,
    "opt_muon": 5,
    "opt_adamw": 4,
    "opt_steps": 1,
    "ema": 1,
}


def _sequence(trainer: Any) -> list[tuple[Any, ...]]:
    """The step journal's parity fields, in order."""
    return [
        (
            row["step"],
            row["loss"],
            row["grad_norm"],
            row["clip_events"],
            row["skipped"],
            row["batch"],
            row["bpb"],
        )
        for row in lines(trainer.journal, "step")
    ]


def _assert_sequence(trainer: Any) -> None:
    """Compare the journal to the golden: exact discrete fields, ulp-tolerant floats."""
    actual = _sequence(trainer)
    assert len(actual) == len(GOLDEN)
    for step, (got, want) in enumerate(zip(actual, GOLDEN, strict=False)):
        assert (got[0], got[3], got[4], got[5]) == (
            want[0],
            want[3],
            want[4],
            want[5],
        ), f"step {step}: a discrete field moved"
        for index, name in ((1, "loss"), (2, "grad_norm"), (6, "bpb")):
            assert got[index] == pytest.approx(want[index], rel=0, abs=2e-6), (
                step,
                name,
                got[index],
                want[index],
            )


def test_graph_off_reproduces_the_eager_sequence(tmp_path: Path) -> None:
    """graph=False reproduces the day the graph was still unwired, to one ulp."""
    trainer = build(config(16), tmp_path / "run.jsonl")
    trainer.train(steps=16)
    _assert_sequence(trainer)
    assert {k: int(v) for k, v in trainer.counters.counts.items() if v} == GOLDEN_COUNTS
    assert all("graph_replays" not in row for row in lines(trainer.journal, "step"))


@dataclass
class _FakeGraph:
    """A counting stand-in: the replay path, with the eager arithmetic behind it."""

    model: torch.nn.Module
    optimizer: Any
    accum: int
    replays: int = 0

    def run(
        self,
        ids: torch.Tensor,
        *,
        counters: Any = None,
        clip: float = CLIP,
        pre_step: Any = None,
    ) -> StepResult:
        """Count the replay and take the real step through the eager function."""
        self.replays += 1
        if counters is not None:
            counters.count("graph_replays")
        return train_step(
            self.model,
            ids,
            optimizer=self.optimizer,
            accum=self.accum,
            clip=clip,
            counters=counters,
            pre_step=pre_step,
        )


def test_graph_on_replays_after_warmup_and_logs_the_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay is used after warm-up and `graph_replays` reaches the journal."""
    made: list[_FakeGraph] = []

    def factory(
        model: torch.nn.Module, optimizer: Any, ids: torch.Tensor, *, accum: int = 1
    ) -> _FakeGraph:
        graph = _FakeGraph(model, optimizer, accum)
        made.append(graph)
        return graph

    monkeypatch.setattr(loop_mod, "StepGraph", factory)
    trainer = build(config(16, warmup_steps=2), tmp_path / "run.jsonl")
    trainer.use_graph = True
    trainer.train(steps=6)
    assert len(made) == 1, "one graph is built, once, after warm-up"
    assert made[0].replays == 4, "steps 2..5 replay; the warm-up steps stay eager"
    rows = lines(trainer.journal, "step")
    assert [row.get("graph_replays", 0) for row in rows] == [0, 0, 1, 1, 1, 1]
    assert trainer.counters.counts["graph_replays"] == 1


def test_the_hot_path_takes_no_host_read_when_reads_are_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `.item()`/`.tolist()` in five steps, and the loss/norm never cross the seam.

    `torch.optim` reads its own CPU step counter with `.item()` — exempt, as in the
    device-firewall gate it mirrors. The step's one declared crossing is the finite
    flag; at `log_every=10` the loss and the norm stay device-resident for all 5 steps.
    """
    real_item, real_tolist = torch.Tensor.item, torch.Tensor.tolist

    def boom(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        caller = inspect.currentframe().f_back
        module = caller.f_globals.get("__name__", "") if caller else ""
        if module.startswith("torch.optim"):
            return real_item(self, *args, **kwargs)
        raise AssertionError(f"host read in the hot path: {module} called .item()/.tolist()")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)

    from dormouse.train import step as step_mod

    calls: list[str] = []
    real_seam = step_mod.host_scalar

    def counted(value: torch.Tensor) -> float:
        calls.append("read")
        return real_seam(value)

    monkeypatch.setattr(step_mod, "host_scalar", counted)
    trainer = build(config(16), tmp_path / "run.jsonl")
    trainer.train(steps=5, log_every=10)
    assert len(calls) == 5, "one flag read per step; no loss/norm reads at log_every=10"
    torch.Tensor.item, torch.Tensor.tolist = real_item, real_tolist


def test_a_refused_capture_is_loud_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --graph run whose capture fails names the flag; it does not run eager."""

    def refuses(*args: Any, **kwargs: Any) -> Any:
        raise LoudError(
            "the step could not be captured: a device sync in MoE", "make it device-side"
        )

    monkeypatch.setattr(loop_mod, "StepGraph", refuses)
    trainer = build(config(16), tmp_path / "run.jsonl")
    trainer.use_graph = True
    with pytest.raises(LoudError, match="--graph requested"):
        trainer.train(steps=16)
    assert trainer.step == 1, "the warm-up step ran; no step ran after the refusal"


def test_cli_passes_graph_and_plan_prints_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`dorm train --graph` reaches entry, and `dorm plan --graph` prints the same state."""
    from dormouse import cli
    from dormouse.train import entry

    seen: dict[str, Any] = {}

    def fake_train(config_path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "plan_flops": 1,
            "plan_source": "bill",
            "directory": str(tmp_path),
            "steps": 0,
            "bpb": 0.0,
        }

    monkeypatch.setattr(entry, "train", fake_train)
    with redirect_stdout(io.StringIO()):
        assert (
            cli.main(["train", str(tmp_path / "x.toml"), "--graph", "--runs", str(tmp_path)]) == 0
        )
    assert seen["graph"] is True

    # The plan needs a real stack spec; use the committed smoke config.
    root = Path(__file__).resolve().parents[1]
    assert cli.main(["plan", str(root / "configs" / "dormouse_1b.toml"), "--graph"]) == 0
    assert "graph=on" in capsys.readouterr().out
