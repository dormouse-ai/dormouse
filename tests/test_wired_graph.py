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

# The 16-step eager sequence: (step, loss, grad_norm, clip_events, skipped, batch
# hash, bpb). Recaptured 2026-11-16 after the tied-head 1/sqrt(d_model) scale (PR #74,
# fix/tied-head-scale) moved the loss level: step-0 loss 12.074 -> 3.355, and every
# grad_norm/bpb fell with it, so fewer grad norms clear CLIP — clip_events drops 7 -> 6
# at steps 3 and 15 and GOLDEN_COUNTS["clip_events"] follows. That is loss magnitude,
# not a broken mechanism: step index, skip flags, batch hashes and every other counter
# (including `tied_head`, opt_* and ema) are bit-identical; test_model.py's independent
# init-CE ~= ln(V) gate pins the new scale. Remove the sqrt(d) divisor and these floats
# move back to the old level.
# One float32 ulp drifts across builds (CPU reduction width / torch wheel):
# step 7 grad_norm; step 9 loss and bpb; step 10 grad_norm; step 11 loss and
# bpb; step 14 loss, grad_norm and bpb; step 15 loss and bpb. Running this exact
# test against pristine unwired master shows the same drift, so it is not a
# behaviour change from deferring host reads. Floats are compared to 2e-6
# (about two ulps at these magnitudes); clip events, skip flags, batch hashes,
# the counter set and every other field stay exact.
GOLDEN = [
    (0, 3.354806661605835, 1.0694636106491089, 0, False, "45d795f783ff2f73", 1.2099907334599993),
    (1, 3.0090034008026123, 0.865772545337677, 1, False, "c3209df410fd9f54", 1.0852685710889887),
    (2, 3.0713627338409424, 1.0557721853256226, 5, False, "3264732e96a1ae44", 1.1077599462208743),
    (3, 3.0693202018737793, 0.9755256772041321, 6, False, "5ff4c7dcc408993a", 1.1070232585359034),
    (4, 2.8468103408813477, 0.8700788021087646, 6, False, "a12f987bdaeea10a", 1.026769790285235),
    (5, 2.68221116065979, 0.8584632277488708, 7, False, "c72c3647cd0a93ad", 0.9674031850252275),
    (6, 3.1665616035461426, 1.1004821062088013, 7, False, "ef88d55d1cd516e3", 1.142095680526356),
    (7, 2.309046506881714, 0.7381711006164551, 7, False, "8c6831a96d97123f", 0.8328124861650581),
    (8, 2.6729798316955566, 1.040021538734436, 7, False, "774c5e92f22d297f", 0.9640736868958489),
    (9, 2.4285171031951904, 0.9641995429992676, 7, False, "2f40a5e15addf1af", 0.8759023953734331),
    (10, 2.6522834300994873, 1.2329951524734497, 7, False, "b1c123284c7bbd9c", 0.956609037909125),
    (11, 2.4393813610076904, 0.9891476035118103, 7, False, "b0e23bab55edb538", 0.8798208480906913),
    (12, 2.19451642036438, 1.0376875400543213, 7, False, "62552c184d0af98d", 0.7915044892022727),
    (13, 1.999880313873291, 0.8581827878952026, 7, False, "7672976ebbe2b914", 0.7213043527991152),
    (14, 1.942959189414978, 0.7194468379020691, 7, False, "2add7dc5cd9bb5fd", 0.7007743968046573),
    (15, 1.8842633962631226, 0.6673699021339417, 6, False, "0b47a29316a0049b", 0.6796043643793507),
]
GOLDEN_COUNTS = {
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
    "clip_events": 6,
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
