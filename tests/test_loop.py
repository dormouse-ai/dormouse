"""The loop: eight gates, each one failing a plausible wrong version of the recipe.

The gates are the recipe's own failure modes: a resume that is smooth but not exact, a
BPB taken from a token count, a clip that never fires or fires everywhere, an EMA that is
a copy, a spike that is rolled back instead of skipped, an anneal whose tail no stage
covers, a stalled stream nobody notices, and a run where a mechanism is enabled and
silent.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal
from stacks import reference_stack

from dormouse.cli import journal_row
from dormouse.train import report
from dormouse.train.guards import adaptive_clip
from dormouse.train.loop import Trainer
from dormouse.train.optim import MixOptimizer
from dormouse.train.recipe import Batch, LoopConfig, Stage, annealed_schedule

SMALL = {"vocab": 16, "d_model": 16, "n_layers": 1, "n_heads": 2, "head_dim": 8, "d_ffn": 32}
ROWS, SEQ = 2, 16
TOKENS = ROWS * SEQ
LN2 = math.log(2)


class ToyStream:
    """A stream whose batch is a pure function of the step, as `data/bin.py` requires.

    The ids are drawn from a four-symbol alphabet of a sixteen-token vocabulary, so a
    model that learns anything lands near `ln 4` and one that learns nothing sits at
    `ln 16` — a two-nat fall, which is a curve a test can read.
    """

    def __init__(
        self,
        stages: tuple[tuple[int, str], ...] = ((0, "bulk"),),
        *,
        bytes_per_token: float = 4.0,
        stall_at: int | None = None,
    ) -> None:
        self.stages = stages
        self.bytes_per_token = bytes_per_token
        self.stall_at = stall_at

    def batch(self, step: int, tokens: int) -> Batch:
        """Return the batch for `step`; `tokens` is a request the toy ignores."""
        if step == self.stall_at:
            time.sleep(1.5)
        generator = torch.Generator().manual_seed(1000 + step)
        ids = torch.randint(0, 4, (ROWS, SEQ + 1), generator=generator)
        stage = [name for start, name in self.stages if step >= start][-1]
        return Batch(
            ids=ids,
            tokens=TOKENS,
            payload_bytes=round(TOKENS * self.bytes_per_token),
            stage=stage,
        )


def config(steps: int = 20, **overrides: Any) -> LoopConfig:
    """A tiny recipe with the same shape as the real one: bulk, then the anneal stage."""
    anneal = steps - round(0.22 * steps)
    base: dict[str, Any] = {
        "steps": steps,
        "stages": (Stage("bulk", 0, anneal), Stage("clean", anneal, steps)),
        "warmup_steps": max(1, steps // 10),
        "tokens_per_step": TOKENS,
        "spike_window": 10,
    }
    base.update(overrides)
    return LoopConfig(**base)


def toy(cfg: LoopConfig, **overrides: Any) -> ToyStream:
    """A stream whose stage boundaries are the config's, so the two cannot disagree."""
    return ToyStream(tuple((stage.start, stage.name) for stage in cfg.stages), **overrides)


def build(
    cfg: LoopConfig,
    journal_path: Path,
    *,
    stream: ToyStream | None = None,
    eval_stream: ToyStream | None = None,
    seed: int = 23,
) -> Trainer:
    """A fresh tiny stack, its optimizer, and a Trainer over a journal at `journal_path`."""
    torch.manual_seed(seed)
    model = reference_stack(**SMALL)
    stages = tuple((stage.start, stage.name) for stage in cfg.stages)
    return Trainer(
        model,
        stream or ToyStream(stages),
        MixOptimizer(model),
        annealed_schedule(cfg),
        cfg,
        Journal.open(journal_path),
        eval_stream,
    )


def lines(journal: Journal, kind: str) -> list[dict[str, Any]]:
    """Every journal record of one kind, in order."""
    return [record for record in journal.records if record.get("kind") == kind]


def seq_of(journal: Journal) -> list[tuple[Any, Any]]:
    """The batch hash and loss of every step: what a resume has to reproduce."""
    return [(record["batch"], record["loss"]) for record in lines(journal, "step")]


# ------------------------------------------------------------------------- resume


def test_resume_is_exact(tmp_path: Path) -> None:
    """60 steps + reload + 40 must equal one run of 100: same batches, same losses.

    Bit-identical and not "close": the stream is a pure function of the step, so the
    only way the two runs differ is state the checkpoint did not carry — the clip
    estimates or the optimizer's moments.
    """
    whole_cfg, split_cfg = config(100), config(100)
    whole = build(whole_cfg, tmp_path / "whole" / "run.jsonl")
    whole.train(steps=100)

    split = build(split_cfg, tmp_path / "split" / "run.jsonl")
    split.train(steps=60, ckpt_every=60)
    torch.manual_seed(23)
    fresh = reference_stack(**SMALL)
    resumed = Trainer.load(
        tmp_path / "split",
        model=fresh,
        stream=ToyStream(tuple((stage.start, stage.name) for stage in split_cfg.stages)),
        optim=MixOptimizer(fresh),
        schedule=annealed_schedule(split_cfg),
        journal=Journal.open(tmp_path / "split" / "run.jsonl"),
    )
    resumed.train(steps=100)

    assert [record["step"] for record in lines(resumed.journal, "step")] == list(range(100))
    assert seq_of(whole.journal) == seq_of(resumed.journal)
    assert seq_of(whole.journal)[:60] == seq_of(split.journal)[:60]


# -------------------------------------------------------------------- bytes and BPB


def test_bpb_comes_from_the_bytes_consumed(tmp_path: Path) -> None:
    """A stream of 4 B/token must yield `loss/(4·ln2)`, not `loss/ln2`."""
    trainer = build(config(16), tmp_path / "run.jsonl", stream=toy(config(16), bytes_per_token=4.0))
    trainer.train(steps=16)
    steps = lines(trainer.journal, "step")
    assert len(steps) == 16
    for record in steps:
        assert record["tokens"] == TOKENS
        assert record["bytes"] == 4 * TOKENS
        assert record["bpb"] == pytest.approx(record["loss"] / (4 * LN2), rel=1e-9)
    assert trainer.payload_bytes == 16 * 4 * TOKENS
    assert report.total_bpb(trainer.nats, trainer.payload_bytes) == pytest.approx(
        trainer.nats / trainer.payload_bytes / LN2, rel=1e-12
    )
    assert trainer.audit() == {"steps": 16, "tokens": 16 * TOKENS, "bytes": 16 * 4 * TOKENS}


def test_a_manifest_that_disagrees_with_the_data_refuses(tmp_path: Path) -> None:
    """The byte count is the measurement, so a batch that contradicts its manifest stops."""

    class Liar(ToyStream):
        def batch(self, step: int, tokens: int) -> Batch:
            honest = super().batch(step, tokens)
            return Batch(honest.ids, honest.tokens, honest.payload_bytes * 2, honest.stage)

    trainer = build(config(16), tmp_path / "run.jsonl", stream=Liar(toy(config(16)).stages))
    with pytest.raises(LoudError, match="manifest"):
        trainer.train(steps=2)


def test_an_audit_that_cannot_explain_the_bytes_fails(tmp_path: Path) -> None:
    """A journal with a step line missing is a run whose BPB cannot be reconstructed."""
    trainer = build(config(16), tmp_path / "run.jsonl")
    trainer.train(steps=16)
    assert trainer.audit()["steps"] == 16
    kept = [
        line
        for line in trainer.journal.path.read_text().splitlines()
        if json.loads(line)["kind"] != "step"
    ]
    trainer.journal.path.write_text("\n".join(kept) + "\n")
    trainer.journal = Journal.open(trainer.journal.path)
    with pytest.raises(LoudError, match="journal explains"):
        trainer.audit()


# ------------------------------------------------------------------------- clipping


def test_per_tensor_clipping_bounds_a_hundredfold_gradient() -> None:
    """100× the reference on every tensor comes out at the ceiling, per tensor."""
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    running = {name: 1e-3 for name, _ in model.named_parameters()}
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    examined, scaled = adaptive_clip(model, running, 2.5)
    assert examined == len(running)
    assert scaled == len(running)
    for _, param in model.named_parameters():
        assert float(param.grad.norm()) == pytest.approx(2.5e-3, rel=1e-5)


def test_the_loop_clips_and_counts_events(tmp_path: Path) -> None:
    """Through the loop: the policy is on by default and the journal names the events."""
    trainer = build(config(16, clip_z=2.5), tmp_path / "run.jsonl")
    trainer.clip_norms = {name: 1e-3 for name, _ in trainer.model.named_parameters()}
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)
    counters = Counters()
    assert trainer._pre_step(0.0, counters) is True
    assert counters.counts["clip_tensors"] == len(trainer.clip_norms)
    assert counters.counts["clip_events"] == len(trainer.clip_norms)
    assert trainer.clip_events == len(trainer.clip_norms)
    for _, param in trainer.model.named_parameters():
        assert float(param.grad.norm()) == pytest.approx(2.5e-3, rel=1e-5)


def test_with_clipping_off_the_same_gradient_passes_untouched(tmp_path: Path) -> None:
    """`clip=False` is the recipe's counterfactual: no scale, no estimate, no counter."""
    trainer = build(config(16, clip=False), tmp_path / "run.jsonl")
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)
    before = {name: param.grad.clone() for name, param in trainer.model.named_parameters()}
    counters = Counters()
    assert trainer._pre_step(0.0, counters) is True
    assert trainer.clip_norms == {}
    assert counters.counts == {}
    for name, param in trainer.model.named_parameters():
        assert torch.equal(param.grad, before[name])


# ------------------------------------------------------------------------------ EMA


def test_the_ema_differs_from_the_live_weights_and_is_returned(tmp_path: Path) -> None:
    """After five checkpoint updates the average is not the weights, and it is the model."""
    trainer = build(config(20), tmp_path / "run.jsonl")
    trainer.train(steps=20, ckpt_every=5)
    assert trainer.ema
    live = {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
    differing = [name for name in live if not torch.equal(trainer.ema[name], live[name])]
    assert differing, "the EMA is a copy of the live weights, not an average"
    final = trainer.final_model()
    for name, param in final.named_parameters():
        assert torch.equal(param, trainer.ema[name])
    assert trainer.counters.counts["ema"] > 0


def test_no_ema_returns_the_live_weights_and_a_zero_counter(tmp_path: Path) -> None:
    """`ema=False` is visible: no average, no counter, and `final_model()` is the run."""
    trainer = build(config(16, ema=False), tmp_path / "run.jsonl")
    trainer.train(steps=16, ckpt_every=5)
    assert trainer.ema is None
    assert "ema" not in trainer.counters.enabled
    assert trainer.counters.counts.get("ema", 0) == 0
    assert trainer.final_model() is trainer.model
    assert trainer._checkpoint().exists()


# ---------------------------------------------------------------------------- spikes


class Spiky(torch.nn.Module):
    """A stack whose logits are `scale`× the model's: a loss far outside the band."""

    def __init__(self, inner: torch.nn.Module, scale: float = 1.0) -> None:
        super().__init__()
        self.inner = inner
        self.scale = scale

    def forward(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Return the wrapped stack's logits, amplified."""
        return self.inner(ids, counters) * self.scale


def spiky_trainer(cfg: LoopConfig, path: Path) -> tuple[Trainer, Spiky]:
    """A Trainer over a stack the test can make spike between steps."""
    torch.manual_seed(23)
    model = Spiky(reference_stack(**SMALL))
    stages = tuple((stage.start, stage.name) for stage in cfg.stages)
    trainer = Trainer(
        model,
        ToyStream(stages),
        MixOptimizer(model),
        annealed_schedule(cfg),
        cfg,
        Journal.open(path),
    )
    return trainer, model


def test_a_spike_is_skipped_with_the_weights_untouched(tmp_path: Path) -> None:
    """A loss above the band leaves every parameter and moment exactly as they were."""
    cfg = config(20)
    trainer, model = spiky_trainer(cfg, tmp_path / "run.jsonl")
    trainer.train(steps=12)
    assert trainer.spike_skips == 0
    before = {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
    model.scale = 1e3
    trainer.train(steps=13)
    assert trainer.skips == 1
    assert trainer.spike_skips == 1
    assert trainer.consecutive == 1
    for name, param in trainer.model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} moved on a skipped step"
    assert lines(trainer.journal, "step")[-1]["skipped"] is True

    model.scale = float("inf")  # a NaN loss is the same skip by the step's own firewall
    trainer.train(steps=14)
    assert trainer.skips == 2
    assert trainer.consecutive == 2
    for name, param in trainer.model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} moved on a skipped step"


def test_six_consecutive_spikes_are_a_divergence(tmp_path: Path) -> None:
    """Five skips is a spike; the sixth is a run that is not training, and it raises."""
    cfg = config(20, max_skips=5)
    trainer, model = spiky_trainer(cfg, tmp_path / "run.jsonl")
    trainer.train(steps=12)
    model.scale = 1e3
    with pytest.raises(LoudError, match="divergence"):
        trainer.train(steps=20)
    assert trainer.consecutive == 6


# ------------------------------------------------------------------- stages, anneal


def test_an_uncovered_tail_refuses_and_names_the_gap() -> None:
    """The tail with no stage is the anneal that does not happen; it is refused up front."""
    with pytest.raises(LoudError, match=r"\[800, 1000\) has no stage"):
        LoopConfig(steps=1000, stages=(Stage("bulk", 0, 800),), warmup_steps=100)


def test_a_gap_between_stages_refuses() -> None:
    """A hole is the same defect as an uncovered tail, one stage earlier."""
    with pytest.raises(LoudError, match="uncovered"):
        LoopConfig(
            steps=100,
            stages=(Stage("bulk", 0, 40), Stage("clean", 50, 100)),
            warmup_steps=10,
        )


def test_a_schedule_that_never_anneals_refuses(tmp_path: Path) -> None:
    """A flat rate to the last step is remote from the recipe; the loop says so."""
    cfg = config(20)
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    with pytest.raises(LoudError, match="anneal"):
        Trainer(
            model,
            ToyStream(),
            MixOptimizer(model),
            lambda step: 1.0,
            cfg,
            Journal.open(tmp_path / "flat" / "run.jsonl"),
        )


# ------------------------------------------------------------------------- guard


def test_a_stalled_stream_produces_the_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stream that starts sleeping after the baseline window drops 20% and says so."""
    cfg = config(20, guard_window=4)
    stages = tuple((stage.start, stage.name) for stage in cfg.stages)
    trainer = build(cfg, tmp_path / "run.jsonl", stream=ToyStream(stages, stall_at=6))
    with caplog.at_level("WARNING", logger="dormouse.train.loop"):
        trainer.train(steps=10)
    assert trainer.throughput_warnings >= 1
    assert any("data stall" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------- the real run


def test_a_schedule_free_run_anneals_through_the_optimizer(tmp_path: Path) -> None:
    """`sf` derives its own rate: the loop must not write a schedule on top of it."""
    cfg = config(20)
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    trainer = Trainer(
        model,
        toy(cfg),
        MixOptimizer(model, schedule="sf"),
        annealed_schedule(cfg),
        cfg,
        Journal.open(tmp_path / "sf" / "run.jsonl"),
    )
    trainer.train(steps=20, ckpt_every=10)
    assert trainer.audit()["steps"] == 20
    assert all(line["lr"] >= 0 for line in lines(trainer.journal, "step"))
    assert trainer.ema


def test_two_hundred_steps_learn_and_every_mechanism_reports(tmp_path: Path) -> None:
    """The only test that can tell a loop from arithmetic that has the right shape."""
    cfg = config(200, warmup_steps=10)
    trainer = build(cfg, tmp_path / "run.jsonl", eval_stream=ToyStream())
    trainer.train(steps=200, ckpt_every=25, eval_every=50)

    steps = lines(trainer.journal, "step")
    assert len(steps) == 200
    losses = [record["loss"] for record in steps]
    opening, closing = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    assert closing < opening - 0.5, (opening, closing)

    assert trainer.counters.silent() == []
    assert trainer.counters.counts["clip_tensors"] > 0
    assert trainer.counters.counts["ema"] > 0
    assert trainer.ema is not None
    assert trainer.evals == 4 and len(lines(trainer.journal, "eval")) == 4
    assert len(trainer.journal.segments()) == 1  # one stage change, at the anneal
    assert trainer.skips == 0
    assert trainer.audit() == {"steps": 200, "tokens": 200 * TOKENS, "bytes": 200 * 4 * TOKENS}
    assert trainer.journal.last("end") is not None
    assert journal_row(trainer.journal.path, now=time.time())["phase"] == "ended"
