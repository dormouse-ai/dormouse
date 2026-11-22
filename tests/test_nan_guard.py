"""NaN-firewall guard: finite band, cold start, and a clean 100-step random-id run."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from dormouse_core.journal import Journal
from stacks import reference_stack

from dormouse.train.guards import adaptive_clip, is_spike
from dormouse.train.loop import Trainer
from dormouse.train.optim import MixOptimizer
from dormouse.train.recipe import Batch, LoopConfig, Stage, annealed_schedule

SMALL = {"vocab": 16, "d_model": 16, "n_layers": 1, "n_heads": 2, "head_dim": 8, "d_ffn": 32}


def test_a_nonfinite_loss_is_a_spike_before_the_window_is_full() -> None:
    assert is_spike(float("nan"), [], k=6.0, size=50)
    assert is_spike(float("inf"), [3.0], k=6.0, size=50)


def test_a_nonfinite_window_cannot_blind_the_gate() -> None:
    assert is_spike(3.0, [float("nan")] * 50, k=6.0, size=50)


def test_a_finite_loss_keeps_the_cold_start_and_the_floor_band() -> None:
    assert not is_spike(30.0, [3.0] * 49, k=6.0, size=50)
    assert not is_spike(3.3, [3.0] * 50, k=6.0, size=50)
    assert is_spike(4.0, [3.0] * 50, k=6.0, size=50)


def test_a_nonfinite_or_zero_clip_reference_reseeds() -> None:
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    running = {name: float("nan") for name, _ in model.named_parameters()}
    examined, scaled = adaptive_clip(model, running, 2.5)
    assert examined == len(running)
    assert scaled == 0
    assert all(math.isfinite(value) and value > 0 for value in running.values())


class _RandomStream:
    bytes_per_token = 4.0

    def __init__(self, stages: tuple[tuple[int, str], ...]) -> None:
        self.stages = stages

    def batch(self, step: int, tokens: int) -> Batch:
        ids = torch.randint(0, 16, (2, 17), generator=torch.Generator().manual_seed(1000 + step))
        stage = [name for start, name in self.stages if step >= start][-1]
        return Batch(ids, 32, 128, stage)


def test_a_hundred_step_random_id_run_has_no_false_skips(tmp_path: Path) -> None:
    steps = 100
    anneal = steps - round(0.22 * steps)
    cfg = LoopConfig(
        steps=steps,
        stages=(Stage("bulk", 0, anneal), Stage("clean", anneal, steps)),
        warmup_steps=10,
        tokens_per_step=32,
        spike_window=50,
    )
    stages = tuple((stage.start, stage.name) for stage in cfg.stages)
    torch.manual_seed(23)
    model = reference_stack(**SMALL)
    trainer = Trainer(
        model,
        _RandomStream(stages),
        MixOptimizer(model),
        annealed_schedule(cfg),
        cfg,
        Journal.open(tmp_path / "run.jsonl"),
    )
    trainer.train(steps=steps)
    assert trainer.skips == 0
    assert trainer.spike_skips == 0
    assert trainer.consecutive == 0
