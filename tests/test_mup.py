"""μP: the classification gate, the bit-identical SP path, and the transfer measurement.

The last test is the one that makes the parametrisation real. It trains the assembled
stack at two widths, sweeps the base learning rate on a log grid at each, and estimates
each optimum with the parabola `2608.20061` eq. (2) fits to loss against log η:

    L(η) = a·(log η)² + b·log η + c        η* = exp(-b / 2a)

μP is correct if the two μP optima are within 1.5× of each other and the two SP optima
are not. The numbers this repo measures are printed by the test; they are a proxy at
d=8/32 on CPU, not a claim about 1B parameters.
"""

from __future__ import annotations

import math

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import Assembled, assemble
from dormouse.model.spec import StackSpec
from dormouse.train.mup import KINDS, Parametrisation
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import train_step

SMALL = {
    "unit": "byte",
    "attn": "hybrid",
    "d_model": 16,
    "n_layers": 4,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 8,
    "c_lat": 8,
    "d_ffn": 64,
    "n_experts": 2,
    "topk": 1,
    "n_shared": 0,
    "seq_len": 64,
    "batch": 2,
    "attn_k": 8,
    "attn_window": 4,
}


def small(**overrides: object) -> StackSpec:
    """A real assembled stack small enough to build in a CPU test."""
    return StackSpec(**{**SMALL, **overrides})  # type: ignore[arg-type]


def mup_stack(**overrides: object) -> Assembled:
    """A μP stack at `mup_base_width=8`, the proxy this file's widths scale from."""
    options: dict[str, object] = {
        "parametrisation": "mup",
        "mup_base_width": 8,
        "weight_decay": 0.0,
    }
    options.update(overrides)
    return assemble(small(**options), smoke=True)


# --------------------------------------------------------------------- classification


def test_every_parameter_of_the_hybrid_stack_is_classified() -> None:
    """The gate the next lane is caught by: no parameter keeps the SP default silently."""
    built = mup_stack()
    model = built.model
    parametrisation = built.parametrisation
    assert parametrisation is not None
    table = parametrisation.table(model)
    assert set(table) == {name for name, _ in model.named_parameters()}
    assert {rule.kind for rule in table.values()} <= set(KINDS)
    counts: dict[str, int] = dict.fromkeys(KINDS, 0)
    for rule in table.values():
        counts[rule.kind] += 1
    assert counts["embedding"] == 1
    assert counts["router"] == 1
    assert counts["hidden"] > 0
    #: The head is tied to `embed.weight`, so it owns no parameter; its μP rule is the
    #: forward scale, which is what "shares, not doubled" means below.
    assert counts["head"] == 0
    assert model.head_scale == pytest.approx(8 / 16)


def test_a_module_without_a_classification_fails_the_suite() -> None:
    """An owner the table has never seen is a refusal, not a hidden default."""

    class Engram(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(4, 4))

    built = mup_stack()
    built.model.engram = Engram()
    assert built.parametrisation is not None
    with pytest.raises(LoudError, match="no rule"):
        built.parametrisation.table(built.model)


def test_the_fan_in_scaling_only_touches_width_scaled_matrices() -> None:
    """`c_lat` and `d_ffn` are fixed by the architecture; a fan-in over them must not move."""
    built = mup_stack()
    parametrisation = built.parametrisation
    assert parametrisation is not None
    table = parametrisation.table(built.model)
    ratio = math.sqrt(16 / 8)
    assert table["blocks.0.attn.q_lat.weight"].init_std == pytest.approx(0.02 / ratio)
    assert table["blocks.0.attn.q_up.weight"].init_std == pytest.approx(0.02)
    assert table["blocks.0.attn.kv_up.weight"].init_std == pytest.approx(0.02)
    assert table["blocks.0.ffn.bank.router.weight"].init_std == pytest.approx(0.02 / ratio)
    #: `1/√(2L)` on the residual write, the same number `blocks.Block` draws.
    write = 0.02 / math.sqrt(2.0 * 4)
    assert table["blocks.0.attn.proj.weight"].init_std == pytest.approx(write / ratio)
    assert table["blocks.0.ffn.bank.experts.0.down.weight"].init_std == pytest.approx(write)
    #: A norm gain is a constant, not a draw; the rule says so with 0.
    assert table["blocks.0.norm1.weight"].init_std == 0.0


# --------------------------------------------------------------------- the SP path


def test_sp_is_the_unscaled_path_bit_for_bit() -> None:
    """`parametrisation="sp"` must leave the head, the init and the optimizer where they were."""
    built = assemble(small(), smoke=True)
    assert built.parametrisation is None
    assert built.model.head_scale == 1.0
    assert getattr(built.model, "parametrisation", None) is None
    x = torch.randn(2, 3, 16)
    assert torch.equal(built.model.head(x), torch.nn.functional.linear(x, built.model.embed.weight))
    optimizer = MixOptimizer(built.model, lr_muon=0.02, lr_adamw=0.003)
    assert {group["lr"] for group in optimizer.muon.param_groups} == {0.02}
    assert {group["lr"] for group in optimizer.adamw.param_groups} == {0.003}
    assert {group["weight_decay"] for group in optimizer.muon.param_groups} == {0.0}
    assert {group["weight_decay"] for group in optimizer.adamw.param_groups} == {0.1}


def test_mup_at_the_base_width_is_the_sp_model() -> None:
    """At `d_model == mup_base_width` every factor is one, so the two builds are identical."""
    torch.manual_seed(3)
    sp = assemble(small(d_model=8, n_heads=1), smoke=True).model
    torch.manual_seed(3)
    mup = mup_stack(d_model=8, n_heads=1, mup_base_width=8).model
    assert mup.head_scale == 1.0
    for (name, left), (other, right) in zip(
        sp.named_parameters(), mup.named_parameters(), strict=True
    ):
        assert name == other
        assert torch.equal(left, right), name
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (2, 64))
    assert torch.equal(sp(ids), mup(ids))


# ------------------------------------------------------------- the embedding/decay knobs


def test_the_embedding_rate_is_its_own_knob_and_the_tied_head_shares_it() -> None:
    """`2605.21486` §5: one parameter, one rate — not the table twice."""
    built = mup_stack(embedding_lr=2.0, d_model=16, weight_decay=0.1)
    parametrisation = built.parametrisation
    assert parametrisation is not None
    optimizer = MixOptimizer(built.model, lr_muon=0.02, lr_adamw=0.003)
    embedding = dict(built.model.named_parameters())["embed.weight"]
    groups = [
        group
        for group in optimizer.adamw.param_groups
        if any(param is embedding for param in group["params"])
    ]
    assert len(groups) == 1
    group = groups[0]
    assert group["lr"] == pytest.approx(0.006)
    assert len(group["params"]) == 1
    #: `η_l · λ_l = η · λ` (2605.21486 eq. 17.9): doubling the rate halves the coefficient.
    assert group["weight_decay"] == pytest.approx(0.05)
    assert group["lr"] * group["weight_decay"] == pytest.approx(0.003 * 0.1)


def test_the_decay_coefficient_keeps_the_papers_product() -> None:
    """The μP `λ` is the coefficient at the base rate, not a per-group rate."""
    built = mup_stack(embedding_lr=4.0, weight_decay=0.1)
    optimizer = MixOptimizer(built.model, lr_muon=0.02, lr_adamw=0.003)
    for group in optimizer.muon.param_groups:
        assert group["lr"] * group["weight_decay"] == pytest.approx(0.02 * 0.1)
    for group in optimizer.adamw.param_groups:
        assert group["lr"] * group["weight_decay"] == pytest.approx(0.003 * 0.1)
    optimizer.set_lr(0.5)
    for group in optimizer.adamw.param_groups:
        assert group["lr"] * group["weight_decay"] == pytest.approx(0.5 * 0.003 * 0.1)


def test_mup_refuses_the_schedule_free_pair() -> None:
    built = mup_stack()
    with pytest.raises(LoudError, match="schedule-free"):
        MixOptimizer(built.model, schedule="sf")


def test_a_bad_embedding_multiplier_is_refused() -> None:
    with pytest.raises(LoudError, match="not a rate"):
        Parametrisation(
            width=8,
            base_width=8,
            vocab=256,
            n_layers=2,
            n_heads=1,
            head_dim=8,
            n_kv_heads=1,
            embedding_lr=0.0,
        )


# ------------------------------------------------------------------------ the transfer


def _make_data(rows: int, seq: int, seed: int = 7) -> torch.Tensor:
    """A noisy deterministic bigram: learnable, but not a lookup a table alone can fit."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.zeros(rows, seq + 1, dtype=torch.long)
    ids[:, 0] = torch.randint(0, 256, (rows,), generator=generator)
    for t in range(1, seq + 1):
        nxt = (ids[:, t - 1] * 7 + 3) % 256
        noise = torch.randint(0, 256, (rows,), generator=generator)
        mask = torch.rand(rows, generator=generator) < 0.05
        ids[:, t] = torch.where(mask, noise, nxt)
    return ids


def _spec(d_model: int, mode: str) -> StackSpec:
    heads = max(1, d_model // 8)
    return small(
        attn="kda",
        n_layers=2,
        d_model=d_model,
        n_heads=heads,
        n_kv_heads=max(1, heads // 2),
        head_dim=8,
        c_lat=8,
        d_ffn=64,
        batch=4,
        parametrisation=mode,
        mup_base_width=8,
        weight_decay=0.0,
    )


def _loss(d_model: int, mode: str, eta: float, data: torch.Tensor) -> float:
    """Train one model at `eta` and return the held-out cross-entropy."""
    torch.manual_seed(0)
    model = assemble(_spec(d_model, mode), smoke=True).model
    #: Weight decay is held at zero on BOTH parametrisations so the sweep measures the
    #: rate, not `λ`. `test_the_decay_coefficient_keeps_the_papers_product` is the test
    #: for the decay term; putting it in this sweep would compare two changes at once.
    optimizer = MixOptimizer(
        model, lr_muon=eta, lr_adamw=eta, weight_decay=0.0, muon_weight_decay=0.0
    )
    train, held = data[:4], data[4:]
    model.train()
    for _ in range(40):
        train_step(model, train, optimizer=optimizer)
    model.eval()
    with torch.no_grad():
        logits = model(held[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), held[:, 1:].reshape(-1))
    return float(loss) if torch.isfinite(loss) else math.inf


def _vertex(grid: tuple[float, ...], losses: list[float]) -> float:
    """The parabola's minimum through the three points around the best one, in log η."""
    best = min(range(len(losses)), key=lambda i: losses[i])
    assert 0 < best < len(losses) - 1, f"the optimum is on the grid's edge: {losses}"
    (x1, y1), (x2, y2), (x3, y3) = (
        (math.log(grid[i]), losses[i]) for i in (best - 1, best, best + 1)
    )
    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denominator
    assert a > 0, f"the three points do not open upward: {losses}"
    return math.exp(-b / (2 * a))


def test_the_optimal_rate_transfers_under_mup_and_not_under_sp() -> None:
    """The gate: four optima, one ratio per parametrisation, no tuning of the threshold."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = _make_data(8, 64)
        grid = (0.015, 0.020, 0.027, 0.034, 0.042, 0.052, 0.068)
        optima: dict[tuple[str, int], float] = {}
        best_loss: dict[tuple[str, int], float] = {}
        for mode in ("sp", "mup"):
            for width in (8, 32):
                losses = [_loss(width, mode, eta, data) for eta in grid]
                optima[(mode, width)] = _vertex(grid, losses)
                best_loss[(mode, width)] = min(losses)
                print(
                    f"{mode:3s} d={width:2d} lr*={optima[(mode, width)]:.4f} "
                    f"loss={best_loss[(mode, width)]:.4f}  {[f'{x:.4f}' for x in losses]}"
                )
        sp_ratio = optima[("sp", 8)] / optima[("sp", 32)]
        mup_ratio = optima[("mup", 8)] / optima[("mup", 32)]
        print(f"sp  ratio (d=8 / d=32) = {sp_ratio:.3f}")
        print(f"mup ratio (d=8 / d=32) = {mup_ratio:.3f}")
        assert mup_ratio <= 1.5, f"muP optima moved by {mup_ratio:.3f}x across width"
        assert sp_ratio > 1.5, f"SP optima agreed to {sp_ratio:.3f}x; the gate is blind"
    finally:
        torch.set_num_threads(threads)
