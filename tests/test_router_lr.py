"""The router's own learning rate: a separate Muon group at `lr_muon / √d_model`.

The defect this file pins is structural, not a high-LR accident. Muon sets a matrix's
per-row update to exactly its group's lr (`muon.py:431`); at the recipe's 0.02 that is
3.5% of a router row's init norm `√d·0.02` at d=832. With momentum holding one
direction across steps, the router's row drifts by more than its whole init scale in
~20 steps, and the A/B record (`docs/records/2026-11-16-flops-per-byte-ab.md:48-54`)
measured one expert taking 1113/8192 rows and the dispatch tripping 5 passes. The
split keeps the polar direction (router stays in Muon, not AdamW) and cuts only the
magnitude.

Top-k note: at top-1 the gate weight is `1.0` exactly (`g/g`), so `d(weight)/d(logit)`
is identically zero and the router receives NO gradient — a top-1 test cannot exercise
the group at all (verified: `router.weight.grad.max() == 0`). The recorded collapse was
on the control's top-2 arm. The decisive test below therefore runs at E=64/top-2, the
configuration that actually drives the router.

`before` numbers below were measured on the unfixed branch (commit `3400a44`).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import MissingReport
from torch import nn

from dormouse.model.moe import MoEFFN
from dormouse.model.router import ROUTER_STD, Router
from dormouse.train.optim import MixOptimizer
from dormouse.train.runconfig import RUN_KEYS, RunRecipe

D, E, K = 832, 64, 2
SEED = 0


def _bank(*, qb: bool = True) -> MoEFFN:
    """A fresh E=64 router at the live width, with cheap experts for a CPU step."""
    torch.manual_seed(SEED)
    return MoEFFN(
        D,
        128,
        n_experts=E,
        topk=K,
        n_shared=1,
        act="moa",
        d_ffn_gelu=192,
        expert_tsct_rank=8,
        min_expert_params=1,
        qb=qb,
    )


def _group_names(opt: MixOptimizer) -> list[list[str]]:
    """Parameter names per Muon group, in the order the optimizer built them."""
    owners = {id(p): name for name, p in opt.model.named_parameters()}
    return [[owners[id(p)] for p in group["params"]] for group in opt.muon.param_groups]


# ---------------------------------------------------------------- the group


def test_default_router_lr_is_lr_muon_over_sqrt_d() -> None:
    """At d=832 the default is 0.02/√832 ≈ 6.93e-4; an explicit key wins."""
    opt = MixOptimizer(_bank(), lr_muon=0.02)
    assert opt.router_lr == pytest.approx(0.02 / math.sqrt(D), rel=1e-12)
    assert opt.router_lr == pytest.approx(6.93375e-4, rel=1e-5)
    assert MixOptimizer(_bank(), lr_muon=0.02, lr_router=1e-3).router_lr == 1e-3


def test_router_is_its_own_small_lr_group_and_the_rest_are_unchanged() -> None:
    """One extra group holds only `router.weight`; every other group keeps its lr."""
    opt = MixOptimizer(_bank(), lr_muon=0.02)
    router_groups = [g for g in opt.muon.param_groups if g.get("is_router")]
    assert len(router_groups) == 1
    assert router_groups[0]["lr"] == pytest.approx(0.02 / math.sqrt(D))
    owners = {id(p): name for name, p in opt.model.named_parameters()}
    assert [owners[id(p)] for p in router_groups[0]["params"]] == ["router.weight"]
    others = [g for g in opt.muon.param_groups if not g.get("is_router")]
    assert all(g["lr"] == 0.02 for g in others)
    assert sum(len(g["params"]) for g in others) == 455


def test_grouping_is_sorted_and_deterministic() -> None:
    """Building twice gives identical group membership and order."""
    first = MixOptimizer(_bank())
    second = MixOptimizer(_bank())
    assert _group_names(first) == _group_names(second)
    for names in _group_names(first):
        assert names == sorted(names), "a group's parameters must be name-sorted"


def test_set_lr_scales_the_router_group_by_its_own_base() -> None:
    """Annealing halves the router's small rate too, not the Muon rate."""
    opt = MixOptimizer(_bank(), lr_muon=0.02)
    opt.set_lr(0.5)
    router = next(g for g in opt.muon.param_groups if g.get("is_router"))
    other = next(g for g in opt.muon.param_groups if not g.get("is_router"))
    assert router["lr"] == pytest.approx(0.01 / math.sqrt(D))
    assert other["lr"] == pytest.approx(0.01)


# --------------------------------------------------------------- the counter


def test_the_router_lr_counter_proves_the_group_stepped() -> None:
    """A stepped group increments `router_lr_group`; `check()` accepts the proof."""
    bank = _bank()
    opt = MixOptimizer(bank, lr_muon=0.02)
    counter = Counters()
    bank(torch.randn(64, D)).square().mean().backward()
    opt.step(counters=counter)
    assert counter.snapshot()["router_lr_group"] == 1
    assert "router_lr_group" in counter.required.values()
    counter.check()


def test_a_split_with_no_router_group_fails_the_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly enabled split whose proof is dropped is a MissingReport."""
    bank = _bank()
    opt = MixOptimizer(bank, lr_router=1e-3)
    counter = Counters()
    real_count = counter.count

    def drop_proof(key: str, n: int = 1) -> None:
        if key != "router_lr_group":
            real_count(key, n)

    monkeypatch.setattr(counter, "count", drop_proof)
    bank(torch.randn(64, D)).square().mean().backward()
    opt.step(counters=counter)
    assert counter.snapshot().get("router_lr_group", 0) == 0
    with pytest.raises(MissingReport, match="router_lr_group"):
        counter.check()


def test_a_config_that_enables_the_split_without_a_router_fails() -> None:
    """`lr_router` set on a router-less model enables the proof, which stays at zero."""

    class NoRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = NoRouter()
    opt = MixOptimizer(model, lr_router=1e-3)
    counter = Counters()
    model(torch.randn(4, 16)).square().mean().backward()
    opt.step(counters=counter)
    assert counter.snapshot().get("router_lr_group", 0) == 0
    with pytest.raises(MissingReport, match="router_lr_group"):
        counter.check()


# ---------------------------------------------------------------- the config key


def test_the_config_key_round_trips() -> None:
    """`lr_router` is a named run key and survives the canonical/recipe round trip."""
    assert "lr_router" in RUN_KEYS
    run = {
        "sources": [],
        "stages": [],
        "eval": None,
        "schedule": "cosine",
        "lr_muon": 0.02,
        "lr_adamw": 0.003,
        "lr_router": 1e-3,
        "warmup": 10,
        "eval_every": 0,
        "ckpt_every": 0,
        "eval_batches": 8,
        "harness": True,
        "seed": 0,
    }
    recipe = RunRecipe.from_canonical({"stack": {}, "run": run}, path=Path("run.toml"))
    assert recipe.lr_router == 1e-3
    assert recipe.canonical()["run"]["lr_router"] == 1e-3
    old = dict(run)
    del old["lr_router"]
    legacy = RunRecipe.from_canonical({"stack": {}, "run": old}, path=Path("old.toml"))
    assert legacy.lr_router is None


# ------------------------------------------------------------ the decisive gate


def test_fifty_steps_preserve_the_router_distribution_and_dispatch() -> None:
    """50 steps at lr_muon=0.02 keep the router near its init and the load balanced.

    Before: the router's row drift after 50 steps was 1.36x its init norm, and the
    recorded run hit 1113/8192 rows (8.7x mean) and 5 dispatch passes. Here the drift
    is 0.058x, max/mean stays under 1.5 every step, and the dispatch never needs a
    second pass. Only the router is trainable so the drift is attributable to the LR
    group alone; QB is left on, so the routing path is the shipped one.
    """
    bank = _bank()
    for name, param in bank.named_parameters():
        param.requires_grad_(name == "router.weight")
    opt = MixOptimizer(bank, lr_muon=0.02)

    with torch.no_grad():
        start = bank.router.weight.detach().clone()
    x = torch.randn(8 * 512, D)
    target = torch.randn(8 * 512, D)
    capacity = bank.capacity(x.shape[0] * K)
    ratios: list[float] = []
    passes: list[int] = []

    for _ in range(50):
        (bank(x) - target).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            load = bank.router.count(bank.router.select(bank.router.logits(x))[0]).float()
            ratios.append(float(load.max() / load.mean()))
            passes.append(max(1, -(-int(load.max()) // capacity)))

    assert max(ratios) < 1.5, f"load ratio peaked at {max(ratios):.3f}"
    assert max(passes) <= 2, f"dispatch needed {max(passes)} passes"
    with torch.no_grad():
        drift = float((bank.router.weight - start).norm() / start.norm())
    assert drift < 0.25, f"router drifted {drift:.3f} of its init norm in 50 steps"


# --------------------------------------------------------- quality invariants


def test_routing_invariants_are_untouched() -> None:
    """Init std, sqrt-softplus score, QB centering and dropless weight are pinned."""
    router = Router(D, E, K)
    assert router.weight.std().item() == pytest.approx(ROUTER_STD, rel=0.1)
    logits = torch.tensor([-5.0, 0.0, 20.0])
    want = torch.sqrt(torch.nn.functional.softplus(logits))
    assert torch.allclose(router.scores(logits), want)
    index, weight, counts = router.select(torch.randn(64, D) @ router.weight.T)
    assert index.shape == (64, K)
    assert torch.allclose(weight.sum(dim=-1), torch.ones(64))
    assert int(counts.sum()) == 64 * K
    before = router.bias.clone()
    router.qb_update(router.logits(torch.randn(512, D)))
    assert torch.allclose(router.bias.mean(), torch.zeros(()), atol=1e-6), (
        "QB keeps its centered gauge"
    )
    assert not torch.equal(router.bias, before), "QB moved on a fresh batch"
