"""The batched dispatch against the loop it replaced: parity, gradients, and no syncs.

`moe.py` no longer loops over experts, so the anchor a reader can check the new path
against lives here: `_loop` is the deleted per-expert dispatch, verbatim in what it
computes (sort by expert, gather one group's rows, `index_add_` its write), and every
gate below compares the two.

Three claims, each one a number:

* **parity** — same routing in, the same write out: `allclose` at `1e-6` relative for the
  dense arm and the rank-`r` factors, per expert and for the whole stack's logits and
  loss, with every parameter's gradient alongside;
* **duplicates** — two identical rows routed to the same expert are two assignments, not
  one slot: `moe_rows_run` stays `n·topk` and the write is the sum, exactly as the loop
  accumulated it;
* **skew** — 80% of the rows on one expert is a SECOND pass, not a refusal: the write and
  every expert's gradient are still the loop's, and `moe_dispatch_passes` says 2;
* **one host read in eager, none in capture** — the eager pass count is
  `ceil(largest group / capacity)`, read once from the device histogram; a captured
  forward instead runs a fixed `MAX_DISPATCH_PASSES` loop with the same write mask
  (extra passes add exact zeros) and a device overflow flag the eager half checks.
  Both paths are held to the same write and the same gradients below.
"""

from __future__ import annotations

import builtins
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import MissingReport
from torch import nn

import dormouse.model.moe as moe_mod
from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec

D, F, E, K = 16, 12, 4, 2
GELU_F = 3 * F // 2

#: The one config at widths a CPU test can afford; every slot is kept.
SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "n_layers": 4,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 32,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _bank(**kwargs: object) -> MoEFFN:
    """A small bank with the 10M-parameter floor raised out of the way."""
    settings: dict[str, object] = {
        "n_experts": E,
        "topk": K,
        "n_shared": 1,
        "act": "moa",
        "d_ffn_gelu": GELU_F,
        "expert_tsct_rank": 0,
        "min_expert_params": 0,
    }
    settings.update(kwargs)
    return MoEFFN(D, F, **settings)  # type: ignore[arg-type]


# --- the loop this replaced -------------------------------------------------


def _loop(bank: MoEFFN, x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The deleted per-expert dispatch: expert-major, one `index_add_` per group."""
    n, k = index.shape
    order = torch.argsort(index.reshape(-1), stable=True)
    tokens = order // k
    gates = weight.reshape(-1)[order]
    sizes = bank.router.count(index).tolist()
    y = torch.zeros_like(x)
    pos = 0
    for expert, size in zip(bank.experts, sizes, strict=True):
        if size == 0:
            continue
        tok = tokens[pos : pos + size]
        out = expert(x.index_select(0, tok))
        y.index_add_(0, tok, out * gates[pos : pos + size, None])
        pos += size
    assert pos == n * k, "the loop executed every routed row"
    return y


def _loop_dispatch(
    self: MoEFFN,
    x: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    counts: torch.Tensor,
    counters: Counters | None = None,
) -> torch.Tensor:
    """`_loop` in `_dispatch`'s signature, so a whole model can be run through it."""
    y = _loop(self, x, index, weight)
    if counters is not None:
        counters.count("moe_rows_run", x.shape[0] * self.topk)
        counters.count("moe_experts_used", self.n_experts)
        counters.count("moe_bank_builds", self.n_experts)
        counters.count("moe_capacity", self.capacity(x.shape[0] * self.topk))
        counters.count("moe_pad_rows", self.n_experts * self.capacity(x.shape[0] * self.topk))
        counters.count("moe_dropped", 0)
    return y


def _worse(got: torch.Tensor, want: torch.Tensor) -> float:
    """The largest deviation relative to the write's own scale (not to its near-zeros)."""
    scale = float(want.detach().abs().max().clamp_min(1e-12))
    return float((got - want).detach().abs().max()) / scale


# --- parity ----------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 8])
def test_the_batched_dispatch_is_the_loop_within_fp32_epsilon(rank: int) -> None:
    """Same routing in, the same write out — dense arm and rank-`r` factors alike."""
    torch.manual_seed(0)
    bank = _bank(n_shared=0, expert_tsct_rank=rank)
    bank.eval()
    x = torch.randn(96, D)
    index, weight, counts = bank.router(x)
    got = bank._dispatch(x, index, weight, counts)
    want = _loop(bank, x, index, weight)
    assert _worse(got, want) < 1e-6, (got - want).abs().max()
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-7)


def test_the_whole_bank_write_is_the_loop_plus_the_shared_expert() -> None:
    """The forward, not just the dispatch: routed bank plus shared, same numbers."""
    torch.manual_seed(1)
    bank = _bank(qb=False)
    bank.eval()
    x = torch.randn(48, D)
    index, weight, _ = bank.router(x)
    want = _loop(bank, x, index, weight) + bank.shared(x)  # type: ignore[misc]
    assert _worse(bank(x), want) < 1e-6


def test_a_token_routed_twice_is_two_assignments() -> None:
    """Two identical rows go to the same experts: two slots, accumulated, not overwritten."""
    torch.manual_seed(2)
    bank = _bank(qb=False, n_shared=0)
    bank.eval()
    x = torch.randn(20, D)
    x[7] = x[3]
    index, weight, _ = bank.router(x)
    dupe = (index[3] == index[7]).all()
    assert bool(dupe), "the fixture must route the duplicate to the same experts"
    counter = Counters()
    got = bank(x, counter)
    assert counter.snapshot()["moe_rows_run"] == 20 * K
    assert _worse(got, _loop(bank, x, index, weight)) < 1e-6


@pytest.mark.parametrize("seed", [0, 1])
def test_every_experts_gradient_is_the_loops_gradient(seed: int) -> None:
    """Per-parameter gradients: the batched `bmm` must not redistribute an expert's rows."""
    torch.manual_seed(seed)
    batched = _bank(qb=False, expert_tsct_rank=8)
    loop = _bank(qb=False, expert_tsct_rank=8)
    loop.load_state_dict(batched.state_dict())
    batched.eval()
    loop.eval()
    x = torch.randn(64, D)
    target = torch.randn(64, D)
    index, weight, counts = batched.router(x)

    (batched._dispatch(x, index, weight, counts) - target).square().mean().backward()
    loop_index, loop_weight, _ = loop.router(x)
    assert torch.equal(index, loop_index), "same weights must route the same way"
    (_loop(loop, x, loop_index, loop_weight) - target).square().mean().backward()

    _same_gradients(batched, loop)


def _same_gradients(batched: MoEFFN, loop: MoEFFN) -> None:
    """Every routed expert's gradient is the loop's, to fp32 epsilon."""
    for other in loop.experts:
        for param in other.parameters():
            assert param.grad is not None
    worst = 0.0
    for mine, theirs in zip(batched.experts, loop.experts, strict=True):
        for name, param in mine.named_parameters():
            want = dict(theirs.named_parameters())[name].grad
            assert param.grad is not None and want is not None
            worst = max(worst, float((param.grad - want).abs().max()))
    assert worst < 1e-6, worst


# --- skew: the second pass -------------------------------------------------


def _skewed(n: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """80% of the rows on expert 0, a duplicate row among them, and one gate each."""
    x = torch.randn(n, D)
    x[7] = x[3]
    index = torch.zeros(n, 1, dtype=torch.long)
    index[80:] = torch.tensor([1, 2, 3, 0] * 5).reshape(-1, 1)
    weight = torch.full((n, 1), 0.5)
    return x, index, weight


def test_a_skewed_routing_is_two_passes_and_the_loops_numbers() -> None:
    """80% on one expert: a second pass places the spill, and the write is the loop's.

    The gate is bf16 epsilon because that is the card's stream; on this fp32 CPU the two
    paths agree far below it, which is the stronger claim and costs nothing to assert.
    """
    torch.manual_seed(5)
    bank = _bank(qb=False, n_shared=0, topk=1)
    bank.eval()
    x, index, weight = _skewed()
    counts = bank.router.count(index)
    counter = Counters()
    got = bank._dispatch(x, index, weight, counts, counter)
    snapshot = counter.snapshot()
    assert snapshot["moe_dispatch_passes"] == 2, snapshot
    assert snapshot["moe_rows_run"] == x.shape[0] * index.shape[1]
    assert snapshot["moe_dropped"] == 0, "a second pass exists so nothing is dropped"
    assert _worse(got, _loop(bank, x, index, weight)) < 1e-6


def test_every_gradient_under_skew_is_the_loops_gradient() -> None:
    """The spill's rows carry their gradient to the right expert, once each."""
    torch.manual_seed(6)
    batched = _bank(qb=False, n_shared=0, topk=1, expert_tsct_rank=8)
    loop = _bank(qb=False, n_shared=0, topk=1, expert_tsct_rank=8)
    loop.load_state_dict(batched.state_dict())
    batched.eval()
    loop.eval()
    x, index, weight = _skewed()
    counts = batched.router.count(index)
    target = torch.randn(x.shape[0], D)

    (batched._dispatch(x, index, weight, counts) - target).square().mean().backward()
    (_loop(loop, x, index, weight) - target).square().mean().backward()

    _same_gradients(batched, loop)


def test_a_forward_under_skew_reports_its_passes_and_checks() -> None:
    """The counter is on the log line and every enabled mechanism reported itself."""
    torch.manual_seed(7)
    bank = _bank(qb=False, n_shared=0, capacity_ratio=1.0, dispatch_sigma=0.0)
    bank.eval()
    with torch.no_grad():
        #: A router bias this large routes every slot to expert 0: `2·n` rows against a
        #: slot of `n` — the skew that used to be a refusal and is now a second pass.
        bank.router.bias[0] = 50.0
    counter = Counters()
    bank(torch.randn(100, D), counter)
    assert counter.snapshot()["moe_dispatch_passes"] == 2
    assert "moe_dispatch_passes=2" in counter.line(), counter.line()
    counter.check()


def test_the_stack_logits_and_loss_are_the_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: the same model, the same loss, once batched and once through the loop."""
    spec = replace(StackSpec.from_toml(CONFIGS / "dormouse_1b.toml"), **SMOKE)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))

    torch.manual_seed(7)
    batched = assemble(spec, device="cpu", smoke=True)
    batched.model.eval()
    logits = batched.model(ids[:, :-1])
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    grads = {
        name: param.grad.clone()
        for name, param in batched.model.named_parameters()
        if param.grad is not None
    }
    assert batched.bank is not None

    with monkeypatch.context() as patch:
        patch.setattr(MoEFFN, "_dispatch", _loop_dispatch)
        torch.manual_seed(7)
        looped = assemble(spec, device="cpu", smoke=True)
        looped.model.eval()
        loop_logits = looped.model(ids[:, :-1])
        loop_loss = nn.functional.cross_entropy(
            loop_logits.reshape(-1, loop_logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        loop_loss.backward()

    assert _worse(logits, loop_logits) < 1e-5
    assert abs(float(loss.detach() - loop_loss.detach())) / abs(float(loop_loss.detach())) < 1e-6
    scale = max(
        float(param.grad.detach().abs().max())
        for param in looped.model.parameters()
        if param.grad is not None
    )
    worst = max(
        float((grads[name] - param.grad.detach()).abs().max())
        for name, param in looped.model.named_parameters()
        if param.grad is not None and grads[name].abs().sum() > 0
    )
    assert worst / scale < 1e-5, worst / scale


# --- one host read ---------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a sync is a card-side fact")
def test_the_forward_reads_the_histogram_once() -> None:
    """One device->host read: the pass count. `warn` mode names every sync that happens."""
    torch.manual_seed(3)
    bank = _bank().to("cuda")
    x = torch.randn(256, D, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank(x).square().mean().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    syncs = [str(w.message) for w in caught if "synchronizing" in str(w.message)]
    assert len(syncs) == 1, syncs


# --- capture: fixed passes, masked, no host read ----------------------------


def test_fixed_passes_are_bit_identical_to_the_data_dependent_dispatch() -> None:
    """The captured loop's extra passes add exact zeros: same write, same gradients."""
    torch.manual_seed(11)
    eager = _bank(qb=False, n_shared=0, topk=1)
    fixed = _bank(qb=False, n_shared=0, topk=1)
    fixed.load_state_dict(eager.state_dict())
    eager.eval()
    fixed.eval()
    x, index, weight = _skewed()
    counts = eager.router.count(index)

    got = eager._dispatch(x, index, weight, counts)
    fixed.fixed_passes = MAX_DISPATCH_PASSES
    want = fixed._dispatch(x, index, weight, counts)
    assert torch.equal(got, want), "the masked extra passes must add exact zeros"
    assert not fixed.overflowed(), "the skew fixture fits inside the fixed slots"

    target = torch.randn_like(got)
    (got - target).square().mean().backward()
    (want - target).square().mean().backward()
    for (name, mine), (_, theirs) in zip(
        eager.named_parameters(), fixed.named_parameters(), strict=True
    ):
        if mine.grad is None or theirs.grad is None:
            assert mine.grad is None and theirs.grad is None, name
            continue
        assert torch.equal(mine.grad, theirs.grad), f"gradient diverged at {name}"


def test_fixed_passes_set_the_overflow_flag_on_device() -> None:
    """A routing past the fixed slots sets a device flag instead of syncing to raise."""
    torch.manual_seed(12)
    bank = _bank(
        qb=False, n_shared=0, topk=1, n_experts=16, capacity_ratio=0.25, dispatch_sigma=0.0
    )
    bank.fixed_passes = MAX_DISPATCH_PASSES
    x = torch.randn(100, D)
    weight = torch.ones(100, 1)

    collapsed = torch.zeros(100, 1, dtype=torch.long)
    bank._dispatch(x, collapsed, weight, bank.router.count(collapsed))
    assert bank.overflowed(), "all rows on one expert past four slots must be flagged"

    balanced = torch.arange(100).reshape(-1, 1) % 16
    bank._dispatch(x, balanced, weight, bank.router.count(balanced))
    assert not bank.overflowed(), "the flag is reset by the next dispatch"


# --- static grid: the fused default, its proof counter, and zero host reads --


def _no_tensor_int() -> Any:
    """A `builtins.int` that allows Python ints and refuses anything device-backed."""
    real = builtins.int

    def guarded(value: object, *args: object, **kwargs: object) -> int:
        if isinstance(value, torch.Tensor):
            raise AssertionError("a device value was read into the host")
        return real(value, *args, **kwargs)  # type: ignore[arg-type, call-arg]

    return guarded


def _no_host_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Python-side device read raise inside the patched context."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a device value was read into the host")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(builtins, "int", _no_tensor_int())


def test_the_fused_plan_is_a_fixed_grid_and_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fused=True` plans MAX_DISPATCH_PASSES slots without touching the histogram."""
    bank = _bank(qb=False, n_shared=0, capacity_ratio=1.0, dispatch_sigma=0.0)
    counts = torch.tensor([50, 50, 0, 0])  # a two-slot routing at capacity 30
    assert bank._pass_plan(counts, 30, 100, False) == (2, False)
    assert bank._pass_plan(counts, 30, 100, True) == (MAX_DISPATCH_PASSES, True)
    bank.fixed_passes = 3
    assert bank._pass_plan(counts, 30, 100, False) == (3, True)

    bank.fixed_passes = None
    with monkeypatch.context() as patch:
        _no_host_read(patch)
        assert bank._pass_plan(counts, 30, 100, True) == (MAX_DISPATCH_PASSES, True)


@pytest.mark.parametrize("n_experts,topk", [(1, 1), (4, 2)])
def test_static_grid_is_the_data_dependent_path_bit_for_bit(n_experts: int, topk: int) -> None:
    """Live-width shapes, duplicates and one expert: same write, same gradients, finite.

    The gate is bf16 epsilon (the card's stream); this fp32 CPU run is `torch.equal`,
    which is the stronger claim and costs nothing to assert.
    """
    torch.manual_seed(20 + n_experts)
    settings: dict[str, object] = {"n_experts": n_experts, "topk": topk, "expert_tsct_rank": 8}
    eager = _bank(qb=False, n_shared=0, **settings)  # type: ignore[arg-type]
    fixed = _bank(qb=False, n_shared=0, **settings)  # type: ignore[arg-type]
    fixed.load_state_dict(eager.state_dict())
    eager.eval()
    fixed.eval()
    x = torch.randn(96, D)
    x[5] = x[1]
    x[9] = x[1]
    x.requires_grad_(True)
    index, weight, counts = eager.router(x)

    got = eager._dispatch(x, index, weight, counts)
    fixed.fixed_passes = MAX_DISPATCH_PASSES
    fi, fw, fc = fixed.router(x)
    want = fixed._dispatch(x, fi, fw, fc)
    assert torch.allclose(got, want, rtol=0, atol=0.0039)
    assert torch.equal(got, want), "extra masked slots add exact zeros"
    assert bool(torch.isfinite(got).all())

    target = torch.randn_like(got)
    (got - target).square().mean().backward(retain_graph=True)
    (want - target).square().mean().backward()
    for (name, mine), (_, theirs) in zip(
        eager.named_parameters(), fixed.named_parameters(), strict=True
    ):
        if mine.grad is None or theirs.grad is None:
            assert mine.grad is None and theirs.grad is None, name
            continue
        assert bool(torch.isfinite(mine.grad).all()), name
        assert torch.allclose(mine.grad, theirs.grad, rtol=0, atol=0.0039), name
        assert torch.equal(mine.grad, theirs.grad), f"gradient diverged at {name}"


def test_five_static_forwards_read_nothing_from_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.item()`, `.tolist()` and `int(tensor)` all raise for five consecutive forwards."""
    torch.manual_seed(21)
    bank = _bank(qb=False, n_shared=0)
    bank.fixed_passes = MAX_DISPATCH_PASSES
    x = torch.randn(64, D, requires_grad=True)
    with monkeypatch.context() as patch:
        _no_host_read(patch)
        for _ in range(5):
            y = bank(x)
    assert bool(torch.isfinite(y).all())


def test_the_static_counter_proves_the_grid_ran() -> None:
    """A static forward counts its slots; `check()` is what reads the proof."""
    torch.manual_seed(22)
    bank = _bank(qb=False, n_shared=0)
    bank.fixed_passes = MAX_DISPATCH_PASSES
    counter = Counters()
    bank(torch.randn(48, D), counter)
    assert counter.snapshot()["moe_static_passes"] == MAX_DISPATCH_PASSES
    assert "moe_static" in counter.enabled
    assert "moe_static_passes" in counter.required.values()
    counter.check()


def test_a_static_flag_with_no_counter_is_a_missing_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enabling run must increment its proof, or the step gate fails loudly."""
    torch.manual_seed(23)
    bank = _bank(qb=False, n_shared=0)
    bank.fixed_passes = MAX_DISPATCH_PASSES
    counter = Counters()
    real_count = counter.count

    def drop_proof(key: str, n: int = 1) -> None:
        if key != "moe_static_passes":
            real_count(key, n)

    monkeypatch.setattr(counter, "count", drop_proof)
    bank(torch.randn(48, D), counter)
    assert "moe_static" in counter.enabled
    assert counter.snapshot().get("moe_static_passes", 0) == 0
    with pytest.raises(MissingReport, match="moe_static"):
        counter.check()


def test_the_bank_is_one_checkpoint_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapsed from passes x groups to one region for the whole bank call."""
    real = moe_mod.checkpoint
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(moe_mod, "checkpoint", counted)
    bank = _bank(qb=False, n_shared=0)
    x = torch.randn(64, D, requires_grad=True)
    bank(x).square().mean().backward()
    assert len(calls) == 1, len(calls)


def test_a_forward_backward_is_sync_debug_clean() -> None:
    """Static forward+backward raises nothing under `set_sync_debug_mode('error')`.

    The CPU wheel has no CUDA, so the host-read guard above is the CPU-side gate and
    the card probe below is the same claim on a device.
    """
    torch.manual_seed(24)
    bank = _bank(qb=False, n_shared=0)
    bank.fixed_passes = MAX_DISPATCH_PASSES
    x = torch.randn(64, D, requires_grad=True)

    def run() -> None:
        bank(x).square().mean().backward()

    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("error")
        try:
            run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    else:
        run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a free card")
def test_cuda_graph_capture_probe_on_a_free_card() -> None:
    """Ready-to-run card probe — do NOT run while another job holds the card.

        DORMOUSE_BANK=0 pytest -q tests/test_moe_dispatch.py -k cuda_graph_capture_probe

    The anchor is deliberate: the fused kernel's own backward gather
    (`kernels/bank.py:718`) still indexes with a device tensor and is a separate lane.
    This probe isolates the scheduling change owned by moe.py/ffn.py.
    """
    bank = _bank(qb=False, n_shared=0)
    bank.fixed_passes = MAX_DISPATCH_PASSES
    bank = bank.to("cuda")
    x = torch.randn(64, D, device="cuda", requires_grad=True)
    for _ in range(3):
        bank(x).square().mean().backward()
        x.grad = None
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.graph(graph):
            loss = bank(x).square().mean()
            loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    graph.replay()
    assert bool(torch.isfinite(loss).all())
    assert bank.overflowed() is False
