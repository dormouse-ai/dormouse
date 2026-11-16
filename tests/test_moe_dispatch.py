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
* **one host read** — the pass count is `ceil(largest group / capacity)`, read once from
  the histogram on the card (a Python loop cannot be sized by a device value). The
  forward is therefore not CUDA-graph capturable any more, and the test that used to
  assert it says so instead of asserting a property the dispatch no longer has.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from torch import nn

from dormouse.model.assemble import assemble
from dormouse.model.moe import MoEFFN
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
