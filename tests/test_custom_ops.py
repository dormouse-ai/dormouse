"""The opaque custom-op boundary: one dynamo graph, no kernel breaks, replay parity.

The kernels are card-side, so CI holds the parts that do not need a card: the explicit
schemas and fake kernels (exercised on `meta` tensors), the single-graph compile of a
CPU-constructible step, and the parity of a captured replay against the eager step.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from dormouse_kernels.engine import bank as kb
from dormouse_kernels.engine import sparse

from dormouse.model.assemble import assemble
from dormouse.model.moe import MAX_DISPATCH_PASSES, MoEFFN
from dormouse.model.spec import StackSpec
from dormouse.train import compiled
from dormouse.train.graph import StepGraph
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import CHUNK_TOKENS, chunked_cross_entropy

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"
#: Inductor's CPU pointwise fusion moves a handful of logits by ~1.1e-5 absolute.
#: The relative check is fp32-tight; the absolute floor covers those fused near-zeros.
FP32_RTOL, FP32_ATOL = 1e-5, 5e-5


def _spec() -> StackSpec:
    """The CI smoke stack, at the smallest shape its arms accept."""
    raw = tomllib.loads(SMOKE.read_text())
    keys = {f.name for f in fields(StackSpec) if f.init}
    return StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(SMOKE))


def _bank_meta() -> tuple[torch.Tensor, ...]:
    """Twelve `meta` operands at a tiny shape, enough to exercise the fake kernel."""
    m = lambda *shape: torch.empty(*shape, device="meta")  # noqa: E731
    return (
        m(32, 16),
        torch.empty(33, dtype=torch.int32, device="meta"),
        torch.empty(4, dtype=torch.int32, device="meta"),
        m(4, 16, 256),
        m(4, 256, 64),
        m(4, 16, 256),
        m(4, 256, 64),
        m(4, 64, 256),
        m(4, 256, 16),
        m(4, 16, 2),
        m(4, 2),
        m(4, 2),
    )


def _sparse_meta() -> tuple[torch.Tensor, ...]:
    """The five sparse operands on `meta`, for the fake kernel."""
    q = torch.empty(2, 8, 4, 16, device="meta")
    at = torch.empty(2, 8, 4, dtype=torch.int64, device="meta")
    vis = torch.empty(2, 8, 4, dtype=torch.uint8, device="meta")
    return q, q, q, at, vis


def test_bank_and_sparse_ops_have_explicit_schemas() -> None:
    """Dynamo needs a stable op identity; both kernels declare one, with a fake kernel."""
    names = {str(schema.name) for schema in torch._C._jit_get_all_schemas()}
    assert "dormouse::bank" in names
    assert "dormouse::sparse_attention" in names
    assert "int[] kinds" in str(kb._bank_op._opoverload._schema)
    assert "float scale" in str(sparse._sparse_op._opoverload._schema)


def test_fake_kernels_declare_shapes_without_data() -> None:
    """The meta implementations carry every output the backward saves, with no card."""
    bank = kb._bank_op(*_bank_meta(), 8, (0,), "bi", True)
    assert [tuple(t.shape) for t in bank] == [
        (32, 16),
        (32, 256),
        (32, 256),
        (32, 256),
        (32, 16),
        (32, 16),
    ]
    assert bank[0].dtype is torch.bfloat16

    attn = sparse._sparse_op(*_sparse_meta(), 0.25)
    assert tuple(attn[0].shape) == (2, 8, 4, 16)
    assert tuple(attn[4].shape) == (2, 8, 4) and attn[4].dtype is torch.int32
    assert tuple(attn[7].shape) == (64, 2) and attn[7].dtype is torch.float32


@pytest.mark.parametrize(
    ("call", "operands"),
    [
        (lambda *xs: kb._bank_op(*xs, 8, (0,), "bi", True)[0].sum(), _bank_meta()),
        (lambda *xs: sparse._sparse_op(*xs, 0.25)[0].sum(), _sparse_meta()),
    ],
)
def test_a_custom_op_is_one_opaque_graph_node(
    call: object, operands: tuple[torch.Tensor, ...]
) -> None:
    """Dynamo traces the launch site as one node: one graph, zero breaks."""
    compiled.reset()
    try:
        torch.compile(call)(*operands)  # type: ignore[operator]
        found = compiled.proof()
        assert found["unique_graphs"] == 1, found
        assert not {
            k: v
            for k, v in found["graph_break"].items()
            if any(part in k.lower() for part in ("bank", "sparse", "attention", "triton"))
        }, found["graph_break"]
    finally:
        compiled.reset()


def test_a_compiled_cpu_step_keeps_one_graph_across_two_steps() -> None:
    """The mutable launch tally no longer guards the graph: two steps, one compilation."""
    torch.manual_seed(0)
    spec = _spec()
    built = assemble(spec, smoke=True)
    for bank in (m for m in built.model.modules() if isinstance(m, MoEFFN)):
        bank.fixed_passes = MAX_DISPATCH_PASSES
    optimizer = MixOptimizer(built.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    compiled.reset()
    try:
        compiled.checked(built.model)
        for _ in range(2):
            built.step(ids, optimizer=optimizer)
        assert built.counters.snapshot()[compiled.COUNTER] == 1, "a second graph is a recompile"
        assert compiled.unique_graphs() == 1, compiled.proof()
        assert not {
            k: v
            for k, v in compiled.breaks().items()
            if any(p in k.lower() for p in ("bank", "sparse", "attention", "triton"))
        }, compiled.breaks()
    finally:
        compiled.reset()


def test_capture_replays_the_compiled_step_with_fp32_parity() -> None:
    """On a CPU-constructible shape, replay matches eager loss, logits and every grad."""
    spec = _spec()
    torch.manual_seed(11)
    built = assemble(spec, smoke=True)
    initial = copy.deepcopy(built.model.state_dict())
    optimizer = MixOptimizer(built.model)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    compiled.reset()
    try:
        graph = StepGraph(built.model, optimizer, ids)
        assert graph.proof()["unique_graphs"] == 1, graph.proof()
        assert graph.graph_breaks() == {}, graph.graph_breaks()
        x, y = ids[:, :-1], ids[:, 1:]
        # Warm-up and capture advanced QB's state; put both arms back at the same point.
        # One forward per arm, from that same state, is the logits comparison.
        built.model.load_state_dict(initial)
        with torch.no_grad():
            got_logits = built.model(x, None)

        built.model.load_state_dict(initial)
        graph.replay(ids)
        assert graph.replays() == 1
        got_loss = float(graph._loss)
        got_grads = {
            n: p.grad.detach().clone() if p.grad is not None else None
            for n, p in built.model.named_parameters()
        }

        torch.manual_seed(11)
        eager = assemble(spec, smoke=True).model
        eager.load_state_dict(initial)
        eager_opt = MixOptimizer(eager)
        eager_opt.zero_grad(set_to_none=False)
        want_logits = eager(x, None)
        want_loss = chunked_cross_entropy(want_logits, y, chunk=CHUNK_TOKENS)
        want_loss.backward()  # type: ignore[no-untyped-call]

        assert got_logits.shape == want_logits.shape
        assert torch.allclose(got_logits, want_logits, rtol=FP32_RTOL, atol=FP32_ATOL)
        assert got_loss == pytest.approx(float(want_loss.detach()), rel=FP32_RTOL, abs=FP32_ATOL)
        for name, grad in got_grads.items():
            other = dict(eager.named_parameters())[name].grad
            if other is None or grad is None:
                assert other is grad, name
                continue
            assert torch.allclose(grad, other, rtol=FP32_RTOL, atol=FP32_ATOL), name
    finally:
        compiled.reset()
