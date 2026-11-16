"""The fused bank against the anchor: parity, the live shapes, and what it refuses.

The anchor is the PyTorch arm the module shipped with (`MoEFFN` with `DORMOUSE_BANK=0`),
and every gate here is a way the kernel could be a lie:

* **parity** — same seed and ids, fp32 at the order-of-summation bound derived in
  `kernels/bank.py`, bf16 at the step's own rounding. Forward AND every parameter
  gradient, because a kernel whose forward matches and whose backward does not is the
  version that trains a different model;
* **duplicate rows** — two assignments of one token into the same expert, which is the
  case a slot map can silently overwrite;
* **placement, not arithmetic** — one expert at top-1 at three capacities is the same
  bits, `torch.equal`. The anchor cannot be `torch.equal` here: it is a different GEMM
  implementation, and this file says so rather than widening a tolerance to hide it;
* **launches** — two per expert group forward, four with the backward, read off the
  counter the kernel itself increments, so the anchor cannot run under the kernel's name;
* **refusals** — a rank that is not a power of two, a dictionary the epilogue does not
  know, and the dense `rank=0` control arm each name their escape instead of running.

The CPU job skips the card gates visibly and still runs the arithmetic ones.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.kernels import bank as kb
from dormouse.model import moe
from dormouse.model.moe import MoEFFN

#: The desk's shape, wide enough that every `tl.dot` is a real tile and small enough that
#: the whole file runs in a second on the card.
D, F, R, E, TK = 64, 128, 32, 3, 2

REQUIRES_CARD = torch.cuda.is_available()
NO_CARD = "the fused bank needs a CUDA device; the anchor is what CI runs"
ON_CARD = pytest.mark.skipif(not REQUIRES_CARD, reason=NO_CARD)

#: The fp32 gate, as a fraction of the output's own largest entry. Three chained
#: `r`-term matmul reductions, each with `√r·u`-order rounding at `u = 2⁻²⁴`, and a
#: six-entry mixture on top: `3·5.7·6e-8 ≈ 1e-6` typical, `5e-6` at the padded-tree
#: worst case. 2e-5 is that bound with the margin a second order of summation needs.
FP32_PARITY = 2e-5

#: Four units in the last place of bf16, relative. `kernels/fused.py`'s two-ulp band is
#: for a single elementwise chain; a bank gradient crosses four bf16 quantisations on its
#: longest path (`y ← q·v_d`, `q ← h·u_d`, `h ← mixture`, `z ← xr·v`), each of which
#: rounds its operand to `2⁻⁸` before the MMA accumulates it, so the band is 4·2⁻⁸ =
#: 1.56% — derived, not fitted. The measured worst case on this card is 1.8 ulp.
BF16_PARITY = 4.0 * 2.0**-8

#: The live f=6144 strip walks 48 f-tiles, against the desk shape's 4, so a gradient
#: accumulates about sqrt(48/4)=3.5x more bf16 rounding on its longest path. Measured
#: worst over ten seeds at d=832/f=6144/r=256 is 10.9 ulp (the prior's small denominator
#: amplifies it); 16 ulp is that with margin. Forward and the weight gradients stay inside
#: this band; the desk band above is the tighter one and is kept for the desk test.
LIVE_BF16_PARITY = 16.0 * 2.0**-8

#: The card's per-block shared-memory ceiling (101376 B = 99 KiB), the exact number in
#: Triton's `OutOfResources` when a tile is too wide. A kernel over this cannot launch.
SMEM_LIMIT = 101_376

#: The live config's widths (`configs/dormouse_1b.toml`). Shared memory is a function of
#: the tile constants and these three widths, not of the group count or row count, so the
#: kernel that refuses at b8 is compiled identically with a small expert count here.
LIVE_D, LIVE_F, LIVE_R = 832, 6144, 256


def _bank(**kwargs: object) -> MoEFFN:
    """A small bank at the rank the fused chain covers, with the parameter floor off."""
    settings: dict[str, object] = {
        "n_experts": E,
        "topk": TK,
        "n_shared": 1,
        "act": "moa",
        "d_ffn_gelu": 3 * F // 2,
        "expert_tsct_rank": R,
        "min_expert_params": 0,
        "device": "cuda" if REQUIRES_CARD else "cpu",
    }
    settings.update(kwargs)
    bank = MoEFFN(D, F, **settings)  # type: ignore[arg-type]
    #: `eval()` because the queue-balancer updates the router bias on every training
    #: forward: two runs of the same weights would route the same rows to different
    #: experts and the parity gate would be comparing the router, not the kernel.
    bank.eval()
    return bank


def _inputs(bank: MoEFFN, rows: int = 24, seed: int = 0) -> torch.Tensor:
    """A seeded residual stream on the bank's device."""
    generator = torch.Generator(device=bank.router.weight.device).manual_seed(seed)
    return torch.randn(rows, D, generator=generator, device=bank.router.weight.device)


def _grads(bank: MoEFFN, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward/backward and return `(output, {name: grad})`.

    The loss is `(y·proj).sum()` with a fixed projection, not `y.square().mean()`: a
    squared loss on a small residual stream puts every gradient at ~1e-8, where a relative
    comparison is a comparison of cancellation noise — a gate on the loss's scale rather
    than on the kernel.
    """
    bank.zero_grad(set_to_none=True)
    y = bank(x)
    proj = torch.randn(
        y.shape, generator=torch.Generator(device=y.device).manual_seed(7), device=y.device
    )
    (y.float() * proj).sum().backward()
    return y, {
        name: p.grad.detach().clone() for name, p in bank.named_parameters() if p.grad is not None
    }


def _within(got: torch.Tensor, want: torch.Tensor, tol: float) -> bool:
    """Whether two gradients agree to `tol` of *their own* largest entry, elementwise.

    A parity gate on a gradient is relative to its own scale or it is a gate on the
    loss's — the argument `kernels/fused.py` makes for `BF16_PARITY` being relative.
    """
    scale = float(max(got.float().abs().max(), want.float().abs().max()))
    return bool(((got.float() - want.float()).abs().max() <= tol * scale).item())


#: One path's `(output, {name: grad})`, as `_grads` returns it.
Run = tuple[torch.Tensor, dict[str, torch.Tensor]]


def _both(bank: MoEFFN, x: torch.Tensor) -> tuple[Run, Run]:
    """The same call with the flag off and on, on the same weights and rows."""
    kb.set_bank(False)
    anchor = _grads(bank, x)
    kb.set_bank(True)
    fused = _grads(bank, x)
    kb.set_bank(False)
    return anchor, fused


# ------------------------------------------------------------------ the flag itself


def test_the_anchor_runs_where_no_kernel_can() -> None:
    """No Triton (or no card) is the anchor path, and calling the kernel refuses."""
    if REQUIRES_CARD and kb._build() is not None:
        pytest.skip(NO_CARD if not REQUIRES_CARD else "this host has the card and Triton")
    assert not kb.bank_enabled()
    with pytest.raises(LoudError, match=r"fused bank off or on a cpu tensor"):
        kb.bank(
            torch.zeros(2, D),
            torch.zeros(2, dtype=torch.int32),
            torch.tensor([2]),
            2,
            None,
            kinds=(1,),
            sides="bi",
        )


def test_a_dense_control_arm_keeps_the_anchor_and_launches_nothing() -> None:
    """`rank=0` has no rank-space tile: the arm is the anchor and no kernel was asked for."""
    bank = _bank(expert_tsct_rank=0)
    assert not bank._fused_ready(_inputs(bank))
    kb.reset_bank_launches()
    bank(_inputs(bank))
    assert kb.bank_launches() == 0


def test_a_rank_that_is_not_a_power_of_two_is_refused_by_name() -> None:
    """A tile is an aligned `tl.dot` operand; 24 is not one, and the escape is named."""
    with pytest.raises(LoudError, match="power of two"):
        kb._validate(24, D, F, (0,), "bi")


def test_a_dictionary_the_epilogue_does_not_know_is_refused() -> None:
    """The kernel computes codes it has formulas for; an unknown entry is not guessed.

    The code is an integer because `ffn._codes` resolves names to `fused.MIX_CODES`, and a
    raw 99 reaching the epilogue would fall through to the last branch of `_activate` and
    compute `situ` under another entry's name — the failure this refusal exists for.
    """
    with pytest.raises(LoudError, match=r"no entry for code 99"):
        kb._validate(R, D, F, (99,), "bi")
    with pytest.raises(LoudError, match=r"entries"):
        kb._validate(R, D, F, (0, 1, 2, 3, 4, 5, 0), "bi")


def test_the_refused_tile_is_arithmetic_not_an_opinion() -> None:
    """The docstring's claim, as arithmetic: `[BM, f]` in registers is past the file.

    A program that held the whole hidden tile would need `BM·f` fp32 values for `h`
    alone — 384 KiB at `BM=16, f=6144`, i.e. 1536 registers per thread over 256 threads,
    against the 255 a thread has. The f-loop is not an optimisation of that tile; it is
    the only one that fits.
    """
    per_thread = kb.BM * 6144 * 4 / (kb.WARPS * 32 * 4)
    assert per_thread > 255, "the refused tile must be the impossible one the docstring says"


def test_the_switch_the_counter_and_the_refusals_are_pure() -> None:
    """The module's CPU surface: the flag, the launch counter, and two refusals."""
    kb.set_bank(True)
    try:
        kb.reset_bank_launches()
        assert kb.bank_launches() == 0
        kb._tally(2)
        assert kb.bank_launches() == 2
        with pytest.raises(LoudError, match="cannot run here"):
            kb._require("forward")
    finally:
        kb.set_bank(False)
    assert not kb._BANK


def test_every_shape_the_tile_schedule_cannot_cover_is_refused() -> None:
    """Four refusals, each naming its escape; a kernel that guessed would compute a lie."""
    for args, match in (
        ((8, 32, 64, (0,), "bi"), "power of two"),
        ((17, 32, 64, (0,), "bi"), "power of two"),
        ((16, 8, 64, (0,), "bi"), "widths are"),
        ((16, 32, 64, (1,), "trio"), "not an FFN topology"),
    ):
        with pytest.raises(LoudError, match=match):
            kb._validate(*args)


def test_gather_copies_the_rows_and_leaves_the_padding_zero() -> None:
    """The one gather in the module: real rows in, zeros where the dispatch padded."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tok = torch.tensor([0, 2, 1, 1, 2, 0], dtype=torch.int32)
    cnt = torch.tensor([2, 3], dtype=torch.int32)
    got = kb.gather(x, tok, cnt, 3, torch.float32)
    assert got.shape == (2, 3, 2)
    assert torch.equal(got[0, 0], x[0]) and torch.equal(got[0, 1], x[2])
    assert torch.equal(got[1, 0], x[1]) and torch.equal(got[1, 2], x[0])
    assert torch.equal(got[0, 2], torch.zeros(2)), "the padding row must not be a real row"


def test_the_fused_group_refuses_a_dictionary_the_kernel_lacks() -> None:
    """`_kinds is None` is an arm the epilogue cannot mix, and the refusal names it."""
    bank = _bank()
    arm = bank.experts[0]
    arm._kinds = None
    x = _inputs(bank, rows=4)
    with pytest.raises(LoudError, match="cannot mix"):
        moe._fused_group(bank, [arm], x, torch.zeros(4, dtype=torch.int32), torch.tensor([4]), 4)


def test_the_padded_tiles_are_half_the_grid_and_return_early() -> None:
    """`capacity` is 2x the mean, so the kernel's `m0 >= cnt` guard is arithmetic.

    The guard is inside the Triton program (`fwd_kernel`), which a CPU job cannot run; what
    it rests on is the slot width the dispatch sizes, and that is a number here: at any
    batch the padded grid holds twice the rows the routing has, so half of its tiles have
    no row to work on and the early return is the schedule's normal case, not a corner.
    """
    for routed in (4, 64, 1024, 49_152):
        bank = MoEFFN(
            16,
            128,
            n_experts=8,
            topk=2,
            n_shared=0,
            act="moa",
            d_ffn_gelu=192,
            expert_tsct_rank=0,
            min_expert_params=0,
        )
        width = bank.capacity(routed)
        mean = routed / 8
        assert width >= 2 * mean * 0.99, (routed, width)
        padding = 8 * width - routed
        assert padding >= routed * 0.9, (routed, width, padding)


def test_the_bf16_step_is_read_from_the_device_not_guessed() -> None:
    """`bf16_step` answers for the dtype of the call, and the tally proves it counted."""
    assert kb.bf16_step(torch.device("cpu")) is False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert kb.bf16_step(torch.device("cpu")) is True


# ------------------------------------------------------------------ parity, on the card


@ON_CARD
def test_fp32_parity_forward_and_every_gradient() -> None:
    """The same seed and ids: the fused chain against the anchor, four orders down."""
    torch.manual_seed(0)
    bank = _bank()
    x = _inputs(bank)
    (y0, g0), (y1, g1) = _both(bank, x)
    scale = float(y0.abs().max())
    assert float((y1 - y0).abs().max()) <= FP32_PARITY * scale
    assert set(g0) == set(g1), "every parameter that moved under one path must move under both"
    for name in g0:
        assert _within(g1[name], g0[name], FP32_PARITY), (
            f"gradient {name} moved by more than the summation bound"
        )


@ON_CARD
def test_bf16_parity_forward_and_every_gradient() -> None:
    """A bf16 step: the same two chains, gated at the step's own quantisation."""
    torch.manual_seed(1)
    bank = _bank()
    x = _inputs(bank)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        #: The canary `tools/gpu_check.py` uses: a matmul inside the block must return
        #: bf16, or a silent fp32 fallback would make the match closer, not worse.
        canary = bank.experts[0].gate(x[:2])
        assert canary.dtype is torch.bfloat16, "the step is not the bf16 one"
        assert kb.bf16_step(x.device), "the kernel's own switch must see the bf16 step"
        (y0, g0), (y1, g1) = _both(bank, x)
    scale = float(y0.abs().max())
    assert float((y1.float() - y0.float()).abs().max()) <= BF16_PARITY * scale
    for name in g0:
        assert _within(g1[name], g0[name], BF16_PARITY), f"gradient {name} left the bf16 ulp band"


@ON_CARD
def test_the_live_widths_compile_under_the_shared_memory_ceiling() -> None:
    """The reported blocker as a gate: every fused kernel fits, and the path is visible.

    A tile-constant edit that pushes a kernel over the card's 101376 B per-block ceiling
    used to surface only as a full step that refuses to launch. This pins the arithmetic
    at d=832/f=6144/r=256, checks the counter says the fused path ran, and checks forward
    and every gradient against the anchor inside the step's bf16 band.
    """
    torch.manual_seed(9)
    bank = MoEFFN(
        LIVE_D,
        LIVE_F,
        n_experts=4,
        topk=2,
        n_shared=0,
        act="moa",
        d_ffn_gelu=3 * LIVE_F // 2,
        expert_tsct_rank=LIVE_R,
        min_expert_params=0,
        device="cuda",
    ).eval()
    x = torch.randn(64, LIVE_D, device="cuda")
    proj = torch.randn_like(x)

    def once() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bank.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = bank(x)
            (y.float() * proj).sum().backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in bank.named_parameters()
            if p.grad is not None
        }
        return y.detach(), grads

    kb.set_bank(False)
    anchor = once()
    kb.set_bank(True)
    kb.reset_bank_launches()
    fused = once()
    launches = kb.bank_launches()
    kb.set_bank(False)

    assert launches > 0, "the flag was on and no fused kernel launched — the anchor ran"
    for name in ("prep", "fwd", "bwd_rows", "bwd_cols"):
        for cache in getattr(kb._KERNELS[name], "device_caches", {}).values():
            for compiled in cache[0].values():
                used = int(compiled.metadata.shared)
                assert used <= SMEM_LIMIT, f"{name} asks {used} B > {SMEM_LIMIT} B"

    y0, g0 = anchor
    y1, g1 = fused
    assert torch.isfinite(y1).all() and all(torch.isfinite(g).all() for g in g1.values())
    assert float((y1.float() - y0.float()).abs().max()) <= LIVE_BF16_PARITY * float(y0.abs().max())
    assert set(g0) == set(g1)
    for name in g0:
        assert _within(g1[name], g0[name], LIVE_BF16_PARITY), (
            f"gradient {name} left the live-width bf16 band"
        )


@ON_CARD
def test_a_duplicate_row_is_summed_not_overwritten() -> None:
    """One token, both assignments to the same expert: the same row twice in a slot map.

    A dispatch that wrote a slot instead of accumulating it would pass a unique-row test
    and lose one of the two contributions here, which is the whole `index_add_` contract.
    """
    torch.manual_seed(2)
    bank = _bank(n_experts=2, topk=2, n_shared=0)
    x = _inputs(bank, rows=6)
    with torch.no_grad():
        #: One router row owns both slots of one token: the assignment is duplicated.
        bank.router.weight.zero_()
        bank.router.weight[0, 0] = 1.0
    (y0, g0), (y1, g1) = _both(bank, x)
    scale = float(y0.abs().max())
    assert float((y1 - y0).abs().max()) <= FP32_PARITY * scale
    for name in g0:
        assert _within(g1[name], g0[name], FP32_PARITY), f"gradient {name} moved"


@ON_CARD
def test_one_expert_at_top_one_is_the_anchor_beside_it() -> None:
    """One expert, top-1: the fused write is the anchor's, and the slots are placement.

    The `torch.equal` claim this file can make is the kernel's own: the same rows at
    three capacities — no padding, 2x, 3x — are bitwise identical, because a slot is
    where a row sits and not what it computes. (Against the anchor the claim is the fp32
    bound: `torch.equal` across two GEMM libraries is not a statement either one can
    honour, and `tests/test_moe.py`'s bit-for-bit gate is torch-against-torch.)
    """
    torch.manual_seed(3)
    bank = _bank(n_experts=1, topk=1, n_shared=0)
    x = _inputs(bank, rows=9)
    kb.set_bank(False)
    y_anchor = bank(x)
    kb.set_bank(True)
    outputs = []
    for ratio in (1.0, 2.0, 3.0):
        bank.capacity_ratio = ratio
        outputs.append(bank(x))
    kb.set_bank(False)
    for other in outputs[1:]:
        assert torch.equal(outputs[0], other), "capacity moved the arithmetic, not the slots"
    assert float((outputs[0] - y_anchor).abs().max()) <= FP32_PARITY * float(y_anchor.abs().max())


@ON_CARD
def test_the_launch_count_is_two_per_group_forward_and_four_with_the_backward() -> None:
    """The number that says the kernel replaced the chain, not joined it."""
    torch.manual_seed(4)
    bank = _bank()
    x = _inputs(bank).requires_grad_(True)
    groups = -(-bank.n_experts // bank.group(bank.capacity(x.shape[0] * bank.topk), 4))
    kb.set_bank(True)
    kb.reset_bank_launches()
    y = bank(x)
    assert kb.bank_launches() == 2 * groups, "prep + fwd per group, forward"
    y.square().mean().backward()
    #: Four in the backward: the checkpointer re-runs prep and fwd (the `Stacked` copies
    #: are 54 MiB a group and retaining twelve layers of them is 4.5 GiB), then
    #: `bwd_rows` and `bwd_cols` write the gradients. `moe.py._fused_group` says why.
    assert kb.bank_launches() == 6 * groups, "recompute + two gradient kernels per group"
    kb.set_bank(False)


@ON_CARD
def test_the_forward_grid_is_tile_major() -> None:
    """The fwd grid is ``(tiles, experts)`` so one wave stays on one expert's factors.

    The expert used to be the fast-varying axis (``(experts, tiles)``), which walks every
    expert of the group before returning to a row tile — each CTA evicts the previous
    CTA's factor slab from L2. Tile-major is the whole change here; this gate stops a
    future edit from silently restoring the expert-major walk, which leaves the arithmetic
    correct but gives back the locality.
    """
    torch.manual_seed(11)
    bank = _bank()
    x = _inputs(bank)
    kb.set_bank(True)
    built = kb._require("forward")
    real = built["fwd"]
    seen: dict[str, tuple[int, ...]] = {}

    class Recorder:
        def __getitem__(self, grid: tuple[int, ...]):  # type: ignore[no-untyped-def]
            seen["grid"] = tuple(grid)
            return real[grid]

    built["fwd"] = Recorder()
    try:
        bank(x)
    finally:
        built["fwd"] = real
        kb.set_bank(False)
    e = bank.n_experts
    tiles = -(-bank.capacity(x.shape[0] * bank.topk) // kb.BM)
    assert seen["grid"] == (tiles, e), "the fwd grid must walk row tiles fastest"


@ON_CARD
def test_the_counter_reports_the_launches_the_forward_ran() -> None:
    """A flag-on forward with a zero counter is the defect; `bank_ops` must not be zero."""
    from dormouse_core.counters import Counters

    torch.manual_seed(5)
    bank = _bank()
    counter = Counters()
    kb.set_bank(True)
    bank(_inputs(bank), counter)
    counts = counter.snapshot()
    kb.set_bank(False)
    assert counts["bank_ops"] >= 2, "prep + fwd at least"
    counter.check()


# ------------------------------------------------------------------ the traffic claim


def _bench() -> object:
    """`tools/bench_bank.py`, loaded the way `tests/test_gpu_check.py` loads its tool."""
    spec = importlib.util.spec_from_file_location(
        "bench_bank", Path(__file__).resolve().parent.parent / "tools" / "bench_bank.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fused_schedule_moves_far_less_than_the_anchor() -> None:
    """The traffic that explains the speedup, from the two hand counts, at the live shape.

    The anchor's bill is dominated by `[rows, f]` slabs and the fused chain's is not: the
    f-wide terms are the expert's `v`/`u_d` factors (read once per row tile) and there is
    no `[rows, f]` activation at all. The gate is a ratio, so a schedule that grew a slab
    back fails it without anyone deciding what "far less" means.
    """
    bench = _bench()
    rows, d, f, r, k = 8 * 1024 * 2, 832, 6144, 256, 6
    fused = bench.fused_bytes(rows, 512, 64, d, f, r, k)  # type: ignore[attr-defined]
    anchor = sum(bench.anchor_bytes(rows, d, f, r, k).values())  # type: ignore[attr-defined]
    floor = sum(v for key, v in fused.items() if "re-read" not in key)
    assert floor < anchor / 3, f"the fused floor is {floor:,} B against {anchor:,} B"
    #: The claim that matters: the fused chain's ACTIVATION traffic is rank-space, not
    #: `[rows, f]`. At the live shape the anchor's `[rows, f]` passes are 5.2 GiB; the
    #: kernel's whole activation set is the `[rows, r]` scratches and the write — 152 MiB.
    activations = fused["x (index map)"] + fused["rank-space scratches"] + fused["write"]
    assert activations < anchor / 20, "an `[rows, f]` slab came back into the schedule"
    #: And the honest weakness, stated where a reader of the gate will see it: the
    #: expert's factors are re-read once per row tile (`ceil(rows/expert/BM)` times), which
    #: is why the ceiling is much larger than the floor. `tools/bench_bank.py` prints both.
    assert fused["factors (re-read per row tile)"] > fused["factors (unique)"] * 8
