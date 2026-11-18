"""The two residual mechanisms, and the gates an A/B owes them.

Both arms were written off on measurements a later audit found invalid: mHC at the
wrong corner of the Birkhoff polytope (`b_res = 0` → `H_res = J/n`, uniform mixing) and
AttnRes at a batch where the score tensor OOMed. The gates below are the ones that
would have caught both, and they are the point of this file:

1. **The identity init is bit-exact.** `torch.equal`, not `isclose`: an arm that starts
   anywhere but the plain residual measures its own perturbation, and a scale error of
   `1/n` or `n` is four orders above float noise.
2. **mHC's mixing matrix is the identity**, and a test that FAILS if someone sets
   `b_res` back to zero — the exact line the two contradictory verdicts on record were
   measured through.
3. **The plain arm never moved**: no parameters, same forward.
4. **Causality**: a later sublayer cannot move an earlier residual.
5. **It learns**: 200 CPU steps, three arms, three losses printed.
6. **Memory**: the live shape (b24x1024, d=832, one source per sublayer), with the
   `[T, S, d]` stack the old implementation built measured in this file as the contrast.
7. **The counters are on the line**, and they are zero when the arm is off.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport
from stacks import REFERENCE, reference_stack
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from dormouse.model.model import Transformer
from dormouse.model.residual import B_RES, MHC, AttnResFull, sinkhorn_knopp
from dormouse.model.spec import StackSpec

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}
#: The live training shape the old arm OOMed at: b24 x 1024 tokens, d = 832.
LIVE_T, LIVE_D, LIVE_LAYERS = 24576, 832, 12
MIB = 2**20

ATTNRES = ("attnres", "attnres_sources", "full-depth attention over the writes")
MHC_DECL = ("mhc", "mhc_mix", "n lanes mixed by a Sinkhorn-Knopp matrix")


def _ids(seed: int = 7, batch: int = 2, seq: int = 9) -> torch.Tensor:
    """A fixed batch of ids, so two arms see the same tokens."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SMALL["vocab"], (batch, seq), generator=generator)


def _arm(name: str, *, seed: int = 5, **over: object) -> Transformer:
    """A reference stack carrying `name`'s residual mechanism, declared to the counters."""
    d_model = int(over.get("d_model", SMALL["d_model"]))  # type: ignore[arg-type]
    n_layers = int(over.get("n_layers", SMALL["n_layers"]))  # type: ignore[arg-type]
    mixer = None
    declared: tuple[tuple[str, str, str], ...] = REFERENCE
    if name == "attnres":
        mixer = AttnResFull(d_model, 2 * n_layers)
        declared = (*REFERENCE, ATTNRES)
    elif name == "mhc":
        mixer = MHC(d_model, 2)
        declared = (*REFERENCE, MHC_DECL)
    return reference_stack(**{**SMALL, **over}, mixer=mixer, declarations=declared, seed=seed)


# ---------------------------------------------------------------- 1. the identity init


def test_attnres_writes_the_plain_residual_bit_for_bit_at_init() -> None:
    """The mixer's own gate: `write(u) == x + u` with `torch.equal`, 1..24 sources.

    The proof is the `log n` anchor (module docstring): at the zero query the weights are
    `(n, -30 …, 1)`, the denominator is exactly `n + 1`, so the current source's weight
    is exactly 1.0. Written without the anchor, a plain `softmax` gives `1/(n+1)` and
    this test is off by a factor of `n + 1`; written with `exp(log n)` instead of the
    integer `n` it is off by one ulp and `torch.equal` fails.
    """
    torch.manual_seed(11)
    mixer = AttnResFull(8, 24)
    x = torch.randn(2, 5, 8)
    mixer.begin(x)
    stream = x
    for _ in range(24):
        u = torch.randn(2, 5, 8) * 0.3
        before = stream
        stream = mixer.write(u)
        assert torch.equal(mixer.read(), stream)
        assert torch.equal(stream, before + u), "the write is not the plain residual"


def test_attnres_is_the_plain_arm_bit_for_bit_on_a_whole_model() -> None:
    """One seed, the same weights, the mixer added: logits `torch.equal`."""
    plain = _arm("plain")
    ours = _arm("attnres")
    ours.load_state_dict(plain.state_dict(), strict=False)  # type: ignore[attr-defined]
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(ours(ids), plain(ids)), (  # type: ignore[operator]
            "AttnRes at init is not the plain residual: the A/B would measure a "
            "perturbation, not the mechanism"
        )


def test_attnres_is_exact_at_every_depth() -> None:
    """`N = L` at 1, 2, 4 and 8 blocks: the identity is a property of the anchor, not luck."""
    for n_layers in (1, 2, 4, 8):
        plain = _arm("plain", n_layers=n_layers)
        ours = _arm("attnres", n_layers=n_layers)
        ours.load_state_dict(plain.state_dict(), strict=False)  # type: ignore[attr-defined]
        ids = _ids(seq=7)
        with torch.no_grad():
            assert torch.equal(ours(ids), plain(ids)), f"{n_layers} layers"  # type: ignore[operator]


# ------------------------------------------------------------------- 2. mHC's identity


def _mixings(mixer: MHC) -> torch.Tensor:
    """`H_res` for a unit-RMS stream (the first token's), `[n, n]`."""
    mixer.begin(torch.randn(1, 4, mixer.d_model))
    _pre, _post, res = mixer.mappings()
    return res[0, 0]


@pytest.mark.parametrize("n_streams", [2, 3, 4])
def test_mhc_starts_at_the_identity_corner(n_streams: int) -> None:
    """`||H_res - I||inf <= 1e-5` at init, and nowhere near the uniform corner.

    The `> 0.5` is the gate the audit's finding is written as, in the norm its numbers
    were recorded in (`||J/n - I|| = 1.0` at n = 2): an init that mixes lanes cannot
    pass it.
    """
    torch.manual_seed(3)
    mixer = MHC(64, n_streams)
    h = _mixings(mixer)
    eye = torch.eye(n_streams)
    error = (h - eye).abs().max().item()
    away = torch.linalg.matrix_norm(h - torch.full_like(h, 1.0 / n_streams)).item()
    print(f"[mhc] n={n_streams} ||H - I||inf = {error:.2e}  ||H - J/n|| = {away:.3f}")
    assert error <= 1e-5, error
    assert away > 0.5, f"H is {away:.3f} from J/n: the init is mixing the lanes"
    assert h.min().item() > -1e-7, "a negative entry is not a doubly stochastic matrix"
    # Doubly stochastic, which is the other half of the constraint.
    assert torch.allclose(h.sum(-1), torch.ones_like(h.sum(-1)), atol=1e-5)
    assert torch.allclose(h.sum(-2), torch.ones_like(h.sum(-2)), atol=1e-5)


def test_the_b_res_zero_init_is_uniform_mixing_and_this_gate_catches_it() -> None:
    """THE REGRESSION GATE for the audit's finding.

    `b_res = 0` puts `H_res` at `J/n` — every lane the average of all lanes — which is
    the opposite corner of the same polytope, and the reason the October verdict
    (-0.078) and the November one (+0.0853) contradict: both measured a model that
    started by averaging all its residual streams. The two asserts below are the pair
    the audit's numbers were recorded as (`||H - J/n|| = 0.0000`, `||H - I|| = 1.0000`).
    """
    mixer = MHC(64, 2)
    with torch.no_grad():
        mixer.b_res.zero_()
    h = _mixings(mixer)
    uniform = torch.full((2, 2), 0.5)
    near = torch.linalg.matrix_norm(h - uniform).item()
    away = torch.linalg.matrix_norm(h - torch.eye(2)).item()
    print(f"[mhc] b_res=0: ||H - J/n|| = {near:.4f}, ||H - I|| = {away:.4f}")
    assert near < 0.01, "b_res=0 is not J/n here"
    assert away > 0.5, "the identity gate would pass"


def test_mhc_is_the_plain_residual_at_init_on_a_whole_model() -> None:
    """The identity matrix is the anchor; the read-out must land on the plain stream."""
    plain = _arm("plain")
    ours = _arm("mhc")
    ours.load_state_dict(plain.state_dict(), strict=False)  # type: ignore[attr-defined]
    ids = _ids()
    with torch.no_grad():
        diff = (ours(ids) - plain(ids)).abs().max().item()  # type: ignore[operator]
    print(f"[mhc] identity at init: max|dlogits| = {diff:.3e}")
    assert diff <= 2e-3, diff


def test_a_one_lane_mhc_and_a_zero_source_attnres_are_refused() -> None:
    with pytest.raises(LoudError) as lane:
        MHC(32, 1)
    assert "plain residual" in str(lane.value)
    with pytest.raises(LoudError) as sources:
        AttnResFull(32, 0)
    assert "at least one source" in str(sources.value)


def test_the_sinkhorn_projection_is_doubly_stochastic_and_reaches_the_identity() -> None:
    """`b·I` → `I` as `b` grows; the log domain never overflows on an unbounded logit."""
    for b in (0.0, 2.0, 12.0, 200.0):
        h = sinkhorn_knopp(torch.as_tensor([[b, 0.0], [0.0, b]]))
        assert torch.isfinite(h).all()
        assert torch.allclose(h.sum(-1), torch.ones(2), atol=1e-5)
    assert sinkhorn_knopp(torch.tensor([[0.0, 0.0], [0.0, 0.0]])).sub(0.5).abs().max() < 1e-6
    high = sinkhorn_knopp(torch.as_tensor([[B_RES, 0.0], [0.0, B_RES]]))
    assert (high - torch.eye(2)).abs().max().item() <= 1e-5


# -------------------------------------------------------------------- 3. the plain arm


def test_the_plain_arm_has_no_mixer_and_no_extra_parameters() -> None:
    model = _arm("plain")
    assert model.mixer is None  # type: ignore[attr-defined]
    report = model.param_report()  # type: ignore[attr-defined]
    assert report["residual"] == 0
    assert not any(k.startswith("mixer.") for k in model.state_dict())  # type: ignore[attr-defined]


def test_attnres_costs_one_query_and_one_prior_per_sublayer() -> None:
    """`L·(d+1)` — the number the arm has to earn back: 19,992 at d=832, L=24."""
    plain = _arm("plain")
    ours = _arm("attnres")
    extra = (
        sum(p.numel() for p in ours.parameters())  # type: ignore[attr-defined]
        - sum(p.numel() for p in plain.parameters())  # type: ignore[attr-defined]
    )
    assert extra == 2 * SMALL["n_layers"] * (SMALL["d_model"] + 1), extra
    assert ours.param_report()["residual"] == extra  # type: ignore[attr-defined]


# ----------------------------------------------------------------------- 4. causality


@pytest.mark.parametrize("name", ["attnres", "mhc"])
def test_a_later_sublayer_cannot_move_an_earlier_residual(name: str) -> None:
    """Perturbing the third write leaves the first two outputs bit-identical."""

    def _mixer() -> AttnResFull | MHC:
        torch.manual_seed(19)  # the same hyper-network for both runs
        return AttnResFull(8, 4) if name == "attnres" else MHC(8, 2)

    torch.manual_seed(23)
    d = 8
    x = torch.randn(2, 3, d)
    writes = [torch.randn(2, 3, d) for _ in range(3)]

    mixer = _mixer()
    mixer.begin(x)
    first = [mixer.write(writes[0]), mixer.write(writes[1])]

    later = _mixer()
    later.begin(x)
    again = [later.write(writes[0]), later.write(writes[1])]
    later.write(writes[2] + 1.0)  # a later write must not reach the earlier ones

    for a, b in zip(first, again, strict=True):
        assert torch.equal(a, b), f"{name}: a later write moved an earlier residual"


# ------------------------------------------------------------------------ 5. the state


def test_the_mixer_refuses_a_write_before_begin() -> None:
    for mixer in (AttnResFull(8, 2), MHC(8, 2)):
        with pytest.raises(LoudError) as e:
            mixer.write(torch.randn(1, 2, 8))
        assert "before begin" in str(e.value)


def test_a_stack_deeper_than_the_mixer_was_built_for_is_refused() -> None:
    mixer = AttnResFull(8, 2)
    mixer.begin(torch.randn(1, 2, 8))
    mixer.write(torch.randn(1, 2, 8))
    mixer.write(torch.randn(1, 2, 8))
    with pytest.raises(LoudError) as e:
        mixer.write(torch.randn(1, 2, 8))
    assert "one source per sublayer" in str(e.value)


# -------------------------------------------------------------------------- 6. counters


def test_the_counters_are_on_the_line_and_zero_when_the_arm_is_off() -> None:
    """`Counters.line()` is the log line: every non-zero counter, `k=v`, sorted."""
    ids = _ids(seq=6)
    for name, key in (("plain", None), ("attnres", "attnres_sources"), ("mhc", "mhc_mix")):
        model = _arm(name)
        counters = Counters()
        with torch.no_grad():
            model(ids, counters)  # type: ignore[operator]
        if key is None:
            assert counters.counts.get("attnres_sources", 0) == 0
            assert counters.counts.get("mhc_mix", 0) == 0
            assert "attnres_sources" not in counters.line()
        else:
            assert counters.counts[key] > 0, counters.line()
            assert f"{key}=" in counters.line(), counters.line()
    print(f"[counters] {counters.line()}")


def test_attnres_counts_every_source_it_mixes() -> None:
    """`attnres_sources` is the source count, not the write count: 1+2+…+2L."""
    n_layers = 3
    model = _arm("attnres", n_layers=n_layers)
    counters = Counters()
    with torch.no_grad():
        model(_ids(seq=5), counters)  # type: ignore[operator]
    writes = 2 * n_layers
    assert counters.counts["attnres_sources"] == writes * (writes + 1) // 2


def test_an_enabled_mixer_that_never_wrote_raises() -> None:
    """The defect class: a mechanism enabled and silent is an error, not a log line."""
    model = reference_stack(**SMALL, declarations=(*REFERENCE, ATTNRES))
    with pytest.raises(MissingReport), torch.no_grad():
        model(_ids(), Counters())


# -------------------------------------------------------------------------- 7. spec


def test_the_spec_names_the_three_residual_arms_and_refuses_a_one_lane_mhc() -> None:
    assert StackSpec(residual="plain").residual == "plain"
    assert StackSpec(residual="attnres").residual == "attnres"
    assert StackSpec(residual="mhc").mhc_streams == 2
    with pytest.raises(LoudError) as e:
        StackSpec(residual="lora")
    assert "residual='lora'" in str(e.value)
    with pytest.raises(LoudError) as lane:
        StackSpec(residual="mhc", mhc_streams=1)
    assert "mhc_streams=1" in str(lane.value)


# ------------------------------------------------------------------- 8. the live shape


class _Allocations(TorchDispatchMode):
    """Every tensor the dispatched call returns, by `numel` — the allocator's view."""

    def __init__(self) -> None:
        super().__init__()
        self.numel: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        out = func(*args, **(kwargs or {}))
        self.numel.extend(
            leaf.numel() for leaf in tree_leaves(out) if isinstance(leaf, torch.Tensor)
        )
        return out


#: One arm at the live shape, in its OWN process: the resident high-water mark is a
#: property of a process, so two arms measured in one interpreter share an arena and
#: the second one inherits the first one's freed blocks. `arm` is `plain` or `attnres`.
_LIVE_PROGRAM = """
import sys
import torch
from dormouse.model.residual import AttnResFull

t, d, s = 24576, 832, 24
torch.manual_seed(0)
sources = [torch.randn(t, d) for _ in range(s)]
x = torch.randn(t, d)
mixer = AttnResFull(d, s)
with torch.no_grad():
    mixer.begin(x)
    for u in sources:
        if sys.argv[1] == "plain":
            x = x + u
        else:
            x = mixer.write(u)
assert x.shape == (t, d)
for line in open("/proc/self/status", encoding="ascii"):
    if line.startswith("VmHWM:"):
        print("PEAK", int(line.split()[1]))
"""


def _peak_mib(arm: str) -> float:
    """Peak resident MiB of one arm at the live shape, measured in a fresh process."""
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "-c", _LIVE_PROGRAM, arm],
        cwd=root,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )
    peak = [line.split()[1] for line in run.stdout.splitlines() if line.startswith("PEAK")]
    assert peak, run.stdout + run.stderr
    return float(peak[0]) / 1024.0


@torch.no_grad()
def test_the_scores_are_one_source_at_a_time_at_the_live_shape() -> None:
    """b24x1024, d=832, one source per sublayer: no `[T, S, d]` tensor is ever built.

    The old implementation's score tensor is `[T, S, d]`: at `S = 24` that is
    `24576·24·832·4 = 1872 MiB` in fp32 (507 MiB at S = 13 bf16 — the recorded number,
    and the reason the arm OOMed at b24). The bound here is one `[T, d]` fp32
    transient (78.0 MiB) plus the `[T, S]` scores (2.4 MiB).
    """
    sources = [torch.randn(LIVE_T, LIVE_D) for _ in range(2 * LIVE_LAYERS)]
    mixer = AttnResFull(LIVE_D, 2 * LIVE_LAYERS)
    x = torch.randn(LIVE_T, LIVE_D)
    with _Allocations() as rec:
        mixer.begin(x)
        for u in sources:
            mixer.write(u)
    biggest = max(rec.numel) * 4
    stack = LIVE_T * len(sources) * LIVE_D * 4
    plain_peak = _peak_mib("plain")
    attnres_peak = _peak_mib("attnres")
    print(
        f"[attnres] live shape T={LIVE_T} d={LIVE_D} S={len(sources)} fp32: "
        f"peak {attnres_peak:.0f} MiB vs plain {plain_peak:.0f} MiB; "
        f"largest allocation {biggest / MIB:.1f} MiB; "
        f"the old [T,S,d] stack would be {stack / MIB:.0f} MiB"
    )
    assert stack not in [n * 4 for n in rec.numel], "the [T, S, d] score stack is back"
    assert biggest <= LIVE_T * LIVE_D * 4 * 4, f"largest allocation {biggest / MIB:.1f} MiB"
    extra = (attnres_peak - plain_peak) * MIB
    print(f"[attnres] peak over the plain write: {extra / MIB:.1f} MiB")
    assert extra <= 0.5 * 1024 * MIB, f"{extra / MIB:.1f} MiB over plain"


@torch.no_grad()
def test_mhc_costs_n_lanes_and_a_mixing_matrix_and_nothing_per_source() -> None:
    """mHC's own footprint: `n` lanes and an `[n, n]` matrix per sublayer."""
    x = torch.randn(24, 1024, LIVE_D)
    u = torch.randn(24, 1024, LIVE_D)
    mixer = MHC(LIVE_D, 2)
    lanes = x.numel() * 2 * 4
    mixer.begin(x)
    print(f"[mhc] live shape b24 x 1024: two lanes are {lanes / MIB:.1f} MiB")
    assert mixer.write(u).shape == x.shape


# -------------------------------------------------------------------- 9. it learns


def test_two_hundred_steps_descend_on_all_three_arms() -> None:
    """The gate a mechanism owes before it is worth an A/B: it trains, all three arms.

    The fixture is `test_step.py`'s: a 3x17 batch over a 16-token vocabulary, whose
    ceiling is memorisation. The number to read is the RATIO — a residual arm that
    starts at the identity must not train slower than the control it replaces.
    """
    from dormouse.train import MixOptimizer, train_step

    ids = _ids(seed=7, batch=3, seq=17) % SMALL["vocab"]
    curves: dict[str, list[float]] = {}
    for name in ("plain", "attnres", "mhc"):
        model = _arm(name, seed=23)
        optimizer = MixOptimizer(model)
        curves[name] = [train_step(model, ids, optimizer=optimizer).loss for _ in range(200)]
        print(
            f"[learn] {name:8s} 200 steps: {curves[name][0]:.2f} -> "
            f"{curves[name][-1]:.2f} (min {min(curves[name]):.2f})"
        )
    for name, losses in curves.items():
        assert math.isfinite(losses[-1]), name
        assert losses[-1] < 0.5 * losses[0], f"{name} did not descend: {losses[-1]:.2f}"
