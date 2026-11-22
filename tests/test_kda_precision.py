"""KDA precision gate: fp32 state + real fp32 log-decay in the FLA path.

FlashKDA's bf16 state (3.45% final-state RMSE at 8k) and fp16 decay LUT (9%
decay error) are a different failure class from the chunked algorithm. This
file checks both halves: an external fp64 cosine on the FLA kernel, and the
full-width paired learning A/B that the low-precision hypothesis is meant to
close. The speed gate is a live-layer fwd+bwd measurement, not a claim from a
kernel benchmark.

CPU-only CI runs the flag contract and the eager fp64 anchor; the card tests
skip loudly without CUDA.
"""

from __future__ import annotations

import math
import statistics
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.kernels import kda_fla
from dormouse.metrics import from_nats
from dormouse.model.assemble import assemble
from dormouse.model.attn_variants.kda import KDA
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kda_precision.toml"
SEEDS_3 = (4, 5, 6)
STEPS = 200
BPB_TAIL = 20
GATE = 0.01
#: t_{0.95, df=2} = 2.920 (one-sided 95%).
T95_DF2 = 2.920
#: The task's ms/layer bar for the FLA path at the live b2 x s1024 shape.
MS_PER_LAYER = 8.0

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the FLA kernel needs CUDA")


def _numpy_recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """External fp64 S_t = (I - β k kᵀ) Diag(alpha) S_{t-1} + β k vᵀ, o_t = S_tᵀ q_t."""
    state = np.zeros((q.shape[1], v.shape[1]), dtype=np.float64)
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        state = (np.eye(q.shape[1]) - beta[t] * np.outer(k[t], k[t])) @ (np.diag(alpha[t]) @ state)
        state += beta[t] * np.outer(k[t], v[t])
        out[t] = state.T @ q[t]
    return out, state


def _as_kernel(a: np.ndarray) -> torch.Tensor:
    """One-batch, one-head [b, h, s, d] tensor (beta [b, h, s]) for the kernel."""
    return torch.from_numpy(a).reshape(1, 1, *a.shape).double()


def test_external_fp64_anchor_survives_the_backend_swap() -> None:
    """The 3-token paper recurrence, ~2e-16 against numpy, on the eager path."""
    q = np.array([[0.5, -0.3], [1.0, 0.2], [-0.7, 0.9]])
    k = np.array([[0.4, 0.8], [-0.6, 0.5], [0.9, -0.2]])
    v = np.array([[1.0, 0.5], [-0.5, 2.0], [0.3, -1.0]])
    alpha = np.array([[0.9, 0.8], [0.95, 0.7], [0.85, 0.6]])
    beta = np.array([0.5, 0.7, 0.3])
    want_out, want_state = _numpy_recurrence(q, k, v, alpha, beta)

    layer = KDA(8, 1, 2, chunk=3).double()
    got, state = layer._chunkwise(*(_as_kernel(a) for a in (q, k, v, alpha, beta)))
    assert np.allclose(got[0, 0].detach().numpy(), want_out, atol=2e-16, rtol=2e-16)
    assert np.allclose(state[0, 0].detach().numpy(), want_state, atol=2e-16, rtol=2e-16)


def test_flag_contract_and_loud_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit 0/1 select the arm; a missing FLA is a loud error, never an eager time."""
    monkeypatch.setenv("DORMOUSE_KDA_FLA", "0")
    assert not kda_fla.fla_requested()
    monkeypatch.setenv("DORMOUSE_KDA_FLA", "1")
    assert kda_fla.fla_requested()
    monkeypatch.setitem(sys.modules, "fla.ops.kda", None)
    with pytest.raises(LoudError, match="install flash-linear-attention"):
        kda_fla.load_chunk_kda()


@CUDA
def test_fla_fp64_cosine_and_fp32_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FLA kernel's write/state against the fp64 recurrence, with fp32 carry."""
    rng = np.random.default_rng(11)
    batch, heads, seq, width = 1, 2, 64, 32
    q = rng.normal(size=(batch, heads, seq, width))
    k = rng.normal(size=(batch, heads, seq, width))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = rng.normal(size=(batch, heads, seq, width))
    alpha = rng.uniform(0.8, 0.999, size=(batch, heads, seq, width))
    beta = rng.uniform(0.1, 0.9, size=(batch, heads, seq))

    want_out = np.empty_like(q)
    want_state = np.empty((batch, heads, width, width))
    for b in range(batch):
        for h in range(heads):
            want_out[b, h], want_state[b, h] = _numpy_recurrence(
                q[b, h], k[b, h], v[b, h], alpha[b, h], beta[b, h]
            )

    args = tuple(torch.from_numpy(a).float().cuda() for a in (q, k, v, alpha, beta))
    with monkeypatch.context() as m:
        m.setenv("DORMOUSE_KDA_FLA", "1")
        got, state = KDA(64, heads, width, chunk=64, device="cuda")._chunkwise(*args)

    assert state.dtype == torch.float32
    want_o = torch.from_numpy(want_out).float().cuda()
    want_s = torch.from_numpy(want_state).float().cuda()
    cos_o = float(torch.nn.functional.cosine_similarity(got.flatten(), want_o.flatten(), dim=0))
    cos_s = float(torch.nn.functional.cosine_similarity(state.flatten(), want_s.flatten(), dim=0))
    print(f"\nFLA fp64 cosine: write {cos_o:.9f}, state {cos_s:.9f}; state dtype {state.dtype}")
    assert cos_o >= 1.0 - 1e-6, cos_o
    assert cos_s >= 1.0 - 1e-6, cos_s


def _spec() -> StackSpec:
    """The shipped architecture at the measurement shape; no field overridden."""
    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(CONFIG))
    assert (spec.batch, spec.seq_len, spec.pattern.count("kda")) == (2, 768, 9)
    return spec


def _train(seed: int, fla: bool, *, steps: int = STEPS) -> float:
    """One arm: `steps` full b2 x s768 steps; return the tail BPB.

    Both arms get the same seed, weight init, id stream and optimizer; only
    ``DORMOUSE_KDA_FLA`` changes. ``accum=2`` splits the two rows into two
    one-row micro-batches, keeping Muon inside the 16 GiB card.
    """
    spec = _spec()
    with pytest.MonkeyPatch.context() as m:
        m.setenv("DORMOUSE_KDA_FLA", "1" if fla else "0")
        torch.manual_seed(seed)
        assembled = assemble(spec, device="cuda")
        optimizer = MixOptimizer(assembled.model)
        #: 1% of the recipe's peak (muon 0.002, adamw 0.0003). At 10% the b2 stack
        #: on random ids hits the repo's NaN firewall around step 80 on BOTH arms
        #: at this HEAD (verified on unmodified app/), so no paired delta exists.
        #: The fraction is identical in both arms, so the A/B is unchanged.
        optimizer.set_lr(0.01)
        gen = torch.Generator().manual_seed(seed + 9000)
        ids = torch.randint(0, spec.vocab, (steps + 1, spec.batch, spec.seq_len + 1), generator=gen)
        ids = ids.cuda()
        bpbs: list[float] = []
        for step in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result, _ = assembled.step(ids[step], optimizer=optimizer, accum=2)
            bpbs.append(from_nats(float(result.loss), ids[step].numel()).bpb)
            if (step + 1) % 25 == 0:
                print(
                    f"  seed {seed} fla={int(fla)} step {step + 1}/{steps}: "
                    f"bpb(now)={bpbs[-1]:.5f} bpb(tail)={statistics.fmean(bpbs[-BPB_TAIL:]):.5f}",
                    flush=True,
                )
        del optimizer, assembled, ids
    torch.cuda.empty_cache()
    return statistics.fmean(bpbs[-BPB_TAIL:])


def _paired(seeds: tuple[int, ...]) -> tuple[list[float], float, float]:
    """Signed per-seed deltas, their mean, and the one-sided 95% t upper bound."""
    deltas: list[float] = []
    for seed in seeds:
        eager = _train(seed, False)
        fla = _train(seed, True)
        deltas.append(fla - eager)
        print(f"  seed {seed}: dBPB {deltas[-1]:+.5f}", flush=True)
    mean = statistics.fmean(deltas)
    upper = mean + T95_DF2 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return deltas, mean, upper


@CUDA
def test_bwd_counter_fires_on_the_triton_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """kda_bwd keeps parity with kda_fwd when the Triton backward runs."""
    torch.manual_seed(2)
    layer = KDA(64, 2, 64, chunk=64, device="cuda")
    counters = Counters()
    counters.enable("kda", counter="kda_bwd", why="the Triton path must backprop")
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    with monkeypatch.context() as m:
        m.setenv("DORMOUSE_KDA_FLA", "1")
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = layer(torch.randn(2, 64, 64, device="cuda"), counters)
            loss = y.float().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert counters.counts["kda_fwd"] > 0
    assert counters.counts["kda_bwd"] == counters.counts["kda_fwd"]
    counters.check()
    cold = [n for n, p in layer.named_parameters() if p.grad is None or not p.grad.any()]
    assert cold == []
    fwd_only = Counters()
    fwd_only.enable("kda", counter="kda_bwd")
    with torch.autocast("cuda", dtype=torch.bfloat16), monkeypatch.context() as m:
        m.setenv("DORMOUSE_KDA_FLA", "1")
        layer(torch.randn(2, 64, 64, device="cuda"), fwd_only)
    with pytest.raises(MissingReport):
        fwd_only.check()


@CUDA
@pytest.mark.xfail(
    strict=True,
    reason="REFUTED on-card: FLA fwd+bwd 13.48 ms/layer at b2 x s1024, above the "
    "8 ms bar (eager 66.80 ms). The 480 ms/9-layer saving is real but not at the gate.",
)
def test_live_layer_cosine_and_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live b2 x s1024 layer: cosine to eager and interleaved fwd+bwd ms/layer."""
    sys_path = Path(__file__).resolve().parents[1]
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from tools.bench_kda import LIVE, kernel_args, timed

    torch.manual_seed(0)
    layer = KDA(**LIVE, device="cuda")
    x = torch.randn(2, 1024, LIVE["d_model"], device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        args = kernel_args(layer, x)
    with monkeypatch.context() as m:
        m.setenv("DORMOUSE_KDA_FLA", "0")
        eager, eager_state = layer._chunkwise(*args)
        m.setenv("DORMOUSE_KDA_FLA", "1")
        fla, fla_state = layer._chunkwise(*args)
    cos = float(
        torch.nn.functional.cosine_similarity(
            eager.detach().float().flatten(), fla.detach().float().flatten(), dim=0
        )
    )
    print(
        f"\nlive-layer write cosine {cos:.8f}; state max diff "
        f"{float((fla_state - eager_state).detach().abs().max()):.3e}"
    )
    assert cos >= 0.999

    def measure(fla_on: bool) -> float:
        with monkeypatch.context() as m:
            m.setenv("DORMOUSE_KDA_FLA", "1" if fla_on else "0")

            def step() -> None:
                for p in layer.parameters():
                    p.grad = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    write, state = layer._chunkwise(*kernel_args(layer, x))
                    (write.float().square().mean() + state.float().square().mean()).backward()

            return timed(step, warmup=2, runs=5)

    eager_ms, fla_ms = measure(False), measure(True)
    print(f"live-layer fwd+bwd ms: eager {eager_ms:.2f}, FLA {fla_ms:.2f}")
    assert fla_ms <= MS_PER_LAYER, (eager_ms, fla_ms)


@CUDA
@pytest.mark.xfail(
    strict=True,
    reason="REFUTED on-card: seed 4 paired dBPB +0.01692 at b2 x s768 / lr 0.01 "
    ">(0.01); seed 5's eager arm hits the repo's pre-existing NaN firewall, so the "
    "3-seed upper bound is not reachable at this HEAD. FLA stays opt-in.",
)
def test_fullwidth_three_seed_delta_bpb_gate() -> None:
    """3 seeds x 200 steps at b2 x s768: one-sided t upper bound <= 0.01 BPB."""
    deltas, mean, upper = _paired(SEEDS_3)
    print("\nseed | dBPB(fla-eager) | steps | shape")
    for seed, delta in zip(SEEDS_3, deltas, strict=True):
        print(f"{seed:4} | {delta:+.5f}       | {STEPS} | b2 x s768")
    print(f"mean {mean:+.5f}, one-sided 95% t upper bound {upper:.5f}, gate {GATE}")
    assert upper <= GATE, (deltas, mean, upper)
