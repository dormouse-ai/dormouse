"""External numpy-fp64 polar anchor for the opt-in fp16 Newton–Schulz chain.

The shipped fp32 path keeps its own self-consistency anchors untouched
(`tests/test_muon_foreach.py`: ortho <= 1e-5, direction <= 1e-6). A 10-bit fp16
mantissa puts the chain's arithmetic floor around 1e-3, so 1e-5 is unreachable by
construction and is not this rung's gate. The fp16 rung is gated against the external
numpy fp64 SVD polar (cos >= 0.9999, ortho <= 2e-2, rel <= 2e-2) on all eight live
buckets, plus the wiring that makes the flag reach the optimizer from the CLI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError, MissingReport

from dormouse.cli import main
from dormouse.train import entry, runconfig
from dormouse.train.muon import NS_FP16_COUNTER, Muon, _polar_many, _require_foreach, polar
from dormouse.train.optim import MixOptimizer
from dormouse.train.step import commit_step

#: Six CUDA-probe buckets (tools/bench_muon_foreach.py) plus the two TSCT expert shapes.
BUCKETS: tuple[tuple[int, int], ...] = (
    (64, 832),
    (832, 2048),
    (2048, 832),
    (2, 4),
    (4, 4),
    (832, 64),  # tall: transpose arm
    (256, 6144),  # 65 TSCT experts
    (8192, 6144),  # dominant FLOP bucket
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: The 8192x6144 fp16 chain + fp64 numpy SVD takes minutes on CPU; skip that ONE
#: bucket loudly on a card-less CI run. The card run is the 8/8 gate.
_PARAMS = [
    pytest.param(
        (rows, cols),
        id=f"{rows}x{cols}",
        marks=pytest.mark.skipif(
            DEVICE == "cpu" and (rows, cols) == (8192, 6144),
            reason="8192x6144 fp16 chain + fp64 numpy SVD is minutes on CPU; run on-card",
        ),
    )
    for rows, cols in BUCKETS
]

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.toml"


def _conditioned(shape: tuple[int, int], seed: int, device: str = DEVICE) -> torch.Tensor:
    """A matrix with its spectrum in the quintic's basin (the live-momentum regime)."""
    torch.manual_seed(seed)
    left, _, right = torch.linalg.svd(torch.randn(*shape, device=device), full_matrices=False)
    spectrum = torch.linspace(0.8, 1.2, min(shape), device=device)
    return left @ torch.diag(spectrum) @ right


def _fp64_polar(x: torch.Tensor) -> np.ndarray:
    """The external polar factor U Vᵀ from numpy's fp64 SVD — not a torch program."""
    u, _, vt = np.linalg.svd(x.detach().cpu().numpy().astype(np.float64), full_matrices=False)
    return u @ vt


def _cosine(got: np.ndarray, ref: np.ndarray) -> float:
    return float((got.flatten() @ ref.flatten()) / (np.linalg.norm(got) * np.linalg.norm(ref)))


def _orthogonality(q: np.ndarray) -> float:
    """‖QQᵀ−I‖∞ on the short side; the transpose arm makes Q wide, not square."""
    gram = q @ q.T if q.shape[0] <= q.shape[1] else q.T @ q
    return float(np.abs(gram - np.eye(gram.shape[0])).max())


@pytest.mark.parametrize("shape", _PARAMS)
def test_fp16_chain_against_numpy_fp64_polar(shape: tuple[int, int]) -> None:
    """cos >= 0.9999 on 8/8 buckets; ortho <= 2e-2 and rel <= 2e-2 to the fp64 polar."""
    x = _conditioned(shape, seed=7)
    ref = _fp64_polar(x)
    fp32 = _polar_many([x])[0].detach().cpu().numpy().astype(np.float64)
    fp16 = _polar_many([x], ns_dtype="fp16")[0].detach().cpu().numpy().astype(np.float64)
    cos32, cos16 = _cosine(fp32, ref), _cosine(fp16, ref)
    ortho = _orthogonality(fp16)
    rel = float(np.linalg.norm(fp16 - ref) / np.linalg.norm(ref))
    print(
        f"\n{shape[0]}x{shape[1]}: fp32 cos={cos32:.7f}  fp16 cos={cos16:.7f}  "
        f"ortho={ortho:.3e}  rel={rel:.3e}"
    )
    assert cos32 >= 0.9999, (shape, cos32)
    assert cos16 >= 0.9999, (shape, cos16)
    assert ortho <= 2e-2, (shape, ortho)
    assert rel <= 2e-2, (shape, rel)
    assert bool(np.isfinite(fp16).all()), shape


def test_heavy_spectrum_fp16_delta_stays_within_two_percent_of_fp32() -> None:
    """Where five steps cannot orthogonalise, only the fp16-vs-fp32 delta is meaningful."""
    torch.manual_seed(11)
    x = torch.randn(256, 6144) * torch.logspace(0, -2, 6144)
    fp32 = _polar_many([x])[0]
    fp16 = _polar_many([x], ns_dtype="fp16")[0]
    rel = float((fp16 - fp32).norm() / fp32.norm())
    assert rel <= 2e-2, rel
    assert bool(torch.isfinite(fp16).all())


def test_fp32_paths_are_bitwise_unchanged_by_the_dtype_argument() -> None:
    """`fp32` is the shipped program: the extra argument must be a no-op on it."""
    torch.manual_seed(13)
    batch = [torch.randn(16, 32), torch.randn(32, 16)]
    default = _polar_many(batch)
    explicit = _polar_many(batch, ns_dtype="fp32")
    assert all(torch.equal(a, b) for a, b in zip(default, explicit, strict=True))
    assert all(torch.equal(got, polar(x)) for x, got in zip(batch, default, strict=True))


def test_fp16_counter_fires_and_a_silent_run_is_refused() -> None:
    """Flag on but no fp16 chain is a failed run, not a quiet fp32 step."""
    torch.manual_seed(17)
    params = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(3)]
    for param in params:
        param.grad = torch.randn_like(param)
    opt = Muon([{"params": params, "head_split": None}], ns_dtype="fp16", grouped=False)
    counters = Counters()
    opt.step(counters=counters)
    assert counters.counts[NS_FP16_COUNTER] == len(params)
    assert all(param.dtype == torch.float32 for param in params)
    counters.reset()
    with (
        patch.object(counters, "count", lambda _key, _n=1: None),
        pytest.raises(MissingReport, match="ns_fp16"),
    ):
        opt.step(counters=counters)


def test_dtype_literals_are_closed() -> None:
    """bf16 is not a chain dtype here."""
    torch.manual_seed(19)
    matrix = torch.randn(4, 4)
    with pytest.raises(LoudError, match="ns_dtype"):
        _polar_many([matrix], ns_dtype="bf16")
    with pytest.raises(LoudError, match="fp32 work tensors"):
        _polar_many([matrix.double()], ns_dtype="fp16")
    with pytest.raises(LoudError, match="fp16 work tensors"):
        _require_foreach([matrix], work_dtype=torch.float16)
    with pytest.raises(LoudError, match="ns_dtype"):
        Muon([[torch.nn.Parameter(matrix)]], ns_dtype="bf16")

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.gain = torch.nn.Parameter(torch.ones(8))  # 1D: AdamW's arm

    net = Net()
    with pytest.raises(LoudError, match="muon_ns_dtype"):
        MixOptimizer(net, muon_ns_dtype="bf16")
    assert MixOptimizer(net, muon_ns_dtype="fp16").muon.ns_dtype == "fp16"


def test_fp16_direction_is_a_polar_direction_not_a_scaled_input() -> None:
    """The quintic must actually run: the fp16 output is orthogonal to a tolerance."""
    torch.manual_seed(23)
    x = _conditioned((64, 128), seed=23)
    q = polar(x, ns_dtype="fp16")
    assert q.dtype == torch.float32
    gram = q @ q.T
    eye = torch.eye(gram.shape[0], device=gram.device)
    assert float((gram - eye).abs().max()) <= 2e-2


def test_the_step_fills_the_fp16_counter_before_check() -> None:
    """The counter lives inside `optimizer.step`: the seam must pass counters to it."""
    torch.manual_seed(29)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.gain = torch.nn.Parameter(torch.ones(8))

    model = Net()
    optimizer = MixOptimizer(model, muon_ns_dtype="fp16")
    model.proj.weight.grad = torch.randn_like(model.proj.weight)
    counters = Counters()
    result = commit_step(model, optimizer, torch.tensor(1.0), counters=counters)
    assert not result.skipped
    assert counters.counts[NS_FP16_COUNTER] == 1


def test_config_key_and_cli_flag_reach_the_muon_arm(tmp_path: Path) -> None:
    """`muon_ns_dtype` survives config.json and the CLI override reaches MixOptimizer."""
    config = tmp_path / "fp16.toml"
    head, table, tail = SMOKE.read_text(encoding="utf-8").partition("[[source]]")
    config.write_text(head + '\nmuon_ns_dtype = "fp16"\n' + table + tail)
    recipe = runconfig.RunRecipe.read(config)
    assert recipe.muon_ns_dtype == "fp16"
    canonical = recipe.canonical()
    assert canonical["run"]["muon_ns_dtype"] == "fp16"
    assert runconfig.RunRecipe.from_canonical(canonical, path=config).muon_ns_dtype == "fp16"
    from dormouse.model.assemble import assemble

    model = assemble(recipe.spec, device="cpu", smoke=True).model
    optimizer, _ = entry._optimizer(model, recipe, recipe.loop(20))
    assert optimizer.muon.ns_dtype == "fp16"

    with patch.object(entry, "train", return_value={"plan_flops": 1}) as train_mock:
        assert main(["train", str(config), "--smoke", "--muon-ns-dtype", "fp16"]) == 0
        assert train_mock.call_args.kwargs["muon_ns_dtype"] == "fp16"


def test_delta_bpb_over_three_seeds_is_within_001() -> None:
    """Paired 3-seed A/B: the fp16 NS chain versus the shipped fp32 chain, byte-unit BPB."""
    from dormouse.metrics import from_nats

    d, vocab, steps, batch = 64, 256, 16, 64

    def run(ns_dtype: str, seed: int) -> float:
        torch.manual_seed(seed)
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(vocab, d)  # adamw arm
        model.mid = torch.nn.Linear(d, d, bias=False)  # muon arm
        model.head = torch.nn.Linear(d, vocab, bias=False)  # muon arm
        opt = MixOptimizer(model, muon_ns_dtype=ns_dtype)
        gen = torch.Generator().manual_seed(seed + 9000)
        ids = torch.randint(0, vocab, (steps + 1, batch), generator=gen)
        nats = 0.0
        measured = 0
        for step in range(steps):
            logits = model.head(model.mid(model.embed(ids[step])))
            loss = torch.nn.functional.cross_entropy(logits.float(), ids[step + 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step >= steps // 2:
                nats += float(loss.detach())
                measured += 1
        return from_nats(nats / measured, measured * batch).bpb

    deltas = [abs(run("fp16", seed) - run("fp32", seed)) for seed in (1, 2, 3)]
    delta = sum(deltas) / len(deltas)
    print(f"\nDelta BPB over 3 seeds: {delta:.5f} (gate 0.01), per seed {deltas}")
    assert delta <= 0.01, (delta, deltas)
