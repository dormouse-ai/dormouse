"""KDA's chunk form: the same number as the pairwise Γ, and the milliseconds that proved it.

WHY THIS FILE EXISTS. The chunk form's Γ ratio used to be a `[batch, heads, chunk,
chunk, head_dim]` tensor, multiplied and reduced once per term — measured at b8 x s1024:
164 ms of a 6.1 s step per KDA layer, 3 133 launches, 27 GB of elementwise traffic. It is
now two matmuls per chunk (`(a ⊙ k)·(k/a)ᵀ`), and a faster kernel is not evidence that it
computes the same number. So this prints three things from one run at the live shape:

* **parity**, both forms against a float64 run of the anchor — the write, the final state
  and every parameter gradient, as a relative difference. In bf16 the two differ by more
  than one bf16 ulp, so the pair alone says nothing: the question is which one is closer
  to the arithmetic, and that is the float64 column;
* **speed**: ms for one layer's forward and forward+backward, then the nine layers of a
  step, with the anchor timed in the same process on the same weights.

    flock /tmp/dormouse-gpu.lock .venv-cuda/bin/python tools/bench_kda.py

The anchor is imported from `tests/test_kda.py` rather than restated here: one
transcription of "the arithmetic being replaced" is the point, and a second copy in this
file is a copy that can drift from the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch
from test_kda import _pairwise_chunkwise

from dormouse.model.attn_variants.kda import KDA

#: The live layer, from `configs/dormouse_1b.toml`: nine of these in the step.
LIVE: dict[str, int] = {
    "d_model": 832,
    "n_heads": 8,
    "head_dim": 64,
    "n_kv_heads": 2,
    "c_lat": 512,
    "chunk": 64,
}
LAYERS = 9


def timed(fn, warmup: int = 2, runs: int = 5) -> float:
    """Return the median device milliseconds of `fn()`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    marks = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        marks.append(start.elapsed_time(end))
    return sorted(marks)[len(marks) // 2]


def kernel_args(layer: KDA, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the layer's projections in the kernel's layout, GQA broadcast as `run` does."""
    q, k, v, alpha, beta = layer._project(x)
    groups = layer.n_heads // layer.n_kv_heads
    if groups > 1:
        k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
    return (*[tile.transpose(1, 2) for tile in (q, k, v, alpha)], beta.transpose(1, 2))


def run_form(
    layer: KDA, x: torch.Tensor, *, batched: bool, amp: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One form end to end: projection, chunk form, a scalar loss, every gradient."""
    for parameter in layer.parameters():
        parameter.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        args = kernel_args(layer, x)
        write, state = (
            layer._chunkwise(*args) if batched else _pairwise_chunkwise(*args, chunk=layer.chunk)
        )
        (write.float().square().mean() + state.float().square().mean()).backward()
    grads = {n: p.grad.detach().double() for n, p in layer.named_parameters() if p.grad is not None}
    assert all(bool(torch.isfinite(g).all()) for g in grads.values()), "a gradient is not finite"
    return write.detach().double(), state.detach().double(), grads


def relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Max absolute difference, in units of the reference's own maximum."""
    return float((actual - reference).abs().max() / reference.abs().max().clamp_min(1e-30))


def compare(
    got: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    want: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Relative differences of the write, the state and every gradient."""
    errors = {"write": relative(got[0], want[0]), "state": relative(got[1], want[1])}
    for name, gradient in got[2].items():
        errors[name] = relative(gradient, want[2][name])
    return errors


def main() -> int:
    """Measure the batched form and the pairwise anchor at the live shape."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--fp32", action="store_true", help="also time and check the fp32 path")
    args = parser.parse_args()
    tokens = args.batch * args.seq
    print(f"KDA at the live layer: {LIVE}, b{args.batch} x s{args.seq} = {tokens:,} tokens")
    print(f"a step holds {LAYERS} of them, so the per-layer ms below are what the step pays")

    torch.manual_seed(0)
    layer = KDA(**LIVE, device="cuda")
    x = torch.randn(args.batch, args.seq, LIVE["d_model"], device="cuda")

    truth_layer = KDA(**LIVE, device="cuda").double()
    truth_layer.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
    truth = run_form(truth_layer, x.double(), batched=False, amp=False)
    del truth_layer
    torch.cuda.empty_cache()
    print("\nparity, relative to a float64 run of the anchor (the arithmetic itself):")
    print(f"  {'form':10} {'write':>10} {'state':>10} {'worst gradient':>15}  name")
    for dtype in (torch.bfloat16, torch.float32):
        amp = dtype is torch.bfloat16
        for name, batched in (("batched", True), ("anchor", False)):
            errors = compare(run_form(layer, x, batched=batched, amp=amp), truth)
            worst = max(errors, key=lambda key: errors[key])
            print(
                f"  {name + ' ' + str(dtype)[6:]:10} {errors['write']:10.3e} "
                f"{errors['state']:10.3e} {errors[worst]:15.3e}  {worst}"
            )

    for dtype in (torch.bfloat16, torch.float32) if args.fp32 else (torch.bfloat16,):
        amp = dtype is torch.bfloat16

        print(f"\n--- {dtype} ms per layer (the step holds {LAYERS})")
        print(f"  {'form':10} {'fwd+bwd':>9} {'step ms':>9} {'tok/s':>9} {'kB/s':>7}")
        for name, batched in (("batched", True), ("anchor", False)):

            def step(batched: bool = batched, amp: bool = amp) -> None:
                run_form(layer, x, batched=batched, amp=amp)

            both = timed(step, runs=3)
            print(
                f"  {name:10} {both:9.1f} {both * LAYERS:9.1f} "
                f"{tokens / (both * LAYERS * 1e-3):9,.0f} "
                f"{tokens / (both * LAYERS * 1e-3) / 1000:7.1f}"
            )
            if name == "batched":

                def forward_only(amp: bool = amp) -> None:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        layer._chunkwise(*kernel_args(layer, x))

                print(f"  {'(forward)':10} {timed(forward_only, runs=3):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
