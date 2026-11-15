"""ms/step for the four FFN arms at the live micro-shape.

`b24 x s1024, d=832, f=2048` is one micro-batch of the recipe; the four arms are
the plain SwiGLU, the plain SiTU (both three matrices), LA and bi-MoA over the
paper's five-entry dictionary, which is the default. One step is a forward, a
scalar loss and a backward — the training step's FFN share, with no optimizer.

    python scripts/bench_ffn.py                 # CPU is the expectation here
    python scripts/bench_ffn.py --device cuda   # when the card is free

The card is usually busy with another lane's run; the script says which device it
measured, so a CPU number is never quoted as a GPU one.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from dormouse.model.ffn_spec import FFNSpec

ARMS: tuple[tuple[str, str], ...] = (
    ("silu", "silu"),
    ("situ", "situ"),
    ("la", "la"),
    ("moa", "moa"),
)


def _step(ffn: torch.nn.Module, x: torch.Tensor) -> None:
    """One forward + backward, the training step's FFN work."""
    ffn.zero_grad(set_to_none=True)
    ffn(x).sum().backward()


def _median_ms(ffn: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    """Median wall time of `iters` steps after `warmup` untimed ones."""
    for _ in range(warmup):
        _step(ffn, x)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _step(ffn, x)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> None:
    """Build every arm at the same shape and print the per-step table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=832)
    parser.add_argument("--d-ffn-gelu", type=int, default=3072)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch reports no CUDA")
    x = torch.randn(args.batch, args.seq, args.d_model, device=device)
    base = 3 * args.d_model * (2 * args.d_ffn_gelu // 3)

    print(
        f"device={device.type} threads={torch.get_num_threads()} "
        f"b={args.batch} s={args.seq} d={args.d_model} f={2 * args.d_ffn_gelu // 3}"
    )
    print(f"{'arm':6} {'params':>10} {'added':>6} {'ms/step':>9} {'tok/s':>10}")
    for label, act in ARMS:
        spec = FFNSpec(
            args.d_model,
            2 * args.d_ffn_gelu // 3,
            act=act,
            d_ffn_gelu=args.d_ffn_gelu,
        )
        ffn = spec.build(device=device)
        params = sum(p.numel() for p in ffn.parameters())
        ms = _median_ms(ffn, x, args.warmup, args.iters)
        tokens = args.batch * args.seq
        print(
            f"{label:6} {params:>10,} {params - base:>6} {ms:>9.2f} {tokens / (ms / 1e3):>10,.0f}"
        )
        del ffn


if __name__ == "__main__":
    main()
