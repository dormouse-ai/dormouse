"""AOT CUBIN GATE — prove a Triton kernel loaded as a raw cubin from C++ matches.

Compiles `dormouse_kernels.engine.fused.glu_kernel` AOT for sm_120 ('cuda:120:32'),
runs the identical kernel through the JIT launch path (golden) and through a
CUDA driver-API C++ harness, then checks bitwise parity and GPU time.

Run (from the worktree root):
  flock -w 2400 /tmp/dormouse-gpu.lock \
    /home/sehaxe/dormouse2/app/.venv-cuda/bin/python tools/cubin_gate/gate.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Triton 3.9 defaults FP fusion ON; pin it so AOT and JIT see the same flags.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.backends.compiler as tbc  # noqa: E402
from dormouse_kernels.engine import fused  # noqa: E402
from torch.nn.functional import silu  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler.compiler import compile as tcompile  # noqa: E402

BLOCK = 1024
WARPS = 4
STAGES = 3


def compile_cubin(build: Path) -> Path:
    """AOT-compile the repo's glu_kernel for 'cuda:120:32' -> .cubin."""
    kernel = fused._build()["glu"]
    target = tbc.GPUTarget("cuda", 120, 32)
    backend = triton.compiler.make_backend(target)
    opts = backend.parse_options({"num_warps": WARPS, "num_stages": STAGES})
    src = ASTSource(
        fn=kernel,
        constexprs={"BLOCK": BLOCK},
        signature={"G": "*fp32", "U": "*fp32", "OUT": "*fp32", "n": "i32", "BLOCK": "constexpr"},
    )
    cc = tcompile(src, target=target, options=opts.__dict__)
    cubin = cc.asm["cubin"]
    assert cubin[:4] == b"\x7fELF", "expected an ELF cubin"
    assert cc.metadata.name == "glu_kernel"
    out = build / "glu_sm120.cubin"
    out.write_bytes(cubin)
    return out


def jit_golden(kernel, g, u, n, iters):
    """Run the same JITFunction the runtime uses; return (output, ms)."""
    out = torch.empty_like(g)
    grid = ((n + BLOCK - 1) // BLOCK,)
    launch = lambda: kernel[grid](g, u, out, n, BLOCK)  # noqa: E731
    launch()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    stop.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(stop) / iters


def main() -> int:
    """Compile, launch both arms, compare, and print the gate table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--harness", type=Path, default=ROOT / "tools/cubin_gate/glu_harness")
    ap.add_argument("--build", type=Path, default=ROOT / "tools/cubin_gate/build")
    args = ap.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    if not args.harness.exists():
        print(f"SKIP: harness not built at {args.harness}")
        return 0

    cubin = compile_cubin(args.build)
    g = torch.rand(
        args.n,
        device="cuda",
        dtype=torch.float32,
        generator=torch.Generator(device="cuda").manual_seed(1234),
    )
    u = torch.rand(
        args.n,
        device="cuda",
        dtype=torch.float32,
        generator=torch.Generator(device="cuda").manual_seed(5678),
    )

    gf, uf = g.reshape(-1), u.reshape(-1)
    (args.build / "g.bin").write_bytes(gf.cpu().numpy().tobytes())
    (args.build / "u.bin").write_bytes(uf.cpu().numpy().tobytes())

    kernel = fused._build()["glu"]
    golden, _ = jit_golden(kernel, gf, uf, args.n, args.iters)
    (args.build / "golden.bin").write_bytes(golden.cpu().numpy().tobytes())

    # Interleave the two arms across rounds; GPU clocks cannot be locked here,
    # so take the min of each arm (contention/clock ramp only adds time).
    triton_times, cpp_times = [], []
    for _ in range(args.rounds):
        _, ms_t = jit_golden(kernel, gf, uf, args.n, args.iters)
        proc = subprocess.run(
            [
                str(args.harness),
                str(cubin),
                str(args.build / "g.bin"),
                str(args.build / "u.bin"),
                str(args.build / "out_cpp.bin"),
                str(args.n),
                str(BLOCK),
                str(WARPS),
                str(args.iters),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            print(proc.stdout, proc.stderr)
            raise SystemExit(f"C++ harness failed: {proc.returncode}")
        ms_c = float(
            next(line.split("ms_cpp=")[1] for line in proc.stdout.splitlines() if "ms_cpp=" in line)
        )
        triton_times.append(ms_t)
        cpp_times.append(ms_c)
    ms_triton, ms_cpp = min(triton_times), min(cpp_times)

    cpp = torch.frombuffer(
        bytearray((args.build / "out_cpp.bin").read_bytes()), dtype=torch.float32
    ).cuda()
    bitwise = torch.equal(cpp, golden)
    ref = silu(gf) * uf
    max_abs = (cpp - ref).abs().max().item()
    win_pct = abs(ms_cpp - ms_triton) / ms_triton * 100.0
    parity_ok = bitwise or max_abs <= 1e-6
    verdict = "PASS" if parity_ok and win_pct <= 2.0 else "REFUTE"

    print(f"cubin: {cubin} ({cubin.stat().st_size} bytes, sm_120)")
    print("kernel | parity | ms_triton | ms_cpp | verdict")
    print(
        f"glu_kernel | {'bitwise' if bitwise else ('1e-6' if parity_ok else 'FAIL')} | "
        f"{ms_triton:.4f} | {ms_cpp:.4f} | {verdict}"
    )
    print(
        f"max_abs_vs_torch={max_abs:.3e}  timing_delta={win_pct:.2f}%  "
        f"(min of {args.rounds} interleaved rounds: triton={[round(x, 4) for x in triton_times]}, "
        f"cpp={[round(x, 4) for x in cpp_times]})"
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
