"""F2/F3/F4: paired IQ-smoke A/B of the S6 loop graft on a real checkpoint.

    PYTHONPATH=src taskset -c 4 python tools/demo_loop_eval.py \
      --checkpoint runs/demo_200m/export/model.pt

Scores IQ_smoke_base and IQ_smoke_loop on 3 seeds, counts per-task wins/regressions,
measures CUDA-event min-of-7 decode latency, and refuses to call a non-green result a
quality win. The wrapper adds zero training FLOP and leaves state_dict bitwise identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dormouse.eval.iq import MockJudge, build_suite, evaluate  # noqa: E402
from dormouse.eval.model_adapter import CheckpointAdapter  # noqa: E402
from tools.demo_loops import graft, polynomial_weights  # noqa: E402


def score(adapter: object, seed: int) -> dict[str, float | int]:
    """Return the per-task scores of one smoke suite."""
    suite = build_suite("smoke", seed=seed)
    rows = evaluate(suite.tasks, adapter, MockJudge())
    return {row.task: row.score for row in rows}


def decode_latency_ms(model: torch.nn.Module, ids: torch.Tensor, reps: int = 7) -> float:
    """Min CUDA-event time of one greedy decode step, after three warm-ups."""
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            model(ids)
        if ids.is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            if ids.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(ids)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time

                t0 = time.perf_counter()
                model(ids)
                times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def main() -> int:
    """Score the paired A/B, measure latency, and print the gate verdict."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--window", type=int, nargs=2, default=[7, 10])
    p.add_argument("--out", type=Path, default=Path("/tmp/opencode/demo_loop_results.json"))
    args = p.parse_args()

    base = CheckpointAdapter(args.checkpoint)
    model = base._talker.model
    state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    results: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "seeds": args.seeds,
        "K": args.K,
        "beta": args.beta,
        "window": tuple(args.window),
    }
    per_seed = {}
    for seed in args.seeds:
        a = score(base, seed)
        graft(model, window=tuple(args.window), K=args.K, beta=args.beta, mode="layer")
        b = score(base, seed)
        # Restore the plain forward by rebuilding the adapter's talker model reference:
        # graft patched forward only; reloading weights is unnecessary (state is intact).
        from dormouse.train.entry import load_artifact

        _spec, fresh_model = load_artifact(args.checkpoint)
        base._talker.model = fresh_model
        model = fresh_model
        wins = sum(1 for k in a if b[k] > a[k])
        regressions = sum(1 for k in a if b[k] < a[k])
        per_seed[str(seed)] = {
            "IQ_smoke_base": sum(a.values()) / len(a),
            "IQ_smoke_loop": sum(b.values()) / len(b),
            "wins": wins,
            "regressions": regressions,
            "per_task_base": a,
            "per_task_loop": b,
        }
    results["seeds_detail"] = per_seed

    # Latency on the reconstructed base model, then on a grafted copy of it.
    _spec, plain = load_artifact(args.checkpoint)
    ids = torch.randint(
        0,
        _spec.vocab,
        (1, min(128, _spec.seq_len)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    plain = plain.to(ids.device)
    torch.set_num_threads(1)
    t_base = decode_latency_ms(plain, ids)
    graft(plain, window=tuple(args.window), K=args.K, beta=args.beta, mode="layer")
    t_loop = decode_latency_ms(plain, ids)
    results["latency_ms"] = {"base": t_base, "loop": t_loop, "ratio": t_loop / t_base}

    state_after = plain.state_dict()
    results["state_dict_bitwise"] = all(
        torch.equal(state_before[k], state_after[k]) for k in state_before
    )

    total_wins = sum(v["wins"] for v in per_seed.values())
    total_reg = sum(v["regressions"] for v in per_seed.values())
    ratio = results["latency_ms"]["ratio"]
    green = total_wins >= 3 and total_reg == 0 and ratio <= 2.0
    results["gate"] = {
        "wins_ge_3": total_wins >= 3,
        "zero_regressions": total_reg == 0,
        "latency_le_2x": ratio <= 2.0,
        "verdict": "GREEN" if green else "REFUTE",
    }
    # F3/F4 are conditional on F2 green; an honest kill is recorded otherwise.
    results["F3_anchor"] = {
        "status": "not_run" if not green else "pending",
        "rule": "h'=x1+omega(x1-x~); omega sweep, kill at delta<=0",
    }
    results["F4_polynomial"] = {
        "status": "not_run" if not green else "pending",
        "weights_K3": polynomial_weights(3),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=str) + "\n")
    print("baseline|looped|Δtasks|regressions|latency ratio|state bitwise|verdict")
    print(
        f"{sum(v['IQ_smoke_base'] for v in per_seed.values()) / len(per_seed):.3f}|"
        f"{sum(v['IQ_smoke_loop'] for v in per_seed.values()) / len(per_seed):.3f}|"
        f"{total_wins}|{total_reg}|{ratio:.3f}|{results['state_dict_bitwise']}|"
        f"{results['gate']['verdict']}"
    )
    print(f"results: {args.out}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
