"""Numerical anchors that fail when a formula moves, on CPU only.

    uv run --no-sync python tools/check_anchors.py

Each check recomputes a published equation OUTSIDE the repo's forward — numpy
fp64 for KDA's delta recurrence and an explicit `h // groups` map for GQA, numpy
for the router's sqrt-softplus and quantile balancing — and compares the module's
own numbers to it on hand-set inputs. A change to any formula, even one that a
shape or counter test would not see, moves one of these numbers and fails here.

The grid is deliberately small (3 tokens d=2, two GQA groupings, E in {2, 8, 16}
x topk in {1, 2}) so the whole gate runs in a second on a CPU core. It reuses the
exact anchor functions the tests call, so the gate and the suite cannot drift.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch
from test_gqa_latent import anchor_gqa_group_map
from test_kda import anchor_kda_recurrence
from test_router import ROUTER_CASES, anchor_router_qb, anchor_router_score

EXPERTS = (2, 8, 16)


def checks() -> list[tuple[str, Callable[[], object]]]:
    """Return the name and zero-argument anchor callable of every gate check."""
    items: list[tuple[str, Callable[[], object]]] = [
        ("kda recurrence (numpy fp64, 3 tokens d=2)", anchor_kda_recurrence),
        ("gqa group map (h//groups, hand-set kv)", anchor_gqa_group_map),
    ]
    items += [(f"router score E={e}", lambda e=e: anchor_router_score(e)) for e in EXPERTS]
    items += [
        (f"router qb E={e} topk={k}", lambda e=e, k=k: anchor_router_qb(e, k))
        for e, k in ROUTER_CASES
    ]
    return items


def main(argv: list[str] | None = None) -> int:
    """Run every anchor; exit 1 with each failed formula named on stderr."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="run only checks whose name contains this substring")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    selected = [(name, check) for name, check in checks() if not args.only or args.only in name]
    if not selected:
        parser.error(f"--only {args.only!r} matched no anchor")

    failures: list[str] = []
    for name, check in selected:
        try:
            detail = check()
            print(f"PASS  {name}  {detail}")
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL  {name}: {exc}", file=sys.stderr)
    print(f"\n{len(selected) - len(failures)}/{len(selected)} anchors passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
