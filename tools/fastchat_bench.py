"""Price configs/fastchat_400m.toml without a card.

    uv run --no-sync python tools/fastchat_bench.py
    uv run --no-sync python tools/fastchat_bench.py --config configs/dormouse_1b.toml

The stack is built on the meta device, so this runs in CPU CI. Two bills are printed:
the headline 6*P_stored*D the two-hour envelope was set with, and the engine's own
6*P_active*D (407 MFLOP/token at the shipped widths, measured in decisive_bench.py).
The headline is the gate; the engine bill says how fast the routed model really is.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dormouse.model.assemble import assemble  # noqa: E402
from dormouse.model.spec import StackSpec  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "fastchat_400m.toml"
#: Measured dense peak of one 5070 Ti (tools/decisive_bench.py:CARD_PEAK_TFLOPS).
CARD_PEAK_TFLOPS = 92.0
#: Whole-step MFU locked at commit 6e210b6 (.bulba/memory.md, 2026-11-22).
MFU = 0.3992
#: The envelope this recipe is sized for.
BUDGET_HOURS = 2.0


def load(config: Path) -> tuple[StackSpec, dict[str, object]]:
    """Parse a config with the stack's own validation and return the raw keys too."""
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping(
        {key: value for key, value in raw.items() if key in keys}, source=str(config)
    )
    return spec, raw


def price(spec: StackSpec) -> dict[str, float]:
    """Return both FLOP bills and their hours, from a meta-built parameter report."""
    report = assemble(replace(spec, torch_compile=False), device="meta").report()
    stored = report["stored"]
    active = report["active"]
    tokens = spec.tokens
    effective = CARD_PEAK_TFLOPS * MFU * 1e12
    headline = 6 * stored * tokens
    engine = 6 * active * tokens
    return {
        "stored": float(stored),
        "active": float(active),
        "tokens": float(tokens),
        "headline_flops": float(headline),
        "headline_hours": headline / effective / 3600,
        "engine_flops": float(engine),
        "engine_hours": engine / effective / 3600,
        "effective_tflops": effective / 1e12,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the price table; exit 1 if the headline bill breaks the envelope."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec, _raw = load(args.config)
    bill = price(spec)
    print(f"config    {args.config.name}")
    print(f"P         stored={bill['stored']:,.0f}  active={bill['active']:,.0f}")
    print(f"D         {bill['tokens']:,.0f} tokens")
    print(
        f"card      peak={CARD_PEAK_TFLOPS:g} TFLOP/s  MFU={MFU:.2%}  "
        f"effective={bill['effective_tflops']:.2f} TFLOP/s"
    )
    print(
        f"headline  6*P_stored*D = {bill['headline_flops']:.4e} FLOP -> "
        f"{bill['headline_hours']:.3f} h  (the 2h gate)"
    )
    print(
        f"engine    6*P_active*D = {bill['engine_flops']:.4e} FLOP -> "
        f"{bill['engine_hours']:.3f} h  (routed bank)"
    )
    if bill["headline_hours"] > BUDGET_HOURS:
        print(
            f"REFUSED: {bill['headline_hours']:.3f} h exceeds the {BUDGET_HOURS:g} h envelope",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
