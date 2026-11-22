"""Price configs/smart_fast.toml and gate its lever manifest without a card.

    uv run --no-sync python tools/smart_fast_bench.py
    uv run --no-sync python tools/smart_fast_bench.py --config configs/fastchat_400m.toml

The stack is built on the meta device, so this runs in CPU CI. Two bills are printed:
the headline 6*P_stored*D the two-hour envelope was set with, and the engine's own
6*P_active*D. The headline is the gate; the engine bill says how fast the routed
model really is.

The lever manifest below is the contract with research/night/iq-04-synthesis.md:
only SURVIVED levers may be enabled, GATED levers are off in this mixed-source run,
and every REFUTED lever is hard-off. `iq_gate()` is the sole quality gate: random=0
and oracle=1 on the smoke suite, with the blind-judge counter proving it ran.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dormouse_core.counters import Counters  # noqa: E402

from dormouse.eval.iq import MockJudge, build_suite, evaluate  # noqa: E402
from dormouse.eval.model_adapter import KnownAdapter, RandomAdapter  # noqa: E402
from dormouse.model.assemble import assemble  # noqa: E402
from dormouse.model.spec import StackSpec  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "smart_fast.toml"
#: Measured dense peak of one 5070 Ti (tools/decisive_bench.py:CARD_PEAK_TFLOPS).
CARD_PEAK_TFLOPS = 92.0
#: Whole-step MFU locked at commit c553a57.
MFU = 0.3992
#: The envelope this recipe is sized for.
BUDGET_HOURS = 2.0
#: The FLOP budget that follows from peak * MFU * the envelope.
BUDGET_FLOPS = CARD_PEAK_TFLOPS * MFU * 1e12 * BUDGET_HOURS * 3600
#: Every lever the night research REFUTEd. None may ever be enabled here.
REFUTED = frozenset({"xmod", "fff", "loop_l", "engram", "mtp", "jepa", "fsz", "sc_k5"})


@dataclass(frozen=True)
class Lever:
    """One lever, its research status, whether this run enables it, and its gate."""

    name: str
    status: str  # "survived" | "gated" | "refuted"
    enabled: bool
    counter: str  # named evidence gate; iq_judge_calls is a real runtime counter
    gate: str


LEVERS: tuple[Lever, ...] = (
    Lever(
        "synth_static_25",
        "survived",
        True,
        "synth_share",
        "25% static corpus, 0 FLOP, never x10; delta-test, 0 shared SHA, abstention>=0.6",
    ),
    Lever(
        "fp16_ns",
        "survived",
        True,
        "ns_cos",
        "optimizer only; cos>=0.9999 on 8 buckets, 3.06x chain (commit 97630f4)",
    ),
    Lever(
        "graph_epilogue",
        "survived",
        True,
        "whole_step_mfu",
        "torch_compile=true plus `dorm train --graph`; MFU 39.92% (commit c553a57)",
    ),
    Lever(
        "active_68.5m",
        "survived",
        True,
        "assemble_report",
        "401M stored / 68.5M active, the honest topology; not 400M active / 3.8B stored",
    ),
    Lever(
        "lns_replace_init",
        "survived",
        True,
        "bpb_regression",
        "REPLACE write_std=0.02/sqrt(2L); 2.714696 +/- 0.10; never combine with scaled init",
    ),
    Lever(
        "sft_pika_30k",
        "survived",
        True,
        "sft_valid",
        "30k multi-turn PiKa, assistant-only loss, re-init chat embeddings, 0 leaks",
    ),
    Lever(
        "verifier_1rollout",
        "survived",
        True,
        "rollout_count",
        "1 rollout per prompt, verifier on held-out; k>1 is red",
    ),
    Lever(
        "grpo_outcome_only",
        "survived",
        True,
        "rlvr_gate",
        "outcome-only reward; dense-MC and sc_k5 are red",
    ),
    Lever(
        "iq_smoke",
        "survived",
        True,
        "iq_judge_calls",
        "random=0 / oracle=1 on smoke; the sole quality gate",
    ),
    Lever(
        "tst_r_0.3",
        "gated",
        False,
        "tst_bwd",
        "conflicts with mix_plan (COMPATIBILITY.md:38); single-source run only, 3 seeds",
    ),
    Lever("xmod", "refuted", False, "", "local 0.799 FLOP, dBPB +6..+14, 1.022x slower"),
    Lever("fff", "refuted", False, "", "MoE integration unverified; conflicts with CUDA graph"),
    Lever("loop_l", "refuted", False, "", "each loop adds FLOP and worsens BPB"),
    Lever("engram", "refuted", False, "", "0 downstream at 0.5-4B; flag is unwired"),
    Lever("mtp", "refuted", False, "", "+26% FLOP for +0.05pp at 1.2B; worse at our scale"),
    Lever("jepa", "refuted", False, "", "+28% step; EBR is 345x FLOP"),
    Lever("fsz", "refuted", False, "", "no surviving evidence in iq-04 synthesis; off"),
    Lever("sc_k5", "refuted", False, "", "+400% FLOP, n_eff about 2.1; 1 rollout optimal"),
)


def load(config: Path) -> tuple[StackSpec, str]:
    """Parse a config with the stack's own validation and return its text."""
    text = config.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    spec = StackSpec.from_mapping(raw, source=str(config))
    return spec, text


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


def problems(spec: StackSpec, text: str) -> list[str]:
    """Return every reason this config is not the smart-fast run; empty means green."""
    bad: list[str] = []
    if spec.objective != "ntp":
        bad.append("objective != ntp: TST is gated off in a mixed-source run")
    if not spec.torch_compile:
        bad.append("torch_compile=false: graph+epilogue wiring is missing")
    for lever in LEVERS:
        if lever.status == "refuted" and lever.enabled:
            bad.append(f"REFUTED lever enabled: {lever.name}")
    for name in REFUTED:
        if f"{name} = true" in text or f"{name}=true" in text:
            bad.append(f"REFUTED flag set true in config: {name}")
    if price(spec)["headline_flops"] > BUDGET_FLOPS:
        bad.append("headline bill exceeds the two-hour envelope")
    return bad


def iq_gate() -> dict[str, int]:
    """Run the smoke suite at both anchors; return the floor, ceiling and counter."""
    suite = build_suite("smoke", seed=0)
    counters = Counters()
    counters.enable("iq_smoke", counter="iq_judge_calls", why="sole quality gate")
    floor = evaluate(suite.tasks, RandomAdapter(0), MockJudge())
    ceiling = evaluate(suite.tasks, KnownAdapter(suite.tasks), MockJudge(), counters=counters)
    counters.check()
    return {
        "n": len(suite),
        "floor": int(sum(row.score for row in floor)),
        "ceiling": int(sum(row.score for row in ceiling)),
        "counter": counters.snapshot()["iq_judge_calls"],
    }


def main(argv: list[str] | None = None) -> int:
    """Print the lever table and the price; exit 1 on a gate failure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec, text = load(args.config)
    bill = price(spec)
    print(f"config    {args.config.name}")
    print("lever                     status     enabled  gate")
    for lever in LEVERS:
        print(f"  {lever.name:<23} {lever.status:<10} {lever.enabled!s:<7}  {lever.gate}")
    print(f"P         stored={bill['stored']:,.0f}  active={bill['active']:,.0f}")
    print(f"D         {bill['tokens']:,.0f} tokens")
    print(
        f"headline  6*P_stored*D = {bill['headline_flops']:.4e} FLOP -> "
        f"{bill['headline_hours']:.3f} h  (budget {BUDGET_FLOPS:.4e} FLOP / "
        f"{BUDGET_HOURS:g} h)"
    )
    print(
        f"engine    6*P_active*D = {bill['engine_flops']:.4e} FLOP -> {bill['engine_hours']:.3f} h"
    )
    iq = iq_gate()
    print(
        f"iq_smoke  n={iq['n']}  random={iq['floor']}  oracle={iq['ceiling']}  "
        f"counter={iq['counter']}"
    )
    bad = problems(spec, text)
    if iq["floor"] != 0 or iq["ceiling"] != iq["n"] or iq["counter"] != iq["n"]:
        bad.append("IQ smoke floor/ceiling/counter gate failed")
    if bad:
        for reason in bad:
            print(f"REFUSED: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
