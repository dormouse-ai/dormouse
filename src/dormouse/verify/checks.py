"""Run every registered proof on CPU fixtures; a false proof is a failed check.

`check_all` is the single entry point for the CI gate and for the pytest test
that puts these lines under the coverage floor. It runs two seeds per mechanism:
one to compare against the anchor, a second to prove the anchor is a function of
the fixture rather than a golden remembered from the first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from dormouse_core.counters import Counters

from .contract import Contract
from .errors import ProofError
from .fixture import Fixture
from .fixtures import dsa_fixture, kda_fixture, tsct_fixture
from .registry import REGISTRY

SEEDS = (0, 1)

FIXTURES: dict[str, Callable[[int], Fixture]] = {
    "kda": kda_fixture,
    "dsa": dsa_fixture,
    "tsct": tsct_fixture,
}


@dataclass(frozen=True)
class Result:
    """One mechanism's verdict, with the largest errors that decided it."""

    name: str
    degeneration: float
    anchor: float
    invariants: tuple[str, ...]
    ok: bool
    reason: str = ""


def _largest(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest absolute difference, or infinity for incomparable shapes."""
    if got.shape != want.shape:
        return float("inf")
    return float((got.detach().double() - want.detach().double()).abs().max())


def enable(contract: Contract, counters: Counters) -> None:
    """Bind a contract's counter into `Counters`; `check()` then refuses silence."""
    counters.enable(
        contract.name,
        counter=contract.counter,
        why=f"{contract.name} must report {contract.counter} whenever it is enabled",
    )


def check_contract(contract: Contract, fixtures: Sequence[Fixture] | None = None) -> Result:
    """Run the degeneration, the anchor and every invariant; no model is mocked."""
    if fixtures is None:
        maker = FIXTURES.get(contract.name)
        if maker is None:
            raise ProofError(
                f"mechanism {contract.name!r} has no CPU fixture",
                "add a fixture builder to verify.fixtures so the gate can run its proofs",
            )
        fixtures = tuple(maker(seed) for seed in SEEDS)
    if not fixtures:
        raise ProofError(
            f"mechanism {contract.name!r} has no fixtures to check",
            "pass at least one seeded Fixture",
        )
    worst_degeneration = 0.0
    worst_anchor = 0.0
    names: list[str] = []
    anchors: list[torch.Tensor] = []
    failures: list[str] = []
    for fixture in fixtures:
        on, off = contract.degeneration(fixture)
        degener = _largest(on, off)
        reference = contract.anchor.evaluate(fixture)
        anchor_error = _largest(on, reference)
        worst_degeneration = max(worst_degeneration, degener)
        worst_anchor = max(worst_anchor, anchor_error)
        anchors.append(reference)
        if degener > contract.epsilon:
            failures.append(f"degeneration off by {degener:.3g}")
        if anchor_error > contract.epsilon:
            failures.append(f"anchor off by {anchor_error:.3g}")
        for invariant in contract.invariants:
            names.append(invariant.name)
            try:
                invariant.check(on, fixture)
            except ProofError as exc:
                failures.append(str(exc))
    if len(anchors) > 1 and all(
        _largest(anchors[0], other) <= contract.epsilon for other in anchors[1:]
    ):
        raise ProofError(
            f"anchor of {contract.name!r} does not depend on the fixture",
            "it is a captured golden, not an external reference: compute it from "
            "the fixture with a paper formula",
        )
    return Result(
        contract.name,
        worst_degeneration,
        worst_anchor,
        tuple(dict.fromkeys(names)),
        not failures,
        "; ".join(failures),
    )


def check_all() -> tuple[Result, ...]:
    """Run the gate over every registered mechanism, collecting failures."""
    results: list[Result] = []
    for contract in REGISTRY.all():
        try:
            results.append(check_contract(contract))
        except ProofError as exc:
            results.append(Result(contract.name, 0.0, 0.0, (), False, str(exc)))
    return tuple(results)


def check_ran(contract: Contract, counters: Counters) -> None:
    """Refuse a step whose enabled mechanism never moved its counter."""
    enable(contract, counters)
    counters.check()


def render(results: Sequence[Result]) -> str:
    """Print the gate's table: errors beside each named proof."""
    header = f"{'mechanism':<12} {'degeneration':>14} {'anchor':>12} {'invariants':<32} verdict"
    lines = [header, "-" * len(header)]
    for result in results:
        invariants = ",".join(result.invariants)
        lines.append(
            f"{result.name:<12} {result.degeneration:>14.3g} {result.anchor:>12.3g} "
            f"{invariants[:32]:<32} {'PASS' if result.ok else 'FAIL'}"
        )
        if not result.ok:
            lines.append(f"  {result.reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Gate entry point: zero only when every registered proof passes."""
    results = check_all()
    print(render(results))
    return 0 if results and all(result.ok for result in results) else 1
