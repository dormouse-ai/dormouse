"""One process-wide registry: no contract, no active mechanism, no allocation."""

from __future__ import annotations

from dormouse_core.errors import LoudError

from .contract import Contract


class MechanismRegistry:
    """The contracts of the mechanisms this process can enable."""

    def __init__(self) -> None:
        """Start empty; mechanism modules register themselves when imported."""
        self._contracts: dict[str, Contract] = {}

    def register(self, contract: Contract) -> None:
        """Add a contract once; a second, different contract for a name is refused."""
        held = self._contracts.get(contract.name)
        if held is not None:
            if held is contract:
                return  # importing the same module twice is not a second mechanism
            raise LoudError(
                f"two contracts claim the name {contract.name!r}",
                "give the mechanism a distinct name or fix the duplicate registration",
            )
        self._contracts[contract.name] = contract

    def get(self, name: str) -> Contract:
        """Return a contract, or refuse the mechanism that has none."""
        try:
            return self._contracts[name]
        except KeyError as exc:
            raise LoudError(
                f"active mechanism {name!r} has no contract",
                "register a Contract with a counter, a degeneration, an external "
                "anchor and invariants before enabling it — an unproven mechanism "
                "must not allocate",
            ) from exc

    def require(self, *names: str) -> None:
        """Refuse before allocation if any named mechanism lacks a proof."""
        for name in names:
            self.get(name)

    def all(self) -> tuple[Contract, ...]:
        """Return every registered contract, sorted by name."""
        return tuple(self._contracts[name] for name in sorted(self._contracts))

    def clear(self) -> None:
        """Forget every registration — only for test isolation."""
        self._contracts.clear()

    def table(self) -> str:
        """Render the proofs as a table for `dorm doctor` and `dorm plan`."""
        header = f"{'mechanism':<12} {'counter':<20} {'anchor':<44} {'invariants':<28} eps"
        rows = [header, "-" * len(header)]
        for contract in self.all():
            invariants = ",".join(inv.name for inv in contract.invariants)
            anchor = f"{contract.anchor.source.value}: {contract.anchor.formula}"
            rows.append(
                f"{contract.name:<12} {contract.counter:<20} {anchor[:44]:<44} "
                f"{invariants[:28]:<28} {contract.epsilon:g}"
            )
        return "\n".join(rows)


REGISTRY = MechanismRegistry()


def register(contract: Contract) -> None:
    """Register a contract in the process-wide registry."""
    REGISTRY.register(contract)


def require(*names: str) -> None:
    """Refuse before allocation if any active mechanism is unproven."""
    REGISTRY.require(*names)


def registry_table() -> str:
    """Return the table of registered mechanisms and their proofs."""
    return REGISTRY.table()
