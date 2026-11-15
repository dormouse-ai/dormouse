"""One checkpoint's table: task, metric, score, n, seed — and BPB with its window.

The row count is not optional. A three-example task that scores 1.000 and a
six-hundred-example task that scores 0.990 are not the same result, and a table
that hides `n` lets the first masquerade as the second.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dormouse_core.errors import LoudError

from dormouse.metrics import BitsPerByte


@dataclass(frozen=True)
class Result:
    """One row: a task's score, on how many examples, from which seed."""

    task: str
    metric: str
    score: float
    n: int
    seed: int

    def __post_init__(self) -> None:
        """Refuse a row that cannot be read as a result."""
        if self.n <= 0:
            raise LoudError(
                f"{self.task}: n={self.n}",
                "score at least one example, or report no row at all",
            )
        if not 0.0 <= self.score <= 1.0:
            raise LoudError(
                f"{self.task}: score {self.score!r} is not a rate",
                "report passes / n, not a sum",
            )


COLUMNS = ("task", "metric", "score", "n", "seed")


def _row(result: Result) -> tuple[str, ...]:
    """Render one result as table cells."""
    return (result.task, result.metric, f"{result.score:.3f}", str(result.n), str(result.seed))


def render(results: Sequence[Result], bpb: BitsPerByte | None = None) -> str:
    """Render the table as text, with the BPB's byte window on its own line."""
    if not results:
        raise LoudError(
            "no results to report",
            "run the suite; an empty table is not a result",
        )
    rows: Iterable[tuple[str, ...]] = [COLUMNS, *(_row(result) for result in results)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    lines = [
        "  ".join(
            cell.ljust(widths[i]) if i < 2 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        ).rstrip()
        for row in rows
    ]
    if bpb is not None:
        lines.append("")
        lines.append(f"bits per byte: {bpb}")
    return "\n".join(lines)
