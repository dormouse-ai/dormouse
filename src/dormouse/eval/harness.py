"""Run a suite against one model and count what passed.

The harness owns exactly one loop and knows exactly one thing about the model:
`generate(prompt, max_tokens, stop)`. A checkpoint wrapper, a stub with a dict of
answers, and a random-weights floor all look the same from here.
"""

from __future__ import annotations

from collections.abc import Iterable

from dormouse_core.errors import LoudError

from .report import Result
from .tasks import Generate, Task


def run(tasks: Iterable[Task], generate: Generate, *, seed: int = 0) -> list[Result]:
    """Score every task; each row carries the number of examples behind it."""
    results: list[Result] = []
    for task in tasks:
        if not task.examples:
            raise LoudError(
                f"{task.name}: no examples",
                "a task that scores nothing is not a task",
            )
        passed = 0
        for example in task.examples:
            output = generate(example.prompt, task.max_tokens, task.stop)
            if task.check(example, output):
                passed += 1
        results.append(
            Result(
                task=task.name,
                metric=task.metric,
                score=passed / len(task.examples),
                n=len(task.examples),
                seed=seed,
            )
        )
    return results
