"""The two things every eval task shares: an example and the way it is scored.

A task is data plus one check. There is no task base class and no registry: a task
is a frozen value the suite builds from a seed, and the harness knows only
`generate(prompt, max_tokens, stop)`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__concepts__ = ("Example", "Task")

#: The only model interface the harness may use. Anything callable this way is a
#: model: a real checkpoint, a stub in a test, or the floor a random init sets.
type Generate = Callable[[str, int, Sequence[str]], str]


@dataclass(frozen=True)
class Example:
    """One prompt and the answer a correct model must produce."""

    prompt: str
    expected: str


@dataclass(frozen=True)
class Task:
    """A deterministic task: fixed examples, one metric, one check."""

    name: str
    metric: str
    examples: tuple[Example, ...]
    check: Callable[[Example, str], bool]
    max_tokens: int = 64
    stop: tuple[str, ...] = ()
