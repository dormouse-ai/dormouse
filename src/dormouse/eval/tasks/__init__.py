"""The deterministic suite: one call, every task, same examples every time.

`suite(seed)` is the whole contract. There is no task selection, no sampling and no
per-task option: a checkpoint that cannot run all five is a checkpoint whose table
has a hole in it, and a hole is not a result.
"""

from __future__ import annotations

from .base import Example, Generate, Task
from .code import code
from .text import END, arithmetic, format_following, needle, reverse_copy


def suite(seed: int = 0) -> tuple[Task, ...]:
    """Build every task from one seed, in a fixed order."""
    return (
        reverse_copy(seed),
        arithmetic(seed),
        needle(seed),
        format_following(seed),
        code(seed),
    )


__all__ = ["END", "Example", "Generate", "Task", "code", "suite"]
