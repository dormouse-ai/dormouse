"""Model adapters: the harness sees only ``generate(prompt) -> str``.

The core in ``iq.py`` never imports torch or OpenCode. An adapter is the single
place where a real generator is turned into that one call: mocks here are
stdlib-only, and the checkpoint adapter imports the heavy stack lazily so a
plain Python process can still run the smoke suite.
"""

from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Protocol, runtime_checkable

from .iq import Task

__concepts__ = ("CheckpointAdapter", "EchoAdapter", "KnownAdapter", "ModelAdapter", "RandomAdapter")


@runtime_checkable
class ModelAdapter(Protocol):
    """The one call a model must expose to the harness."""

    def generate(self, prompt: str) -> str:
        """Return one answer for one prompt."""
        ...


class KnownAdapter:
    """The deterministic ceiling: return the reference answer for each prompt."""

    def __init__(self, tasks: tuple[Task, ...] | list[Task]) -> None:
        """Index prompts to reference answers once, at construction."""
        self._answers = {task.prompt: task.expected for task in tasks}

    def generate(self, prompt: str) -> str:
        """Return the answer the suite expects for this prompt, else nothing."""
        return self._answers.get(prompt, "")


class EchoAdapter:
    """A floor: echo the prompt, which is never the answer."""

    def generate(self, prompt: str) -> str:
        """Return the prompt unchanged."""
        return prompt


class RandomAdapter:
    """A floor: deterministic random letters, seeded so a run can be replayed."""

    def __init__(self, seed: int = 0) -> None:
        """Hold one seeded generator; two adapters with a seed are identical."""
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> str:
        """Return 24 random ASCII letters, ignoring the prompt."""
        return "".join(self._rng.choice(string.ascii_letters) for _ in range(24))


class CheckpointAdapter:  # pragma: no cover - needs torch and a real artifact
    """A real checkpoint through ``_build`` and the Talker's greedy decode."""

    def __init__(self, path: str | Path, *, max_tokens: int = 96) -> None:
        """Load the artifact lazily; importing this module needs no torch."""
        from ..train.entry import load_artifact
        from ..train.talker import Talker, vocabulary

        spec, model = load_artifact(Path(path))
        encode, decode = vocabulary(spec)
        self._talker = Talker(model, spec, encode=encode, decode=decode)
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Greedy-decode one answer, with no stop sequence for this short suite."""
        return self._talker(prompt, self._max_tokens, ())


__all__ = [
    "CheckpointAdapter",
    "EchoAdapter",
    "KnownAdapter",
    "ModelAdapter",
    "RandomAdapter",
]
