"""Verifiable tasks that need no judge: exact answers, fixed seeds.

Each one targets a failure that a loss number cannot see:

* `reverse_copy` — state and induction, checked character by character;
* `arithmetic` — exactness, on answers that are computable by hand;
* `needle` — retrieval at 4k words, the reach the DSA layer exists to provide;
* `format_following` — does generation stop at our end marker, or answer and ramble.
  The old model answered correctly and kept going because nothing checked the stop.

`CODE`, the fifth task, lives in `code.py` because it executes what it scores.
"""

from __future__ import annotations

import random

from ...data.tokenizer import END, format_turn, role_marker
from .base import Example, Task

#: No `l`, `o`, `0` or `1`: a transcription error in the prompt is visible.
ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"

#: Filler for the needle task. Plain words, no punctuation that could turn into a
#: structure the model can lean on instead of reading the plant.
FILLER = (
    "the",
    "a",
    "and",
    "of",
    "to",
    "in",
    "that",
    "it",
    "is",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "at",
    "be",
    "this",
    "from",
    "or",
    "one",
    "had",
    "by",
    "word",
    "but",
    "not",
    "what",
    "all",
    "were",
    "when",
    "we",
    "there",
    "can",
    "an",
    "your",
    "which",
    "their",
    "said",
    "if",
    "do",
    "will",
    "each",
)


def _exact(example: Example, output: str) -> bool:
    """Return whether the answer, stripped, is the expected text."""
    return output.strip() == example.expected


def _number(example: Example, output: str) -> bool:
    """Return whether the answer is the expected integer, forgiving one full stop."""
    return output.strip().rstrip(".").strip() == example.expected


def reverse_copy(seed: int, n: int = 6) -> Task:
    """Reverse or copy a string exactly, character by character.

    Short strings test state; the 32-character copy tests induction at a length the
    model cannot hold as a lookup table.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(n):
        length = 8 if i % 2 == 0 else 32
        text = "".join(rng.choice(ALPHABET) for _ in range(length))
        if i % 4 < 2:
            prompt = f"Reverse the string exactly.\n{text}\nAnswer:"
            examples.append(Example(prompt, text[::-1]))
        else:
            prompt = f"Copy the string exactly.\n{text}\nAnswer:"
            examples.append(Example(prompt, text))
    return Task(
        name="reverse-copy",
        metric="exact-match",
        examples=tuple(examples),
        check=_exact,
        max_tokens=48,
    )


def arithmetic(seed: int, n: int = 8) -> Task:
    """Add, subtract or multiply two three-digit numbers, exactly."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(n):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        op = "+-*"[i % 3]
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        examples.append(Example(f"What is {a} {op} {b}?\nAnswer:", str(result)))
    return Task(
        name="arithmetic",
        metric="exact-match",
        examples=tuple(examples),
        check=_number,
        max_tokens=16,
    )


def needle(seed: int, n: int = 3, *, context_words: int = 4000) -> Task:
    """Find one planted fact in ~4k words of filler and return it exactly."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n):
        code = str(rng.randint(100_000, 999_999))
        position = rng.randrange(context_words // 4, 3 * context_words // 4)
        words = [rng.choice(FILLER) for _ in range(context_words)]
        words[position] = f"The vault access code is {code}."
        context = " ".join(words)
        prompt = (
            "Read the passage, then answer with the number only.\n\n"
            f"{context}\n\nWhat is the vault access code?\nAnswer:"
        )
        examples.append(Example(prompt, code))
    return Task(
        name="needle-4k",
        metric="exact-match",
        examples=tuple(examples),
        check=_number,
        max_tokens=16,
    )


def _stops_at_end(example: Example, output: str) -> bool:
    """Return whether the answer is exact and nothing follows the end marker.

    A model that answers, emits the end marker and keeps talking is scored failed, which
    is the check the old project never had. Text after the marker is the ramble.
    """
    answer, _, rest = output.partition(END)
    return not rest.strip() and answer.strip() == example.expected


def format_following(seed: int, n: int = 6) -> Task:
    """Answer in one word, inside this project's chat format, and stop.

    The prompt is a system turn, a user turn and an open assistant turn, each closed
    by `END`. The answer must be exact and the generation must not continue past the
    marker. A model that ignores the stop sequence fails even when the word is right.
    """
    rng = random.Random(seed)
    answers = ["yes", "no", "blue", "seven", "north", "quiet"]
    rng.shuffle(answers)
    examples: list[Example] = []
    for answer in answers[:n]:
        prompt = (
            format_turn("system", "Answer with a single word and stop.")
            + format_turn("user", f"Reply with exactly: {answer}")
            + role_marker("assistant")
        )
        examples.append(Example(prompt, answer))
    return Task(
        name="format-following",
        metric="stops-at-end",
        examples=tuple(examples),
        check=_stops_at_end,
        max_tokens=16,
        stop=(END,),
    )


__all__ = ["END", "arithmetic", "format_following", "needle", "reverse_copy"]
