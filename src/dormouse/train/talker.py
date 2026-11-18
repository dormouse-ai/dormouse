"""A checkpoint as the harness sees it: `generate(prompt, max_tokens, stop)` and no more.

`eval/harness.py` knows exactly one thing about a model — that call — so this is the whole
adapter between a checkpoint and a score. Greedy, because a sample from a 20-step checkpoint
is noise and the harness's checks are exact-match; the task is to exercise the seam and, at
a trained checkpoint, to measure what "knows how to stop" means rather than to flatter it.

The two shape rules are the model's, not preferences:

* the context is cut to `seq_len`, the window the model was built and priced for, so a
  4k-word needle prompt is read as the last `seq_len` bytes and the table says so;
* the right edge is padded to the attention chunk and at least to `attn_k + 1`, because KDA
  refuses a window that does not tile its chunk and DSA refuses `k >= seq` (that forward is
  dense attention with extra steps). Padding on the right cannot change a causal model's
  logits at the last real position, which is the only one this reads.

`vocabulary` is the other half and it is deliberately used in two places (`dorm eval` and
the run's final eval): ids that are read with a different vocabulary than they were written
with are a table of numbers that look fine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from dormouse_core.errors import LoudError
from torch import nn

from ..model.spec import StackSpec

__all__ = ["Talker", "chunk", "vocabulary"]

#: The two halves of a vocabulary: text to ids, ids to text.
type Vocabulary = tuple[Callable[[str], list[int]], Callable[[Sequence[int]], str]]


class Talker:
    """A checkpoint read as text: greedy decoding, one prompt at a time."""

    def __init__(
        self,
        model: nn.Module,
        spec: StackSpec,
        *,
        encode: Callable[[str], list[int]],
        decode: Callable[[Sequence[int]], str],
    ) -> None:
        """Hold the model, the window it was built at, and the vocabulary's two halves."""
        self.model = model
        self.window = spec.seq_len
        self.chunk = chunk(model)
        self.floor = self.chunk * -(-(spec.attn_k + 1) // self.chunk)
        self.encode = encode
        self.decode = decode

    def __call__(self, prompt: str, max_tokens: int, stop: Sequence[str]) -> str:
        """Greedy-decode `max_tokens` ids, stopping on the end-marker ID.

        The marker is an id, not a decoded substring: the corpus deliberately teaches
        the marker's spelling as ordinary content, so a string search here would stop
        generation on a quote of the marker instead of on the control token.
        """
        ids = list(self.encode(prompt))[-(self.window - 1) :] or [0]
        produced: list[int] = []
        text = ""
        for _ in range(max_tokens):
            context = ids[-(self.window - 1) :]
            width = max(len(context) + (-len(context) % self.chunk), self.floor)
            padded = context + [0] * (width - len(context))
            with torch.no_grad():
                logits: torch.Tensor = self.model(torch.tensor([padded], dtype=torch.long))
            next_id = int(logits[0, len(context) - 1].argmax())
            from ..data.tokenizer import END_ID

            if next_id == END_ID:
                break
            ids.append(next_id)
            produced.append(next_id)
            text = self.decode(produced)
        return text


def chunk(model: nn.Module) -> int:
    """Return the attention chunk a window must tile; 1 when the stack does not care."""
    for block in getattr(model, "blocks", ()):
        size = getattr(getattr(block, "attn", None), "chunk", None)
        if isinstance(size, int) and size > 1:
            return size
    return 1


def vocabulary(spec: StackSpec) -> Vocabulary:
    """Return the two halves of the vocabulary: text to ids, ids to text."""
    if spec.unit == "byte":
        return (
            lambda text: list(text.encode("utf-8")),
            lambda ids: bytes(int(item) for item in ids).decode("utf-8", "replace"),
        )
    from ..data.tokenizer import load

    if not spec.tokenizer:
        raise LoudError(
            f"unit={spec.unit} with no tokenizer: ids cannot be turned back into text",
            'name the run\'s tokenizer in the config (tokenizer = "tokenizer.json"): the '
            "harness scores text, and a table read with the wrong vocabulary is noise",
        )
    tokenizer = load(spec.tokenizer)
    return (
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        lambda ids: tokenizer.decode(list(ids)),
    )
