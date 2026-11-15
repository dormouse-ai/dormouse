"""The evaluation harness: BPB you can compare and tasks that can fail.

`dorm eval` is built from these four pieces and nothing else moves into it:

* `bpb` — bits per byte over the measured evaluation text, with the conversion
  anchored to a hand-computable entropy;
* `leakage` — an eval split that shares a document with training is refused;
* `tasks`/`harness` — five deterministic, judge-free tasks scored against any
  `generate(prompt, max_tokens, stop)`;
* `report` — one table per checkpoint, every row carrying its `n`.
"""

from __future__ import annotations

from .bpb import bytes_per_token, entropy_bits_per_byte, from_token_nats
from .harness import run
from .leakage import Split, doc_hash, require_held_out
from .report import Result, render
from .sandbox import SandboxError, run_definition
from .tasks import suite

__all__ = [
    "Result",
    "SandboxError",
    "Split",
    "bytes_per_token",
    "doc_hash",
    "entropy_bits_per_byte",
    "from_token_nats",
    "render",
    "require_held_out",
    "run",
    "run_definition",
    "suite",
]
