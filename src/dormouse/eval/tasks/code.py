"""One of our own signatures: a generated body must load.

The task is deliberately narrow. It does not judge style, it does not run a test
suite, and it does not compare text: it compiles and executes the body in the
sandbox, then checks the named function exists. A model that writes plausible
English fails; a model that writes Python passes. That is the signal this buys.

The signatures are copied from this repo, so the task measures the domain the
checkpoint is being trained on, not a benchmark from somewhere else.

The ceiling is named: a body that loads and returns nonsense passes. That is the
question asked; a call-and-compare check is the next rung once the model emits
plausible bodies at all.
"""

from __future__ import annotations

from ..sandbox import SandboxError, run_definition
from .base import Example, Task

#: (signature, docstring, a body that satisfies it), from this repo's own code.
CASES: tuple[tuple[str, str, str], ...] = (
    (
        "def eval_window(batches: int, batch: int, seq_len: int) -> int:",
        "Return how many positions an evaluation of these sizes covers.",
        "return batches * batch * seq_len",
    ),
    (
        "def bytes_per_token(text: str, token_count: int) -> float:",
        "Return the average number of UTF-8 bytes behind each token.",
        'return len(text.encode("utf-8")) / token_count',
    ),
    (
        "def doc_hash(document: str) -> str:",
        "Return the SHA-256 hex digest of a document's exact text.",
        'import hashlib\n\nreturn hashlib.sha256(document.encode("utf-8")).hexdigest()',
    ),
)

#: Prompt prefix, identical for every case so only the signature varies.
INSTRUCTION = "Complete this function. Reply with the body only, indented four spaces.\n\n"


def _header(signature: str, docstring: str) -> str:
    """Render the signature and its docstring for the model to fill in."""
    return f'{signature}\n    """{docstring}"""'


def _source(header: str, body: str) -> str:
    """Indent the generated body under the header so it becomes a definition."""
    lines = body.strip("\n").splitlines()
    indented = "\n".join("    " + line if line.strip() else line for line in lines)
    return f"{header}\n{indented}"


def _name(signature: str) -> str:
    """Return the function name the check looks up after execution."""
    return signature.removeprefix("def ").partition("(")[0].strip()


def code(seed: int = 0) -> Task:
    """Execute a generated body against our own signature; score if it loads."""
    headers: dict[str, tuple[str, str]] = {}
    examples: list[Example] = []
    for signature, docstring, body in CASES:
        header = _header(signature, docstring)
        prompt = f"{INSTRUCTION}{header}\n"
        headers[prompt] = (header, _name(signature))
        examples.append(Example(prompt, body))

    def check(example: Example, output: str) -> bool:
        if not output.strip():
            return False
        header, name = headers[example.prompt]
        try:
            namespace = run_definition(_source(header, output))
        except SandboxError:
            return False
        return callable(namespace.get(name))

    return Task(
        name="code-executes",
        metric="executes",
        examples=tuple(examples),
        check=check,
        max_tokens=96,
    )


__all__ = ["CASES", "code"]
