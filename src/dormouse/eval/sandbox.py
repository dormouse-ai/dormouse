"""Run a model's generated Python with a deadline and no way out of the process.

The code under test is untrusted by construction: it is whatever the model wrote.
Three things are refused here — the modules that reach the OS, the process table or
the network, imports outside a small allowlist, and a body that never returns.

This is a seatbelt for honest code, not a jail: Python's object graph can be walked
around a guarded `__import__` by a model that is trying to escape. The eval asks
"did it write code that works", so the seatbelt is the right size.
"""

from __future__ import annotations

import builtins
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from dormouse_core.errors import LoudError

#: Modules a generated body may import. Anything else is refused: a task needs
#: `math` or `re`, never the machine.
ALLOWED = frozenset(
    {
        "collections",
        "dataclasses",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
)

#: Named separately so the refusal for these reads as the real hazard, not as a
#: missing allowlist entry.
DENIED = frozenset({"os", "subprocess", "socket"})


class SandboxError(LoudError):
    """Generated code was refused, did not compile, or outlived its deadline."""


def _guarded_import(
    name: str,
    module_globals: Mapping[str, object] | None = None,
    module_locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = None,
    level: int = 0,
) -> object:
    """Import only from the allowlist; name os, subprocess and socket explicitly."""
    root = name.partition(".")[0]
    if root in DENIED:
        raise SandboxError(
            f"import {name!r} is refused in the eval sandbox",
            "generated code gets no OS, no process and no socket",
        )
    if root not in ALLOWED:
        raise SandboxError(
            f"import {name!r} is not on the sandbox allowlist",
            f"allowed: {', '.join(sorted(ALLOWED))}",
        )
    return builtins.__import__(name, module_globals, module_locals, fromlist, level)


def _builtins() -> dict[str, object]:
    """Return the real builtins with the import hook replaced."""
    namespace: dict[str, object] = dict(vars(builtins))
    namespace["__import__"] = _guarded_import
    return namespace


@contextmanager
def _deadline(seconds: float) -> Iterator[None]:
    """Raise `SandboxError` when the body runs past `seconds`."""

    def expire(signum: int, frame: object) -> None:
        raise SandboxError(
            f"generated code ran longer than {seconds:g}s",
            "it is looping; score the example as failed",
        )

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_definition(source: str, *, timeout: float = 2.0) -> dict[str, object]:
    """Compile and execute a generated definition, returning its namespace."""
    try:
        code = compile(source, "<generated>", "exec")
    except SyntaxError as exc:
        raise SandboxError(
            f"generated code does not compile: {exc.msg} (line {exc.lineno})",
            "score the example as failed and keep the output in the report",
        ) from exc
    namespace: dict[str, object] = {"__builtins__": _builtins()}
    with _deadline(timeout):
        exec(code, namespace)
    return namespace
