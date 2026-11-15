"""The sandbox gate: generated code gets no OS, no process, no socket, no loop.

These are the three imports a model should never need and the one endless loop that
would otherwise hang the suite.
"""

from __future__ import annotations

import pytest

from dormouse.eval.sandbox import SandboxError, run_definition


@pytest.mark.parametrize("module", ["os", "subprocess", "socket"])
def test_a_dangerous_import_is_refused(module: str) -> None:
    with pytest.raises(SandboxError) as error:
        run_definition(f"import {module}\nx = 1")
    assert module in str(error.value)


def test_an_import_from_a_denied_module_is_refused() -> None:
    with pytest.raises(SandboxError):
        run_definition("from socket import socket")


def test_an_import_outside_the_allowlist_is_refused() -> None:
    with pytest.raises(SandboxError) as error:
        run_definition("import ctypes")
    assert "allowlist" in str(error.value)


def test_a_body_that_never_returns_is_stopped() -> None:
    with pytest.raises(SandboxError) as error:
        run_definition("while True:\n    pass", timeout=0.05)
    assert "longer than" in str(error.value)


def test_a_body_that_does_not_compile_is_a_sandbox_error() -> None:
    with pytest.raises(SandboxError) as error:
        run_definition("def broken(")
    assert "does not compile" in str(error.value)


def test_honest_code_runs_and_can_import_math() -> None:
    source = "import math\n\ndef root(x: float) -> float:\n    return math.sqrt(x)"
    namespace = run_definition(source)
    assert namespace["root"](9.0) == 3.0
