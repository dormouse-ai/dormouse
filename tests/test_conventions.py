"""One concept per module, declared in the module when it holds more than one.

The rule exists because a 2000-line `model.py` is where this project's silent defects
lived: a block, six arms and a router in one file means no reader can say what the
live configuration builds. A public class is the unit of a concept; a module with two
of them must say so out loud with `__concepts__`, which keeps the escape legal and
visible instead of quiet.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "dormouse"
MODULES = sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def _public_classes(tree: ast.Module) -> list[str]:
    """Top-level classes a caller can import, in source order."""
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")]


def _declared(tree: ast.Module) -> tuple[str, ...] | None:
    """The module's `__concepts__` tuple, or None when it declares nothing."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "__concepts__" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return None


def test_a_module_with_two_concepts_declares_them() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text())
        classes = _public_classes(tree)
        declared = _declared(tree)
        if len(classes) < 2:
            assert declared is None, f"{path.name}: one concept needs no __concepts__"
            continue
        assert declared is not None, f"{path.name} holds {classes} and declares no __concepts__"
        assert sorted(declared) == sorted(classes), f"{path.name}: {declared} != {classes}"


def test_the_scan_actually_sees_the_model() -> None:
    """A discovery bug would make the rule above pass vacuously."""
    assert any(p.name == "model.py" and p.parent.name == "model" for p in MODULES)
