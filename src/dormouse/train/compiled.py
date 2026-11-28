"""The `torch.compile` lever: one place that turns it on, and one counter that proves it.

WHY A MODULE FOR ONE CALL. `torch.compile(model)` that silently runs the eager path is
the defect class this project has paid for twice — a mechanism that was enabled, never
ran, and left every test green. It has two silent ways to do that here:

* the backend falls back (a graph break at every interesting region leaves the eager
  kernels in place and reports nothing), and
* dynamo **recompiles** the whole stack per step because a Python object threaded through
  the forward changed identity, so the "compiled" run is a compile loop.

So this module owns three facts and reports all of them: whether compile was asked for
(`enable`), how many graphs dynamo actually built (`unique_graphs`, read from dynamo's own
counters and never from our own bookkeeping), and every reason it stopped (`breaks`). A
run that asked for compile and built zero graphs raises through `Counters` instead of
reporting a speed-up that came from eager kernels.

WHAT IS *NOT* CLAIMED. A graph break is not a failure by itself: the counter calls in
`model/` are Python side effects dynamo cannot trace, and each one ends a region. The
claim is only what the counters show — N graphs built, M frames traced, the break reasons
by name — and `counters.check()` fails if the graphs are zero.
"""

from __future__ import annotations

import signal
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from dormouse_core.errors import LoudError
from torch import nn

#: The counter a compiled step reports. `Counters.check()` fails when it stays at zero,
#: which is what makes "compile ran" a measurement instead of an assumption.
COUNTER = "compile_graphs"

#: Ten minutes. `torch.compile` is lazy: the cost lands on the first forward, so the cap
#: belongs around that call, not around `enable`. A compile that has not landed by then is
#: NO ARTIFACT — the earlier card probe was killed at 540 s (tools/bench_compile_step.py:33).
COMPILE_BUDGET_S = 600

#: Set while the model is compiled, so callers can tell a compiled step from an eager one
#: without inspecting `torch._dynamo`'s globals.
_ENABLED = False


def enable(model: nn.Module, *, fullgraph: bool = False, backend: str = "inductor") -> None:
    """Compile `model` in place — the same module object, so parameter names survive.

    `nn.Module.compile` is used rather than wrapping the model in `torch.compile(...)`:
    the wrapper renames every parameter to `_orig_mod.…`, and `train/optim.py` routes
    parameters *by name* (the token table by suffix, the packed qkv by suffix), so a
    wrapper silently routes the whole model to the wrong rule.

    `dynamic=False` is the design's own shape regime: every tensor in a step has a static
    shape (the dispatch's capacity is a formula over the batch, not a read), so a dynamic
    graph buys nothing and costs guards.
    """
    global _ENABLED
    reset()
    model.compile(backend=backend, dynamic=False, fullgraph=fullgraph)
    _ENABLED = True


def enabled() -> bool:
    """Return whether `enable` has been called in this process."""
    return _ENABLED


def reset() -> None:
    """Clear dynamo's state and counters so the next measurement starts at zero."""
    global _ENABLED
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    _ENABLED = False


def _counters() -> Mapping[str, Any]:
    """Dynamo's own counters; private API, and the only place the truth lives."""
    return torch._dynamo.utils.counters


def unique_graphs() -> int:
    """How many graphs dynamo has built since the last `reset` — the engagement proof."""
    stats = _counters().get("stats", {})
    return int(stats.get("unique_graphs", 0)) if isinstance(stats, Mapping) else 0


def frames() -> dict[str, int]:
    """Frames dynamo traced, ran, or refused, keyed by dynamo's own names."""
    found = _counters().get("frames", {})
    return {str(k): int(v) for k, v in found.items()} if isinstance(found, Mapping) else {}


def breaks() -> dict[str, int]:
    """Every graph-break reason dynamo recorded, largest first: the names, not a count."""
    found = _counters().get("graph_break", {})
    if not isinstance(found, Mapping):
        return {}
    return {str(k): int(v) for k, v in sorted(found.items(), key=lambda kv: -int(kv[1]))}


def proof() -> dict[str, Any]:
    """Everything a report needs: graphs, frames, and the break reasons by name."""
    return {"unique_graphs": unique_graphs(), "frames": frames(), "graph_break": breaks()}


def line() -> str:
    """One log line: the graphs built and the top of the break list."""
    if not _ENABLED:
        return "compile off"
    found = breaks()
    top = ", ".join(f"{name} x{count}" for name, count in list(found.items())[:3])
    return f"compile on: {unique_graphs()} graphs, breaks {sum(found.values())} ({top or 'none'})"


@contextmanager
def compile_budget(seconds: int = COMPILE_BUDGET_S) -> Iterator[None]:
    """Bound the first compiled call with a named refusal when inductor never lands.

    The SIGALRM handler raises instead of returning, so an expired budget cannot be logged
    and then ignored. The previous handler is restored on both the success and failure path.
    """
    if seconds < 1:
        raise LoudError(
            f"compile budget {seconds}s is not a budget",
            f"use a positive number of seconds (the default is {COMPILE_BUDGET_S})",
        )
    previous = signal.getsignal(signal.SIGALRM)

    def _expired(signum: int, frame: object) -> None:
        """Turn the alarm into a refusal that names the missing artifact."""
        del signum, frame
        raise LoudError(
            f"inductor did not produce a graph within {seconds}s",
            "an unfinished compile is NO ARTIFACT, never an eager speed-up under a "
            "compiled name; run with torch_compile=false or raise the budget deliberately",
        )

    signal.signal(signal.SIGALRM, _expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def require_graph() -> dict[str, Any]:
    """Refuse a compiled run that built no graph, naming dynamo's break reasons.

    A graph break is not by itself a failure — the counter calls in `model/` end regions.
    Zero graphs with the flag on is: the eager kernels ran under the compiled name, which
    is the defect this counter exists to catch.
    """
    found = proof()
    if found["unique_graphs"] <= 0:
        breaks = ", ".join(f"{name} x{count}" for name, count in found["graph_break"].items())
        raise LoudError(
            "torch.compile built 0 inductor graphs: the eager path ran under a compiled name",
            f"graph breaks: {breaks or 'none'}; no ms/step may be reported from this run. "
            "A historical live-shape blocker is recorded in "
            "docs/records/2026-11-16-step-graph-live-shape.md §5",
        )
    return found


def checked(model: nn.Module, **kwargs: Any) -> None:
    """`enable`, and refuse a torch that could not compile the model at all."""
    try:
        enable(model, **kwargs)
    except (RuntimeError, AttributeError) as exc:  # pragma: no cover - a broken torch
        raise LoudError(
            f"torch.compile could not be enabled: {exc}",
            "run with torch_compile=false; a compile that failed at the first call would "
            "report eager timings under the compiled name",
        ) from exc
