"""One training step as one CUDA graph: counted before capture, proved after, never faked.

WHY A GRAPH AT ALL. A step of this stack is ~900 kernel launches of a few microseconds
each; at 3-5 µs of launch overhead per kernel the host is a fifth of the step before a
matmul is issued. A graph replays the recorded launch sequence from one call, which is
the only way to remove that cost without moving any of the arithmetic.

THE THREE THINGS THAT MADE THE OLD ENGINE'S GRAPH A LIE, AND THEIR GATES HERE.

1. **It fell back to eager silently.** A capture failed, the code caught the exception
   and ran the eager step, and every "+0 % from the graph" reading for weeks was a
   reading of the eager path. Here a failed capture raises `LoudError` naming the
   offending op (`StepGraph.failed_on`), `capture()` returns no graph, and
   `graph_replays` stays 0 — so a caller that asserts a replay fails instead of reading
   a number that came from somewhere else.
2. **It never said what the private pool needed.** A graph keeps every live allocation
   of the captured region in its own pool, so a card with room for the step and not for
   the pool dies inside the allocator of a different op, with a message about the wrong
   thing. `preflight()` counts the requirement from the allocator before capture and
   refuses below it.
3. **Nothing proved the graph ran.** `graph_replays` counts every `replay()`.

WHY THE STEP IS WRITTEN OUT HERE, AND WHAT KEEPS IT HONEST. `train/step.py` reads three
values back to the host — `torch.isfinite(loss)`, `float(loss)`, and the clip norm —
because it is a step written to *branch* on them. A host read inside a capture is a hard
error: the capture fails at `aten::_local_scalar_dense`. `device_step` below is the same
order with the branching removed, and `tests/test_graph.py` runs twenty steps through
both and requires the weights to agree to 1e-6. That gate is what stops the two from
drifting: the duplicate is the price of not putting a capture branch into the one step
module every other lane trains through.

TWO CONSEQUENCES OF CAPTURE THAT ARE STATED RATHER THAN HIDDEN:

* **The learning rate is baked in.** The optimizer reads `group["lr"]` as a Python float
  at capture time; changing it afterwards does not change the replayed step. `replay()`
  compares the optimizer's rate with the captured one and refuses a mismatch, because a
  graph silently training at yesterday's rate is the same defect class as above.
* **The NaN firewall cannot skip inside the graph.** A branch on `isfinite(loss)` is a
  host read. The body instead multiplies every gradient by the device-side
  `isfinite(loss)` indicator, so a non-finite batch takes a zero-gradient step: the
  weights move by momentum only (one step of decay) instead of being poisoned. The loop
  still counts and journals the skip from the loss it reads after the replay.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .optim import MixOptimizer
from .step import chunked_cross_entropy

__concepts__ = ("Preflight", "StepGraph")

#: Headroom the preflight keeps free beside the pool: the allocator's fragmentation and
#: the graph's own bookkeeping are not in any count this module can take, and the cost
#: of being wrong is an OOM inside someone else's kernel.
MARGIN_BYTES = 512 * 1024 * 1024

try:  # pragma: no cover - the shim keeps a torch without the API from crashing
    from torch.utils._python_dispatch import TorchDispatchMode

    _DISPATCH: Any = TorchDispatchMode
except ImportError:  # pragma: no cover
    _DISPATCH = None

__all__ = ["Preflight", "StepGraph", "device_step", "preflight"]


@dataclass(frozen=True)
class Preflight:
    """What a graph is about to cost, counted before it is taken."""

    #: Weights, gradients and both optimizers' moments: what the allocator holds at rest.
    persistent_bytes: int
    #: The private pool the graph needs: the activation high-water mark of one step.
    pool_bytes: int
    #: Device free memory at the moment of the check.
    free_bytes: int
    #: Headroom kept beside the pool (`MARGIN_BYTES`).
    margin_bytes: int

    @property
    def total_bytes(self) -> int:
        """The requirement a capture must be allowed to take."""
        return self.pool_bytes + self.margin_bytes

    def fits(self) -> bool:
        """Return whether the card has room for the pool plus the margin."""
        return self.free_bytes >= self.total_bytes

    def line(self) -> str:
        """One line, in the units a card is argued in."""
        return (
            f"graph preflight: state {self.persistent_bytes / 2**20:.0f} MiB, "
            f"pool {self.pool_bytes / 2**20:.0f} MiB, margin "
            f"{self.margin_bytes / 2**20:.0f} MiB, free "
            f"{self.free_bytes / 2**20:.0f} MiB"
        )


def state_bytes(model: nn.Module, optimizer: MixOptimizer) -> int:
    """Return the bytes weights, gradients and optimizer moments hold at rest.

    Counted from the tensors and from the allocator, never from a per-parameter
    constant: a Muon matrix keeps one momentum buffer and an AdamW vector keeps two, so
    any single factor is wrong for half the model.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    for arm in (optimizer.muon, optimizer.adamw):
        for group in arm.param_groups:
            for param in group["params"]:
                for value in arm.state.get(param, {}).values():
                    if torch.is_tensor(value):
                        total += value.numel() * value.element_size()
    return total


def preflight(
    *, persistent_bytes: int, pool_bytes: int, free_bytes: int, margin_bytes: int = MARGIN_BYTES
) -> Preflight:
    """Build the report and refuse a capture the card cannot hold.

    `pool_bytes` is the activation high-water mark of a real step (the difference between
    `max_memory_allocated()` after one step and the resting set), because the pool has to
    hold every live allocation of the captured region and the region is the step.
    """
    report = Preflight(
        persistent_bytes=persistent_bytes,
        pool_bytes=pool_bytes,
        free_bytes=free_bytes,
        margin_bytes=margin_bytes,
    )
    if not report.fits():
        raise LoudError(
            f"{report.line()}: the pool does not fit — "
            f"short by {(report.total_bytes - report.free_bytes) / 2**20:.0f} MiB",
            "run a smaller micro-batch for the graphed step, or free the card; capturing "
            "anyway fails inside the graph's allocator, naming an op that is not the "
            "cause",
        )
    return report


_BASE: Any = _DISPATCH if _DISPATCH is not None else object


class _OpTrace(_BASE):
    """Record the name of the last dispatched ATen op, to name a capture failure.

    A capture fails at the *launch* of the op that is not allowed, and torch's message
    does not say which one: it says the operation is not permitted while capturing. The
    dispatcher sees every op before its kernel, so the last name this records is the op
    that failed.
    """

    def __init__(self) -> None:
        """Start with no op recorded."""
        super().__init__()
        self.last = "unknown"

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        """Note `func`, then run it."""
        self.last = str(func) if "profiler" not in str(func) else self.last
        return func(*args, **(kwargs or {}))


def device_step(
    model: nn.Module,
    ids: torch.Tensor,
    optimizer: MixOptimizer,
    *,
    accum: int = 1,
    clip: float = math.inf,
) -> torch.Tensor:
    """`train_step`'s order, with every host read removed, returning the loss on device.

    The order is the contract (`train/step.py`): zero the gradients, then per micro-batch
    forward, chunked cross-entropy, backward divided by `accum`, then the global clip,
    then the optimizer. What is dropped is the *branching*: the NaN firewall and the
    `StepResult` floats, because a graph cannot be branched on a value it cannot read.
    The firewall survives as a device-side multiply, described in the module docstring.
    """
    if ids.dim() != 2 or ids.shape[1] < 2:
        raise LoudError(
            f"a graphed step takes ids [batch, seq+1], got {tuple(ids.shape)}",
            "the +1 is the target shift: inputs ids[:, :-1], targets ids[:, 1:]",
        )
    if accum < 1 or ids.shape[0] % accum:
        raise LoudError(
            f"accum={accum} does not split {ids.shape[0]} rows evenly",
            "pass a row count that is a multiple of accum; an uneven split weights the "
            "last micro-batch differently under the same 1/accum",
        )
    inputs, targets = ids[:, :-1], ids[:, 1:]
    rows = ids.shape[0] // accum
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    for micro in range(accum):
        x = inputs[micro * rows : (micro + 1) * rows]
        y = targets[micro * rows : (micro + 1) * rows]
        loss = chunked_cross_entropy(model(x, None), y) / accum
        loss.backward()  # type: ignore[no-untyped-call]
        total = total + loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    ok = torch.isfinite(total).to(total.dtype).detach()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(ok)
    optimizer.step()
    return total * ok


class StepGraph:
    """A captured `device_step`, its preflight, its failure record and its replay count."""

    def __init__(
        self, model: nn.Module, optimizer: MixOptimizer, *, ids: torch.Tensor, accum: int = 1
    ) -> None:
        """Hold the objects the step runs and the static buffer it reads."""
        if not torch.cuda.is_available():
            raise LoudError(
                "a CUDA graph needs a CUDA device and this process has none",
                "install the CUDA torch build and run on the card; the eager step is "
                "what runs here, and it is not a graph",
            )
        self.model = model
        self.optimizer = optimizer
        self.static_ids = ids.detach().clone()
        self.accum = accum
        self.graph_replays = 0
        #: The op a failed capture stopped at, or None while the capture is sound.
        self.failed_on: str | None = None
        self.report: Preflight | None = None
        #: Pool bytes actually taken, measured as the device free delta across capture.
        self.pool_bytes = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        self._loss: torch.Tensor | None = None
        self._lr: tuple[float, float] = self._rates()

    def _rates(self) -> tuple[float, float]:
        """Return the two arms' rates as the optimizer will read them at capture."""
        muon = float(self.optimizer.muon.param_groups[0]["lr"])
        adamw = float(self.optimizer.adamw.param_groups[0]["lr"])
        return muon, adamw

    def body(self) -> torch.Tensor:
        """Return the captured closure: one device-only step over the static buffer."""
        return device_step(
            self.model, self.static_ids, self.optimizer, accum=self.accum, clip=math.inf
        )

    def _make_capturable(self) -> None:
        """Switch the torch AdamW arm to its capture-safe step, before the warmup.

        `torch.optim.AdamW` refuses a capture until its groups say `capturable`, because
        the default `step()` reads the learning rate to the host — and it builds its
        per-parameter `step` counter on the device only if the flag was already set, so
        this has to run before the recording's warmup. Setting it here rather than in the
        optimizer's constructor keeps the eager path, and its exact arithmetic, exactly
        as it was for every other caller.
        """
        if not isinstance(self.optimizer.adamw, torch.optim.AdamW):
            return
        for group in self.optimizer.adamw.param_groups:
            group["capturable"] = True
        #: A run that trained eagerly first has counters on the host; a captured step
        #: cannot read them, and torch asserts the device rather than moving it.
        for param, state in self.optimizer.adamw.state.items():
            step = state.get("step")
            if torch.is_tensor(step) and step.device != param.device:
                state["step"] = step.to(param.device)

    def _arms(self) -> tuple[Any, Any]:
        """Both optimizer arms, in the order `MixOptimizer.step` runs them."""
        return (self.optimizer.muon, self.optimizer.adamw)

    def _tracked(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every tensor a step mutates, paired with a host copy of it.

        Tensors are identified by *object*, not by name or order: a warmup creates the
        momentum and moment buffers that did not exist before it, and a restore that
        rebuilt them from a state dict would leave the graph writing into buffers the
        optimizer no longer reads — a step that runs and changes nothing.
        """
        found: list[tuple[torch.Tensor, torch.Tensor]] = []
        for param in self.model.parameters():
            found.append((param, param.detach().to("cpu", copy=True)))
        for arm in self._arms():
            for group in arm.param_groups:
                for param in group["params"]:
                    for value in arm.state.get(param, {}).values():
                        if torch.is_tensor(value):
                            found.append((value, value.detach().to("cpu", copy=True)))
        return found

    def _state_keys(self) -> set[tuple[int, str]]:
        """Return the optimizer state entries that exist now, by parameter identity and key."""
        keys: set[tuple[int, str]] = set()
        for arm in self._arms():
            for group in arm.param_groups:
                for param in group["params"]:
                    keys.update((id(param), key) for key in arm.state.get(param, {}))
        return keys

    def _reset_new_state(self, keys: set[tuple[int, str]]) -> None:
        """Zero the state entries a warmup created, so the restore is exact not additive.

        Zeroed rather than deleted on purpose: the graph recorded writes into those
        buffers, so removing the entry would orphan a buffer that still holds the
        warmup's momentum — the replay would then take a step with momentum from a batch
        that the restore claims never happened. A zeroed buffer is what the eager path
        creates on its own first step, so the two agree.
        """
        with torch.no_grad():
            for arm in self._arms():
                for group in arm.param_groups:
                    for param in group["params"]:
                        state = arm.state.get(param)
                        if state is None:
                            continue
                        for key, value in state.items():
                            if (id(param), key) in keys:
                                continue
                            if torch.is_tensor(value):
                                value.zero_()
                            elif isinstance(value, float):
                                state[key] = 0.0
                            elif isinstance(value, int):
                                state[key] = 0

    def capture(self, *, margin_bytes: int = MARGIN_BYTES, restore: bool = True) -> Preflight:
        """Warm up, check the pool, capture, and record what it cost.

        The warmup runs on a side stream because capture cannot allocate: every block the
        step will need has to be in the caching allocator before the recording starts.

        THE WARMUP IS TWO REAL OPTIMIZER STEPS, and that is a trap: a capture taken
        mid-run would silently train two extra batches. `restore=True` copies the weights
        and both arms' state to the host first and puts them back afterwards, in place, so
        capturing is side-effect free — at the price of one host copy of the run's state
        (~16 B/parameter), which is a start-up cost and not a step cost. Pass
        `restore=False` when capturing before the first step, where there is nothing to
        undo and no copy to pay for.
        """
        snapshot = self._tracked() if restore else []
        keys = self._state_keys() if restore else set()
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        self._make_capturable()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        persistent = state_bytes(self.model, self.optimizer)
        peak = torch.cuda.max_memory_allocated()
        free, _ = torch.cuda.mem_get_info()
        self.report = preflight(
            persistent_bytes=persistent,
            pool_bytes=max(peak - persistent, 0),
            free_bytes=free,
            margin_bytes=margin_bytes,
        )
        free_before, _ = torch.cuda.mem_get_info()
        graph = torch.cuda.CUDAGraph()
        trace = _OpTrace() if _DISPATCH is not None else None
        recorder: Any = trace if trace is not None else contextlib.nullcontext()
        try:
            with recorder, torch.cuda.graph(graph):
                self._loss = self.body()
        except RuntimeError as exc:
            self.failed_on = trace.last if trace is not None else "unknown"
            raise LoudError(
                f"the graph capture stopped at {self.failed_on!r}: {exc}",
                "a capture cannot branch on a host value or allocate: make the step "
                "device-only, or run it eagerly — there is no fallback here on purpose, "
                "because a silent one reports an eager step under a graph's name",
            ) from exc
        self._graph = graph
        self._lr = self._rates()
        free_after, _ = torch.cuda.mem_get_info()
        self.pool_bytes = max(free_before - free_after, 0)
        if snapshot:
            with torch.no_grad():
                for tensor, saved in snapshot:
                    tensor.copy_(saved)
            self._reset_new_state(keys)
            self._make_capturable()
            self._lr = self._rates()
        return self.report

    def replay(self, ids: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """Copy `ids` into the static buffer, replay the graph, return the loss tensor."""
        if self._graph is None or self._loss is None:
            raise LoudError(
                f"replay before a successful capture (failed on {self.failed_on!r})",
                "call capture() first; an eager step returned from here would be the "
                "silent fallback this module exists to make impossible",
            )
        if ids.shape != self.static_ids.shape:
            raise LoudError(
                f"ids {tuple(ids.shape)} do not match the static buffer "
                f"{tuple(self.static_ids.shape)}",
                "a graph replays one shape: capture a graph per micro-batch shape",
            )
        if self._rates() != self._lr:
            raise LoudError(
                f"the graph was captured at rates {self._lr} and the optimizer is at "
                f"{self._rates()}",
                "the learning rate is a Python float baked into the recorded step; "
                "re-capture, or the graph trains at the rate it was born with",
            )
        self.static_ids.copy_(ids)
        self._graph.replay()
        self.graph_replays += 1
        if counters is not None:
            counters.count("graph_replays")
        return self._loss
