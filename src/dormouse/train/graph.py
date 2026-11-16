"""One step's forwards and backwards as one CUDA graph, and the counter that proves it.

WHY A GRAPH. After the mixture became one kernel the step still issues ~112 000 kernel
launches, and those launches are not the arithmetic: the profiler's top kernels are dtype
conversions and `elementwise_kernel`s of a few microseconds each. A graph records the
launch sequence once and replays it from one call, which takes the host out of the loop —
the only lever that addresses launches without touching the arithmetic.

WHAT IS INSIDE THE GRAPH, AND WHY NOT THE WHOLE STEP. `train/step.py` is written to
*branch* on three values it reads back to the host: `torch.isfinite(loss)`, `float(loss)`
and the clip norm. A host read inside a capture is a hard error, so the graph holds the
part with no host reads — `zero_grad`, the micro-batch loop, the chunked cross-entropy and
`backward()` — and the clip, the firewall's decision and `optimizer.step()` stay eager.
That split needs no second copy of the step, which is what a captured branch would.

TWO CONSEQUENCES THAT ARE STATED, NOT HIDDEN:

* **The shapes are frozen.** The graph replays the batch it was captured at, and `step`
  refuses an `ids` tensor of another shape rather than copying part of it.
* **The bandwidth is the copy.** Every replay copies `ids` into the graph's own buffer;
  that copy is one kernel and the only per-step host work left.

NO SILENT EAGER. A capture that fails raises `LoudError` naming the op that stopped it,
`replays()` stays zero, and `graph_replays` at zero under an enabled mechanism is the
defect class `dormouse_core.counters` exists for.
"""

from __future__ import annotations

import math

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from torch import nn

from .optim import MixOptimizer
from .step import CHUNK_TOKENS, CLIP, PreStep, StepResult, chunked_cross_entropy

__all__ = ["StepGraph"]

#: Headroom a capture keeps free beside its private pool. The allocator's fragmentation
#: and the graph's own bookkeeping are in no count this module can take, and being wrong
#: costs an OOM inside someone else's kernel.
MARGIN_BYTES = 384 * 1024 * 1024

#: Warm-up iterations before a capture: every kernel compiled, every lazy buffer grown, so
#: the graph's private pool is the step's steady state and not a first-call high-water mark.
WARMUP = 3


class StepGraph:
    """`zero_grad → forward → chunked CE → backward`, recorded once and replayed per step.

    `run` is the same step `train_step` is — same order, same clip, same firewall — with
    the captured half replayed instead of issued. The two are held against each other in
    `tests/test_graph.py`, so they cannot drift apart silently.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: MixOptimizer,
        ids: torch.Tensor,
        *,
        accum: int = 1,
    ) -> None:
        """Capture the forward/backward half at `ids`'s shape."""
        if ids.dim() != 2 or ids.shape[1] < 2:
            raise LoudError(
                f"a step graph needs ids [batch, seq+1], got {tuple(ids.shape)}",
                "the +1 is the target shift; a one-column batch predicts nothing",
            )
        if accum < 1 or ids.shape[0] % accum:
            raise LoudError(
                f"accum={accum} does not split {ids.shape[0]} rows evenly",
                "the captured step divides the batch the way `train_step` does",
            )
        self.model = model
        self.optimizer = optimizer
        self.accum = accum
        self.rows = ids.shape[0]
        self.seq = ids.shape[1] - 1
        self.ids = ids.clone()
        self._loss = torch.zeros((), device=ids.device, dtype=torch.float32)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._replays = 0
        #: The op a failed capture stopped at, or `None` — the name a report needs.
        self.failed_on: str | None = None
        self._warm()
        self._capture()

    # ------------------------------------------------------------------ capture
    def _body(self) -> torch.Tensor:
        """Run the captured arithmetic: no host reads, one device scalar out."""
        self.optimizer.zero_grad(set_to_none=False)
        inputs, targets = self.ids[:, :-1], self.ids[:, 1:]
        rows = self.rows // self.accum
        total = torch.zeros((), device=self.ids.device, dtype=torch.float32)
        for micro in range(self.accum):
            x = inputs[micro * rows : (micro + 1) * rows]
            y = targets[micro * rows : (micro + 1) * rows]
            logits = self.model(x, None)
            loss = chunked_cross_entropy(logits, y, chunk=CHUNK_TOKENS)
            (loss / self.accum).backward()  # type: ignore[no-untyped-call]
            total = total + loss.detach() / self.accum
        return total

    def _warm(self) -> None:
        """Run the body off-graph: every kernel compiled, every buffer grown."""
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _capture(self) -> None:
        """Capture `_body` into a graph, naming the op that stopped it."""
        free, _total = torch.cuda.mem_get_info()
        if free < MARGIN_BYTES:
            raise LoudError(
                f"a step graph needs {MARGIN_BYTES / 2**20:.0f} MiB of headroom, "
                f"{free / 2**20:.0f} MiB is free",
                "free the card before capturing; the pool is allocated beside the live "
                "tensors, and an OOM inside the allocator names the wrong op",
            )
        graph = torch.cuda.CUDAGraph()
        #: A host sync inside a capture is the commonest way a step is not capturable, and
        #: CUDA reports it as "an error during capture" with no name attached. Sync-debug
        #: turns the offending call into a Python error that names the op, which is the
        #: whole point of this class's error path.
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.cuda.graph(graph):
                self._loss = self._body()
        except RuntimeError as exc:
            self.failed_on = str(exc).splitlines()[0][:200]
            raise LoudError(
                f"the step could not be captured: {self.failed_on}",
                "the op named above is the one to make device-side (a host read, an "
                "allocation that depends on a device value, or a library call that is not "
                "capturable); a graph that replays nothing is not a fallback",
            ) from exc
        finally:
            torch.cuda.set_sync_debug_mode("default")
        self._graph = graph

    # ------------------------------------------------------------------- replay
    def replay(self, ids: torch.Tensor, *, counters: Counters | None = None) -> None:
        """Copy `ids` into the graph's buffer and replay; the loss lands in `self._loss`."""
        if ids.shape != self.ids.shape:
            raise LoudError(
                f"this graph was captured at {tuple(self.ids.shape)} and got {tuple(ids.shape)}",
                "capture one graph per batch shape, or feed the shape it was built for",
            )
        if self._graph is None:  # pragma: no cover - `_capture` raises instead
            raise LoudError(f"replay without a capture ({self.failed_on})", "see failed_on")
        self.ids.copy_(ids)
        self._graph.replay()
        self._replays += 1
        if counters is not None:
            counters.count("graph_replays")

    def loss(self) -> float:
        """Return the last replay's loss, read back to the host — outside the capture."""
        return float(self._loss)

    def replays(self) -> int:
        """How many times the graph has been replayed; zero means it never ran."""
        return self._replays

    def run(
        self,
        ids: torch.Tensor,
        *,
        counters: Counters | None = None,
        clip: float = CLIP,
        pre_step: PreStep | None = None,
    ) -> StepResult:
        """One optimizer step: the replay, then the eager half `train_step` documents."""
        self.replay(ids, counters=counters)
        loss = self.loss()
        if not math.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=False)
            if counters is not None:
                counters.count("nan_guard")
                counters.check()
            return StepResult(loss=loss, grad_norm=float("nan"), skipped=True)
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
        if counters is not None:
            counters.count("clip")
        if not math.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=False)
            if counters is not None:
                counters.count("nan_guard")
                counters.check()
            return StepResult(loss=loss, grad_norm=norm, skipped=True)
        if pre_step is not None and not pre_step(loss):
            self.optimizer.zero_grad(set_to_none=False)
            return StepResult(loss=loss, grad_norm=norm, skipped=True)
        sizes = self.optimizer.sizes
        self.optimizer.step()
        if counters is not None:
            counters.count("opt_muon", sizes["muon"])
            counters.count("opt_adamw", sizes["adamw"])
            counters.check()
        return StepResult(loss=loss, grad_norm=norm, skipped=False)

    def __repr__(self) -> str:
        """Print the shape the graph was built at and how often it has run."""
        return (
            f"StepGraph(rows={self.rows}, seq={self.seq}, replays={self._replays}, "
            f"captured={self._graph is not None})"
        )
