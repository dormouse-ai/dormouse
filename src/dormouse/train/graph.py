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

from typing import Any

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from dormouse_kernels.engine import mega_block
from torch import nn

from ..model.attn_variants.kda import KDA
from ..model.moe import MAX_DISPATCH_PASSES, MoEFFN
from . import compiled
from .optim import MixOptimizer
from .step import (
    CHUNK_TOKENS,
    CLIP,
    PreStep,
    StepResult,
    chunked_cross_entropy,
    commit_step,
    host_scalar,
)

__all__ = ["StepGraph"]

#: Headroom a capture keeps free beside its private pool. The allocator's fragmentation
#: and the graph's own bookkeeping are in no count this module can take, and being wrong
#: costs an OOM inside someone else's kernel. The project's rule is the larger of this floor
#: and 8% of VRAM; refuse below it rather than faulting inside capture.
MARGIN_BYTES = 384 * 1024 * 1024
POOL_FRACTION = 0.08

#: Warm-up iterations before a capture: every kernel compiled, every lazy buffer grown, so
#: the graph's private pool is the step's steady state and not a first-call high-water mark.
WARMUP = 3

#: Break-reason substrings that mean an opaque kernel leaked back into the graph. Counter
#: side effects and other Python-only seams end regions too, but they are not the kernels.
_KERNEL_BREAKS = ("bank", "sparse", "attention", "triton", "custom_op")


class _CallableGraph:
    """The CPU stand-in for a CUDA graph: the compiled body, replayed by calling it.

    The captured arithmetic is identical; only the host's way of re-issuing it differs.
    """

    def __init__(self, body: Any) -> None:
        """Hold the callable body."""
        self.body = body

    def replay(self) -> None:
        """Re-run the compiled body; `capture` binds its loss to the graph's output."""
        self.body()


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
        epilogue: bool = False,
    ) -> None:
        """Capture the forward/backward half at `ids`'s shape.

        `epilogue=True` routes each ideal dense bank through the fused SiLU+residual
        epilogue before capture. Only the dense bank has a capture-safe fused path.
        """
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
        #: A captured forward cannot size its pass loop from `counts.max()`. Switch every
        #: bank to the fixed, masked loop before warm-up and capture; the eager path keeps
        #: its data-dependent count. Stored so the eager half can read the overflow flag.
        self._banks = [m for m in model.modules() if isinstance(m, MoEFFN)]
        self.epilogue = bool(epilogue)
        self.dense = any(getattr(b, "ideal_bank", False) for b in self._banks)
        if self.epilogue:
            if not self.dense:
                raise LoudError(
                    "epilogue needs a dense ideal bank to fuse",
                    "build the stack with ideal_bank=true; the routed and A4/ternary banks "
                    "read counts and pack codes on the host and cannot be captured",
                )
            mega_block.set_epilogue(True)
            for bank in self._banks:
                if bank.ideal_bank:
                    mega_block.install_fused_bank(bank)
        for bank in self._banks:
            bank.fixed_passes = MAX_DISPATCH_PASSES
        self.rows = ids.shape[0]
        self.seq = ids.shape[1] - 1
        self.ids = ids.clone()
        self._loss = torch.zeros((), device=ids.device, dtype=torch.float32)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._replays = 0
        #: The op a failed capture stopped at, or `None` — the name a report needs.
        self.failed_on: str | None = None
        self.capture()

    # ------------------------------------------------------------------ capture
    def capture(self) -> None:
        """Compile the step into one dynamo graph, then record it (or bind it on CPU).

        The bank and sparse-attention kernels are opaque custom ops, so dynamo sees one
        node per kernel instead of breaking at every Triton launch site. A capture that
        produced more than one graph, or a break naming a kernel, is refused: an eager
        step under a graph's name is the defect this method exists to prevent.
        """
        try:
            if not compiled.enabled():
                compiled.checked(self.model)
            self._warm()
        except Exception as exc:
            raise LoudError(
                f"the 200M step graph could not be captured: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}",
                "the eager fast path stays intact; an eager step under a graph name is "
                "never reported as replay",
            ) from exc
        proof = compiled.proof()
        graphs = proof["unique_graphs"]
        # The dense bank's Python fwd/bwd tallies split dynamo into device regions;
        # the CUDA graph below still records every region. Zero graphs or a kernel
        # break is refused for every arm.
        if graphs < 1 or (graphs != 1 and not self.dense):
            raise LoudError(
                f"the step compiled into {graphs} dynamo graphs",
                f"breaks: {proof['graph_break']}; only the dense bank's Python tallies may "
                "split the step into device regions; a kernel break is never allowed",
            )
        kernel_breaks = {
            name: count
            for name, count in proof["graph_break"].items()
            if any(part in name.lower() for part in _KERNEL_BREAKS)
        }
        if kernel_breaks:
            raise LoudError(
                f"a kernel leaked out of the opaque graph: {kernel_breaks}",
                "the bank and sparse-attention launches must stay behind their custom ops",
            )
        if self.ids.is_cuda:
            self._capture()
        else:

            def replay_body() -> None:
                self._loss = self._body()

            self._graph = _CallableGraph(replay_body)  # type: ignore[assignment]
            replay_body()

    def proof(self) -> dict[str, Any]:
        """Return dynamo's own account of the capture: graphs, frames and breaks."""
        return compiled.proof()

    def graph_breaks(self) -> dict[str, int]:
        """Return the break reasons recorded while capturing, by name."""
        return compiled.breaks()

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
        if not self.ids.is_cuda:
            for _ in range(WARMUP):
                self._body()
            return
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _capture(self) -> None:
        """Capture `_body` into a graph, naming the op that stopped it."""
        free, total = torch.cuda.mem_get_info()
        needed = max(MARGIN_BYTES, int(POOL_FRACTION * total))
        if free < needed:
            raise LoudError(
                f"a step graph needs max(384 MiB, 8% VRAM)={needed / 2**20:.0f} MiB of "
                f"headroom, {free / 2**20:.0f} MiB is free",
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

    def _prove_dense(self, counters: Counters) -> None:
        """Report the dense bank's fwd/bwd receipts, which replay cannot tally.

        A CUDA replay runs the kernels but not the Python counters inside the bank's
        forward and hook. Grads are the host-side receipt that the backward ran; the
        rows are the frozen shape the graph was captured at. Without this the step's
        `MissingReport` gate would refuse an ideal bank that actually replayed.
        """
        if not getattr(self, "dense", False):
            return
        proved = int(any(p.grad is not None for p in self.model.parameters()))
        for name in ("ideal_bank_fwd", "ideal_bank_bwd"):
            counters.enable(name, counter=name, why="the dense bank replayed")
            counters.count(name, self.rows if name.endswith("fwd") else proved)
        if self.epilogue:
            for name in ("epilogue_fwd", "epilogue_bwd"):
                counters.enable(name, counter=name, why="the fused epilogue replayed")
                counters.count(name, self.rows if name.endswith("fwd") else proved)

    def loss(self) -> float:
        """Return the last replay's loss through the one declared host-read seam."""
        return host_scalar(self._loss)

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
        if counters is not None:
            self._prove_dense(counters)
        for bank in getattr(self, "_banks", ()):
            if bank.overflowed():
                raise LoudError(
                    f"the replayed routing needed more than {MAX_DISPATCH_PASSES} dispatch "
                    "passes: the router has collapsed",
                    "the captured loop is fixed at MAX_DISPATCH_PASSES and cannot raise on "
                    "device; this is the same refusal the eager path makes on counts.max(), "
                    "checked here because the host read sits outside the capture",
                )
        for module in self.model.modules():
            if isinstance(module, KDA):
                module.check_alpha_floor()
        return commit_step(
            self.model,
            self.optimizer,
            self._loss,
            counters=counters,
            clip=clip,
            pre_step=pre_step,
        )

    def __repr__(self) -> str:
        """Print the shape the graph was built at and how often it has run."""
        return (
            f"StepGraph(rows={self.rows}, seq={self.seq}, replays={self._replays}, "
            f"captured={self._graph is not None})"
        )
