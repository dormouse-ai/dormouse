"""Optimizer state at 8 bits, block-wise: one scale and one dynamic code per block.

    BLOCK SCALE (Dettmers et al. 2021, arXiv:2110.02861 §2 and Figure 1)
        N_j = max(|x_i| : i ∈ block j)                  # one normaliser per block
        q_i = sgn(x_i)·nearest M[k] to |x_i|/N_j        # the 8-bit code
        x̂_i = sgn(q_i)·M[|q_i|]·N_j                     # dequantise

    CODEBOOK (their dynamic map, Appendix F; one sign bit, 127 magnitudes)
        M = {0} ∪ { 10^-d·(0.1 + 0.9k/C_d) : k = 1..C_d, d = 0..4, C_d = 2^(5-d) }
            ∪ { 0.55·10^-4·10^(-k/2) : k = 1..65 }

    STATE UPDATE (§2, Figure 1's loop)
        s ← quantise(dequantise(s) + Δ)                 # Δ is the step's change

    ERROR (this module's bound, derived in §3)
        |x_i - x̂_i| ≤ N_j/71       for |x_i| ≥ N_j/10      # the top decade's grid
        |x_i - x̂_i| ≤ 54%·|x_i|    below it, to 10^-37·N_j, then zero

§1. WHY A BLOCK, AND WHY 256. A moment is not one distribution: Adam's `v` varies over
several orders of magnitude across a model, and inside one tensor the rows of a table
touched by a rare token sit orders below the rows touched by every token. A tensor-wide
scale is set by that largest value, so every other value lands on a handful of codes —
or on zero. A block confines an outlier to its own 256 values: the scale is the largest
magnitude *there*, so the code step is proportional to the local magnitude. 256 is the
paper's default and the point where the scales cost 4/256 = 1.6% of the saving: 8-bit
state is 1.0156 bytes per parameter against 4.

§2. WHY THE CODE IS NOT LINEAR, WITH THE MEASUREMENTS THAT DECIDED IT. A linear 8-bit
code over a block asks for `1/254` of the block's maximum in absolute error, and the
moment that decides the step does not live in that range. The first step of AdamW is
`v = (1-β₂)·g²`: over 256 neighbouring gradients of a gaussian `g`, `g²` spans five
orders, so every element below `N/254` quantises to *zero* — and zero in `v` is
`m/(√0 + ε)`, a step of `m/ε`. Measured on this module's fixture: a linear code made
the first step `2.6e5` where the correct one is `1.4e-2`, and 20 steps drifted by 1.0e3
in relative norm. The paper's answer is a *dynamic* code (§1.2-1.3): the exponent is
part of the value, so precision is spent where the value is large and range where it is
small. The map in the docstring is that construction — powers of ten, `2^(5-d)` levels
in the decade `10^-d` — flattened into a table sorted ascending, so one `bucketize` is
the quantiser and one `gather` the dequantiser.

A floor at `10^-6·N`, which is what the paper's own 7-decade 8-bit map gives, was
measured to be *not deep enough for `v`*: `g²` of a gaussian spans `10^-9` of its block
maximum, and the first step still came out `20.6` against `1.4e-2`, because the update
is `m/√v` and an element whose `v` is floored sees its update magnified by `√(v̂/v)`.
So the tail below the linear head is `√10` apart down to `10^-37·N` — every ratio a
positive fp32 moment can have with its block's maximum — which costs the top decade one
bit of resolution (32 levels, not 64) and takes the 20-step drift to `1.1e-3`. What is
left is a 54% relative error on values four or more decades below their block's maximum;
their update is still the right sign at the right order.

Our table spends one bit on the sign, because one code keeps `v`, `m` and Muon's
momentum alike; the paper's second-moment code is unsigned and so has twice the levels
in the top decade. What that costs is a measurement, not an argument —
`tools/lowbit_gates.py` reports it.

§3. THE BOUND, AND WHERE IT IS EXACT. Adjacent codes in the top decade are `0.9/32` of
`N` apart, so rounding to the nearest moves an element by at most `N/71`; the decades
below hold half as many levels each, and the tail is `√10` apart, which is where the 54%
line comes from. The bound is not uniform because the code is not: absolute above
`N/10`, relative below. Two cases are exact. A block of one magnitude has every code at
`M[126] = 1`, and `sgn·1·N = ±N` bit for bit. A block of zeros has every code at
`M[0] = 0`, and a moment that is zero stays zero rather than reading the smallest code.

§4. THE STABLE EMBEDDING, AND WHAT IT COSTS. The paper's §2.3: the token table keeps
32-bit optimizer state — "this is the only layer that uses 32-bit optimizer states" —
because its gradient is not i.i.d. across positions: a token's row is touched only when
that token appears, so a rare token's gradient is 100× the mean and its second moment is
where a quantiser underflows. `MixOptimizer` applies that rule by name, before anyone can
ask for 8 bits, and the table's state is a plain dense tensor that never meets the
quantiser. The cost is 32768·832·4 = 109 MB for `v` and the same for `m`, out of the
~3 GB an 8-bit Muon momentum would save, so the rule is cheap — and it is the recipe's
own, not a tuning choice.

§5. WHY THE FIELDS ARE TENSORS AND INTS. The state lives in `torch.optim.Optimizer.state`
and is written into a checkpoint that `checkpoint.load` reads with `weights_only=True`,
which refuses to construct an arbitrary class. So `to_plain` writes `{"bits", "block",
"shape", "dense"}` or `{"codes", "scales", ...}`, and `plain_state_dict` is what both
arms return from `state_dict()`: a saved moment is an int8 tensor, a float tensor, and
the ints that say how to read them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from dormouse_core.errors import LoudError

__all__ = [
    "BITS",
    "BLOCK",
    "CHUNK",
    "MAGNITUDE",
    "LowBitState",
    "adopt",
    "check",
    "plain_state_dict",
    "revive_state_dict",
]

#: Values that share one scale: the paper's default block (§1).
BLOCK = 256

#: The two widths a moment can have: the dense tensor it always was, or 8 bits.
BITS = (32, 8)

#: Values per pass in `LowBitState.apply`. The transient is this many fp32 values, so a
#: 1B-parameter moment is quantised 1M values at a time rather than allocated at 4 GB.
CHUNK = 1 << 20

#: Decades the linear head spans; the geometric tail below it reaches `10^TAIL_END` (§2).
DECADES = 5
TAIL = 65
TAIL_END = -37


def _magnitude() -> torch.Tensor:
    """Build `M`, ascending: one level of zero, 127 nonzero, `1` last (§2).

    The head is linear inside each decade — `2^(5-d)` levels in the decade `10^-d`, so
    the absolute resolution near a value of `N` is `N/71` — and the tail below the last
    head level is geometric, `√10` apart, down to `10^-37`: every ratio a positive fp32
    moment can have with its block's maximum, at a factor 1.8. The top level is exactly
    `1`, so a block whose values all sit at its maximum dequantises to that maximum — the
    case §3 says is exact.
    """
    levels: list[float] = []
    for decade in range(DECADES):
        count = 2 ** (DECADES - decade)
        steps = torch.linspace(0.1, 1.0, count + 1)[1:]
        levels.extend((steps * 10.0**-decade).tolist())
    deepest = 0.5**0.5 * min(levels)
    levels.extend(torch.logspace(math.log10(deepest), TAIL_END, TAIL).tolist())
    return torch.tensor([0.0, *sorted(levels)], dtype=torch.float32)


#: The codebook, ascending, `0` first and `1` last: one level of zero plus 127 levels.
MAGNITUDE = _magnitude()

#: The midpoints between adjacent codes: `bucketize` against these is "nearest code".
_BOUNDARIES = (MAGNITUDE[:-1] + MAGNITUDE[1:]) / 2


def check(bits: int, block: int) -> None:
    """Refuse a state width or block size this module cannot keep, naming the ones it can."""
    if bits not in BITS:
        raise LoudError(
            f"optimizer state bits={bits}",
            "a moment is 32 bits (the dense tensor `torch.optim` keeps) or 8 (block-wise, "
            "this module); a third width is a format no checkpoint here can load",
        )
    if block < 1:
        raise LoudError(
            f"optimizer state block={block}",
            "a block is a positive number of values; 256 is the paper's default, and a "
            "block larger than the tensor is per-tensor quantisation — the thing §1 avoids",
        )


class LowBitState:
    """One optimizer moment, held as 8-bit blocks of `block` values or as a dense tensor.

    The 32-bit path is the tensor `torch.optim` would have allocated, and every operation
    over it is the same in-place call on the same values: a run at `bits=32` is the run
    this module did not exist for. The 8-bit path dequantises a window, applies the same
    in-place call in fp32, and re-quantises the blocks the window covers.
    """

    def __init__(self, like: torch.Tensor, *, bits: int = 8, block: int = BLOCK) -> None:
        """Allocate a zeroed state shaped like `like`; see §5 for the layout."""
        check(bits, block)
        self.bits = bits
        self.block = block
        self.shape = like.shape
        self.numel = like.numel()
        padded = self._padded()
        self.data: dict[str, torch.Tensor]
        if bits == 32:
            #: `zeros_like`, not fp32: a 32-bit moment has always followed its parameter.
            self.data = {"dense": torch.zeros_like(like)}
        else:
            self.data = {
                "codes": torch.zeros(padded, dtype=torch.int8, device=like.device),
                "scales": torch.zeros(padded // block, dtype=torch.float32, device=like.device),
            }

    def __repr__(self) -> str:
        """Name the width, the block and the shape — what a checkpoint diff shows."""
        return f"LowBitState(bits={self.bits}, block={self.block}, shape={tuple(self.shape)})"

    def zero_(self) -> None:
        """Clear every value, exactly as `torch.zeros_like` would have started it."""
        for tensor in self.data.values():
            tensor.zero_()

    def dense(self) -> torch.Tensor:
        """Return the state as a tensor: the stored one at 32 bits, a copy at 8.

        At 32 bits this is the tensor itself, not a copy: `Muon` hands it to the polar
        chain, which allocates its own intermediates and does not write to its argument.
        """
        if "dense" in self.data:
            return self.data["dense"]
        return self._dequantise(0, self._padded())[: self.numel].view(self.shape)

    def apply(self, update: Callable[..., None], *others: torch.Tensor) -> None:
        """Apply `update` to the state in fp32, one window at a time, then re-quantise.

        `update(window, *windows)` must write into its first argument; the window is a
        view of the fp32 buffer the state was dequantised into, and the blocks it covers
        are written back after the call returns. At 32 bits the window is a view of the
        stored tensor, so the same call is the call `torch.optim` would have made.
        """
        for other in others:
            if other.numel() != self.numel:
                raise LoudError(
                    f"a {tuple(other.shape)} tensor into a {tuple(self.shape)} moment",
                    "a moment is shaped like its parameter; a mismatch is a parameter "
                    "that changed shape under a state that did not",
                )
        step = CHUNK if self.bits == 32 else max(self.block, (CHUNK // self.block) * self.block)
        for start in range(0, self.numel, step):
            stop = min(start + step, self.numel)
            parts = [other.reshape(-1)[start:stop] for other in others]
            if "dense" in self.data:
                update(self.data["dense"].reshape(-1)[start:stop], *parts)
                continue
            #: Whole blocks only, so the last window can overhang `numel` into the pad.
            hi = min(-(-stop // self.block) * self.block, self._padded())
            window = self._dequantise(start, hi)
            update(window[: stop - start], *parts)
            self._quantise(window, start)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> None:
        """Move the state to `device`; a dense state also follows its parameter's dtype.

        Called once after `load_state_dict`: the checkpoint is read on the CPU, and a
        resumed run on a card would otherwise step a CPU moment against GPU gradients.
        """
        for name, tensor in self.data.items():
            self.data[name] = tensor.to(device=device, dtype=dtype if name == "dense" else None)

    def to_plain(self) -> dict[str, Any]:
        """Write the state as tensors and ints: what `torch.save` can keep (§5)."""
        return {"bits": self.bits, "block": self.block, "shape": tuple(self.shape), **self.data}

    @classmethod
    def from_plain(cls, raw: Mapping[str, Any]) -> LowBitState:
        """Rebuild the container from what `to_plain` wrote, shape and dtype included."""
        dense = raw.get("dense")
        like = torch.zeros(
            tuple(raw["shape"]), dtype=dense.dtype if dense is not None else torch.float32
        )
        state = cls(like, bits=int(raw["bits"]), block=int(raw["block"]))
        for name, tensor in state.data.items():
            tensor.copy_(raw[name])
        return state

    def _padded(self) -> int:
        """`numel` rounded up to a whole block: the last block is zero-padded, never ragged."""
        return -(-self.numel // self.block) * self.block

    def _quantise(self, window: torch.Tensor, start: int) -> None:
        """Write `window` (whole blocks from `start`) as one scale and one code per value."""
        rows = window.numel() // self.block
        grid = window.view(rows, self.block)
        scale = grid.abs().amax(dim=1)
        lo = start // self.block
        self.data["scales"][lo : lo + rows] = scale
        #: An all-zero block has scale 0 and must quantise to 0, not to `0/0`: the floor
        #: is the dtype's `tiny`, so `0/tiny = 0` and every code in the block is 0.
        floor = scale.clamp_min(torch.finfo(torch.float32).tiny)
        code = torch.bucketize(grid.abs() / floor[:, None], _BOUNDARIES).to(torch.int8)
        self.data["codes"][start : start + window.numel()] = (
            code * torch.sign(grid).to(torch.int8)
        ).reshape(-1)

    def _dequantise(self, start: int, stop: int) -> torch.Tensor:
        """`sgn(q)·M[|q|]·N` for the whole blocks in `[start, stop)` (§2)."""
        rows = (stop - start) // self.block
        lo = start // self.block
        codes = self.data["codes"][start:stop].view(rows, self.block)
        scale = self.data["scales"][lo : lo + rows]
        magnitude = MAGNITUDE.to(codes.device)[codes.abs().to(torch.long)]
        signed = magnitude * codes.to(torch.float32).sign()
        return (signed * scale[:, None]).reshape(-1)


def plain_state_dict(saved: dict[str, Any]) -> dict[str, Any]:
    """`Optimizer.state_dict()` with every `LowBitState` written as tensors and ints (§5)."""
    return {**saved, "state": {key: _plain(entry) for key, entry in saved["state"].items()}}


def revive_state_dict(saved: dict[str, Any]) -> dict[str, Any]:
    """Reverse `plain_state_dict`: turn a loaded checkpoint back into the live container."""
    return {**saved, "state": {key: _revive(entry) for key, entry in saved["state"].items()}}


def adopt(states: Mapping[torch.Tensor, dict[str, Any]]) -> None:
    """Put each restored state on its parameter's device, and a dense one in its dtype."""
    for param, entry in states.items():
        for value in entry.values():
            if isinstance(value, LowBitState):
                value.to(param.device, param.dtype)


def _plain(entry: dict[str, Any]) -> dict[str, Any]:
    """One parameter's state entries with the container replaced by its plain form."""
    return {
        name: value.to_plain() if isinstance(value, LowBitState) else value
        for name, value in entry.items()
    }


def _revive(entry: dict[str, Any]) -> dict[str, Any]:
    """One parameter's state entries with `_plain`'s form replaced by the container."""
    return {
        name: LowBitState.from_plain(value) if _is_plain(value) else value
        for name, value in entry.items()
    }


def _is_plain(value: Any) -> bool:
    """Return True for a mapping `to_plain` wrote: no `torch.optim` state is a dict."""
    return isinstance(value, dict) and {"bits", "block", "shape"} <= set(value)
