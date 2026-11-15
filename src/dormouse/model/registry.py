"""The parts a stack can be built from, each with what it needs and what proves it ran.

This module is a table, not behaviour: one `Piece` per arm, with the `Coeffects` its
class already declares (for KDA and DSA those are the classes' own `COEFFECTS`, so the
refusal comes from the mechanism rather than from a copy of it) and the counter that
has to move for the arm to have actually run.

Why a table instead of `if kind == ...` at the build site: the cross-slot refusals are
the whole point of assembling rather than hand-writing a model, and a refusal can only
be trusted if it runs before the first allocation. `check` walks the table against the
spec's declared context and refuses with the conflicting pair named; `assemble` then
builds exactly the pieces that survived.

The attention slot is per LAYER, not per stack: the stack is a 3:1 recipe, and the
pieces it produces are one attention arm per layer. That is why the slot key carries
the layer index — two arms in one layer is a real conflict, and the same two arms in
different layers is the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

from dormouse_core.coeffects import Coeffects
from dormouse_core.errors import LoudError

from .attn_variants.dsa import DSA
from .attn_variants.kda import KDA
from .spec import StackSpec

#: Slots whose counter the model's forward produces. The rest (optimizer,
#: objective) are counted by the step, which is why they are not declared on the
#: stack: a mechanism the forward cannot count would raise `MissingReport` there.
MODEL_SLOTS = frozenset({"attn", "ffn", "expert", "tsct", "dictionary", "weight_qat", "act_quant"})


@dataclass(frozen=True)
class Piece:
    """One arm in one slot: its name, its coeffects, and the counter that proves it."""

    slot: str
    name: str
    mechanism: str
    counter: str
    coeffects: Coeffects
    why: str

    def conflict(self, ctx: dict[str, str]) -> str:
        """Return the sentence naming what this arm needs and the context lacks."""
        gap = self.coeffects.unsatisfied(ctx)
        if not gap:
            return ""
        needs = ", ".join(f"{k}={v!r}" for k, v in self.coeffects.requires.items())
        return (
            f"{self.slot}={self.name!r} cannot run in this context: it {gap} (it declares {needs})"
        )

    def escape(self) -> str:
        """Return the way out: what to change, in the slot's own vocabulary."""
        if self.slot.startswith("attn"):
            want = next(iter(self.coeffects.requires.values()), "none")
            return (
                f"the context must declare attn.pos={want!r} for this arm: no attention "
                "mechanism here rotates positions"
            )
        return f"drop {self.slot}={self.name!r}"


#: One expert bank's coefficient: it holds the FFN slot, and it needs no context.
BANK = Coeffects(provides={"ffn.slot": "bank", "ffn.arm": "bank"})

#: The attention mechanisms. Two, and the refusal names both.
ATTN: dict[str, Piece] = {
    "kda": Piece("attn", "kda", "kda", "kda_fwd", KDA.COEFFECTS, "delta rule, no RoPE"),
    "dsa": Piece("attn", "dsa", "dsa", "dsa_fwd", DSA.COEFFECTS, "512-key selector, no RoPE"),
}

#: The FFN slot is a bank of experts, always: it is the architecture's one choice.
FFN: dict[str, Piece] = {
    "moe": Piece("ffn", "moe", "moe", "moe_fwd", BANK, "64 routed experts + one shared"),
}

#: The arm inside an expert: the bi-sided mixture the architecture names.
EXPERT: dict[str, Piece] = {
    "moa": Piece("expert", "moa", "moa", "moa_fwd", Coeffects(), "gated activation mix"),
}

#: A mixture's dictionary is a mechanism of its own: `K` entries times `d` coefficients.
DICTIONARY = Piece(
    "dictionary",
    "dictionary",
    "ffn_dictionary",
    "ffn_dict_entries_used",
    Coeffects(),
    "an enabled dictionary entry must contribute",
)

#: The experts' rank-`r` read: a compute path, so it has a counter like any other.
TSCT = Piece(
    "tsct",
    "tsct",
    "tsct",
    "tsct_fwd",
    Coeffects(),
    "the expert's matrices are read through their rank-256 spectrum",
)

OPTIMIZER: dict[str, Piece] = {
    "muon": Piece("optimizer", "muon", "opt_muon", "opt_muon", Coeffects(), "matrices under Muon"),
    "adamw": Piece("optimizer", "adamw", "opt_adamw", "opt_adamw", Coeffects(), "AdamW only"),
}

OBJECTIVE: dict[str, Piece] = {
    "ntp": Piece("objective", "ntp", "ntp", "ntp_loss", Coeffects(), "next-token CE, chunked"),
}

QAT: dict[str, Piece] = {
    "ternary": Piece(
        "weight_qat", "ternary", "weight_qat", "weight_ternary", Coeffects(), "1.58-bit weights"
    ),
    "fp4": Piece(
        "weight_qat", "fp4", "weight_qat", "weight_fp4", Coeffects(), "e2m1 weights, e4m3 per 16"
    ),
}

QUANT: dict[str, Piece] = {
    "int4": Piece("act_quant", "int4", "act_quant", "act_int4", Coeffects(), "int4 activations"),
    "fp4": Piece(
        "act_quant", "fp4", "act_quant", "act_fp4", Coeffects(), "NVFP4 at the block boundary"
    ),
}


def pieces(spec: StackSpec, *, ctx: dict[str, str] | None = None) -> tuple[Piece, ...]:
    """Return the arms this spec enables, refusing any whose coeffects do not fit.

    `ctx` overrides the context the spec declares; it exists for the refusal tests
    and for a host that knows something the config cannot (a checkpoint's scheme).
    """
    declared = dict(spec.ctx if ctx is None else ctx)
    chosen: list[Piece] = []
    for layer, kind in enumerate(spec.pattern):
        piece = ATTN[kind]
        chosen.append(
            Piece(
                f"attn[{layer}]",
                piece.name,
                piece.mechanism,
                piece.counter,
                piece.coeffects,
                piece.why,
            )
        )
    chosen.append(FFN[spec.ffn])
    chosen.append(EXPERT[spec.expert_arm])
    chosen.append(DICTIONARY)
    chosen.append(TSCT)
    chosen.append(OPTIMIZER[spec.optimizer])
    chosen.append(OBJECTIVE["ntp"])
    if spec.weight_qat != "none":
        chosen.append(QAT[spec.weight_qat])
    if spec.act_quant != "none":
        chosen.append(QUANT[spec.act_quant])
    _check(chosen, declared)
    return tuple(chosen)


def _check(chosen: list[Piece], ctx: dict[str, str]) -> None:
    """Refuse a conflicting pair (coeffects or occupancy) before anything is built."""
    held: dict[str, Piece] = {}
    for piece in chosen:
        if piece.slot in held:
            other = held[piece.slot]
            raise LoudError(
                f"{other.name!r} and {piece.name!r} both occupy the {piece.slot} slot",
                f"one arm per slot: drop {piece.name!r}, or move it to another slot",
            )
        gap = piece.conflict(ctx)
        if gap:
            raise LoudError(gap, piece.escape())
        held[piece.slot] = piece


def declarations(enabled: tuple[Piece, ...]) -> list[tuple[str, str, str]]:
    """Return `(mechanism, counter, why)` for the arms the model's forward counts."""
    return [(p.mechanism, p.counter, p.why) for p in enabled if p.slot.split("[")[0] in MODEL_SLOTS]
