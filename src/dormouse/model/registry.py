"""The parts a stack can be built from, each with what it needs and what proves it ran.

This module is a table, not behaviour: one `Piece` per arm, per slot, with the
`Coeffects` its class already declares (for KDA and CSA those are the classes' own
`COEFFECTS`, so the refusal comes from the mechanism rather than from a copy of it)
and the counter that has to move for the arm to have actually run.

Why a table instead of `if kind == ...` at the build site: the cross-slot refusals
are the whole point of assembling rather than hand-writing a model, and a refusal
can only be trusted if it runs before the first allocation. `check` walks the table
against the spec's declared context and refuses with the conflicting pair named;
`assemble` then builds exactly the pieces that survived.

The attention slot is per LAYER, not per stack: `attn=hybrid` is a recipe
(`attn_pattern`), and the pieces it produces are one attention arm per layer. That
is why the slot key carries the layer index — two arms in one layer is a real
conflict, and the same two arms in different layers is the hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass

from dormouse_core.coeffects import Coeffects
from dormouse_core.errors import LoudError

from .attn_variants.csa import CSA
from .attn_variants.dsa import DSA
from .attn_variants.kda import KDA
from .spec import StackSpec

#: Slots whose counter the model's forward produces. The rest (optimizer,
#: objective) are counted by the step, which is why they are not declared on the
#: stack: a mechanism the forward cannot count would raise `MissingReport` there.
MODEL_SLOTS = frozenset({"attn", "pos", "ffn", "expert", "dictionary", "weight_qat", "act_quant"})


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
            return f"set attn_pos={want!r}, or pick an arm that runs in this context"
        return f"drop {self.slot}={self.name!r}"


#: Dense causal attention with RoPE: the control every variant is measured against.
FULL = Coeffects(requires={"attn.pos": "rope"}, provides={"attn.state": "dense"})

#: One expert bank's coefficient: it holds the FFN slot, and it needs no context.
BANK = Coeffects(provides={"ffn.slot": "bank", "ffn.arm": "bank"})

#: The table. Ordered by slot so a plan prints in the order it assembles.
ATTN: dict[str, Piece] = {
    "full": Piece("attn", "full", "attention", "attn_fwd", FULL, "dense causal + RoPE"),
    "kda": Piece("attn", "kda", "kda", "kda_fwd", KDA.COEFFECTS, "delta rule, no RoPE"),
    "csa": Piece("attn", "csa", "csa", "csa_fwd", CSA.COEFFECTS, "sparse block selector, no RoPE"),
    "dsa": Piece(
        "attn",
        "dsa",
        "dsa",
        "dsa_fwd",
        DSA.COEFFECTS,
        "exact over k selected tokens + window, no RoPE",
    ),
}

FFN: dict[str, Piece] = {
    "gated": Piece("ffn", "gated", "ffn", "ffn_fwd", Coeffects(), "SwiGLU, the control"),
    "la": Piece("ffn", "la", "ffn_la", "ffn_la", Coeffects(), "learnable activation mix"),
    "moa": Piece("ffn", "moa", "ffn_moa", "ffn_moa", Coeffects(), "gated activation mix"),
    "moe": Piece("ffn", "moe", "moe", "moe_fwd", BANK, "64 routed experts + one shared"),
}

#: The arm inside an expert, when the slot is a bank. `silu`/`situ` are the plain
#: arms and add no mechanism of their own over `ffn`.
EXPERT: dict[str, Piece] = {
    "silu": Piece("expert", "silu", "ffn", "ffn_fwd", Coeffects(), "gated expert"),
    "situ": Piece("expert", "situ", "ffn", "ffn_fwd", Coeffects(), "gated expert"),
    "la": Piece("expert", "la", "ffn_la", "ffn_la", Coeffects(), "expert mixes a dictionary"),
    "moa": Piece("expert", "moa", "ffn_moa", "ffn_moa", Coeffects(), "expert gates a dictionary"),
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
#: The positional scheme, when the arm that carries one is in the stack.
ROPE = Piece(
    "pos", "rope", "rope", "rope_fwd", Coeffects(), "the rotations are the position signal"
)

#: A mixture's dictionary is a mechanism of its own: `K` entries times `d` coefficients.
DICTIONARY = Piece(
    "dictionary",
    "dictionary",
    "ffn_dictionary",
    "ffn_dict_entries_used",
    Coeffects(),
    "an enabled dictionary entry must contribute",
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
}

QUANT: dict[str, Piece] = {
    "int4": Piece("act_quant", "int4", "act_quant", "act_int4", Coeffects(), "int4 activations"),
}

#: Which table each single-valued slot reads, so `plan` and `check` cannot disagree.
TABLES: dict[str, dict[str, Piece]] = {
    "ffn": FFN,
    "optimizer": OPTIMIZER,
    "objective": OBJECTIVE,
    "weight_qat": QAT,
    "act_quant": QUANT,
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
    if spec.pos == "rope":
        chosen.append(ROPE)
    if spec.is_moe:
        chosen.append(EXPERT[spec.expert_arm])
    if spec.ffn in ("la", "moa") or (spec.is_moe and spec.expert_arm in ("la", "moa")):
        chosen.append(DICTIONARY)
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
