"""The stack as one validated value: every slot, its literals, and their refusals.

A slot is a position in the stack that holds exactly one arm, and every slot is a
closed set of literals. The set is closed because this project's silent failures
were not wrong numbers — they were flags nobody read: a key that did not exist in
the schema and was ignored, a boolean that meant "off" to one module and "on" to
another. So an unknown value here raises `LoudError` naming the alternatives, and
the cross-slot contradictions are refused at construction, before a tensor exists.

The slots, and what each one decides:

* `unit` — the sequence unit. `byte` is the vocab-256 recipe every module in this
  repo was written against; `bpe` is the vocab-32768 column the token table is
  priced at, and the tokenizer for it is data-layer work that does not live here.
* `attn` — `full` (dense causal + RoPE, the control), `kda`, `csa`, `dsa`, or
  `hybrid`, which mixes them by `attn_pattern` (`"3:1"` = three KDA layers per DSA
  layer; `"3:1:csa"` is the same ratio over the block selector).
* `attn_pos` — the positional scheme the attention slot is told it has. Empty means
  "whatever the arm needs"; a value that contradicts the arm is refused by the
  arm's own `Coeffects` in `assemble`, not here, so the refusal comes from the
  mechanism rather than from a table in this file.
* `ffn` — the FFN slot's arm: `gated`, `la`, `moa` (the dense arms of `model/ffn.py`)
  or `moe` (a bank of them). One slot, one arm.
* `activation` — the arm INSIDE an FFN. It exists for `ffn=moe`, where the expert's
  arm is a different decision from the slot's (`moe` + `activation=moa` is a bank of
  mixtures; `ffn=moa` is one dense mixture). Setting it with a dense `ffn` names a
  second arm for the same slot and is refused.
* `expert_tsct_rank` — `0` means dense experts, which is what PR #18 builds. A
  non-zero rank is a factorised expert, which nothing in this repo assembles yet.
* `optimizer` — `muon` is `MixOptimizer` (matrices under Muon, vectors under AdamW,
  `train/optim.py`). `adamw` is the control arm and is refused by the step, which
  owns a two-arm contract.
* `objective` — `ntp` is the chunked cross-entropy `train/step.py` computes. The
  `+tst` and `+jepa` arms have no loss module in any branch, so they are refused
  with that reason instead of being accepted and silently not run.
* `weight_qat`, `act_quant` — the two quantization slots (`model/quant.py`):
  ternary weights (BitNet b1.58 straight-through) and int4 activations.
* `torch_compile` — whether the stack is handed to `torch.compile`. It is a slot and
  not a flag because it contradicts `act_quant=int4`, and a contradiction a config
  cannot see is one a run discovers at hour six.
* `sources` — the data streams, comma-separated. One name is one stream; a mix is
  what `objective=ntp+tst` cannot use, which is the refusal that makes this a value.

Widths live here too because a plan has to price them without allocating: the same
object answers "what will this cost" (`assemble.plan_lines`) and "what will this
build" (`assemble.assemble`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dormouse_core.config import Config
from dormouse_core.errors import LoudError

#: Every slot's literals, in the order the plan prints them.
UNITS: tuple[str, ...] = ("byte", "bpe")
ATTN: tuple[str, ...] = ("full", "kda", "csa", "dsa", "hybrid")
ATTN_POS: tuple[str, ...] = ("rope", "none")
FFNS: tuple[str, ...] = ("gated", "la", "moa", "moe")
ACTIVATIONS: tuple[str, ...] = ("none", "silu", "situ", "la", "moa")
OPTIMIZERS: tuple[str, ...] = ("muon", "adamw")
OBJECTIVES: tuple[str, ...] = ("ntp", "ntp+tst", "ntp+jepa")
WEIGHT_QAT: tuple[str, ...] = ("none", "ternary")
ACT_QUANT: tuple[str, ...] = ("none", "int4")

#: The vocab each unit prices the token table at.
VOCAB: dict[str, int] = {"byte": 256, "bpe": 32768}

#: One period of the hybrid recipe, as `attn_pattern` spells it.
DEFAULT_PATTERN = "3:1:dsa"

#: The arms that carry a positional table of their own.
POSITIONAL: dict[str, str] = {
    "full": "rope",
    "kda": "none",
    "csa": "none",
    "dsa": "none",
    "hybrid": "none",
}

#: The sparse arms a pattern's third field may name.
SPARSE: tuple[str, ...] = ("dsa", "csa")

#: Objectives this repo can compute. The others are named in the refusal.
COMPUTABLE = frozenset({"ntp"})

#: Values a slot knows but does not assemble, with the reason said out loud.
KNOWN_UNBUILT: dict[str, dict[str, str]] = {
    "weight_qat": {
        "fp8": "weights are quantized once here: ternary (1.58 bit) or not at all; "
        "fp8 would be a second scheme on the same matrix",
    },
}


def _literal(key: str, value: str, options: Sequence[str]) -> str:
    """Return `value` if it is one of `options`; otherwise name the alternatives."""
    unbuilt = KNOWN_UNBUILT.get(key, {}).get(value)
    if unbuilt is not None:
        raise LoudError(
            f"{key}={value!r} is known and not assembled",
            f"{unbuilt}; use one of: {', '.join(options)}",
        )
    if value not in options:
        raise LoudError(
            f"{key}={value!r} is not a value this stack has",
            f"use one of: {', '.join(options)}",
        )
    return value


@dataclass
class StackSpec(Config):
    """One stack: every slot, every width, and the refusals between them."""

    # ---------------------------------------------------------------- the slots
    unit: str = "byte"
    attn: str = "full"
    attn_pattern: str = ""
    attn_pos: str = ""
    ffn: str = "gated"
    activation: str = "none"
    expert_tsct_rank: int = 0
    optimizer: str = "muon"
    objective: str = "ntp"
    weight_qat: str = "none"
    act_quant: str = "none"

    # ------------------------------------------------------------------ wiring
    torch_compile: bool = False
    sources: str = "corpus"

    # ------------------------------------------------------------------ widths
    d_model: int = 832
    n_layers: int = 12
    n_heads: int = 13
    head_dim: int = 64
    d_ffn: int = 2048
    n_experts: int = 64
    topk: int = 2
    n_shared: int = 1
    seq_len: int = 1024
    batch: int = 24

    # -------------------------------------------------------------------- plan
    tokens: int = 10_000_000_000
    throughput_tflops: float = 1.0

    # ------------------------------------------------------------- validation
    def __post_init__(self) -> None:
        """Validate on construction, not only on `from_mapping`.

        `Config.from_mapping` validates after the fact; a caller that builds a spec
        directly (the assembler, a test, a notebook) would otherwise hold a value the
        rest of the stack is not allowed to see.
        """
        self.validate()

    def validate(self) -> None:
        """Refuse an unknown literal or a pair of slots that cannot coexist."""
        _literal("unit", self.unit, UNITS)
        _literal("attn", self.attn, ATTN)
        _literal("ffn", self.ffn, FFNS)
        _literal("activation", self.activation, ACTIVATIONS)
        _literal("optimizer", self.optimizer, OPTIMIZERS)
        _literal("objective", self.objective, OBJECTIVES)
        _literal("weight_qat", self.weight_qat, WEIGHT_QAT)
        _literal("act_quant", self.act_quant, ACT_QUANT)
        if self.attn_pos:
            _literal("attn_pos", self.attn_pos, ATTN_POS)
        if self.expert_tsct_rank:
            raise LoudError(
                f"expert_tsct_rank={self.expert_tsct_rank}: a factorised expert is not "
                "assembled in this stack",
                "use 0: a bank builds dense experts (3·d·d_ffn each) and a dense FFN "
                "has no expert at all; the rank argument needs one built as two factors",
            )
        if self.ffn != "moe" and self.activation != "none":
            raise LoudError(
                f"ffn={self.ffn!r} with activation={self.activation!r}: one FFN slot, "
                "and it already holds an arm",
                "drop activation (the dense arms name their own), or set ffn=moe so "
                "the activation is the arm inside an expert",
            )
        if self.attn != "hybrid" and self.attn_pattern:
            raise LoudError(
                f"attn_pattern={self.attn_pattern!r} with attn={self.attn!r}: the "
                "pattern is the hybrid's recipe",
                "drop attn_pattern, or set attn=hybrid so a pattern has layers to tile",
            )
        if self.attn == "hybrid":
            _pattern(self.attn_pattern or DEFAULT_PATTERN, self.n_layers)
        if "tst" in self.objective and "," in self.sources:
            raise LoudError(
                f"objective={self.objective!r} over sources={self.sources!r}: the tst "
                "arm reads one contiguous stream",
                "name one source, or drop +tst; a mix makes the two terms estimators "
                "of different distributions",
            )
        if self.objective not in COMPUTABLE:
            raise LoudError(
                f"objective={self.objective!r}: no loss module computes it",
                "this stack assembles ntp (train/step.py's chunked cross-entropy); a "
                "tst or jepa term needs a second loss that no branch ships yet",
            )
        if self.act_quant == "int4" and self.torch_compile:
            raise LoudError(
                "act_quant=int4 with torch_compile=true: the straight-through estimator "
                "detaches at the block boundary, and torch._dynamo measures the extra "
                "graph break (11 breaks with int4 against 10 without, on the 4-layer "
                "stack; `torch._dynamo.explain`)",
                "set torch_compile=false, or act_quant=none; a graph that breaks at "
                "every fake-quant boundary runs the eager path under a compiled name, "
                "so the plan would price a kernel that never ran",
            )

    # ---------------------------------------------------------------- derived
    @property
    def vocab(self) -> int:
        """The token table's row count, from the sequence unit."""
        return VOCAB[self.unit]

    @property
    def pos(self) -> str:
        """The positional scheme the attention slot is told it has."""
        return self.attn_pos or POSITIONAL[self.attn]

    @property
    def ctx(self) -> dict[str, str]:
        """The shared context the components' `Coeffects` are checked against."""
        return {"attn.pos": self.pos, "attn.rule": "delta", "attn.selector": "true"}

    @property
    def pattern(self) -> tuple[str, ...]:
        """The attention recipe, one entry per layer."""
        if self.attn != "hybrid":
            return (self.attn,) * self.n_layers
        period = _pattern(self.attn_pattern or DEFAULT_PATTERN, self.n_layers)
        return period * (self.n_layers // len(period))

    @property
    def is_moe(self) -> bool:
        """Whether the FFN slot is a bank rather than one dense arm."""
        return self.ffn == "moe"

    @property
    def expert_arm(self) -> str:
        """The arm inside an expert: the `activation` slot, or the gated default."""
        return "silu" if self.activation == "none" else self.activation

    @property
    def d_ffn_gelu(self) -> int:
        """The GELU width a mixture's `3·d·d_ffn` bill is checked against."""
        return 3 * self.d_ffn // 2


def _pattern(spelling: str, n_layers: int) -> tuple[str, ...]:
    """Turn `"3:1"` into `[KDA, KDA, KDA, DSA]`, refusing one that does not tile.

    A third field names the sparse arm: `"3:1:csa"` is the block selector this
    project measured before DSA existed, `"3:1"` is the DSA default.
    """
    parts = spelling.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts[:2]):
        raise LoudError(
            f"attn_pattern={spelling!r} is not a ratio",
            "spell one period as `kda:sparse`, e.g. 3:1 (three KDA layers per DSA "
            "layer), or 3:1:csa to keep the block selector",
        )
    kda, sparse = (int(p) for p in parts[:2])
    arm = parts[2] if len(parts) == 3 else "dsa"
    if arm not in SPARSE:
        raise LoudError(
            f"attn_pattern={spelling!r} names {arm!r} as the sparse layer",
            f"the third field is one of: {', '.join(SPARSE)}; drop it for the dsa default",
        )
    if kda + sparse < 1:
        raise LoudError(
            f"attn_pattern={spelling!r} names no layer",
            "at least one of the two counts must be positive, e.g. 3:1 or 1:1",
        )
    if n_layers % (kda + sparse):
        raise LoudError(
            f"n_layers={n_layers} is not a multiple of the {kda + sparse}-layer pattern "
            f"{spelling}: {n_layers % (kda + sparse)} layers are left over",
            "the pattern is one period of the stack; change n_layers or shorten the "
            "period so it tiles exactly",
        )
    return ("kda",) * kda + (arm,) * sparse
