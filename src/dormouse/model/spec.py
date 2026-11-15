"""The stack as one validated value: the one architecture, and every refusal around it.

There is ONE model here. Twelve layers, attention in a 3:1 pattern of KDA and DSA
(`attn_variants/hybrid.py`), a bank of 64 rank-256 MoA experts plus one shared, a tied
head, chunked cross-entropy, Muon per head with AdamW on vectors. Every field below is
a width or a knob of that model; nothing is a menu, because a menu is how a mechanism
gets "measured" without running (this project's recorded failure class: a flag set to
a value no code path read).

The literals that remain, and the refusal each one carries:

* `unit` — `bpe` is the recipe's vocabulary (`VOCAB["bpe"] = 32768`, priced into the
  token table); the tokenizer that produces those ids is data-layer work that does not
  live in this package. `byte` stays for a corpus that has no tokenizer yet.
* `attn` — `hybrid` is the architecture. `kda` and `dsa` build a uniform stack of one
  mechanism: the control arms a hybrid is measured against, and the only two names a
  refusal prints.
* `optimizer` — `muon` is `MixOptimizer` (matrices under Muon, vectors under AdamW).
  `adamw` is the control arm.
* `objective` — `ntp` is the chunked cross-entropy `train/step.py` computes. `+tst`
  and `+jepa` have no loss module, so they are refused with that reason.
* `weight_qat`, `act_quant`, `torch_compile` — the quantization slots and the compile
  flag, which contradicts `act_quant=int4` (a graph break at every fake-quant
  boundary, measured) and is refused as that pair.
* `sources` — the data streams. One name is one stream; a mix is what `+tst` cannot
  use, which is the refusal that makes this a value.

Widths live here too because a plan has to price them without allocating: the same
object answers "what will this cost" (`assemble.plan_data`) and "what will this
build" (`assemble.assemble`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dormouse_core.config import Config
from dormouse_core.errors import LoudError

from .attn_variants.hybrid import HYBRID_3_TO_1
from .tsct import TSCT_RANK

#: Every slot's literals, in the order the plan prints them.
UNITS: tuple[str, ...] = ("bpe", "byte")
#: The stack, and the two mechanisms it is made of.
ATTN: tuple[str, ...] = ("hybrid", "kda", "dsa")
OPTIMIZERS: tuple[str, ...] = ("muon", "adamw")
OBJECTIVES: tuple[str, ...] = ("ntp", "ntp+tst", "ntp+jepa")
WEIGHT_QAT: tuple[str, ...] = ("none", "ternary")
ACT_QUANT: tuple[str, ...] = ("none", "int4")

#: The vocab each unit prices the token table at.
VOCAB: dict[str, int] = {"byte": 256, "bpe": 32768}

#: One period of the stack's attention: three KDA layers per DSA layer. It is the
#: mechanism's own declaration (`attn_variants/hybrid.py`), not a copy of it.
HYBRID: tuple[str, ...] = HYBRID_3_TO_1

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
    """The one stack: every slot, every width, and the refusals between them."""

    # ---------------------------------------------------------------- the slots
    unit: str = "bpe"
    attn: str = "hybrid"
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
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    c_lat: int = 512
    d_ffn: int = 6144
    n_experts: int = 64
    topk: int = 2
    n_shared: int = 1
    expert_tsct_rank: int = TSCT_RANK
    attn_k: int = 512
    attn_window: int = 128
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
        """Refuse an unknown literal or a pair of values that cannot coexist."""
        _literal("unit", self.unit, UNITS)
        _literal("attn", self.attn, ATTN)
        _literal("optimizer", self.optimizer, OPTIMIZERS)
        _literal("objective", self.objective, OBJECTIVES)
        _literal("weight_qat", self.weight_qat, WEIGHT_QAT)
        _literal("act_quant", self.act_quant, ACT_QUANT)
        if self.attn == "hybrid" and self.n_layers % len(HYBRID):
            raise LoudError(
                f"n_layers={self.n_layers} is not a multiple of the "
                f"{len(HYBRID)}-layer attention period {list(HYBRID)}: "
                f"{self.n_layers % len(HYBRID)} layers are left over",
                "the stack is 3 KDA layers per DSA layer; change n_layers to a "
                "multiple of 4 so the ratio holds for every layer",
            )
        if self.n_kv_heads < 1 or self.n_heads % self.n_kv_heads:
            raise LoudError(
                f"n_heads={self.n_heads} does not tile n_kv_heads={self.n_kv_heads}",
                "GQA shares one key/value head among a whole number of query heads: "
                "2 of 8 is the recipe",
            )
        if self.c_lat < 1:
            raise LoudError(
                f"c_lat={self.c_lat} is not a width",
                "the q/kv bottleneck is at least one channel; c_lat=d_model turns it off",
            )
        if self.c_lat > self.d_model:
            raise LoudError(
                f"c_lat={self.c_lat} is wider than d_model={self.d_model}",
                "a bottleneck wider than the stream it reads costs more than it saves",
            )
        if self.expert_tsct_rank < 1:
            raise LoudError(
                f"expert_tsct_rank={self.expert_tsct_rank}: this stack's experts are factorised",
                f"use {TSCT_RANK} (the measured rank); 0 was the dense arm the "
                "rank-256 experts replaced",
            )
        if self.attn_k < self.attn_window:
            raise LoudError(
                f"attn_k={self.attn_k} < attn_window={self.attn_window}: the budget is "
                "inside the window",
                "the indexer selects what the window does not already read; keep "
                "attn_k >= attn_window (512 and 128 is the recipe)",
            )
        if self.n_shared not in (0, 1):
            raise LoudError(
                f"n_shared={self.n_shared}: the design has ONE always-on expert",
                "use 0 (the parity control) or 1",
            )
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
    def ctx(self) -> dict[str, str]:
        """The shared context the components' `Coeffects` are checked against."""
        return {"attn.pos": "none", "attn.rule": "delta", "attn.selector": "true"}

    @property
    def pattern(self) -> tuple[str, ...]:
        """The attention recipe, one entry per layer."""
        period = HYBRID if self.attn == "hybrid" else (self.attn,)
        return period * (self.n_layers // len(period))

    @property
    def ffn(self) -> str:
        """The FFN slot's arm. One value: the bank is the architecture."""
        return "moe"

    @property
    def is_moe(self) -> bool:
        """Whether the FFN slot is a bank rather than one dense arm."""
        return True

    @property
    def expert_arm(self) -> str:
        """The arm inside an expert: the mixture the architecture names."""
        return "moa"

    @property
    def d_ffn_gelu(self) -> int:
        """The GELU width a mixture's `3·d·d_ffn` bill is checked against."""
        return 3 * self.d_ffn // 2
