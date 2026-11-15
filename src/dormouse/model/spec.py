"""The stack as one validated value: the one architecture, and every refusal around it.

There is ONE model here. Twelve layers, attention in a 3:1 pattern of KDA and DSA
(`attn_variants/hybrid.py`), a bank of 64 rank-256 MoA experts plus one shared, a tied
head, chunked cross-entropy, Muon per head with AdamW on vectors. Every field below is
a width or a knob of that model; nothing is a menu, because a menu is how a mechanism
gets "measured" without running (this project's recorded failure class: a flag set to
a value no code path read).

The literals that remain, and the refusal each one carries:

* `unit` — `bpe` is the recipe's vocabulary (`VOCAB["bpe"]` is
  `data.tokenizer.DEFAULT_VOCAB`, priced into the token table); `byte` stays for a
  corpus that has no tokenizer yet.
* `attn` — `hybrid` is the architecture. `kda` and `dsa` build a uniform stack of one
  mechanism: the control arms a hybrid is measured against, and the only two names a
  refusal prints.
* `optimizer` — `muon` is `MixOptimizer` (matrices under Muon, vectors under AdamW).
  `adamw` is the control arm.
* `dtype` — the run's numeric policy (`train/precision.py`): `fp32` is the reference and
  `bf16` is what a real run gets. A spec read from a config that omits it is a run's
  spec and takes `bf16`; a `StackSpec(...)` built in code is a test or a notebook and
  keeps `fp32`, so `train/entry.py` is the only place that has to take `fp32` back for
  `--smoke`.
* `objective` — `ntp` is the chunked cross-entropy `train/step.py` computes. `+tst`
  and `+jepa` have no loss module, so they are refused with that reason.
* `weight_qat`, `act_quant`, `torch_compile` — the quantization slots and the compile
  flag, which contradicts `act_quant=int4` (a graph break at every fake-quant
  boundary, measured) and is refused as that pair.
* `sources` — the data streams. One name is one stream; a mix is what `+tst` cannot
  use, which is the refusal that makes this a value.

Widths live here too because a plan has to price them without allocating: the same
object answers "what will this cost" (`assemble.plan_lines`) and "what will this
build" (`assemble.assemble`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from dormouse_core.config import Config
from dormouse_core.errors import LoudError

from ..data.tokenizer import DEFAULT_VOCAB
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
DTYPES: tuple[str, ...] = ("fp32", "bf16")

#: The policy a config that omits `dtype` gets. A run is bf16 because 4 bytes of weights
#: plus 4 of gradients plus fp32 moments is what does not fit the card at batch 2.
RUN_DEFAULT = "bf16"

#: The vocab each unit prices the token table at. `bpe` is the tokenizer's own default:
#: the number exists in `data/tokenizer.py` and nowhere else, so a head priced here and a
#: tokenizer trained there cannot drift apart.
VOCAB: dict[str, int] = {"byte": 256, "bpe": DEFAULT_VOCAB}

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
    dtype: str = "fp32"

    # ------------------------------------------------------------------ wiring
    torch_compile: bool = False
    sources: str = "corpus"

    # ------------------------------------------------------------------ widths
    #: The token table's row count. 0 means "the unit's own": `bpe` takes
    #: `data.tokenizer.DEFAULT_VOCAB`, `byte` takes 256. A config that states a number is
    #: checked against the tokenizer, never trusted.
    vocab: int = 0
    #: The `tokenizer.json` a run will load. When named, the token table is checked
    #: against that file's real vocabulary (`data.tokenizer.check_vocab`), because a
    #: trained tokenizer can come up short of the size it was asked for.
    tokenizer: str = ""

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

        `vocab=0` is resolved here, before anything reads it: the number belongs to the
        tokenizer, and a config that states one keeps it so `validate` can refuse it.
        """
        if not self.vocab:
            self.vocab = VOCAB.get(self.unit, 0)
        self.validate()

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: str = "<mapping>") -> StackSpec:
        """Build from a mapping, where an omitted `dtype` means a RUN, not a test.

        The one field whose default depends on where the spec comes from. A config file —
        or an artifact's `spec` — describes a run, and a run is bf16 (`RUN_DEFAULT`);
        `train/entry.py` hands `--smoke` the reference fp32 back. A `StackSpec(...)` built
        in code is a test, a tool or a notebook, and gets fp32 with no allocator to save.
        """
        spec = super().from_mapping({"dtype": RUN_DEFAULT, **data}, source=source)
        return cast(StackSpec, spec)

    def validate(self) -> None:
        """Refuse an unknown literal or a pair of values that cannot coexist."""
        _literal("unit", self.unit, UNITS)
        _literal("attn", self.attn, ATTN)
        _literal("optimizer", self.optimizer, OPTIMIZERS)
        _literal("dtype", self.dtype, DTYPES)
        _literal("objective", self.objective, OBJECTIVES)
        _literal("weight_qat", self.weight_qat, WEIGHT_QAT)
        _literal("act_quant", self.act_quant, ACT_QUANT)
        if self.vocab != VOCAB[self.unit]:
            raise LoudError(
                f"vocab={self.vocab} is not the {self.unit} vocabulary ({VOCAB[self.unit]})",
                "the token table's size is the tokenizer's, not the config's: it is "
                "data.tokenizer.DEFAULT_VOCAB, and data.tokenizer.check_vocab is the "
                "refusal for a trained tokenizer that came up short",
            )
        if self.tokenizer:
            from ..data.tokenizer import check_vocab, load

            check_vocab(load(self.tokenizer), self.vocab)
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
