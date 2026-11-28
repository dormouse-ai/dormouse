"""The stack as one validated value: the one architecture, and every refusal around it.

There is ONE model here. Attention in a 3:1 pattern of KDA and DSA by default
(`attn_variants/hybrid.py`); a config may name one period of KDA/CSA/HCA instead, and
that period must tile the depth. The FFN is a bank of 64 rank-256 MoA experts plus one
shared (or one dense SwiGLU), with a tied head, chunked cross-entropy, Muon per head
with AdamW on vectors. Every field below is a width or a knob of that model; nothing
is a menu of flags nobody reads.

The literals that remain, and the refusal each one carries:

* `unit` — `bpe` is the recipe's vocabulary (`VOCAB["bpe"]` is
  `data.tokenizer.DEFAULT_VOCAB`, priced into the token table); `byte` stays for a
  corpus that has no tokenizer yet.
* `attn` — `hybrid` is the architecture. `kda` and `dsa` build a uniform stack of one
  mechanism: the control arms a hybrid is measured against, and the only two names a
  refusal prints.
* `ffn` — `moe` is the shared bank; `dense` is one full-rank SwiGLU per layer, with
  stored = active and no bank fields. The two topologies are not a menu of hacks.
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
object answers "what will this cost" (`assemble.plan_lines`) and "what will this
build" (`assemble.assemble`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from dormouse_core.config import Config
from dormouse_core.errors import LoudError

from ..data.tokenizer import DEFAULT_VOCAB
from .attn_variants.csa import CHUNK as CSA_CHUNK
from .attn_variants.csa import HCA_GROUP
from .attn_variants.hybrid import HYBRID_3_TO_1, PATTERN_KINDS
from .quant import a4_on, bank_quant_on, set_a4_config
from .tsct import TSCT_RANK, native_tern, request_native_tern

#: Every slot's literals, in the order the plan prints them.
UNITS: tuple[str, ...] = ("bpe", "byte")
#: The stack, and the uniform control arms. CSA/HCA are named by `attn_pattern`.
ATTN: tuple[str, ...] = ("hybrid", "kda", "dsa")
FFN_SLOTS: tuple[str, ...] = ("moe", "dense")
OPTIMIZERS: tuple[str, ...] = ("muon", "adamw")
OBJECTIVES: tuple[str, ...] = ("ntp", "ntp+tst", "ntp+jepa")
WEIGHT_QAT: tuple[str, ...] = ("none", "ternary", "ternary_native")
ACT_QUANT: tuple[str, ...] = ("none", "int4", "a4")
#: Where the run's tensors live; `auto` falls back to CPU when no card is present.
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

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
    #: One period of the attention slot, tiled to `n_layers`. Empty is the shipped
    #: 3:1 KDA:DSA stack; `["kda"]*6 + ["csa", "hca"]` is the sparse global arm.
    attn_pattern: list[str] = field(default_factory=list)
    ffn: str = "moe"
    optimizer: str = "muon"
    objective: str = "ntp"
    weight_qat: str = "none"
    act_quant: str = "none"

    # ------------------------------------------------------------------ wiring
    torch_compile: bool = False
    sources: str = "corpus"
    device: str = "auto"

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
    #: Dense FFN factorisation: 0 is the full-rank read; 256 stores each of the
    #: three FFN matrices as `[in, r] . [r, out]`. With `weight_qat=ternary_native`
    #: the factors train as ternary from step 0.
    ffn_tsct_rank: int = 0
    #: Recompute each block in the backward instead of retaining its activations.
    block_checkpoint: bool = False
    #: Price the head one sequence chunk at a time; off is the full logits tensor.
    chunked_loss: bool = False
    #: The dense ideal bank: one router-addressed U/V pair of big GEMMs instead of
    #: per-expert dispatch. Off is the shipped bank, bit for bit.
    ideal_bank: bool = False
    #: Slots in the dense bank; each keeps `expert_tsct_rank` channels.
    ideal_slots: int = 32
    attn_k: int = 512
    attn_window: int = 128
    seq_len: int = 1024
    batch: int = 24

    # ---------------------------------------------------------------- loops
    #: Shared-block recurrence over the middle of the stack. Fractions, not layer
    #: indices, so the same architecture scales from 8 layers to 64 without a code change.
    loop_passes: int = 1
    loop_start_frac: float = 0.25
    loop_end_frac: float = 0.75
    loop_rank: int = 8

    # -------------------------------------------------------------------- plan
    tokens: int = 10_000_000_000
    throughput_tflops: float = 1.0

    @property
    def loop_window_fraction(self) -> float:
        """Fraction of the depth re-executed by one extra loop pass."""
        return max(0.0, min(1.0, self.loop_end_frac - self.loop_start_frac))

    @property
    def loop_factor(self) -> float:
        """Step FLOP multiplier from the shared-block passes; 1.0 when off."""
        return 1.0 + (self.loop_passes - 1) * self.loop_window_fraction

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

    def validate(self) -> None:
        """Refuse an unknown literal or a pair of values that cannot coexist."""
        _literal("unit", self.unit, UNITS)
        _literal("attn", self.attn, ATTN)
        _literal("ffn", self.ffn, FFN_SLOTS)
        _literal("optimizer", self.optimizer, OPTIMIZERS)
        _literal("objective", self.objective, OBJECTIVES)
        _literal("weight_qat", self.weight_qat, WEIGHT_QAT)
        _literal("act_quant", self.act_quant, ACT_QUANT)
        _literal("device", self.device, DEVICES)
        #: The config is the mechanism. A4 turns on the bank's quant+codesave here,
        #: with no env flag to forget; every other spec turns it back off.
        set_a4_config(self.act_quant == "a4")
        request_native_tern(self.weight_qat == "ternary_native")
        if self.weight_qat == "ternary_native" and not native_tern():
            raise LoudError(
                "weight_qat='ternary_native' needs DORMOUSE_TSCT_NATIVE_TERN=1",
                "the native arm is explicit: a config that names ternary without the flag "
                "would run fp32 factors under a ternary name",
            )
        if native_tern() and self.weight_qat != "ternary_native":
            raise LoudError(
                f"DORMOUSE_TSCT_NATIVE_TERN=1 with weight_qat={self.weight_qat!r}: the "
                "forward would read fp32 factors",
                "set weight_qat='ternary_native' to train ternaries from step 1, or unset "
                "the flag; a postwar conversion is not this arm",
            )
        if self.weight_qat == "ternary_native" and self.torch_compile:
            raise LoudError(
                "weight_qat='ternary_native' with torch_compile=true",
                "the STE is an autograd.Function boundary that dynamo breaks; the compiled "
                "step would report a graph it cannot keep. Set torch_compile=false",
            )
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
        if self.attn_pattern:
            if self.attn != "hybrid":
                raise LoudError(
                    f"attn_pattern={list(self.attn_pattern)} with attn={self.attn!r}",
                    "a pattern names a hybrid slot; set attn='hybrid', or leave "
                    "attn_pattern empty for a uniform stack",
                )
            if not isinstance(self.attn_pattern, (list, tuple)) or not all(
                isinstance(name, str) for name in self.attn_pattern
            ):
                raise LoudError(
                    f"attn_pattern={self.attn_pattern!r} is not a list of names",
                    f"write a TOML array of {', '.join(PATTERN_KINDS)}",
                )
            unknown = [name for name in self.attn_pattern if name not in PATTERN_KINDS]
            if unknown:
                raise LoudError(
                    f"attn_pattern names {unknown[0]!r}, which is not an attention mechanism",
                    f"use one of: {', '.join(PATTERN_KINDS)}; DSA is the default period",
                )
            if self.n_layers % len(self.attn_pattern):
                raise LoudError(
                    f"n_layers={self.n_layers} is not a multiple of the "
                    f"{len(self.attn_pattern)}-layer pattern {list(self.attn_pattern)}: "
                    f"{self.n_layers % len(self.attn_pattern)} layers are left over",
                    "the pattern is one period of the stack; change n_layers to a "
                    "multiple of it, or shorten the pattern so it tiles",
                )
            if "hca" in self.attn_pattern and self.seq_len % (CSA_CHUNK * HCA_GROUP):
                raise LoudError(
                    f"seq_len={self.seq_len} does not hold a whole HCA coarse block "
                    f"of {CSA_CHUNK * HCA_GROUP} tokens",
                    f"set seq_len to a multiple of {CSA_CHUNK * HCA_GROUP}, or leave "
                    "hca out of the pattern",
                )
        elif self.attn == "hybrid" and self.n_layers % len(HYBRID):
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
        if self.ideal_slots < 1:
            raise LoudError(
                f"ideal_slots={self.ideal_slots} builds no dense bank",
                "32 slots at rank 256 is the measured width; raise it for more slots",
            )
        if self.ideal_bank and self.topk > self.ideal_slots:
            raise LoudError(
                f"topk={self.topk} exceeds ideal_slots={self.ideal_slots}",
                "the gate mask needs one slot per pick; raise ideal_slots or lower topk",
            )
        if self.ffn == "moe" and self.expert_tsct_rank < 1:
            raise LoudError(
                f"expert_tsct_rank={self.expert_tsct_rank}: this stack's experts are factorised",
                f"use {TSCT_RANK} (the measured rank); 0 is the dense arm, which "
                "needs ffn='dense' and no bank",
            )
        if self.ffn_tsct_rank < 0:
            raise LoudError(
                f"ffn_tsct_rank={self.ffn_tsct_rank}: a negative rank stores nothing",
                "0 is the dense read; 256 is the factorised recipe",
            )
            raise LoudError(
                f"expert_tsct_rank={self.expert_tsct_rank}: this stack's experts are factorised",
                f"use {TSCT_RANK} (the measured rank); 0 is the dense arm, which "
                "needs ffn='dense' and no bank",
            )
        if self.ffn == "dense":
            if self.expert_tsct_rank != 0:
                raise LoudError(
                    f"ffn='dense' with expert_tsct_rank={self.expert_tsct_rank}",
                    "a dense FFN stores full matrices: set expert_tsct_rank=0; the "
                    f"factorised arm is ffn='moe' with rank {TSCT_RANK}",
                )
            if self.n_experts or self.topk or self.n_shared:
                raise LoudError(
                    "ffn='dense' names bank fields "
                    f"(n_experts={self.n_experts}, topk={self.topk}, n_shared={self.n_shared})",
                    "a dense stack has no bank: set all three to 0, or use ffn='moe'",
                )
        if self.attn_k < self.attn_window:
            raise LoudError(
                f"attn_k={self.attn_k} < attn_window={self.attn_window}: the budget is "
                "inside the window",
                "the indexer selects what the window does not already read; keep "
                "attn_k >= attn_window (512 and 128 is the recipe)",
            )
        if self.loop_passes < 1:
            raise LoudError(
                f"loop_passes={self.loop_passes}: a loop runs at least once",
                "use 1 for the dense control, 2 for one extra pass",
            )
        if self.loop_passes > 1 and not (0.0 <= self.loop_start_frac < self.loop_end_frac <= 1.0):
            raise LoudError(
                f"loop window [{self.loop_start_frac}, {self.loop_end_frac}] is invalid",
                "loop_start_frac < loop_end_frac inside [0, 1]",
            )
        if self.loop_rank < 1:
            raise LoudError(
                f"loop_rank={self.loop_rank}: a zero-rank adapter writes nothing",
                "use a positive rank; 8 is the recipe",
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
        if bank_quant_on() and self.torch_compile:
            raise LoudError(
                "the INT4 bank boundary (DORMOUSE_BANK_A4=1 or act_quant='a4') with "
                "torch_compile=true crosses a callable that dynamo cannot trace",
                "set torch_compile=false, or leave the bank flags off and use act_quant='none'; "
                "a graph that breaks at every bank read must not be priced as a compiled step",
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
        if self.act_quant == "a4" and self.torch_compile:
            raise LoudError(
                "act_quant=a4 with torch_compile=true: the save/reload boundary keeps "
                "packed codes in autograd, so dynamo splits the step there",
                "set torch_compile=false, or act_quant=none; the A4 path is eager",
            )
        if self.act_quant == "a4" and self.weight_qat != "ternary_native":
            raise LoudError(
                f"act_quant=a4 with weight_qat={self.weight_qat!r}: a half-quantised "
                "stack is not the model (4-bit activations against wider weights)",
                'set weight_qat="ternary_native" from step 0; no warm-up, no ramp',
            )
        if self.weight_qat == "ternary_native" and self.act_quant == "int4":
            raise LoudError(
                "weight_qat='ternary_native' with act_quant='int4': two quant schemes "
                "on one stack is the other half-quantised combination",
                'set act_quant="a4" to pair native ternary weights with A4, or drop '
                "act_quant; ternary_native alone is its own documented arm",
            )
        if a4_on() and self.act_quant != "a4":
            raise LoudError(
                f"DORMOUSE_ACT_A4=1 but act_quant={self.act_quant!r}",
                'the model flag asserts the native a4+ternary path: set act_quant="a4" '
                'and weight_qat="ternary_native" in the config, or unset the flag to '
                "train the bf16 control",
            )

    # ---------------------------------------------------------------- derived
    @property
    def ctx(self) -> dict[str, str]:
        """The shared context the components' `Coeffects` are checked against."""
        return {"attn.pos": "none", "attn.rule": "delta", "attn.selector": "true"}

    @property
    def pattern(self) -> tuple[str, ...]:
        """The attention recipe, one entry per layer."""
        if self.attn_pattern:
            period = tuple(self.attn_pattern)
        else:
            period = HYBRID if self.attn == "hybrid" else (self.attn,)
        return period * (self.n_layers // len(period))

    @property
    def is_moe(self) -> bool:
        """Whether the FFN slot is a bank rather than one full-rank arm per layer."""
        return self.ffn == "moe"

    @property
    def expert_arm(self) -> str:
        """The arm inside an expert: the mixture the architecture names."""
        return "moa"

    @property
    def d_ffn_gelu(self) -> int:
        """The GELU width a mixture's `3·d·d_ffn` bill is checked against."""
        return 3 * self.d_ffn // 2
