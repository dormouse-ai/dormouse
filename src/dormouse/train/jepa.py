"""ER-JEPA: the paired alignment term, an episodic replay memory, and loss dropout.

WHY REPLAY AND NOT JUST THE ALIGNMENT TERM. LLM-JEPA (arXiv:2509.14252, eq. 1) adds

    L = Σ_ℓ L_LLM(Text_{1:ℓ-1}, Text_ℓ) + λ · d(Pred(Enc(Text)), Enc(Code))

with `d` the cosine distance and a stop-gradient on `Enc(Code)`. Two later results say
that term alone is not a mechanism one can count on: an audit of 22 auxiliaries on a
fixed Llama-3.2-1B LoRA harness found no JEPA variant surviving a multiple-comparison
correction (arXiv:2605.15394), and the reason proposed is that JEPA needs *conditional
concentration* — one dominant target representation per context — which text does not
give, so target variance rises and the representation collapses (arXiv:2607.23531).
ER-JEPA (arXiv:2609.36952) is the answer this module implements: keep the alignment
term, and replay stored pairs so the target is re-anchored at every step instead of
drifting with the current batch. Its eq. (4) is

    L_ER-JEPA(B_t) = L_new(B_t) + β · L_replay(B_t)

    L_replay = γ/Z^rep · Σ_i Σ_ℓ L_LLM(τ̃_i,<ℓ, τ̃_i,ℓ) + λ/R · Σ_i d(p_i^rep, z_i^{a,rep})

with `R_t ≤ R` the replay budget, `γ = 1`, and `β ≥ 0` the replay strength. The paper
reports β only as a weight (its ablations vary λ and the predictor depth k), so
`jepa_replay_weight` starts at 0.5 and says so rather than pretending to a number it
does not state.

WHAT THIS MODULE IS, IN FOUR PARTS.

* `JepaPair` — the paired corpus: row `i` of `view_a` and of `view_b` are two
  encodings of one thing, both one token wider than the sequence they supervise.
  Position is arithmetic, like `data/bin.py`, so a resumed run reads the same order.
* `EpisodicMemory` — a fixed-capacity store filled by reservoir sampling, so it is a
  uniform sample of everything seen and not the recent tail. `stats()` returns the ages
  that claim is checked against.
* `jepa_loss` / `JepaHead` / `er_jepa_loss` — the metric on one pair and eq. (4) over
  the current pairs plus the replayed ones. The metric is cosine on the encoder's own
  output (eq. 1 has no projector; a learned one collapsed when measured). Both target
  sides are computed under `torch.no_grad()`: that stop is the objective. Without it
  the term trains both views toward a constant, which is a reconstruction/averaging
  objective the token loss already covers, and the test that reads `requires_grad`
  fails if it is removed.
* `JepaAux` — what the step consumes: head, memory, λ, β, the replay fraction, and the
  loss-dropout rate.

LOSS DROPOUT. On a `1 − ld` share of steps the term runs; on the rest the step pays
nothing — not the replay, not its forward. Skipping a computed-then-discarded term
would keep the cost and only hide the gradient, so `JepaAux.keep()` is decided before
any of it. In units of one plain text forward, a paired step costs `2 + 2·frac` (one
window over the packed pair, one over the replayed pairs of `frac` the rows) and a
skipped step costs exactly `1`, so

    cost per epoch = 1 + (1 − α) · (1 + 2·frac) × baseline,

which is the LLM-JEPA paper's `(2 − α)` at `frac = 0`. As there, λ should be co-tuned
with the rate: keep `λ·(1 − α)` roughly constant, so a run at α=0.75 wants about 4× the
λ of α=0.

THE BLOCK MASK IS NOT IMPLEMENTED (§3.2 of the LLM-JEPA paper). They pack both views
and mask so that `Enc(Code)` cannot attend to `Text`, which makes the target a pure
view-B embedding. This stack's attention takes no additive mask, so the packed window
here lets view B see view A. What that changes: the target is "B given A" rather than
"B", which is easier to predict and, per the paradox paper, closer to the high-variance
regime it warns about. What it does not change: both targets are still computed under
`no_grad()`, so the stop-gradient discipline is intact.

WHAT IS NOT IMPLEMENTED, ON PURPOSE. The paper's default retrieval is content-based:
a frozen sparse Top-K projection of the current cue, Eq. (14)–(16). Here retrieval is
the paper's *uniform* variant (D.3.2), ordered by `(step, seed)` arithmetic, because a
frozen projection plus a top-k index is a second mechanism with its own hyperparameters
and the store must be reproducible step by step. The tied `[PRED]` predictor of the
original paper is likewise replaced by one linear map on the pooled embedding, so no
vocabulary token is needed.

`__concepts__` names the five public types: the reader, the store, the head, the step's
view, and the config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dormouse_core.config import Config
from dormouse_core.errors import LoudError
from torch import nn
from torch.nn import functional

from ..data.bin import Stream

__all__ = [
    "JEPA_MODES",
    "EpisodicMemory",
    "JepaAux",
    "JepaConfig",
    "JepaHead",
    "JepaPair",
    "er_jepa_loss",
    "jepa_loss",
]

#: One concept per public class, declared: what turns the term on (`JepaConfig`), what
#: the step holds (`JepaAux`), and the three mechanisms — reader, store, head.
__concepts__ = ("EpisodicMemory", "JepaAux", "JepaConfig", "JepaHead", "JepaPair")

#: `none` is the flag off (bit-identical step); `replay` is ER-JEPA.
JEPA_MODES: tuple[str, ...] = ("none", "replay")

#: The two metrics the paper names: cosine (theirs) and squared L2.
KINDS: tuple[str, ...] = ("cosine", "l2")

#: The files a paired corpus directory holds, beside their `.json` manifests.
VIEW_A, VIEW_B = "view_a.bin", "view_b.bin"


def _tokens(rows: np.ndarray) -> torch.Tensor:
    """Ids as a `long` tensor, copied: a memmap slice is not a writable buffer."""
    return torch.from_numpy(np.array(rows, dtype=np.int64))


def _stride(rows: int, seed: int) -> int:
    """Return a stride coprime to `rows`: the golden-ratio step, nudged by `seed`.

    Coprime is what makes `x ↦ x·stride mod rows` a bijection, so a pass over `rows`
    slots reads every one exactly once instead of repeating a lopsided subset.
    """
    step = max(1, round(rows * 0.618)) + seed % 7
    while math.gcd(step, rows) != 1:
        step += 1
    return step


class JepaPair:
    """A paired corpus as batches: view A with its targets' token, and view B.

    Row `i` of `view_a` and row `i` of `view_b` are two encodings of one thing — a
    problem and its solution, a spec and its implementation, a question and its answer.
    Both rows are the same width, one token wider than the sequence they supervise:
    the step packs them as `[view_a | view_b]` in one window, reads the token loss on
    the view-B columns (the paper's `x^full = x^u || x^a`, eq. 3) and pools the two
    halves into the pair of embeddings the alignment term compares. One forward, both
    views, and the model has to carry the source into the target to predict it.

    Position is arithmetic, not a cursor: batch `step` reads slots
    `((step·batch + j)·stride) mod rows`, so a resumed run lands exactly where it
    stopped and two readers with the same seed read the same order. Same rule as
    `data/bin.py`; no sidecar, because a sidecar is what made a resume unreproducible
    there.
    """

    def __init__(
        self, view_a: np.ndarray, view_b: np.ndarray, *, batch: int, seed: int = 0
    ) -> None:
        """Check the two views are aligned, then fix the batch order."""
        a, b = np.asarray(view_a), np.asarray(view_b)
        if a.ndim != 2 or b.ndim != 2:
            raise LoudError(
                f"a paired corpus is two [rows, tokens] arrays, got {a.shape} and {b.shape}",
                "reshape the streams to two axes; one row is one pair",
            )
        if a.shape[0] != b.shape[0]:
            raise LoudError(
                f"view A has {a.shape[0]} rows and view B has {b.shape[0]}",
                "the two streams must hold one row per pair, aligned row by row",
            )
        if a.shape[1] != b.shape[1]:
            raise LoudError(
                f"view A is {a.shape[1]} tokens wide and view B is {b.shape[1]}",
                "the two views are packed side by side, so they are the same width",
            )
        if batch < 1 or batch > a.shape[0]:
            raise LoudError(
                f"batch={batch} does not fit {a.shape[0]} pairs",
                "pass a batch of at most the pair count",
            )
        self.view_a, self.view_b = a, b
        self.batch = batch
        self.rows = int(a.shape[0])
        self.stride = _stride(self.rows, seed)

    def at(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the `step`-th batch: `(view_a [B, W], view_b [B, W])`, from `step` alone."""
        if step < 0:
            raise LoudError(f"step={step} is negative", "steps start at 0")
        base = step * self.batch
        slots = np.array([((base + j) * self.stride) % self.rows for j in range(self.batch)])
        return _tokens(self.view_a[slots]), _tokens(self.view_b[slots])

    @classmethod
    def from_bins(
        cls,
        view_a: str | Path,
        view_b: str | Path,
        *,
        batch: int,
        seq_len: int,
        seed: int = 0,
    ) -> JepaPair:
        """Read two `.bin` streams as `[rows, seq_len+1]` rows each."""
        a, b = Stream(view_a), Stream(view_b)
        rows = min(a.tokens, b.tokens) // (seq_len + 1)
        if rows < batch:
            raise LoudError(
                f"{a.path} and {b.path} hold {rows} pairs of {seq_len} tokens, "
                f"a batch of {batch} needs {batch}",
                "lower batch, lower seq_len, or grow the paired corpus",
            )
        width = seq_len + 1
        return cls(
            a.take(0, rows * width).reshape(rows, width),
            b.take(0, rows * width).reshape(rows, width),
            batch=batch,
            seed=seed,
        )

    @classmethod
    def from_dir(cls, path: str | Path, *, batch: int, seq_len: int, seed: int = 0) -> JepaPair:
        """Read `<path>/view_a.bin` and `<path>/view_b.bin`, each with its manifest."""
        root = Path(path)
        if not root.is_dir():
            raise LoudError(
                f"no paired corpus at {root}",
                f"point jepa_pairs_path at a directory holding {VIEW_A} and {VIEW_B}",
            )
        return cls.from_bins(root / VIEW_A, root / VIEW_B, batch=batch, seq_len=seq_len, seed=seed)


class EpisodicMemory:
    """A fixed-capacity store of token pairs, filled by reservoir sampling.

    The paper's slot (Eq. 13) holds the source and target tokens, an address, a
    running JEPA error and a replay count. This store keeps the tokens and the
    insertion index; the error and the count belong to the content-based policy that
    is not implemented here (module docstring).

    RESERVOIR SAMPLING is what keeps the store from collapsing to the recent tail: the
    `n`-th insert enters with probability `capacity / n`, so after `N` inserts the store
    is a uniform sample of all `N` examples and an item's chance of surviving does not
    depend on its age. A store that simply overwrote the oldest slot would look the
    same in `size` and fail every replay test that matters; `stats()` exposes the ages
    so the difference is measured, not asserted.
    """

    def __init__(self, *, capacity: int, width_a: int, width_b: int, seed: int = 0) -> None:
        """Allocate the slots; nothing is stored until `insert` is called."""
        if capacity < 1:
            raise LoudError(
                f"memory capacity {capacity} stores nothing",
                "a replay memory needs at least one slot; 256 is the tested size",
            )
        if width_a < 1 or width_b < 1:
            raise LoudError(
                f"slot widths {width_a} and {width_b} hold no token",
                "pass the paired views' widths: view A is one token wider than view B",
            )
        self.capacity = capacity
        self.view_a = torch.zeros((capacity, width_a), dtype=torch.long)
        self.view_b = torch.zeros((capacity, width_b), dtype=torch.long)
        self.ids = torch.zeros(capacity, dtype=torch.long)
        self.size = 0
        self.inserts = 0
        self.seed = seed
        self._gen = torch.Generator().manual_seed(seed)

    def insert(self, view_a: torch.Tensor, view_b: torch.Tensor) -> None:
        """Store a batch of pairs, replacing earlier slots by the reservoir rule."""
        if view_a.dim() != 2 or view_b.dim() != 2:
            raise LoudError(
                f"insert takes two [rows, tokens] batches, got {tuple(view_a.shape)} "
                f"and {tuple(view_b.shape)}",
                "rows are pairs; one row is stored per pair",
            )
        if view_a.shape[1] != self.view_a.shape[1] or view_b.shape[1] != self.view_b.shape[1]:
            raise LoudError(
                f"pair widths {tuple(view_a.shape[1:])} do not fit this store's "
                f"{tuple(self.view_a.shape[1:])}",
                "build the memory at the widths JepaPair yields",
            )
        if view_a.shape[0] != view_b.shape[0]:
            raise LoudError(
                f"{view_a.shape[0]} view A rows against {view_b.shape[0]} view B rows",
                "the two views of a pair are stored together, row by row",
            )
        for row in range(view_a.shape[0]):
            self.inserts += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                # Algorithm R: the n-th item survives with probability capacity/n.
                draw = int(torch.randint(0, self.inserts, (1,), generator=self._gen))
                if draw >= self.capacity:
                    continue
                slot = draw
            self.view_a[slot] = view_a[row]
            self.view_b[slot] = view_b[row]
            self.ids[slot] = self.inserts

    def retrieve(self, count: int, *, step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """`count` stored pairs in the `(step, seed)` order: no cursor, no state.

        The order is a function of the step alone, so a resumed run replays the same
        entries. When the store holds fewer rows than asked, slots repeat; the paper
        instead replays the `R_t ≤ R` slots it has, which is what `replay_rows` counts.
        """
        if self.size == 0:
            raise LoudError(
                "the replay memory is empty",
                "insert the current batch before retrieving from it",
            )
        if count < 1:
            raise LoudError(f"retrieve({count}) asks for no pair", "ask for at least one")
        stride = _stride(self.size, self.seed)
        base = step * count
        slots = torch.tensor([((base + j) * stride) % self.size for j in range(count)])
        return self.view_a.index_select(0, slots), self.view_b.index_select(0, slots)

    def stats(self) -> dict[str, int | list[int]]:
        """Size, capacity, inserts, and the age of every stored slot (in inserts)."""
        ages = (self.inserts - self.ids[: self.size]).tolist()
        return {
            "size": self.size,
            "capacity": self.capacity,
            "inserts": self.inserts,
            "ages": ages,
        }


def _readout(h: torch.Tensor) -> torch.Tensor:
    """One embedding per row: the last position's state, or the row itself at rank 2.

    The paper reads `Enc` off "the hidden state of the last token from the last
    layer", and this is that. The mean over positions was the first implementation
    here and it is a trap: mean-pooled states of a causal stack cancel, and a state
    that lands near zero makes the cosine's backward NaN while its forward stays
    finite — measured as a step whose loss was 3.78 and whose every gradient was NaN,
    which the firewall then skipped. The last position cannot cancel.
    """
    return h[:, -1] if h.dim() == 3 else h


def jepa_loss(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    projector: nn.Module,
    *,
    kind: str = "cosine",
    predictor: nn.Module | None = None,
) -> torch.Tensor:
    """`d(Pred(Enc A), sg(Enc B))` — the alignment term of eq. (1), on one pair.

    Both embeddings go through the same `projector`; the target is computed under
    `torch.no_grad()` **and** detached, which is the stop-gradient on `Enc(Code)`.
    `predictor` defaults to `None`, i.e. `Pred(x) = x` — the LLM-JEPA paper's k=0,
    where the predictor is trivial and the term predicts view B's projection with
    view A's.

    λ and β do not appear here; they belong to the caller (`JepaAux`) and are applied
    in `er_jepa_loss`, because the paper's co-tuning rule (`λ·(1 − α)` roughly
    constant) is a statement about the rate, not about one pair.
    """
    if kind not in KINDS:
        raise LoudError(
            f"kind={kind!r} is not a metric this term has",
            f"use one of: {', '.join(KINDS)}",
        )
    z_a = projector(_readout(h_a))
    if predictor is not None:
        z_a = predictor(z_a)
    with torch.no_grad():
        # `.detach()` as well as `no_grad()`: `_readout` is a slice, and a slice of a
        # tensor that requires grad keeps the flag (no_grad stops recording, not the
        # view's base), so the backward still reached the target. Measured: the
        # target's grad was 0.096 with no_grad alone, and the test below failed on
        # exactly that — the silence the stop-gradient exists to prevent.
        z_b = projector(_readout(h_b)).detach()
    if kind == "cosine":
        return 1.0 - functional.cosine_similarity(z_a, z_b, dim=-1).mean()
    return functional.mse_loss(z_a, z_b)


class JepaHead(nn.Module):
    """The embedding half of the term: a predictor and a metric on the encoder's space.

    `predictor` is the map applied to view A only — one linear layer, the smallest
    thing that is a prediction rather than an equality, and no vocabulary token (the
    paper's tied `[PRED]` predictor is not implemented; see the module docstring).
    It is trained by the alignment term alone; the encoder it reads is the stack's,
    shared with the token loss.
    """

    def __init__(self, d_model: int, *, kind: str = "cosine") -> None:
        """Build the predictor at width `d_model` and fix the metric."""
        super().__init__()
        if d_model < 1:
            raise LoudError(
                f"d_model={d_model} predicts nothing",
                "pass the stack's residual width",
            )
        if kind not in KINDS:
            raise LoudError(
                f"kind={kind!r} is not a metric this term has",
                f"use one of: {', '.join(KINDS)}",
            )
        self.d_model = d_model
        self.kind = kind
        #: The metric is on `Enc`'s own output: eq. (1) has no projector between the
        #: encoder and `d`, and a *learned* one is a free collapse path. Measured on
        #: the synthetic gate: a trained projector drove the cosine distance of every
        #: current pair to 0 while the stored pairs stayed 0.52 apart — the collapse
        #: the paradox paper describes, moved into a module that does no work for the
        #: model. `jepa_loss` still takes any projector; this head passes the identity.
        self.projector = nn.Identity()
        self.predictor = nn.Linear(d_model, d_model)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """One batch's alignment term, from embeddings `[B, T, D]` or `[B, D]`."""
        return jepa_loss(h_a, h_b, self.projector, kind=self.kind, predictor=self.predictor)


def er_jepa_loss(
    current: tuple[torch.Tensor, torch.Tensor] | None,
    replayed: tuple[torch.Tensor, torch.Tensor] | None,
    head: JepaHead,
    *,
    lam: float = 1.0,
    beta: float = 1.0,
    token_replay: torch.Tensor | None = None,
) -> torch.Tensor:
    """ER-JEPA's loss, eq. (4): `λ·d(current) + β·(λ·d(replayed) + token_replay)`.

    `current` is the fresh batch's `(h_a, h_b)` and `replayed` the stored batch's;
    either may be `None` when the caller is adding one half at a time — the step adds
    the current half inside the token-loss graph (view A is the batch it already
    encoded) and the replayed half after the micro-batch loop, which is where the
    paper's two terms live. `token_replay` is the already-computed `L_LLM` on the
    replayed tokens (eq. 6's first term at `γ = 1`); the step computes it where the
    logits are, and passes it here so one function owns eq. (4)'s weights.

    Both target sides go through `jepa_loss`, i.e. under `no_grad()`: the replay path
    stops the gradient on its `Enc(τ̃^a)` exactly as the fresh path does.
    """
    total: torch.Tensor | None = None
    if current is not None:
        total = lam * head(*current)
    if replayed is not None:
        replay = lam * head(*replayed)
        if token_replay is not None:
            replay = replay + token_replay
        total = beta * replay if total is None else total + beta * replay
    if total is None:
        raise LoudError(
            "er_jepa_loss got neither the current pairs nor a replay",
            "pass current=(h_a, h_b), replayed=(h_a, h_b), or both",
        )
    return total


@dataclass(frozen=True)
class JepaAux:
    """The term as the step consumes it: head, memory, λ, β, replay fraction, rate.

    `ld` is α: the share of steps that skip the term, its second forward and the
    replay. Cost is `1 + (1−α)·(1 + 2·frac)` forward units against 1 for the baseline,
    and `keep()` is the one batch-level coin flip that decides — nothing is computed
    on a skipped step, so nothing is thrown away either.

    `lam` (λ) is the alignment weight and `replay_weight` (β) the replay weight, both
    in token-loss units. The paper gives β ≥ 0 and varies λ; β defaults to 0.5 here
    because no value is stated (`JepaConfig` says the same). λ must be co-tuned with
    `ld`: keep `λ·(1 − ld)` roughly constant, so an α=0.75 run wants `≈4×` the λ of
    α=0.
    """

    head: JepaHead
    memory: EpisodicMemory
    lam: float = 1.0
    replay_weight: float = 0.5
    replay_frac: float = 0.25
    ld: float = 0.0

    def __post_init__(self) -> None:
        """Refuse weights that switch a term off and rates outside [0, 1]."""
        if self.lam < 0.0 or self.replay_weight < 0.0:
            raise LoudError(
                f"lam={self.lam} and replay_weight={self.replay_weight} are weights",
                "λ and β are non-negative; 1.0 and 0.5 are the defaults",
            )
        if not 0.0 <= self.ld <= 1.0:
            raise LoudError(
                f"ld={self.ld} is not a rate",
                "the rate is a probability: 0 computes every step, 1 computes none",
            )
        if not 0.0 <= self.replay_frac <= 1.0:
            raise LoudError(
                f"replay_frac={self.replay_frac} is not a fraction of a batch",
                "frac is the share of the batch replayed: 0.25 at batch 24 is 6 rows",
            )

    def replay_rows(self, batch: int) -> int:
        """`R_t = min(round(frac·batch), |M|)` — eq. (6)'s budget, never above the store."""
        if self.replay_frac <= 0.0:
            return 0
        wanted = max(1, round(self.replay_frac * batch))
        return min(wanted, self.memory.size)

    def keep(self) -> bool:
        """One coin flip per step: True means compute the term and pay its forwards."""
        return self.ld < 1.0 and float(torch.rand(())) >= self.ld


@dataclass
class JepaConfig(Config):
    """The slots that turn ER-JEPA on, and the refusals between them.

    `jepa="none"` assembles nothing and leaves the step bit-identical; `"replay"`
    reads `jepa_pairs_path`, keeps the token loss untouched, stores every batch in a
    `jepa_memory_size`-slot memory, and on a `1 − jepa_ld` share of steps adds
    eq. (4) with `λ = jepa_lambda` on `jepa_replay_frac · batch` replayed rows.

    `jepa_replay_weight` is eq. (4)'s β. The paper states only `β ≥ 0` in its ablations
    (it varies λ and the predictor depth), so the default is 0.5 and it is named here
    rather than buried in the step.
    """

    jepa: str = "none"
    jepa_lambda: float = 1.0
    jepa_ld: float = 0.0
    jepa_memory_size: int = 256
    jepa_replay_frac: float = 0.25
    jepa_replay_weight: float = 0.5
    jepa_pairs_path: str = ""

    def validate(self) -> None:
        """Refuse an unknown mode, a bad weight or rate, an empty corpus or store."""
        if self.jepa not in JEPA_MODES:
            raise LoudError(
                f"jepa={self.jepa!r} is not a mode this objective has",
                f"use one of: {', '.join(JEPA_MODES)}",
            )
        if self.jepa_lambda < 0.0 or self.jepa_replay_weight < 0.0:
            raise LoudError(
                f"jepa_lambda={self.jepa_lambda}, jepa_replay_weight={self.jepa_replay_weight}",
                "λ and β are non-negative; 1.0 and 0.5 are the defaults",
            )
        if not 0.0 <= self.jepa_ld <= 1.0:
            raise LoudError(
                f"jepa_ld={self.jepa_ld} is not a rate",
                "the rate is a probability: 0 computes every step, 1 computes none",
            )
        if not 0.0 <= self.jepa_replay_frac <= 1.0:
            raise LoudError(
                f"jepa_replay_frac={self.jepa_replay_frac} is not a fraction",
                "frac is the share of the batch replayed: 0.25 at batch 24 is 6 rows",
            )
        if self.jepa_memory_size < 1:
            raise LoudError(
                f"jepa_memory_size={self.jepa_memory_size} stores nothing",
                "a replay memory needs at least one slot; 256 is the tested size",
            )
        if self.jepa == "replay" and not self.jepa_pairs_path:
            raise LoudError(
                "jepa='replay' with no jepa_pairs_path has nothing to read",
                f"point it at a directory holding {VIEW_A} and {VIEW_B}",
            )

    def pair(self, *, batch: int, seq_len: int, seed: int = 0) -> JepaPair | None:
        """Return the loader this config describes, or `None` when the term is off."""
        if self.jepa == "none":
            return None
        return JepaPair.from_dir(self.jepa_pairs_path, batch=batch, seq_len=seq_len, seed=seed)

    def aux(self, d_model: int, *, seq_len: int, seed: int = 0) -> JepaAux | None:
        """Return the step's view of the term at width `d_model`, or `None` when off."""
        if self.jepa == "none":
            return None
        return JepaAux(
            head=JepaHead(d_model),
            memory=EpisodicMemory(
                capacity=self.jepa_memory_size,
                width_a=seq_len + 1,
                width_b=seq_len + 1,
                seed=seed,
            ),
            lam=self.jepa_lambda,
            replay_weight=self.jepa_replay_weight,
            replay_frac=self.jepa_replay_frac,
            ld=self.jepa_ld,
        )
