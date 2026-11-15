"""ER-JEPA: the store's ages, the replay's rows, the stop-gradient, and the flag off.

Five properties are tested here, and each one stands for a way this mechanism goes
silently wrong:

* the memory's age distribution — a store that keeps the recent tail looks identical
  in `size` and makes replay a no-op, so the ages are measured against the uniform
  bound reservoir sampling promises;
* the replay count — a store that is filled but never read is the flag-set-mechanism-
  absent defect, so `jepa_replay_rows` is asserted against `replay_frac · batch`;
* the skip — at `ld=1.0` the replay's forward must not run, which is shown by the
  counter and by the step rate;
* the stop-gradient — on the current pairs and on the replayed ones, a target that
  carries a gradient turns alignment into reconstruction;
* retention — pairs the stream showed and the store kept are decoded better when they
  are replayed, which is the paper's claim made measurable on a CPU-sized task.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model import Transformer
from dormouse.train import MixOptimizer, train_step
from dormouse.train.jepa import (
    EpisodicMemory,
    JepaAux,
    JepaConfig,
    JepaHead,
    JepaPair,
    er_jepa_loss,
    jepa_loss,
)
from dormouse.train.step import chunked_cross_entropy

SMALL = {"vocab": 16, "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_ffn": 64}

#: Tokens per view on the small batches: a pair is packed as [view A | view B].
W = 4


def _model() -> Transformer:
    torch.manual_seed(23)
    return Transformer(**SMALL)


def _views(rows: int = 32, width: int = W, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    return (
        generator.integers(0, SMALL["vocab"], size=(rows, width)),
        generator.integers(0, SMALL["vocab"], size=(rows, width)),
    )


def _batch(rows: int = 4, width: int = W) -> tuple[torch.Tensor, torch.Tensor]:
    """`(packed ids [rows, 2·W], view B [rows, W])` for one step."""
    view_a, view_b = JepaPair(*_views(rows, width), batch=rows).at(0)
    return torch.cat([view_a, view_b], dim=1), view_b


def _memory(capacity: int = 4) -> EpisodicMemory:
    return EpisodicMemory(capacity=capacity, width_a=2, width_b=2, seed=1)


def _aux(frac: float = 0.5, ld: float = 0.0, capacity: int = 64, width: int = W) -> JepaAux:
    return JepaAux(
        head=JepaHead(SMALL["d_model"]),
        memory=EpisodicMemory(capacity=capacity, width_a=width, width_b=width, seed=1),
        lam=1.0,
        replay_weight=0.5,
        replay_frac=frac,
        ld=ld,
    )


# ------------------------------------------------------------------- the pair stream


def test_a_pair_batch_is_a_function_of_the_step() -> None:
    """Two readers with one seed read one order; a resume lands where it stopped."""
    a, b = _views(rows=64)
    first, second = JepaPair(a, b, batch=4, seed=3), JepaPair(a, b, batch=4, seed=3)
    assert torch.equal(first.at(5)[0], second.at(5)[0])
    assert torch.equal(first.at(5)[1], second.at(5)[1])
    assert not torch.equal(first.at(5)[0], first.at(6)[0])
    # One pass over the rows reads every pair once: the stride is a bijection.
    seen = {tuple(row.tolist()) for step in range(16) for row in first.at(step)[0]}
    assert len(seen) == 64


def test_two_seeds_read_two_orders() -> None:
    a, b = _views(rows=64)
    assert not torch.equal(
        JepaPair(a, b, batch=4, seed=3).at(2)[0], JepaPair(a, b, batch=4, seed=5).at(2)[0]
    )


@pytest.mark.parametrize(
    ("view_a", "view_b", "batch", "message"),
    [
        (np.zeros((8, W)), np.zeros((7, W)), 2, "rows"),
        (np.zeros((8, W)), np.zeros((8, W - 1)), 2, "wide"),
        (np.zeros((8, W)), np.zeros((8, W)), 9, "does not fit"),
        (np.zeros(8), np.zeros(8), 2, r"two \[rows, tokens\] arrays"),
    ],
)
def test_views_that_do_not_line_up_are_refused(
    view_a: np.ndarray, view_b: np.ndarray, batch: int, message: str
) -> None:
    with pytest.raises(LoudError, match=message):
        JepaPair(view_a, view_b, batch=batch)


def _write_stream(path: Path, values: list[int]) -> None:
    arr = np.asarray(values, dtype=np.uint16)
    path.write_bytes(arr.tobytes())
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"dtype": "uint16", "tokens": len(values)})
    )


def test_a_paired_directory_is_two_streams_with_manifests(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    root.mkdir()
    _write_stream(root / "view_a.bin", list(range(24)))
    _write_stream(root / "view_b.bin", list(range(24)))
    pair = JepaPair.from_dir(root, batch=2, seq_len=3)
    assert pair.rows == 6 and pair.batch == 2  # 24 tokens / 4 per row
    left, right = pair.at(0)
    assert left.shape == right.shape == (2, 4)


def test_a_directory_without_pairs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LoudError, match="no paired corpus"):
        JepaPair.from_dir(tmp_path / "nothing", batch=1, seq_len=3)


def test_a_corpus_too_small_for_the_batch_is_refused(tmp_path: Path) -> None:
    _write_stream(tmp_path / "view_a.bin", list(range(8)))
    _write_stream(tmp_path / "view_b.bin", list(range(8)))
    with pytest.raises(LoudError, match="needs 4"):
        JepaPair.from_bins(tmp_path / "view_a.bin", tmp_path / "view_b.bin", batch=4, seq_len=3)


# ------------------------------------------------------------------------ the memory


def test_the_store_never_grows_past_its_capacity() -> None:
    memory = EpisodicMemory(capacity=8, width_a=W, width_b=W, seed=1)
    a, b = (torch.zeros(4, W, dtype=torch.long), torch.ones(4, W, dtype=torch.long))
    for _ in range(10):
        memory.insert(a, b)
    stats = memory.stats()
    assert stats["size"] == 8 and stats["inserts"] == 40
    assert len(stats["ages"]) == 8


@pytest.mark.parametrize(
    ("view_a", "view_b", "message"),
    [
        (torch.zeros(2, W, dtype=torch.long), torch.zeros(3, W, dtype=torch.long), "rows"),
        (torch.zeros(2, W + 1, dtype=torch.long), torch.zeros(2, W, dtype=torch.long), "widths"),
        (torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), "batches"),
    ],
)
def test_a_pair_of_the_wrong_shape_is_not_stored(
    view_a: torch.Tensor, view_b: torch.Tensor, message: str
) -> None:
    memory = EpisodicMemory(capacity=8, width_a=W, width_b=W, seed=1)
    with pytest.raises(LoudError, match=message):
        memory.insert(view_a, view_b)


def _chi_square(ages: list[int], span: int, bins: int = 8) -> float:
    """Chi-square of the ages against uniform over `[0, span)`, the inserts seen."""
    counts = [0] * bins
    for age in ages:
        counts[min(age * bins // span, bins - 1)] += 1
    expected = len(ages) / bins
    return sum((count - expected) ** 2 / expected for count in counts)


def test_reservoir_sampling_keeps_the_ages_uniform() -> None:
    """2000 inserts into 256 slots must leave a uniform sample of the whole history.

    Chi-square with 8 bins has 7 degrees of freedom; the 0.999 quantile is 24.32, so a
    store that keeps say only the last 256 rows (every age below 256, all in the first
    bin) fails by orders of magnitude — that mutant is the next test.
    """
    memory = EpisodicMemory(capacity=256, width_a=2, width_b=2, seed=1)
    rows = torch.arange(4000, dtype=torch.long).reshape(2000, 2) % 7
    for start in range(0, 2000, 2):
        memory.insert(rows[start : start + 2], rows[start : start + 2])
    assert memory.inserts == 2000
    ages = memory.stats()["ages"]
    assert isinstance(ages, list) and len(ages) == 256
    assert max(ages) > 1500, "the store kept only recent rows"
    assert _chi_square(ages, memory.inserts) < 24.32


def test_an_overwrite_the_oldest_store_fails_the_age_bound() -> None:
    """The bound above is not vacuous: a FIFO store fails it exactly as it should."""
    fifo = list(range(256))  # the last 256 of 2000 inserts, as ages
    assert max(fifo) < 256
    assert _chi_square(fifo, span=2000) > 100


def test_retrieval_is_a_function_of_the_step_and_wraps_a_small_store() -> None:
    memory = EpisodicMemory(capacity=8, width_a=2, width_b=2, seed=1)
    rows = torch.arange(16, dtype=torch.long).reshape(8, 2)
    memory.insert(rows, rows + 100)
    first, second = memory.retrieve(3, step=2), memory.retrieve(3, step=2)
    assert torch.equal(first[0], second[0]) and first[0].shape == (3, 2)
    assert not torch.equal(first[0], memory.retrieve(3, step=3)[0])
    wide = memory.retrieve(20, step=0)[0]  # more rows than slots: slots repeat
    assert wide.shape == (20, 2)


def test_retrieving_from_an_empty_store_is_refused() -> None:
    empty = EpisodicMemory(capacity=4, width_a=2, width_b=2)
    with pytest.raises(LoudError, match="empty"):
        empty.retrieve(1)
    full = EpisodicMemory(capacity=4, width_a=2, width_b=2, seed=1)
    rows = torch.zeros(1, 2, dtype=torch.long)
    full.insert(rows, rows)
    with pytest.raises(LoudError, match="no pair"):
        full.retrieve(0)


# ---------------------------------------------------------------- the two embeddings


def test_the_target_side_carries_no_gradient() -> None:
    """`sg(Enc B)`: the failure mode is a target that trains itself."""
    generator = torch.Generator().manual_seed(7)
    h_a = torch.randn(3, 8, generator=generator, requires_grad=True)
    h_b = torch.randn(3, 8, generator=generator, requires_grad=True)
    loss = jepa_loss(h_a, h_b, torch.nn.Identity())
    loss.backward()
    assert h_b.grad is None, "the target side received a gradient"
    assert h_a.grad is not None, "the predicting side did not"


@pytest.mark.parametrize("replayed", [False, True])
def test_no_gradient_reaches_the_encoder_through_the_target(replayed: bool) -> None:
    """A replayed target is stopped exactly like a current one."""
    model, head = _model(), JepaHead(SMALL["d_model"])
    view_a, view_b = _views(rows=4)
    h_b = model.hidden(torch.from_numpy(view_b))
    with torch.no_grad():
        h_a = model.hidden(torch.from_numpy(view_a))
    if replayed:
        loss = er_jepa_loss(None, (h_a, h_b), head, lam=1.0, beta=0.5)
    else:
        loss = er_jepa_loss((h_a, h_b), None, head, lam=1.0)
    loss.backward()
    assert all(param.grad is None for param in model.parameters()), "the encoder moved"
    assert head.predictor.weight.grad is not None, "the head never learned"


def test_two_identical_embeddings_are_a_zero_cosine_distance() -> None:
    h = torch.randn(4, 8)
    assert float(jepa_loss(h, h, torch.nn.Identity())) == pytest.approx(0.0, abs=1e-6)
    assert float(jepa_loss(h, -h, torch.nn.Identity())) == pytest.approx(2.0, abs=1e-5)


def test_the_metric_is_cosine_or_l2_and_a_third_kind_is_refused() -> None:
    h_a, h_b = torch.randn(2, 5), torch.randn(2, 5)
    identity = torch.nn.Identity()
    assert float(jepa_loss(h_a, h_b, identity, kind="l2")) >= 0.0
    with pytest.raises(LoudError, match="is not a metric this term has"):
        jepa_loss(h_a, h_b, identity, kind="dot")


def test_eq_4_is_the_two_halves_added_at_their_weights() -> None:
    head = JepaHead(8)
    current = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    replayed = (torch.randn(3, 4, 8), torch.randn(3, 4, 8))
    tokens = torch.tensor(0.75)
    together = er_jepa_loss(current, replayed, head, lam=2.0, beta=0.5, token_replay=tokens)
    apart = er_jepa_loss(current, None, head, lam=2.0) + 0.5 * (
        er_jepa_loss(None, replayed, head, lam=2.0) + tokens
    )
    assert float(together.detach()) == pytest.approx(float(apart.detach()), rel=1e-6)


def test_a_term_with_neither_half_is_refused() -> None:
    with pytest.raises(LoudError, match="neither"):
        er_jepa_loss(None, None, JepaHead(8))


# --------------------------------------------------------------- the step's wiring


def _reference_step(model: Transformer, ids: torch.Tensor, optimizer: MixOptimizer) -> float:
    """The step as it was before the term existed: one forward, CE, clip, step."""
    inputs, targets = ids[:, :-1], ids[:, 1:]
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = chunked_cross_entropy(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def test_the_forward_is_exactly_hidden_then_head() -> None:
    """The refactor must be the same call sequence, not a second stack."""
    model = _model()
    ids, _ = _batch(3)
    assert torch.equal(model(ids[:, :-1]), model.head(model.hidden(ids[:, :-1])))


def test_with_the_flag_off_the_step_is_bit_identical() -> None:
    """`jepa=None` must not move a single bit: same logits, same loss, same weights."""
    torch.manual_seed(23)
    model = Transformer(**SMALL)
    twin = Transformer(**SMALL)
    twin.load_state_dict(model.state_dict())
    ids, _ = _batch(6)
    want = _reference_step(twin, ids, MixOptimizer(twin))
    got = train_step(model, ids, optimizer=MixOptimizer(model))
    assert got.jepa_loss == 0.0
    assert got.loss == want
    for name, param in model.named_parameters():
        assert torch.equal(param, dict(twin.named_parameters())[name]), name


def test_the_counters_sum_to_the_steps_and_the_replay_budget() -> None:
    """`jepa_fwd + jepa_skipped == steps`, and the replay rows are the budget."""
    model = _model()
    optimizer = MixOptimizer(model)
    aux = _aux(frac=0.5)
    steps, batch = 6, 4
    totals: Counter[str] = Counter()
    last = Counters()
    for step in range(steps):
        ids, view_b = _batch(batch)
        last = Counters()
        train_step(model, ids, optimizer=optimizer, jepa=aux, pair=view_b, step=step, counters=last)
        totals.update(last.snapshot())
    assert totals["jepa_fwd"] + totals["jepa_skipped"] == steps
    assert totals["jepa_fwd"] == steps
    assert totals["jepa_replay_rows"] == round(0.5 * batch) * steps
    assert last.counts["jepa_memory_size"] == steps * batch  # 24 rows, capacity 64
    assert "jepa_loss" not in totals, "a float does not belong in an integer counter"


def test_a_skipped_step_runs_no_replay_and_costs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ld=1.0` is the baseline: the counter says so, and the clock agrees."""
    steps = 40

    def rate(aux: JepaAux) -> float:
        model = _model()
        optimizer = MixOptimizer(model)
        torch.set_num_threads(2)
        ids, view_b = _batch(4)
        for _ in range(2):  # warm the allocator, not counted
            train_step(model, ids, optimizer=optimizer, jepa=aux, pair=view_b)
        start = time.perf_counter()
        for step in range(steps):
            train_step(model, ids, optimizer=optimizer, jepa=aux, pair=view_b, step=step)
        return steps / (time.perf_counter() - start)

    skipped = _aux(frac=0.5, ld=1.0)
    counters = Counters()
    model = _model()
    ids, view_b = _batch(4)
    token_loss = float(
        chunked_cross_entropy(
            model.head(model.hidden(ids[:, :-1]))[:, W - 1 :], ids[:, 1:][:, W - 1 :]
        ).detach()
    )
    result = train_step(
        model, ids, optimizer=MixOptimizer(model), jepa=skipped, pair=view_b, counters=counters
    )
    assert counters.counts["jepa_skipped"] == 1
    assert counters.counts["jepa_fwd"] == 0
    assert "jepa_replay_rows" not in counters.counts
    assert result.jepa_loss == 0.0
    # The token objective is the same one a kept step reads: dropout drops the
    # alignment term, never the language loss.
    assert result.loss == pytest.approx(token_loss, rel=1e-6)

    slow, fast = rate(_aux(frac=0.5, ld=0.0)), rate(_aux(frac=0.5, ld=1.0))
    with capsys.disabled():
        print(f"\nER-JEPA step rate: ld=0 {slow:.1f} steps/s, ld=1.0 {fast:.1f} steps/s")
    assert fast > slow * 1.05, "the skip is not skipping the replay"


def test_the_term_stores_the_batch_even_when_the_step_skips() -> None:
    aux = _aux(frac=0.5, ld=1.0, capacity=64)
    model = _model()
    optimizer = MixOptimizer(model)
    for step in range(3):
        ids, view_b = _batch(4)
        train_step(model, ids, optimizer=optimizer, jepa=aux, pair=view_b, step=step)
    assert aux.memory.size == 12


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        (None, "come together"),
        (torch.zeros(4, W, dtype=torch.long), "come together"),
    ],
)
def test_the_term_and_its_pair_come_together(pair: torch.Tensor | None, message: str) -> None:
    model, ids = _model(), _batch(4)[0]
    kwargs = {"jepa": _aux()} if pair is None else {"pair": pair}
    with pytest.raises(LoudError, match=message):
        train_step(model, ids, optimizer=MixOptimizer(model), **kwargs)


def test_a_batch_that_is_not_a_packed_pair_is_refused() -> None:
    model = _model()
    ids = _batch(4)[0]
    with pytest.raises(LoudError, match="is not \\[view A \\| view B\\]"):
        train_step(model, ids[:, :-1], optimizer=MixOptimizer(model), jepa=_aux(), pair=ids[:, W:])


# --------------------------------------------------------------- the retention gate


def _square(n: int) -> tuple[list[int], list[int]]:
    """View A: the number. View B: its square, one digit per token."""
    return [n // 10, n % 10, 10, 10], [int(c) for c in f"{n * n:04d}"]


def _train_pairs(frac: float, seed: int = 23, steps: int = 300) -> tuple[float, EpisodicMemory]:
    """Train the small stack on `steps` paired squares, then decode the store's rows.

    The store is the 16-slot reservoir over the 600 inserted rows, so its entries are
    pairs the stream showed early and then stopped showing: a uniform sample of the
    whole stream, most of them hundreds of inserts old. `replay_frac=0` is the same
    run with the same store and no replay — the control.
    """
    torch.manual_seed(seed)
    model = Transformer(vocab=12, d_model=32, n_layers=2, n_heads=2, head_dim=16, d_ffn=64)
    head = JepaHead(32)
    optimizer = MixOptimizer(torch.nn.ModuleList([model, head]))
    memory = EpisodicMemory(capacity=16, width_a=W, width_b=W, seed=1)
    aux = JepaAux(head=head, memory=memory, lam=1.0, replay_weight=0.5, replay_frac=frac)
    for step in range(steps):
        numbers = [((step * 2 + j) * 37) % 100 for j in range(2)]
        view_a = torch.tensor([_square(n)[0] for n in numbers])
        view_b = torch.tensor([_square(n)[1] for n in numbers])
        train_step(
            model,
            torch.cat([view_a, view_b], dim=1),
            optimizer=optimizer,
            jepa=aux,
            pair=view_b,
            step=step,
        )
    stored_a, stored_b = memory.view_a[: memory.size], memory.view_b[: memory.size]
    packed = torch.cat([stored_a, stored_b], dim=1)
    with torch.no_grad():
        logits = model.head(model.hidden(packed[:, :-1]))
        recall = float((logits[:, W - 1 :].argmax(-1) == packed[:, 1:][:, W - 1 :]).float().mean())
    return recall, memory


def test_replay_retains_pairs_the_stream_stopped_showing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The paper's claim, made measurable: stored pairs decode better when replayed.

    Recall here is teacher-forced accuracy on the stored pairs' view-B tokens — the
    token objective the replay adds supervision to, not a metric invented for the
    test. Measured with λ=1, β=0.5, 300 CPU steps and 16 slots: 0.375 → 0.641,
    0.391 → 0.531, 0.375 → 0.688 at seeds 23/41/57 (+0.27/+0.14/+0.31). The bar is
    +0.10, and the two runs are asserted to hold the *same* stored rows so the
    comparison is paired.
    """
    torch.set_num_threads(2)
    control, control_memory = _train_pairs(frac=0.0)
    replayed, replay_memory = _train_pairs(frac=0.5)
    assert torch.equal(
        control_memory.view_a[: control_memory.size], replay_memory.view_a[: replay_memory.size]
    ), "the two arms stored different pairs: the comparison is not paired"
    with capsys.disabled():
        print(f"\nstored-pair recall: no replay {control:.3f}, replay {replayed:.3f}")
    assert replayed - control >= 0.10, (control, replayed)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: JepaPair(*_views(rows=4), batch=2).at(-1), "negative"),
        (lambda: EpisodicMemory(capacity=0, width_a=2, width_b=2), "stores nothing"),
        (lambda: EpisodicMemory(capacity=4, width_a=0, width_b=2), "hold no token"),
        (lambda: JepaHead(0), "predicts nothing"),
        (lambda: JepaHead(8, kind="dot"), "not a metric this term has"),
        (lambda: JepaAux(head=JepaHead(8), memory=_memory(), lam=-1.0), "are weights"),
        (lambda: JepaAux(head=JepaHead(8), memory=_memory(), ld=2.0), "is not a rate"),
        (lambda: JepaAux(head=JepaHead(8), memory=_memory(), replay_frac=1.5), "is not a fraction"),
    ],
)
def test_a_component_that_cannot_run_is_refused(build: object, message: str) -> None:
    with pytest.raises(LoudError, match=message):
        build()  # type: ignore[operator]


# ----------------------------------------------------------------------- the config


def test_a_config_with_no_mode_assembles_nothing() -> None:
    config = JepaConfig()
    assert config.pair(batch=2, seq_len=3) is None
    assert config.aux(32, seq_len=3) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("jepa", "llm", "is not a mode this objective has"),
        ("jepa_lambda", -1.0, "non-negative"),
        ("jepa_ld", 1.5, "is not a rate"),
        ("jepa_replay_frac", -0.1, "is not a fraction"),
        ("jepa_memory_size", 0, "stores nothing"),
        ("jepa", "replay", "no jepa_pairs_path"),
    ],
)
def test_a_config_that_cannot_run_is_refused(field: str, value: object, message: str) -> None:
    with pytest.raises(LoudError, match=message):
        JepaConfig.from_mapping({field: value})


def test_a_config_reads_a_toml_and_builds_the_loader_and_the_term(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    root.mkdir()
    _write_stream(root / "view_a.bin", list(range(32)))
    _write_stream(root / "view_b.bin", list(range(32)))
    path = tmp_path / "jepa.toml"
    path.write_text(
        "\n".join(
            [
                'jepa = "replay"',
                "jepa_lambda = 1.0",
                "jepa_ld = 0.25",
                "jepa_memory_size = 8",
                "jepa_replay_frac = 0.5",
                f'jepa_pairs_path = "{root}"',
            ]
        )
    )
    config = JepaConfig.from_toml(path)
    pair = config.pair(batch=2, seq_len=3)
    aux = config.aux(32, seq_len=3)
    assert pair is not None and pair.at(0)[0].shape == (2, 4)
    assert aux is not None and aux.memory.capacity == 8
    assert (aux.lam, aux.ld, aux.replay_frac, aux.replay_weight) == (1.0, 0.25, 0.5, 0.5)
