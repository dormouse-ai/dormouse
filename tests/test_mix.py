"""The mixture's rules, each one standing in for a way a run reads the wrong data.

Every defect here is silent in the loss: a drifted share, a rewound source, a resumed
run that re-reads a window, an id past the vocabulary, a stage that kept its old mix.
The tests are therefore arithmetic and exact where they can be — a hash, a multiset,
a count — rather than statistical.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal
from hypothesis import given, settings
from hypothesis import strategies as st

from dormouse.data.mix import CYCLE, MixtureStream, Source, Stage, plan_cycle

SHA_A = "a" * 64
SHA_B = "b" * 64

#: One file per (directory, name, size), built once per test.
FILES: dict[tuple[str, str, int], Path] = {}


def _source(tmp_path: Path, name: str, size: int, *, sha: str = "") -> Path:
    """A `.bin` whose ids are `0..size`, plus its manifest."""
    path = tmp_path / f"{name}.bin"
    np.arange(size, dtype=np.uint16).tofile(path)
    path.with_suffix(".bin.json").write_text(
        json.dumps({"dtype": "uint16", "tokens": size, "tokenizer_sha": sha})
    )
    return path


def _file(tmp_path: Path, name: str, size: int, *, sha: str = "") -> Path:
    """`_source`, memoised per (name, size) so one name can appear in several stages."""
    key = (str(tmp_path), name, size)
    if key not in FILES:
        FILES[key] = _source(tmp_path, name, size, sha=sha)
    return FILES[key]


def _mix(
    tmp_path: Path,
    weights: tuple[float, ...] = (0.5, 0.3, 0.2),
    *,
    size: int = 100_000,
    plan: tuple[tuple[str, float, str], ...] | None = None,
    stages: tuple[Stage, ...] = (),
    seed: int = 0,
    tokenizer_sha: str = "",
    wrap: bool = False,
) -> MixtureStream:
    """A mixture over one source per weight, or over an explicit `(name, weight, stage)` plan."""
    if plan is None:
        names = ("web", "math", "code", "wiki")[: len(weights)]
        plan = tuple((name, w, "train") for name, w in zip(names, weights, strict=True))
    sources = [
        Source(_file(tmp_path, name, size, sha=tokenizer_sha), weight, stage)
        for name, weight, stage in plan
    ]
    return MixtureStream(sources, stages=stages, seed=seed, tokenizer_sha=tokenizer_sha, wrap=wrap)


def _digests(stream: MixtureStream, steps: int, *, batch: int = 4, seq_len: int = 16):
    """sha256 per batch, so a difference is one named step, not a numpy diff."""
    out = []
    for step, (x, y) in enumerate(stream.batches(batch=batch, seq_len=seq_len)):
        if step == steps:
            break
        out.append(hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
    return out


# --------------------------------------------------------------- shape and dtype
def test_a_batch_is_int64_inputs_and_shifted_targets(tmp_path: Path) -> None:
    stream = _mix(tmp_path)
    x, y = stream.step(0, batch=3, seq_len=8)
    assert x.shape == (3, 8) and y.shape == (3, 8)
    assert x.dtype == np.int64 and y.dtype == np.int64
    assert x.base is y.base is not None  # one buffer, so the shift is free
    assert (y[:, :-1] == x[:, 1:]).all()
    assert (y[:, -1] == x[:, -1] + 1).all()


def test_a_source_is_read_only_where_it_was_written(tmp_path: Path) -> None:
    """The off-by-one at the end of a source: window by window, the mirror of the file."""
    path = _source(tmp_path, "src", 960)
    stream = MixtureStream([Source(path, 1.0)])
    windows = (stream.step(s, batch=8, seq_len=9) for s in range(12))
    read = np.vstack([np.hstack([x, y[:, -1:]]) for x, y in windows])
    assert read.shape == (96, 10)
    assert (read.ravel() == np.arange(960)).all()
    with pytest.raises(LoudError) as e:
        stream.step(12, batch=8, seq_len=9)
    assert "out of tokens" in str(e.value)
    assert "src.bin" in str(e.value)


# --------------------------------------------------------------------- the share
def test_the_plan_hits_the_declared_share_within_half_a_percent(tmp_path: Path) -> None:
    """200 steps of a 50/30/20 mix: every source's realised share is within 0.5 %."""
    stream = _mix(tmp_path)
    target = np.array([0.5, 0.3, 0.2])
    drawn = np.zeros(3)
    for step in range(200):
        drawn += np.bincount(stream.draws(step, 1), minlength=3)
    assert np.abs(drawn / drawn.sum() - target).max() < 0.005
    drawn = np.zeros(3)
    for step in range(1000):
        drawn += np.bincount(stream.draws(step, 24), minlength=3)
    assert np.abs(drawn / drawn.sum() - target).max() < 0.005


def test_a_prefix_is_as_exact_as_the_whole_plan(tmp_path: Path) -> None:
    """The count at every step boundary is within one draw of its share — not on average."""
    stream = _mix(tmp_path, weights=(0.7, 0.3), size=200_000)
    target = np.array([0.7, 0.3])
    drawn = np.zeros(2)
    for step in range(60):
        drawn += np.bincount(stream.draws(step, 7), minlength=2)
        assert np.abs(drawn - target * (step + 1) * 7).max() < 1.0


def test_shares_are_normalised_per_stage(tmp_path: Path) -> None:
    stream = _mix(tmp_path, weights=(2.0, 1.0, 1.0))
    shares = stream.shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[str(tmp_path / "web.bin")] == pytest.approx(0.5)
    assert shares[str(tmp_path / "math.bin")] == pytest.approx(0.25)


# ------------------------------------------------------- determinism, resumption
def test_two_runs_with_one_seed_are_bit_identical(tmp_path: Path) -> None:
    first = _digests(_mix(tmp_path), 20)
    second = _digests(_mix(tmp_path), 20)
    assert len(first) == 20
    assert first == second
    assert len(set(first)) > 15  # a stuck stream would repeat one batch


def test_a_seed_reorders_a_step_and_not_its_mix(tmp_path: Path) -> None:
    a, b = _mix(tmp_path, seed=0), _mix(tmp_path, seed=1)
    assert _digests(a, 5) != _digests(b, 5)
    assert (np.bincount(a.draws(0, 24)) == np.bincount(b.draws(0, 24))).all()


def test_resuming_at_a_step_replays_the_batches_it_would_have_seen(tmp_path: Path) -> None:
    straight = _digests(_mix(tmp_path), 12)
    assert _digests(_mix(tmp_path), 5) == straight[:5]
    x, y = next(_mix(tmp_path).batches(batch=4, seq_len=16, start_step=7))
    assert hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest() == straight[7]


# ------------------------------------------------------------------------ stages
ANNEAL = (
    ("web", 0.85, "S1"),
    ("wiki", 0.15, "S1"),
    ("web", 0.70, "S2"),
    ("wiki", 0.15, "S2"),
    ("ru", 0.15, "S2"),
    ("math", 0.60, "S3"),
    ("code", 0.40, "S3"),
)
ANNEAL_STAGES = (Stage("S1", until=100), Stage("S2", until=300), Stage("S3"))


def test_the_annealing_pattern_is_three_named_stages(tmp_path: Path) -> None:
    """web → web+math+(ru) → high-quality decay: the 1.7B/2.4B boundary shape."""
    stream = _mix(tmp_path, plan=ANNEAL, stages=ANNEAL_STAGES)
    assert stream.stage_at(0).name == "S1"
    assert stream.stage_at(99).name == "S1"
    assert stream.stage_at(100).name == "S2"
    assert stream.stage_at(299).name == "S2"
    assert stream.stage_at(300).name == "S3"
    s1, s2 = stream.shares("S1"), stream.shares("S2")
    assert sum(s1.values()) == pytest.approx(1.0)
    assert sum(s2.values()) == pytest.approx(1.0)
    assert s1[str(tmp_path / "web.bin")] == pytest.approx(0.85)
    # One file, two stages, two weights: S2 reads web harder than S1 did.
    assert s2[str(tmp_path / "web.bin")] == pytest.approx(0.70)
    assert set(s2) == {str(tmp_path / f"{n}.bin") for n in ("web", "wiki", "ru")}


def test_a_stage_boundary_is_written_to_the_journal(tmp_path: Path) -> None:
    stream = _mix(tmp_path, plan=ANNEAL, stages=ANNEAL_STAGES)
    journal = Journal.open(tmp_path / "run.jsonl")
    for step in range(10):
        stream.record(journal, step)
        if step == 9:
            stream.record(journal, 100)  # the boundary the trainer would hit next
    swaps = journal.segments()
    assert [s["step"] for s in swaps] == [0, 100]
    assert [s["stage"] for s in swaps] == ["S1", "S2"]
    assert sum(swaps[0]["weights"].values()) == pytest.approx(1.0)


def test_a_resumed_run_announces_the_stage_it_starts_in(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "run.jsonl")
    resumed = _mix(tmp_path, plan=ANNEAL, stages=ANNEAL_STAGES)
    assert resumed.record(journal, 150) is not None
    assert journal.segments()[0]["step"] == 150
    assert resumed.record(journal, 151) is None  # one event per boundary, not per step


def test_a_stage_change_changes_what_is_drawn(tmp_path: Path) -> None:
    stream = _mix(tmp_path, plan=ANNEAL, stages=ANNEAL_STAGES)
    first = np.bincount(stream.draws(0, 100), minlength=2)
    assert first[0] == 85 and first[1] == 15  # S1: 85/15 over web/wiki
    later = np.bincount(stream.draws(100, 100), minlength=3)
    assert later[0] == 70  # S2, stage-local order web/wiki/ru: web at 70
    assert later[1] == 15 and later[2] == 15


def test_the_stage_table_must_match_the_sources(tmp_path: Path) -> None:
    sources = [Source(_file(tmp_path, "web", 1000), 1.0, "web")]
    sources.append(Source(_file(tmp_path, "math", 1000), 1.0, "math"))
    with pytest.raises(LoudError) as e:
        MixtureStream(sources)
    assert "no stage table" in str(e.value)
    with pytest.raises(LoudError) as e:
        MixtureStream(sources, stages=(Stage("web", until=3), Stage("math", until=2)))
    assert "not after" in str(e.value)
    with pytest.raises(LoudError) as e:
        MixtureStream(sources, stages=(Stage("web", until=3), Stage("code")))
    assert "do not match" in str(e.value)


def test_a_stage_that_ended_refuses_the_step_instead_of_repeating_it(tmp_path: Path) -> None:
    stream = _mix(tmp_path, weights=(1.0,), stages=(Stage("train", until=3),))
    assert stream.stage_at(2).name == "train"
    with pytest.raises(LoudError) as e:
        stream.stage_at(3)
    assert "past the last stage" in str(e.value)
    with pytest.raises(LoudError) as e:
        next(stream.batches(batch=2, seq_len=4, start_step=3))
    assert "past the last stage" in str(e.value)


# ------------------------------------------------------------- end of the source
def test_an_exhausted_source_is_named_not_rewound(tmp_path: Path) -> None:
    path = _source(tmp_path, "tiny", 100)
    stream = MixtureStream([Source(path, 1.0)])
    with pytest.raises(LoudError) as e:
        list(stream.batches(batch=2, seq_len=32))
    assert "tiny.bin" in str(e.value)
    assert "wrap=True" in str(e.value)
    assert "step 1" in str(e.value)


def test_wrap_is_the_only_way_a_source_is_looped(tmp_path: Path) -> None:
    path = _source(tmp_path, "small", 100)
    x, _ = MixtureStream([Source(path, 1.0)], wrap=True).step(9, batch=2, seq_len=8)
    assert x.min() >= 0 and x.max() < 100


def test_a_source_that_cannot_hold_one_step_is_refused_even_wrapped(tmp_path: Path) -> None:
    path = _source(tmp_path, "four", 4)
    with pytest.raises(LoudError) as e:
        MixtureStream([Source(path, 1.0)], wrap=True).step(0, batch=2, seq_len=8)
    assert "cannot make it longer than a step" in str(e.value)


# ------------------------------------------------------------------- manifests
def test_two_tokenizers_in_one_mixture_are_refused(tmp_path: Path) -> None:
    sources = [
        Source(_source(tmp_path, "a", 1000, sha=SHA_A), 1.0),
        Source(_source(tmp_path, "b", 1000, sha=SHA_B), 1.0),
    ]
    with pytest.raises(LoudError) as e:
        MixtureStream(sources)
    assert "different tokenizers" in str(e.value)


def test_a_tokenizer_mismatch_names_both_hashes(tmp_path: Path) -> None:
    with pytest.raises(LoudError) as e:
        MixtureStream([Source(_source(tmp_path, "bare", 1000), 1.0)], tokenizer_sha=SHA_A)
    assert "(none)" in str(e.value) and SHA_A in str(e.value)
    ok = MixtureStream(
        [Source(_source(tmp_path, "stamped", 1000, sha=SHA_A), 1.0)], tokenizer_sha=SHA_A
    )
    assert ok.sources[0].weight == 1.0


# ------------------------------------------------------------------- refusals
def test_a_mixture_that_cannot_hit_a_share_is_refused(tmp_path: Path) -> None:
    """Every refusal names the value and the way out; these are the run's first line."""
    path = _source(tmp_path, "one", 1000)
    with pytest.raises(LoudError) as e:
        MixtureStream([])
    assert "no sources" in str(e.value)
    with pytest.raises(LoudError) as e:
        MixtureStream([Source(path, 1.0)], seed=-1)
    assert "seed=-1" in str(e.value)
    with pytest.raises(LoudError) as e:
        plan_cycle([1.0, -1.0])
    assert "non-negative" in str(e.value)
    with pytest.raises(LoudError) as e:
        plan_cycle([0.0, 0.0])
    assert "every weight" in str(e.value)
    with pytest.raises(LoudError) as e:
        plan_cycle([[1.0, 2.0]])
    assert "1-D" in str(e.value)
    with pytest.raises(LoudError) as e:
        Source(path, 0.0)
    assert "carries no data" in str(e.value)
    with pytest.raises(LoudError) as e:
        Source(path, 1.0, "")
    assert "no stage" in str(e.value)


def test_a_stage_table_that_cannot_be_ordered_is_refused(tmp_path: Path) -> None:
    sources = [Source(_file(tmp_path, "a", 1000), 1.0, "a")]
    sources.append(Source(_file(tmp_path, "b", 1000), 1.0, "b"))
    with pytest.raises(LoudError) as e:
        MixtureStream(sources, stages=(Stage("a"), Stage("b", until=5)))
    assert "after a stage that has none" in str(e.value)


def test_reading_a_step_that_is_not_a_step_is_refused(tmp_path: Path) -> None:
    stream = _mix(tmp_path, plan=ANNEAL, stages=ANNEAL_STAGES)
    with pytest.raises(LoudError) as e:
        stream.stage_at(-1)
    assert "negative" in str(e.value)
    with pytest.raises(LoudError) as e:
        stream.step(0, batch=0, seq_len=4)
    assert "batch=0" in str(e.value)
    with pytest.raises(LoudError) as e:
        stream.step(0, batch=1, seq_len=0)
    assert "seq_len=0" in str(e.value)
    with pytest.raises(LoudError) as e:
        stream.shares()
    assert "ambiguous" in str(e.value)
    with pytest.raises(LoudError) as e:
        stream.shares("S9")
    assert "no stage named" in str(e.value)


# ------------------------------------------------------------------ timing budget
def test_a_batch_fetch_fits_the_budget(tmp_path: Path) -> None:
    """0.9 MB/s of text is ~9 batches a second; the stream must not be the bottleneck."""
    stream = _mix(tmp_path, size=600_000)
    for step in range(3):  # warm the mmap pages
        stream.step(step, batch=24, seq_len=1024)
    took = []
    for step in range(3, 23):
        start = time.perf_counter()
        stream.step(step, batch=24, seq_len=1024)
        took.append((time.perf_counter() - start) * 1e6)
    median = float(np.median(took))
    assert median < 200.0, f"batch fetch median {median:.0f} us, budget 200 us"


# ------------------------------------------------------------------- properties
@given(
    weights=st.lists(
        st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=25, deadline=None)
def test_the_plan_is_exact_at_every_prefix_of_its_cycle(weights: list[float]) -> None:
    """Every row of the cycle: exact to one draw, and never lower than the row above."""
    cycle, slots = plan_cycle(weights)
    exact = np.asarray(weights) / float(sum(weights)) * np.arange(CYCLE + 1)[:, None]
    assert cycle.shape == (CYCLE + 1, len(weights))
    assert slots.shape == (CYCLE,)
    assert np.array_equal(np.bincount(slots, minlength=len(weights)), cycle[CYCLE])
    assert int(cycle[CYCLE].sum()) == CYCLE
    assert (cycle >= 0).all()
    assert (np.diff(cycle, axis=0) >= 0).all()
    bound = 1.0 + float(np.max(np.asarray(weights) / sum(weights)))
    assert (np.abs(cycle - exact) < bound + 1e-9).all()


@given(
    weights=st.lists(
        st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    )
)
@settings(max_examples=15, deadline=None)
def test_any_weight_vector_mixes_without_stalling(weights: list[float]) -> None:
    """Shares sum to one, every step fills its batch, and no source starves."""
    tmp_path = Path(tempfile.mkdtemp(prefix="dormouse-mix-"))
    stream = _mix(tmp_path, weights=tuple(weights), size=200_000)
    assert sum(stream.shares().values()) == pytest.approx(1.0)
    drawn = np.zeros(len(weights))
    for step in range(50):
        slots = stream.draws(step, 8)
        assert slots.size == 8
        assert set(np.unique(slots)) <= set(range(len(weights)))
        drawn += np.bincount(slots, minlength=len(weights))
    target = np.asarray(weights) / sum(weights) * drawn.sum()
    assert (drawn[target >= 1.0] > 0).all()  # nothing with a draw to get starves
    assert (np.abs(drawn - target) < 1.0 + max(weights) / sum(weights)).all()
