"""Portable IQ harness core: tasks, scoring and a blind judge.

No OpenCode, no torch, no network, no CLI. A model is anything with
``generate(prompt) -> str``; a judge is anything with
``judge(task, answer) -> Score``. Both are wired by the caller, so this module
can run on a bare Python install with only ``dormouse_core`` for the counters.

The word "IQ" here describes the shape of the harness, not a claim about what
it measures. See ``docs/iq-harness.md`` for the ceiling.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

__concepts__ = ("BlindJudge", "MockJudge", "Row", "Score", "Task", "TaskSuite")

SCHEMA_VERSION = 1

#: No number from this harness may be reported as IQ without a named human anchor.
IQ_REFUTED = "IQ_REFUTED"
#: The pre-registered minimum for a quality claim; smoke (n=10) is a signal only.
MIN_QUALITY_N = 50

#: The three ways an answer can be checked, all without a network judge.
type TaskKind = Literal["exact", "code", "choice"]

#: The committed local data. A caller on another machine may pass any path.
DEFAULT_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "corpus_2mb.bin"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Task:
    """One prompt, one reference answer, and how to score it."""

    prompt: str
    expected: str
    kind: TaskKind
    choices: tuple[str, ...] = ()

    @property
    def task_id(self) -> str:
        """A stable id from the prompt: no global counter, no suite ordering."""
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Score:
    """A verdict. A zero without a reason is refused at construction.

    Refusing here rather than in the CLI means a pluggable judge cannot smuggle
    an unexplained zero into any consumer of this module.
    """

    value: float
    reason: str

    def __post_init__(self) -> None:
        """Refuse a non-rate verdict and the failure that says why it failed."""
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise LoudError(
                f"score {self.value!r} is not a finite rate in [0, 1]",
                "a judge reports passes / n, a finite number between zero and one",
            )
        if self.value == 0.0 and not self.reason.strip():
            raise LoudError(
                "score 0 without reason",
                "every failure names what was expected and what was produced",
            )


@runtime_checkable
class BlindJudge(Protocol):
    """Score an answer knowing only the task, never which model wrote it."""

    def judge(self, task: Task, answer: str) -> Score:
        """Return the verdict for one answer, without knowing its author."""
        ...


class MockJudge:
    """The deterministic, pluggable-by-default judge: exact local scoring."""

    def judge(self, task: Task, answer: str) -> Score:
        """Apply the task's own scoring rule to one answer."""
        return score(task, answer)


@dataclass(frozen=True)
class Row:
    """One NDJSON row: a task's id, kind, score and its reason."""

    task: str
    kind: TaskKind
    score: float
    reason: str


class _Model(Protocol):
    """The one call the harness makes on a model."""

    def generate(self, prompt: str) -> str: ...


def normalize(text: str) -> str:
    """Case-fold, Unicode-normalize and flatten an answer for exact comparison."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", text).strip().strip(" .,!?:;")


def _unwrap(answer: str) -> str:
    """Drop a Markdown code fence around a code answer, if there is one."""
    lines = answer.strip().splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return answer.strip()


def score(task: Task, answer: str) -> Score:
    """Score one answer against one task; a zero always carries a reason."""
    expected = normalize(task.expected)
    got = normalize(answer)
    if task.kind == "exact":
        if got == expected:
            return Score(1.0, "exact match after normalization")
        return Score(0.0, f"expected {task.expected!r}, got {answer!r}")
    if task.kind == "choice":
        if got.rstrip(").") == expected.rstrip(")."):
            return Score(1.0, "choice matches")
        return Score(0.0, f"expected choice {task.expected!r}, got {answer!r}")
    if task.kind != "code":
        raise LoudError(f"unknown task kind {task.kind!r}", "use exact, choice or code")
    body = _unwrap(answer)
    try:
        compile(body, "<answer>", "exec")
    except SyntaxError as exc:
        return Score(0.0, f"answer is not valid Python ({exc.msg})")
    if expected not in normalize(body):
        return Score(0.0, f"expected snippet {task.expected!r} missing from {body!r}")
    return Score(1.0, "code compiles and contains the expected snippet")


def evaluate(
    tasks: Iterable[Task],
    model: _Model,
    judge: BlindJudge,
    *,
    counters: Counters | None = None,
) -> list[Row]:
    """Score every task, counting judge calls when a counter is supplied.

    An enabled judge whose count stayed at zero is a ``MissingReport``: a suite
    that never reached the judge has no score, only the appearance of one.
    """
    if counters is not None:
        counters.enable("blind_judge", counter="iq_judge_calls", why="every row needs a verdict")
    rows: list[Row] = []
    for task in tasks:
        answer = model.generate(task.prompt)
        verdict = judge.judge(task, answer)
        if counters is not None:
            counters.count("iq_judge_calls")
        rows.append(Row(task.task_id, task.kind, verdict.value, verdict.reason))
    if counters is not None:
        counters.check()
    return rows


def wilson_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a pass rate, declared before the run."""
    if n <= 0 or not 0 <= successes <= n:
        raise LoudError(
            f"Wilson interval needs 0 <= k <= n, got k={successes}, n={n}",
            "count passes out of a positive number of examples",
        )
    z = 1.959963984540054
    rate = successes / n
    denom = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def iq_verdict(*, blind_kappa: float | None = None, anchor: str = "") -> str:
    """Refuse every IQ claim until blind kappa against a named human anchor exists."""
    if blind_kappa is None or not anchor or not math.isfinite(blind_kappa):
        return IQ_REFUTED
    return (
        f"blind-kappa={blind_kappa:.3f} vs {anchor} measured; "
        "IQ is not an instrument here, no number claimed"
    )


def arithmetic(seed: int = 0, n: int = 4) -> tuple[Task, ...]:
    """SYNTH: two-digit arithmetic, alternating English and Russian prompts."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n):
        a, b = rng.randint(2, 97), rng.randint(2, 97)
        op = "+-*"[i % 3]
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        if i % 2:
            prompt = f"Сколько будет {a} {op} {b}?\nОтвет:"
        else:
            prompt = f"What is {a} {op} {b}?\nAnswer:"
        tasks.append(Task(prompt, str(result), "exact"))
    return tuple(tasks)


#: SYNTH fill-in sentences: one unambiguous word is removed from each.
_FILLING: tuple[tuple[str, str], ...] = (
    ("Fill in the blank: The sky is ____ at noon.\nAnswer:", "blue"),
    ("Fill in the blank: She reads a ____ every evening.\nAnswer:", "book"),
    ("Заполните пропуск: На улице идёт ____ .\nОтвет:", "дождь"),
    ("Заполните пропуск: Кошка спит на ____ .\nОтвет:", "диван"),
)

#: SYNTH factual QA: a fixed set, not a claim about world knowledge.
_FACTS: tuple[tuple[str, str], ...] = (
    ("What is the capital of France?\nAnswer:", "Paris"),
    ("Which planet is known as the Red Planet?\nAnswer:", "Mars"),
    ("Какая столица Франции?\nОтвет:", "Париж"),
    ("Сколько дней в неделе?\nОтвет:", "семь"),
)

#: SYNTH code completion: the answer must compile and contain the snippet.
_CODE: tuple[tuple[str, str], ...] = (
    ("Complete this Python expression for twice x. Reply with the expression only.\n", "x * 2"),
    ("Complete this Python expression for the length of s. Reply with it only.\n", "len(s)"),
    ("Завершите выражение Python: остаток от деления a на b. Только выражение.\n", "a % b"),
)


def _take(cases: Sequence[tuple[str, str]], n: int, kind: TaskKind) -> tuple[Task, ...]:
    """Turn the first `n` (prompt, expected) pairs into tasks of one kind."""
    return tuple(Task(prompt, expected, kind) for prompt, expected in cases[:n])


def filling(seed: int = 0, n: int = 2) -> tuple[Task, ...]:
    """SYNTH: English and Russian cloze from a fixed, unambiguous word list."""
    cases = list(_FILLING)
    random.Random(seed).shuffle(cases)
    return _take(cases, n, "exact")


def factual(seed: int = 0, n: int = 2) -> tuple[Task, ...]:
    """SYNTH: short English and Russian factual questions with one-word answers."""
    cases = list(_FACTS)
    random.Random(seed).shuffle(cases)
    return _take(cases, n, "choice")


def code(seed: int = 0, n: int = 2) -> tuple[Task, ...]:
    """SYNTH: short Python expressions that must compile and contain the snippet."""
    cases = list(_CODE)
    random.Random(seed).shuffle(cases)
    return _take(cases, n, "code")


def from_fixture(path: Path, *, seed: int = 0, n: int = 8) -> tuple[Task, ...]:
    """Build English fill-in tasks from a local text fixture.

    The 2 MB corpus is ordinary prose. A sentence is used only when a word in it
    occurs exactly once, so blanking that word has one unambiguous answer.
    """
    text = path.read_bytes().decode("utf-8", errors="replace")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    tasks: list[Task] = []
    for i in order:
        sentence = " ".join(sentences[i].split())
        words = re.findall(r"[A-Za-z]+", sentence)
        if not 8 <= len(words) <= 40:
            continue
        candidates = [
            word
            for word in words
            if len(re.findall(rf"\b{re.escape(word)}\b", sentence, flags=re.IGNORECASE)) == 1
        ]
        if not candidates:
            continue
        word = candidates[len(candidates) // 2]
        blanked = re.sub(rf"\b{re.escape(word)}\b", "____", sentence, count=1, flags=re.IGNORECASE)
        tasks.append(
            Task(f"Fill in the blank in this English sentence.\n{blanked}\nAnswer:", word, "exact")
        )
        if len(tasks) == n:
            break
    if len(tasks) < n:
        raise LoudError(
            f"{path}: only {len(tasks)} usable sentences, need {n}",
            "point --fixture at a bigger local text file",
        )
    return tuple(tasks)


def build_suite(name: str = "smoke", *, seed: int = 0, fixture: Path | None = None) -> TaskSuite:
    """Build a named suite; `smoke` is local-only, `full` adds fixture tasks."""
    tasks = arithmetic(seed) + filling(seed) + factual(seed) + code(seed)
    if name == "smoke":
        return TaskSuite(name, tasks)
    if name == "full":
        tasks += from_fixture(fixture or DEFAULT_FIXTURE, seed=seed, n=8)
        return TaskSuite(name, tasks)
    raise LoudError(f"unknown suite {name!r}", "use smoke or full")


@dataclass(frozen=True)
class TaskSuite:
    """A named, ordered set of tasks built from one seed."""

    name: str
    tasks: tuple[Task, ...]

    def __iter__(self) -> Iterable[Task]:
        """Iterate the tasks in their fixed order."""
        return iter(self.tasks)

    def __len__(self) -> int:
        """Return how many tasks the suite scores."""
        return len(self.tasks)


__all__ = [
    "DEFAULT_FIXTURE",
    "IQ_REFUTED",
    "MIN_QUALITY_N",
    "SCHEMA_VERSION",
    "BlindJudge",
    "MockJudge",
    "Row",
    "Score",
    "Task",
    "TaskKind",
    "TaskSuite",
    "arithmetic",
    "build_suite",
    "code",
    "evaluate",
    "factual",
    "filling",
    "from_fixture",
    "iq_verdict",
    "normalize",
    "score",
    "wilson_95",
]
