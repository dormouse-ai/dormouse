"""Every task must be failed by a broken model and passed by a correct one.

A task that random weights can pass measures nothing, and a task whose own reference
answer fails the check measures nothing either. Both directions are asserted, per
task, with the random floor printed so it is visible in the test log.

The ramble test is the one the old project was missing: the right word followed by
more text must not score as a right answer.
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence

import pytest

from dormouse.eval.harness import run
from dormouse.eval.tasks import END, Generate, Task, code, suite
from dormouse.eval.tasks.text import format_following


def random_model(seed: int) -> Generate:
    """The floor: random weights sample characters, they never answer."""
    rng = random.Random(seed)

    def generate(prompt: str, max_tokens: int, stop: Sequence[str]) -> str:
        return "".join(rng.choice(string.printable) for _ in range(max_tokens)).strip()

    return generate


def oracle(tasks: Sequence[Task]) -> Generate:
    """The ceiling: a model that always gives the reference answer."""
    answers = {example.prompt: example.expected for task in tasks for example in task.examples}

    def generate(prompt: str, max_tokens: int, stop: Sequence[str]) -> str:
        return answers.get(prompt, "")

    return generate


@pytest.mark.parametrize("task", suite(0), ids=lambda task: task.name)
def test_random_weights_fail_every_task(task: Task) -> None:
    result = run([task], random_model(0))[0]
    print(f"random floor {task.name}: {result.score:.3f} over n={result.n}")
    assert result.score == 0.0


@pytest.mark.parametrize("task", suite(0), ids=lambda task: task.name)
def test_the_reference_answers_pass_every_task(task: Task) -> None:
    assert run([task], oracle([task]))[0].score == 1.0


def test_an_empty_generation_does_not_pass_the_code_task() -> None:
    result = run([code()], lambda prompt, max_tokens, stop: "   ")[0]
    assert result.score == 0.0


def test_a_model_that_answers_then_rambles_fails_format_following() -> None:
    """The old defect: correct answer, no stop, still counted as correct."""
    task = format_following(0)
    rambling = {e.prompt: f"{e.expected}{END} and then it kept talking" for e in task.examples}
    result = run([task], lambda prompt, max_tokens, stop: rambling[prompt])[0]
    assert result.score == 0.0


def test_two_runs_on_the_same_checkpoint_give_identical_numbers() -> None:
    first = run(suite(0), random_model(7))
    second = run(suite(0), random_model(7))
    assert first == second
    assert [row.task for row in first] == [task.name for task in suite(0)]
