"""One config dialect: a file `dorm train` accepts must also be one `dorm plan` prices.

The split was real: `plan` read a price-only StackSpec (so run keys were "unknown"),
`train` read a RunRecipe (so a price-only file had "no [[source]]").  Thirteen shipped
configs, zero that passed both.  This matrix keeps the two readers on one schema, and
keeps a refused run from leaving a config-only zombie directory behind.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from dormouse_core.errors import LoudError

from dormouse import cli
from dormouse.train import entry
from dormouse.train.runconfig import RunRecipe

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.toml"))


def _plan_rc(path: Path) -> int:
    """Run the real `dorm plan` command, swallowing its bill."""
    with redirect_stdout(io.StringIO()):
        return cli.main(["plan", str(path)])


def _trainable() -> list[Path]:
    """Configs `dorm train` reads: they carry a `[[source]]`, eval and tokenizer."""
    out: list[Path] = []
    for path in CONFIGS:
        try:
            RunRecipe.read(path)
        except LoudError:
            continue
        out.append(path)
    return out


def test_at_least_one_shipped_config_passes_both_commands() -> None:
    """The documented plan/train pair works: not zero of thirteen."""
    both = [path.name for path in _trainable() if _plan_rc(path) == 0]
    assert both, "no shipped config passes both `dorm plan` and `dorm train`"


def test_every_trainable_config_uses_one_config_schema() -> None:
    """A file `dorm train` reads is never an 'unknown key' to `dorm plan`.

    `plan` prices on the meta device; `smoke.toml` still refuses there at the
    documented 10M-expert floor, but that is a price refusal, not a dialect one.
    """
    trainable = _trainable()
    assert len(trainable) >= 1
    for path in trainable:
        assert RunRecipe.read_spec(path).as_dict() == RunRecipe.read(path).spec.as_dict(), (
            f"plan and train read different stacks: {path.name}"
        )


def test_a_price_only_stack_gets_a_named_train_refusal() -> None:
    """A stack-only file is not a run; `train` says so by name instead of guessing."""
    with pytest.raises(LoudError, match=r"no \[\[source\]\] table"):
        RunRecipe.read(CONFIGS[0].parent / "dormouse_1b.toml")


def test_a_refusal_before_start_leaves_no_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A priced-out run must not leave config.toml/json that look like a live run."""

    def refused(*_args: object, **_kwargs: object) -> tuple[float, int, str]:
        raise LoudError("the plan refused", "fix the config")

    monkeypatch.setattr(entry, "_plan", refused)
    config = CONFIGS[0].parent / "train_404m_b6.toml"
    with pytest.raises(LoudError, match="the plan refused"):
        entry.train(config, runs=tmp_path / "runs", echo=lambda _msg: None)
    assert not (tmp_path / "runs" / config.stem).exists()
