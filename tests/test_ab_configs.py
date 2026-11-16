"""The A/B is an A/B only if the arm differs from the control in two values.

Verdict comments may be added after the measurement; an assignment may not. Both
checks are here so a third tuned field turns the test red instead of an unnoticed
confound in the BPB delta.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONTROL = CONFIGS / "dormouse_1b.toml"
ARM = CONFIGS / "dormouse_1b_top1_r192.toml"


def _assignments(path: Path) -> dict[str, str]:
    """Return only the live `key = value` lines, ignoring comments and blanks."""
    return {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }


def test_only_topk_and_rank_differ_as_assignments() -> None:
    control, arm = _assignments(CONTROL), _assignments(ARM)
    changed = {key for key in control.keys() | arm.keys() if control.get(key) != arm.get(key)}
    assert changed == {"topk", "expert_tsct_rank"}, changed
    assert control["topk"] == "2" and arm["topk"] == "1"
    assert control["expert_tsct_rank"].replace("_", "") == "256"
    assert arm["expert_tsct_rank"].replace("_", "") == "192"


def test_the_parsed_configs_differ_in_exactly_two_keys() -> None:
    control = tomllib.loads(CONTROL.read_text(encoding="utf-8"))
    arm = tomllib.loads(ARM.read_text(encoding="utf-8"))
    assert control.keys() == arm.keys()
    changed = {key for key in control if control[key] != arm[key]}
    assert changed == {"topk", "expert_tsct_rank"}, changed
    assert (control["topk"], arm["topk"]) == (2, 1)
    assert (control["expert_tsct_rank"], arm["expert_tsct_rank"]) == (256, 192)
