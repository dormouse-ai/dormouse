"""One runnable check for the 500M/20B config: it prices and it stays put."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import fields
from pathlib import Path

from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.spec import StackSpec

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_500m_20b.toml"


def test_config_prices_to_500m_active_20b_stored_and_parses() -> None:
    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    keys = {f.name for f in fields(StackSpec) if f.init}
    spec = StackSpec.from_mapping({k: v for k, v in raw.items() if k in keys}, source=str(CONFIG))

    report = assemble(spec, device="meta").report()
    assert abs(report["active"] - 500e6) / 500e6 <= 0.02, report["active"]
    assert abs(report["stored"] - 20e9) / 20e9 <= 0.02, report["stored"]
    assert any("params: stored=" in line for line in plan_lines(spec))

    resolved = {f.name: getattr(spec, f.name) for f in fields(spec) if f.init}
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode())
    assert digest.hexdigest()[:16] == "17b8af1c1ca7888b"
