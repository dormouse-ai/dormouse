"""Fail when a registered mechanism cannot prove it works, on CPU only.

    uv run --no-sync python tools/check_contracts.py

Every contract is run twice — degeneration against the stack it replaced and
anchor against an external paper formula — on seeded CPU fixtures. A mechanism
with no contract never reaches this script: assembly refuses it before it
allocates. A contract that is present but false fails here.
"""

from __future__ import annotations

import importlib
import sys

from dormouse.verify.checks import main

#: Importing a mechanism module registers its contract. The gate must not depend
#: on another test or command having imported them first.
for _module in (
    "dormouse.model.attn_variants.kda",
    "dormouse.model.attn_variants.dsa",
    "dormouse.model.tsct",
):
    importlib.import_module(_module)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
