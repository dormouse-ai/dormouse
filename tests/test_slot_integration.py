"""The three slots coexist: all on together, any subset alone, and all off as master.

The last contract is a golden, not a claim: five CPU steps of the flag-off stack must
reproduce the loss and parameter digest captured from pristine upstream/master at
`9a71c4db` (Python 3.12, torch 2.15.0+cpu, one thread). A whole step is deterministic
only at one thread on CPU; the golden was recorded with `torch.set_num_threads(1)`.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dormouse.cli import main
from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PAIRS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jepa_pairs"

#: Pristine `upstream/master@9a71c4db`, same seeds, same widths, five steps, one thread.
MASTER_LOSSES = (
    "31.90871238708496",
    "30.46558380126953",
    "28.13888168334961",
    "25.251773834228516",
    "22.06128692626953",
)
MASTER_DIGEST = "f3da08a71e0e335016eed49f453c26342379ea4e7e5c469b16f8062b4b485a80"

#: Every slot kept, only the numbers shrunk — the same smoke widths the other stack
#: tests use. KDA needs a sequence that tiles its 64-token chunk.
SMOKE = {
    "d_model": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 16,
    "n_layers": 4,
    "d_ffn": 64,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 16,
    "attn_k": 32,
    "attn_window": 16,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}


def _spec() -> StackSpec:
    return replace(StackSpec.from_toml(CONFIGS / "slots_all.toml"), jepa_pairs_path=str(PAIRS))


def _all_slots() -> StackSpec:
    return replace(_spec(), **SMOKE)


def _packed(step: int, spec: StackSpec) -> torch.Tensor:
    """The paired batch, packed as `[view A | view B]` for one step."""
    assert spec.jepa != "none"
    from dormouse.train.jepa import JepaPair

    pair = JepaPair.from_dir(PAIRS, batch=spec.batch, seq_len=spec.seq_len)
    view_a, view_b = pair.at(step)
    return torch.cat([view_a, view_b], dim=1)


def _digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for param in model.parameters():
        h.update(param.detach().numpy().tobytes())
    return h.hexdigest()


def test_all_three_slots_run_five_cpu_steps_and_report_every_counter() -> None:
    spec = _all_slots()
    stack = assemble(spec, smoke=True)
    optimizer = MixOptimizer(stack.model, steps=2)
    assert stack.aux is not None and stack.jepa is not None and stack.jepa_pair is not None

    for step in range(5):
        ids = _packed(step, spec)
        result, counters = stack.step(ids, optimizer=optimizer)
        assert not result.skipped, f"step {step} tripped the NaN firewall"
        assert torch.isfinite(torch.tensor(result.loss)), f"step {step}: {result.loss}"
        assert result.aux_loss is not None and torch.isfinite(torch.tensor(result.aux_loss))
        assert torch.isfinite(torch.tensor(result.jepa_loss))
        if step == 4:
            print(f"\n{counters.line()}")
            final = counters
    for counter in (
        "attnres_sources",
        "aux_fwd",
        "aux_loss",
        "jepa_seen",
        "jepa_fwd",
        "jepa_replay_rows",
        "jepa_memory_size",
    ):
        assert final.counts[counter] > 0, counter
    assert final.silent() == []
    final.check()
    assert set(stack.mechanisms().values()) <= set(final.counts)
    # One head per slot, under two distinct names: no second hook, no shared counter.
    modules = dict(stack.model.named_modules())
    assert "aux" in modules and "jepa" in modules and modules["aux"] is not modules["jepa"]
    assert [name for name in modules if name.endswith("aux")] == ["aux"]


@pytest.mark.parametrize(
    ("residual", "aux", "jepa"),
    [
        (r, a, j)
        for r in ("plain", "attnres", "mhc")
        for a in ("none", "lightmtp")
        for j in ("none", "replay")
    ],
)
def test_any_subset_builds_and_steps(residual: str, aux: str, jepa: str) -> None:
    spec = _all_slots()
    changes: dict[str, object] = {"residual": residual, "aux": aux, "jepa": jepa}
    if aux == "none":
        # A config that sets aux_* numbers with the head off is refused on purpose;
        # a subset that turns it off must drop the numbers with it.
        changes.update(aux_k=8, aux_layer=0, aux_weight=0.1, aux_varreg=False)
    spec = replace(spec, **changes)
    stack = assemble(spec, smoke=True)
    optimizer = MixOptimizer(stack.model, steps=2)
    ids = _packed(0, spec) if jepa == "replay" else torch.randint(0, 256, (2, 129))
    result, counters = stack.step(ids, optimizer=optimizer)
    assert not result.skipped
    assert torch.isfinite(torch.tensor(result.loss))
    assert (result.aux_loss is not None) == (aux == "lightmtp")
    expected = {"attnres": "attnres_sources", "mhc": "mhc_mix"}.get(residual)
    if expected:
        assert counters.counts[expected] > 0
    else:
        assert "attnres_sources" not in counters.counts and "mhc_mix" not in counters.counts
    assert ("aux_fwd" in counters.counts) == (aux == "lightmtp")
    assert ("jepa_seen" in counters.counts) == (jepa == "replay")
    assert counters.silent() == []
    counters.check()


def test_flag_off_is_bit_identical_to_master() -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1234)
        spec = StackSpec(
            unit="byte",
            attn="hybrid",
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            c_lat=16,
            n_layers=4,
            d_ffn=64,
            n_experts=4,
            topk=1,
            n_shared=1,
            expert_tsct_rank=16,
            attn_k=32,
            attn_window=16,
            seq_len=64,
            batch=2,
            tokens=10_000,
        )
        stack = assemble(spec, smoke=True)
        optimizer = MixOptimizer(stack.model, steps=2)
        ids = torch.randint(0, 256, (2, 65), generator=torch.Generator().manual_seed(7))
        losses = []
        for _ in range(5):
            result, _counters = stack.step(ids, optimizer=optimizer)
            losses.append(repr(result.loss))
        assert tuple(losses) == MASTER_LOSSES
        assert _digest(stack.model) == MASTER_DIGEST
    finally:
        torch.set_num_threads(threads)


def test_one_plan_line_names_all_three_slots_and_their_counters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _spec()
    lines = plan_lines(spec)
    slots = [line for line in lines if line.startswith("slots: ")]
    assert len(slots) == 1
    line = slots[0]
    for expected in (
        "residual=attnres→attnres_sources",
        "aux=lightmtp→aux_fwd+aux_loss",
        "jepa=replay→jepa_seen+jepa_fwd+jepa_replay_rows+jepa_memory_size",
    ):
        assert expected in line, line
    import os

    cwd = Path.cwd()
    os.chdir(CONFIGS.parent)
    try:
        code = main(["plan", str(CONFIGS / "slots_all.toml")])
    finally:
        os.chdir(cwd)
    assert code == 0
    printed = capsys.readouterr().out
    assert sum(item.startswith("slots: ") for item in printed.splitlines()) == 1
    assert "residual=attnres" in printed and "aux=lightmtp" in printed and "jepa=replay" in printed
