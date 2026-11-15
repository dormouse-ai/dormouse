"""The stack, built from the one config, priced, trained for three steps, and gated.

Three contracts live here, and each one is a number rather than an opinion:

* the model stores its experts as rank-256 factors — 400.3M stored, an FFN active set of
  16.07M — measured from the module tree, with the attention share of the active set
  printed next to it;
* every attention layer is one of the two mechanisms, in the 3:1 pattern, with GQA and
  the bottleneck on both — and the DSA layers are where the plan says they are;
* three CPU steps run with a finite loss and every enabled mechanism moved its
  counter: `kda_fwd`/`kda_bwd`, `dsa_fwd`, `moe_rows_run`, `tsct_fwd`, `moa_fwd`.
  A `MissingReport` here is the defect class the counters exist for.

The smoke widths keep every slot and shrink only the numbers, so the wiring under test
is the config's wiring.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.assemble import assemble, plan_lines
from dormouse.model.registry import pieces
from dormouse.model.spec import StackSpec
from dormouse.train.optim import MixOptimizer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

#: The one config, at widths a CPU test can afford. Every slot is kept.
SMOKE = {
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "c_lat": 32,
    "n_layers": 4,
    "d_ffn": 128,
    "n_experts": 4,
    "topk": 1,
    "expert_tsct_rank": 32,
    "attn_k": 64,
    "attn_window": 32,
    "seq_len": 128,
    "batch": 2,
    "tokens": 1_000_000,
}


def _smoke() -> StackSpec:
    return replace(StackSpec.from_toml(CONFIGS / "dormouse_1b.toml"), **SMOKE)


def test_the_model_stores_the_factors_and_runs_the_ffn_at_sixteen_million() -> None:
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    report = assemble(spec, device="meta").report()
    print(
        f"\ndormouse_1b: stored={report['stored']:,} active={report['active']:,} "
        f"ffn_active={report['active_ffn']:,} attention={report['attention']:,} "
        f"embedding={report['embedding']:,}"
    )
    per_expert = 3 * (832 * 256 + 256 * 6144) + 832 * 6 + 2 * 6
    assert report["active_ffn"] == 3 * per_expert + 64 * 832
    assert report["ffn"] == per_expert * 65 + 64 * 832
    #: 400M and not the dense bank's 1.0B: the `[832, 6144]` masters are not stored.
    assert 0.35e9 <= report["stored"] <= 0.45e9, report["stored"]
    assert report["active_ffn"] < report["stored"] / 20
    assert report["head"] == 0  # tied: a nonzero here is a second matrix


def test_the_attention_share_of_the_step_is_printed_and_is_the_minority() -> None:
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    report = assemble(spec, device="meta").report()
    share = report["attention"] / report["active"]
    print(f"\nattention share of the active set: {share:.1%}")
    assert 0.2 <= share <= 0.4, share


def test_the_layers_are_the_pattern_the_config_names() -> None:
    assembled = assemble(_smoke(), smoke=True)
    kinds = [type(block.attn).__name__ for block in assembled.model.blocks]
    assert kinds == ["KDA", "KDA", "KDA", "DSA"]
    kda = assembled.model.blocks[0].attn
    assert (kda.n_heads, kda.n_kv_heads, kda.c_lat) == (4, 2, 32)
    assert kda.latent, "the architecture's projections are the compressed ones"
    dsa = assembled.model.blocks[3].attn
    assert (dsa.n_kv_heads, dsa.c_lat, dsa.k) == (2, 32, 64)


def test_muon_splits_the_compressed_projections_per_head() -> None:
    """Per-head Muon on `q_up` and the K half of `kv_up`; V and `proj` stay whole."""
    assembled = assemble(_smoke(), smoke=True)
    splits = {group["head_split"] for group in MixOptimizer(assembled.model).muon.param_groups}
    assert (4, 64) in splits  # q_up: 4 heads x 16 channels, every row is a head
    assert (2, 32) in splits  # kv_up: 2 KV heads over the K half, V as the remainder
    assert None in splits  # proj, the FFN and the rest are whole matrices


def test_the_bank_is_one_module_shared_by_every_layer() -> None:
    spec = _smoke()
    assembled = assemble(spec, smoke=True)
    banks = [module for module in assembled.model.modules() if type(module).__name__ == "FlatBank"]
    inner = {id(module.bank) for module in banks}
    assert len(banks) == spec.n_layers
    assert len(inner) == 1, "a bank per layer would store and run n_layers times as much"


def test_the_plan_names_the_stack_the_bill_and_the_mechanisms() -> None:
    lines = plan_lines(StackSpec.from_toml(CONFIGS / "dormouse_1b.toml"))
    text = "\n".join(lines)
    print("\n" + text)
    for expected in (
        "hybrid 3:1 kda:dsa",
        "kda, kda, kda, dsa",
        "experts: moa",
        "tsct rank 256",
        "kv=2, c_lat=512",
        "moe→moe_fwd",
        "moa→moa_fwd",
        "tsct→tsct_fwd",
        "ntp→ntp_loss",
        "attention=",
        "of the active set",
    ):
        assert expected in text, expected


def test_three_cpu_steps_leave_every_enabled_mechanism_reported() -> None:
    spec = _smoke()
    assembled = assemble(spec, smoke=True)
    optimizer = MixOptimizer(assembled.model, steps=2)
    ids = torch.randint(0, spec.vocab, (spec.batch, spec.seq_len + 1))
    for step in range(3):
        result, counters = assembled.step(ids, optimizer=optimizer)
        assert not result.skipped, f"step {step} tripped the NaN firewall"
        assert torch.isfinite(torch.tensor(result.loss)), f"step {step}: loss={result.loss}"
    print(f"\n{counters.line()}")
    print(f"enabled={sorted(counters.enabled)}")
    assert counters.silent() == []
    counters.check()
    for counter in ("kda_fwd", "kda_bwd", "dsa_fwd", "moe_rows_run", "tsct_fwd", "moa_fwd"):
        assert counters.counts[counter] > 0, counter
    declared = {piece.mechanism for piece in assembled.pieces}
    assert declared <= counters.enabled, declared - counters.enabled
    assert {"kda", "dsa", "moe", "moa", "tsct", "ffn_dictionary"} <= counters.enabled


def test_a_mechanism_that_is_enabled_and_silent_still_raises() -> None:
    """The gate above is only worth anything if it can fail."""
    from dormouse_core.counters import Counters, MissingReport

    counters = Counters()
    counters.enable("tsct", counter="tsct_fwd")
    with pytest.raises(MissingReport):
        counters.check()


def test_the_assemble_side_refuses_a_context_the_mechanisms_cannot_take() -> None:
    """The refusal is the mechanism's own `Coeffects`, not a table in the assembler."""
    with pytest.raises(LoudError) as caught:
        pieces(
            _smoke(),
            ctx={"attn.pos": "rope", "attn.rule": "delta", "attn.selector": "true"},
        )
    assert "attn.pos" in str(caught.value) and "kda" in str(caught.value)


def test_dorm_plan_is_wired_and_train_refuses_a_config_with_no_data(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`plan` prices the 1B config; `train` on it refuses by name: the corpus is not here.

    The refusal is the point. A command that is wired must still refuse what it cannot do,
    and the 1B config carries no `[[source]]` table because the 25 GB corpus is not in this
    repository.
    """
    from dormouse.cli import main

    assert main(["plan", str(CONFIGS / "dormouse_1b.toml")]) == 0
    printed = capsys.readouterr().out
    assert "params: stored=" in printed
    assert "mechanisms:" in printed
    assert main(["train", str(CONFIGS / "dormouse_1b.toml"), "--runs", str(tmp_path)]) == 2
    assert "no [[source]] table" in capsys.readouterr().err


def test_plan_needs_no_allocation_so_the_real_stack_is_cheap() -> None:
    """The plan builds on `meta`: the same numbers, no 1.6 GiB of weights."""
    spec = StackSpec.from_toml(CONFIGS / "dormouse_1b.toml")
    measured = assemble(spec, device="meta").report()
    assert measured["stored"] > 3.9e8
    assert all(isinstance(value, int) for value in measured.values())
