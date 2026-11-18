"""Three gates for mechanisms that could report a number without running.

Each gate here is red on the code that merely counted a value: a zero that was
never measured, an entry with no gradient, and a quantization swap that was
undone still reported a positive count. The fix makes silence a `MissingReport`
at the step's final `Counters.check()`, not a plausible-looking log line.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import MissingReport

from dormouse.model import quant
from dormouse.model.ffn import LaFFN, MoaFFN
from dormouse.model.moe import MoEFFN
from dormouse.model.registry import QAT

D, F = 16, 12


def _bank() -> MoEFFN:
    """A small bank; the 10M-parameter floor is irrelevant to the counter gate."""
    return MoEFFN(
        D,
        F,
        n_experts=4,
        topk=2,
        n_shared=1,
        act="moa",
        d_ffn_gelu=3 * F // 2,
        expert_tsct_rank=0,
        min_expert_params=0,
    )


def test_the_drop_measurement_runs_and_is_enabled() -> None:
    """A dropless bank still reports that it MEASURED zero drops."""
    counters = Counters()
    _bank()(torch.randn(16, D), counters)
    assert "moe_dropped" in counters.enabled
    assert counters.counts["moe_dropped"] == 0, "dropless: zero is the measurement"
    assert counters.counts["moe_drop_checks"] == 1, "the measurement itself ran"
    counters.check()


def test_a_silent_drop_measurement_is_a_missing_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress the proof the drop balance was measured; the step must refuse."""
    counters = Counters()
    real_count = counters.count

    def skip_proof(key: str, n: int = 1) -> None:
        if key != "moe_drop_checks":
            real_count(key, n)

    monkeypatch.setattr(counters, "count", skip_proof)
    _bank()(torch.randn(16, D), counters)
    with pytest.raises(MissingReport, match="moe_dropped"):
        counters.check()


@pytest.mark.parametrize("cls", [LaFFN, MoaFFN])
def test_a_dictionary_with_no_live_entry_is_a_missing_report(
    cls: type[LaFFN] | type[MoaFFN],
) -> None:
    """Zero input gives every enabled entry a zero gradient: the step must refuse."""
    ffn = cls(D, F, dictionary=("silu", "relu2"))
    counters = Counters()
    ffn(torch.zeros(1, 3, D), counters).sum().backward()
    with pytest.raises(MissingReport, match="dictionary"):
        counters.check()


@pytest.mark.parametrize("cls", [LaFFN, MoaFFN])
def test_a_live_dictionary_passes_the_step_end_check(cls: type[LaFFN] | type[MoaFFN]) -> None:
    """The same check is green when the entries actually receive gradient."""
    torch.manual_seed(11)
    ffn = cls(D, F, dictionary=("silu", "relu2"))
    counters = Counters()
    ffn(torch.randn(4, D), counters).sum().backward()
    counters.check()


class _Inner(torch.nn.Module):
    """A counters-aware inner block so `QuantBlock` can wrap it."""

    def __init__(self) -> None:
        super().__init__()
        self.linears = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        return self.linears(x)


def test_qat_reports_live_ternary_not_a_static_list() -> None:
    """Undo the swap after the block was built: no ternary forward ran, so refuse."""
    inner = _Inner()
    quant.ternarize(inner)
    block = quant.QuantBlock(inner, matrices=2)
    for module in inner.modules():
        if type(module) is quant.QuantLinear:
            module.__class__ = torch.nn.Linear
    piece = QAT["ternary"]
    counters = Counters()
    counters.enable(piece.mechanism, counter=piece.counter, why=piece.why)
    block(torch.randn(3, 4), counters)
    with pytest.raises(MissingReport, match=piece.mechanism):
        counters.check()


def test_qat_passes_when_the_ternary_weights_are_live() -> None:
    """The same counter is positive when `QuantLinear` really runs the forward."""
    inner = _Inner()
    quant.ternarize(inner)
    block = quant.QuantBlock(inner, matrices=2)
    piece = QAT["ternary"]
    counters = Counters()
    counters.enable(piece.mechanism, counter=piece.counter, why=piece.why)
    block(torch.randn(3, 4), counters)
    assert counters.counts[piece.counter] == 2
    counters.check()
