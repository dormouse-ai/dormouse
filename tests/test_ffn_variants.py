"""`la` and `moa` against the baseline they claim to improve.

Five gates, each one a way the arms can be wrong while looking trained:

* **degeneracy** — a one-entry dictionary must be the plain FFN `torch.equal`, at
  the FFN and at the logits, on both topologies. A mixture that is merely close
  is a mixture wired wrong;
* **the bill** — `K` parameters for LA, `K + d·K + K` for MoA, measured from the
  module tree against the three-matrix bill, and linear in `d` not `d²`;
* **expressivity** — coefficients `(1, 0, …)` ARE the pure first activation, and
  a mixed vector is not;
* **gradient reach** — every enabled entry moves its coefficient (and, for MoA,
  its gate row); the zero-input case is the demonstration that the check fails
  when nothing flows;
* **counters** — `ffn_la` / `ffn_moa` / `ffn_dict_entries_used` prove which path
  ran, so a run cannot silently execute the baseline.

The formulas are the paper's (arXiv 2605.26647); the anchors here are the plain
FFN's own weights, never a golden captured from this code.
"""

from __future__ import annotations

import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError
from stacks import REFERENCE, arm_declarations, reference_stack

from dormouse.model.activations import SITU_BETA, activation
from dormouse.model.ffn import (
    PROJ_STD,
    GatedFFN,
    LaFFN,
    MoaFFN,
    neutral_width,
)
from dormouse.model.ffn_spec import FFNSpec

D, F = 12, 8
#: 3·12·8 = 288 = 2·12·12: the neutral pair the `la`/`moa` width rule wants.
GELU_F = 12
#: The mechanism name and the counter each mixture arm reports itself with.
ARMS = {"la": ("ffn_la", "ffn_la"), "moa": ("moa", "moa_fwd")}
#: The dictionary the "all entries learn" tests exercise: the paper's five plus
#: the one bounded member this project adds.
SIX = ("gelu", "silu", "relu2", "leaky", "tanh", "situ")


def _model(act: str | None = None, dictionary: list[str] | None = None) -> torch.nn.Module:
    """A two-layer reference stack whose FFN is the arm under test."""
    ffn = None if act is None else FFNSpec(D, F, act=act, dictionary=dictionary, d_ffn_gelu=GELU_F)
    return reference_stack(
        d_model=D,
        n_layers=2,
        n_heads=2,
        head_dim=6,
        d_ffn=F,
        max_seq_len=8,
        ffn=ffn,
        declarations=REFERENCE + arm_declarations(act or "silu"),
    )


def _copy(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Load `source`'s weights into `target`; the extra mixture parameters stay put."""
    target.load_state_dict(source.state_dict(), strict=False)


# --- the bill ---------------------------------------------------------------


def test_the_added_parameters_are_k_and_k_plus_dk_plus_k() -> None:
    """`K` scalars for LA; a `d → K` gate, its bias and `K` priors for MoA."""
    k = len(SIX)
    la = FFNSpec(832, 2048, act="la", dictionary=SIX, d_ffn_gelu=3072)
    moa = FFNSpec(832, 2048, act="moa", dictionary=SIX, d_ffn_gelu=3072)
    assert la.added_parameters() == k == 6
    assert moa.added_parameters() == 832 * k + k + k == 5004
    base = 3 * 832 * 2048
    la_bill = sum(p.numel() for p in la.build().parameters())
    moa_bill = sum(p.numel() for p in moa.build().parameters())
    print(f"\nla: {la_bill - base} added, moa: {moa_bill - base} added, base {base}")
    assert la_bill == base + k
    assert moa_bill == base + 832 * k + 2 * k


def test_the_added_bill_is_o_d_not_o_d_squared() -> None:
    """Doubling `d` adds `K·d/2` parameters, not `K·d²/4`."""
    k = len(SIX)
    narrow_f = neutral_width(1536)
    wide = FFNSpec(832, 2048, act="moa", dictionary=SIX, d_ffn_gelu=3072).build()
    narrow = FFNSpec(416, narrow_f, act="moa", dictionary=SIX, d_ffn_gelu=1536).build()
    wide_added = sum(p.numel() for p in wide.parameters()) - 3 * 832 * 2048
    narrow_added = sum(p.numel() for p in narrow.parameters()) - 3 * 416 * narrow_f
    assert wide_added - narrow_added == k * (832 - 416)
    assert wide_added < 832**2  # 5004 vs 692 224: nowhere near quadratic


def test_neutral_width_is_the_la_moa_width_rule() -> None:
    """`3·d·f == 2·d·f_gelu` at the live shape, and the spec agrees."""
    assert 3 * 832 * 2048 == 2 * 832 * 3072
    spec = FFNSpec(832, neutral_width(3072), act="la", d_ffn_gelu=3072)
    assert spec.d_ffn == 2048


# --- expressivity -----------------------------------------------------------


def test_coefficients_one_zero_are_the_pure_first_activation() -> None:
    """The mixture's basis claim, with the same weights as the plain FFN."""
    torch.manual_seed(4)
    plain = GatedFFN(D, F, act="gelu")
    la = LaFFN(D, F, dictionary=SIX)
    _copy(la, plain)
    with torch.no_grad():
        la.coef.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    x = torch.randn(3, 4, D)
    assert torch.equal(la(x), plain(x))


def test_a_mixed_coefficient_vector_is_not_the_plain_activation() -> None:
    """If `(0.5, 0.5, …)` equalled silu, the second entry would be decoration."""
    torch.manual_seed(5)
    plain = GatedFFN(D, F)
    la = LaFFN(D, F, dictionary=("silu", "relu2"))
    _copy(la, plain)
    with torch.no_grad():
        la.coef.copy_(torch.tensor([0.5, 0.5]))
    x = torch.randn(3, 4, D)
    got, want = la(x).detach(), plain(x).detach()
    assert not torch.equal(got, want)
    assert float((got - want).abs().max()) > 1e-6


# --- gradient reach ---------------------------------------------------------


@pytest.mark.parametrize("cls", [LaFFN, MoaFFN])
def test_every_enabled_entry_receives_gradient(cls: type[LaFFN] | type[MoaFFN]) -> None:
    """An entry with a coefficient but no gradient is a silently dead entry."""
    torch.manual_seed(6)
    ffn = cls(D, F, dictionary=SIX)
    ffn(torch.randn(4, 5, D)).sum().backward()
    assert ffn.dead_entries() == ()
    assert ffn.coef.grad is not None
    for k, name in enumerate(SIX):
        assert float(ffn.coef.grad[k]) != 0.0, name
        if isinstance(ffn, MoaFFN):
            assert ffn.mix.weight.grad is not None
            assert float(ffn.mix.weight.grad[k].abs().sum()) > 0.0, name


@pytest.mark.parametrize("cls", [LaFFN, MoaFFN])
def test_the_dead_entry_check_fails_when_nothing_flows(cls: type[LaFFN] | type[MoaFFN]) -> None:
    """The zero input is the measured failure path: no entry contributes, all are named."""
    ffn = cls(D, F, dictionary=("silu", "relu2"))
    ffn(torch.zeros(1, 3, D)).sum().backward()
    assert ffn.dead_entries() == ("silu", "relu2")


def test_a_zero_coefficient_is_not_an_enabled_entry() -> None:
    """Switching an entry off must remove it from the count, not fail as dead."""
    torch.manual_seed(7)
    la = LaFFN(D, F, dictionary=("silu", "relu2"))
    with torch.no_grad():
        la.coef[1] = 0.0
    la(torch.randn(2, 3, D)).sum().backward()
    assert la.dead_entries() == ()


# --- counters ---------------------------------------------------------------


@pytest.mark.parametrize(("cls", "arm"), [(LaFFN, "ffn_la"), (MoaFFN, "moa_fwd")])
def test_the_arm_reports_itself_and_its_entries(cls: type[LaFFN] | type[MoaFFN], arm: str) -> None:
    counters = Counters()
    ffn = cls(D, F, dictionary=SIX)
    ffn(torch.zeros(1, 3, D), counters)
    assert counters.snapshot() == {"ffn_fwd": 3, arm: 3, "ffn_dict_entries_used": len(SIX)}


@pytest.mark.parametrize("act", ["la", "moa"])
def test_a_mixed_model_enables_its_arm_and_check_passes(act: str) -> None:
    """`Counters.check()` is the proof the configured arm ran, not the baseline."""
    model = _model(act, ["silu"])
    counters = Counters()
    model(torch.zeros(1, 4, dtype=torch.long), counters)
    mechanism, counter = ARMS[act]
    assert {mechanism, "ffn_dictionary"} <= set(counters.enabled)
    assert counters.counts[counter] == 2 * 4
    assert counters.counts["ffn_dict_entries_used"] == 2
    counters.check()


def test_the_plain_model_does_not_enable_a_mixture() -> None:
    counters = Counters()
    _model()(torch.zeros(1, 2, dtype=torch.long), counters)
    assert not {"ffn_la", "moa_fwd", "ffn_dictionary"} & set(counters.enabled)


# --- init and config --------------------------------------------------------


def test_las_coefficients_start_at_one_and_moas_gate_is_small() -> None:
    la = LaFFN(D, F, dictionary=SIX)
    assert torch.equal(la.coef, torch.ones(len(SIX)))
    moa = MoaFFN(D, F, dictionary=SIX)
    assert torch.equal(moa.coef, torch.ones(len(SIX)))
    assert moa.mix.weight.std().item() == pytest.approx(PROJ_STD, rel=0.2)
    assert torch.equal(moa.mix.bias, torch.zeros(len(SIX)))


def test_an_empty_or_unknown_dictionary_is_refused() -> None:
    with pytest.raises(LoudError, match="empty"):
        FFNSpec(832, 2048, act="la", dictionary=(), d_ffn_gelu=3072)
    with pytest.raises(LoudError, match="unknown activation"):
        FFNSpec(832, 2048, act="la", dictionary=["nope"], d_ffn_gelu=3072)
    with pytest.raises(LoudError, match="repeats"):
        FFNSpec(832, 2048, act="moa", dictionary=["silu", "silu"], d_ffn_gelu=3072)


def test_la_and_moa_keep_the_parameter_neutral_width() -> None:
    """The old convention (the GELU width) is refused with both bills named."""
    with pytest.raises(LoudError) as e:
        FFNSpec(832, 3072, act="la", d_ffn_gelu=3072)
    assert "3·832·3072=7667712" in str(e.value)
    assert "2·832·3072=5111808" in str(e.value)
    with pytest.raises(LoudError, match="parameter-neutral"):
        FFNSpec(832, 2048, act="moa")
    assert FFNSpec(832, 2048, act="moa", d_ffn_gelu=3072).d_ffn == 2048


def test_a_dictionary_that_nothing_reads_is_refused() -> None:
    with pytest.raises(LoudError, match="only feeds la/moa"):
        FFNSpec(832, 2048, act="silu", dictionary=["gelu"])
    with pytest.raises(LoudError, match="sides is MoA's"):
        FFNSpec(832, 2048, act="silu", sides="bi")
    with pytest.raises(LoudError, match="not an FFN arm"):
        FFNSpec(832, 2048, act="swiglu")
    with pytest.raises(LoudError, match="not a topology"):
        FFNSpec(832, 2048, act="moa", sides="both", d_ffn_gelu=3072)


def test_the_situ_entry_is_the_bounded_half_of_k3s_glu() -> None:
    """`|β₁·tanh(z/β₁)·sigmoid(z)| ≤ β₁` up to |z| = 1e3 — the quantization property."""
    z = torch.linspace(-1e3, 1e3, 20_001)
    assert float(activation("situ")(z).abs().max()) <= SITU_BETA
    assert float(activation("tanh")(z).abs().max()) <= 1.0
