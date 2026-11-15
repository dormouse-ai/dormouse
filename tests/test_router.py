"""The router: scores, top-k, a device-side count, and Quantile Balancing.

Five gates, each one a way a router can be wrong while looking trained:

* **the score** — `sqrt(softplus(logit))` recomputed in numpy from the module's
  own weights, so a swapped softmax or a missing root is O(1) away;
* **route quality** — eight clusters whose correct expert is obvious; 200 Adam
  steps on the routing loss alone must select each token's own cluster, or the
  projection is not learning the pattern a router exists to learn;
* **balance** — QB on a uniform random score bank stays inside the 1.25x
  max/mean band after one update, the recorded skewed fixture converges to
  ±10% of the target, and a balanced bank is a fixed point;
* **device-side** — the controller's AST contains no `.item()`, `.tolist()`,
  `.cpu()`, `.numpy()`, `nonzero` or `bincount`, and `torch.compile(fullgraph)`
  traces the whole router (score → top-k → histogram → bias) into one graph;
* **the bill** — `topk=1` weighs exactly `1.0`, which is what makes the bank's
  dense-FFN parity gate exact rather than approximate.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from dormouse.model.router import Router

D = 64
N = 16384


def _loads(router: Router, logits: torch.Tensor) -> torch.Tensor:
    """The router's share per expert: top-k on the raised logits, `[E]` summing to 1."""
    index = (logits + router.bias).topk(router.topk, dim=-1).indices
    return router.count(index).float() / (logits.shape[0] * router.topk)


# --- the score --------------------------------------------------------------


def test_scores_are_the_square_root_of_softplus() -> None:
    """`sqrt(log(1 + e^z))` in numpy, from the router's own weights."""
    torch.manual_seed(0)
    router = Router(6, 4, 2)
    x = torch.randn(5, 6)
    z = x.detach().numpy() @ router.weight.detach().numpy().T
    want = np.sqrt(np.log1p(np.exp(z)))
    assert np.allclose(router.scores(router.logits(x)).detach().numpy(), want, atol=1e-7)


def test_a_top_one_route_weighs_exactly_one() -> None:
    """`w / w` is 1.0, not 0.999...: the parity gate in `moe.py` rests on this."""
    torch.manual_seed(1)
    router = Router(D, 8, 1)
    _, weight, counts = router(torch.randn(37, D))
    assert torch.equal(weight, torch.ones(37, 1))
    assert int(counts.sum()) == 37


# --- route quality ----------------------------------------------------------


def test_the_router_learns_eight_obvious_clusters() -> None:
    """200 steps on the routing loss must put each token's own cluster in its top-2.

    The projection starts at `N(0, 0.02²)`, so before training the clusters are
    noise (12.5% top-1). The gate fails if the router cannot learn a pattern
    that a single linear map separates by a margin of 2.0.
    """
    torch.manual_seed(0)
    d, e, tokens = 32, 8, 2048
    centers = torch.randn(e, d) * 2.0
    labels = torch.arange(tokens) % e
    x = centers[labels] + 0.1 * torch.randn(tokens, d)
    router = Router(d, e, 2)

    def routed() -> torch.Tensor:
        with torch.no_grad():
            return router.select(router.logits(x))[0]

    before = float((routed()[:, 0] == labels).float().mean())
    optimizer = torch.optim.Adam(router.parameters(), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(router.logits(x), labels)
        loss.backward()
        optimizer.step()
    index = routed()
    assert before < 0.3, f"the fixture is too easy: {before:.3f} top-1 before training"
    assert float(loss.detach()) < 0.2
    assert float((index[:, 0] == labels).float().mean()) >= 0.95
    assert torch.all((index == labels[:, None]).any(dim=1)), "a cluster fell out of top-2"


# --- balance ----------------------------------------------------------------


def test_qb_keeps_a_uniform_random_input_within_1_25() -> None:
    """The gate: one update on `x ~ N(0,1)` must not push the load ratio past 1.25."""
    torch.manual_seed(3)
    n, d, e, k = N, D, 16, 2
    router = Router(d, e, k)
    logits = router.logits(torch.randn(n, d)).detach()
    before = _loads(router, logits)
    router.qb_update(logits)
    after = _loads(router, logits)
    ratio_before = float(before.max() / before.mean())
    ratio_after = float(after.max() / after.mean())
    assert ratio_before <= 1.25, ratio_before
    assert ratio_after <= 1.25, f"QB broke a uniform bank: {ratio_before:.3f} -> {ratio_after:.3f}"
    assert ratio_after < ratio_before, "the update made the load worse than not running"


def test_qb_moves_a_skewed_bank_to_its_target_load() -> None:
    """The recorded fixture: `N(0,1) + linspace(1.5, 0)`, E=8, top-2."""
    torch.manual_seed(0)
    n, e, k = 8192, 8, 2
    logits = torch.randn(n, e) + torch.linspace(1.5, 0.0, e)
    router = Router(4, e, k)
    target = 1.0 / e
    raw = _loads(router, logits)
    assert float((raw - target).abs().max()) > 0.5 * target, (
        f"the fixture is not skewed: loads {[round(v, 4) for v in raw.tolist()]}"
    )
    for _ in range(50):
        router.qb_update(logits)
    got = _loads(router, logits)
    assert torch.all((got - target).abs() <= 0.10 * target), (
        f"target {target:.3f}, loads {[round(v, 4) for v in got.tolist()]}"
    )


def test_an_already_balanced_bank_is_a_fixed_point() -> None:
    """Every expert's logit column is a permutation of the same values: no load gap."""
    generator = torch.Generator().manual_seed(1)
    n, e = 8192, 8
    base = torch.randn(n, 1, generator=generator).repeat(1, e)
    order = torch.stack([torch.randperm(e, generator=generator) for _ in range(n)])
    logits = torch.gather(base, 1, order)
    router = Router(4, e, 2)
    before = router.bias.clone()
    for _ in range(50):
        router.qb_update(logits)
    moved = float((router.bias - before).abs().max())
    assert moved < 1e-5, f"the bias moved {moved:.2e} on a balanced bank"


def test_the_binned_quantile_matches_torch_quantile_within_a_bin() -> None:
    """The bias is `mean(q) - q` of the `1 - topk/E` quantile: anchor it outside."""
    torch.manual_seed(5)
    n, e = 8192, 2
    logits = torch.randn(n, e) * 2.0 + torch.tensor([1.0, 0.0])
    router = Router(4, e, 1)
    router.qb_update(logits)
    quantiles = torch.quantile(logits, 1.0 - router.topk / e, dim=0)
    got = -router.bias  # bias = mean(q) - q, and only the differences are observable
    error = (got - got.mean()) - (quantiles - quantiles.mean())
    span = logits.max(dim=0).values - logits.min(dim=0).values
    assert torch.all(error.abs() <= span / router.qb_bins + 1e-3), error


# --- device-side ------------------------------------------------------------


def test_the_router_never_leaves_the_device() -> None:
    """No host read may sit in the routing path: a CUDA graph cannot hold one."""
    tree = ast.parse(inspect.getsource(Router))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    banned = {"item", "tolist", "bincount", "cpu", "numpy", "nonzero"}
    assert called & banned == set(), f"host-side call(s) in Router: {called & banned}"


def test_the_router_traces_into_one_graph() -> None:
    """`fullgraph=True` raises on a graph break; the outputs must be bit-identical."""
    torch.manual_seed(2)
    reference = Router(8, 4, 2)
    traced = Router(8, 4, 2)
    traced.load_state_dict(reference.state_dict())
    x = torch.randn(16, 8)
    torch._dynamo.reset()
    compiled = torch.compile(traced, backend="eager", fullgraph=True)
    got = compiled(x)
    torch._dynamo.reset()
    want = reference(x)
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert torch.equal(traced.bias, reference.bias), "the QB update left the graph"


# --- state and refusals -----------------------------------------------------


def test_the_bias_is_a_buffer_that_only_moves_in_training() -> None:
    """A controller whose state an optimizer can step is not a load controller."""
    torch.manual_seed(3)
    router = Router(8, 4, 2)
    assert "bias" in dict(router.named_buffers())
    assert "bias" not in dict(router.named_parameters())
    assert torch.equal(router.bias, torch.zeros(4))
    counter = Counters()
    router(torch.randn(9, 8), counter)
    assert counter.snapshot() == {"moe_qb": 1}
    trained = router.bias.clone()
    router.eval()
    router(torch.randn(9, 8), Counters())
    assert torch.equal(router.bias, trained), "eval moved the load controller"


def test_qb_off_never_touches_the_bias_or_the_counter() -> None:
    """The OFF arm must be the plain top-k, to the bit: no parameter, no counter."""
    torch.manual_seed(4)
    router = Router(8, 4, 2, qb=False)
    counter = Counters()
    router(torch.randn(9, 8), counter)
    assert counter.snapshot() == {}
    assert torch.equal(router.bias, torch.zeros(4))


def test_an_impossible_routing_is_refused_with_an_escape() -> None:
    with pytest.raises(LoudError, match="not a width"):
        Router(0, 4)
    with pytest.raises(LoudError, match="not a bank"):
        Router(8, 0)
    with pytest.raises(LoudError, match="topk=3 of n_experts=2"):
        Router(8, 2, 3)
    with pytest.raises(LoudError, match="topk=0"):
        Router(8, 4, 0)
    with pytest.raises(LoudError, match="cannot locate a quantile"):
        Router(8, 4, 2, qb_bins=1)
    with pytest.raises(LoudError) as raised:
        Router(8, 2, 3)
    assert "Escape:" in str(raised.value)
    assert "n_experts=4" in repr(Router(8, 4, 2))
