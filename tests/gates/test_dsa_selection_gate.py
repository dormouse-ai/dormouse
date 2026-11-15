"""DSA's selection gate: can the indexer learn, is the window load-bearing, is MLI real.

Every number here is measured on CPU on a synthetic needle-in-a-haystack task and
printed, because the defect this file exists for was a selector that reported a
selection rate of exactly `k/n_blocks` — chance — while the arm was called sparse.
A gate that only checks shapes cannot see that. So:

* TRAINED, the indexer must put the needle inside a k=512 selection at 4k in >90%
  of trials after 200 steps of the SELECTION objective alone (cross-entropy of its
  own scores at the needle, the objective V3.2 distils into it);
* UNTRAINED, the same indexer must sit near chance — the failure mode, printed;
* the same task is run against block CSA at the same 512-key budget, both trained
  for the same 200 steps, and both rates are printed: if block selection wins, that
  is a finding and it is reported rather than hidden;
* with the needle one position behind the query, the window alone must recall it,
  and removing the window must collapse that recall;
* MLI's premise — adjacent layers select nearly the same tokens — is measured on
  real corpus activations, and an overlap under the floor must be refused.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.model.attn_variants.csa import CSA
from dormouse.model.attn_variants.dsa import DSA
from dormouse.model.attn_variants.dsa_indexer import MLI_MIN_OVERLAP, LightningIndexer

SEQ = 4096
K = 512
WINDOW = 128
BATCH = 8
WIDTH = 16
D_IDX = 16
STEPS = 200
CHUNK = 16

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_2mb.bin"


def _task(
    batch: int, seq: int, width: int, gen: torch.Generator, *, window: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random tokens, one needle marked on channel 0, positions in `[0, seq-window)`.

    The needle is kept clear of the last `window` positions so the local window
    cannot answer the task: the recall measured is the SELECTION's.
    """
    x = torch.randn(batch, seq, width, generator=gen)
    pos = torch.randint(0, seq - window, (batch,), generator=gen)
    x[torch.arange(batch), pos, 0] += 4.0
    return x, pos


def _recall(indexer: LightningIndexer, x: torch.Tensor, pos: torch.Tensor, k: int) -> float:
    """Fraction of trials whose last-query selection contains the needle."""
    with torch.no_grad():
        kept = indexer.select(x, k)
    return float((kept[:, -1] == pos.view(-1, 1)).any(-1).float().mean())


def _trained() -> tuple[LightningIndexer, torch.Tensor, torch.Tensor]:
    """Train the indexer for `STEPS` on the selection objective alone; print the curve."""
    torch.manual_seed(41)
    indexer = LightningIndexer(WIDTH, D_IDX)
    optimiser = torch.optim.Adam(indexer.parameters(), lr=0.05)
    training = torch.Generator().manual_seed(1)
    evals, eval_pos = _task(BATCH, SEQ, WIDTH, torch.Generator().manual_seed(2))
    curve = []
    for step in range(STEPS):
        x, pos = _task(BATCH, SEQ, WIDTH, training)
        loss = torch.nn.functional.cross_entropy(indexer.scores(x), pos)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 40 == 0 or step == STEPS - 1:
            curve.append(
                (
                    step,
                    round(float(loss.detach()), 3),
                    round(_recall(indexer, evals, eval_pos, K), 3),
                )
            )
    print(f"\nindexer recall curve (step, loss, recall@{K}): {curve}")
    return indexer, evals, eval_pos


def test_the_indexer_learns_the_needle_it_selects() -> None:
    """`>90%` recall at 4k after 200 steps — the gate the broken CSA arm failed."""
    indexer, x, pos = _trained()
    rate = _recall(indexer, x, pos, K)
    chance = K / SEQ
    print(f"trained recall {rate:.3f} at k={K}, T={SEQ} (chance {chance:.3f})")
    assert rate > 0.9
    assert rate > 3.0 * chance


def test_an_indexer_that_never_trained_stays_near_chance() -> None:
    """The failure mode, measured: a projection that never learned selects noise."""
    torch.manual_seed(42)
    frozen = LightningIndexer(WIDTH, D_IDX)
    _, x, pos = _trained()
    rate = _recall(frozen, x, pos, K)
    print(f"untrained recall {rate:.3f} at k={K}, T={SEQ} (chance {K / SEQ:.3f})")
    assert rate < 0.35


def test_the_local_window_is_load_bearing() -> None:
    """A needle one position behind the query: the window must carry it, the tail cannot.

    Both layers are untrained, so the indexer contributes chance in both arms and the
    difference is the window alone.
    """
    torch.manual_seed(43)
    x = torch.randn(BATCH, SEQ, WIDTH, generator=torch.Generator().manual_seed(3))
    at = torch.full((BATCH,), SEQ - 2)
    x[torch.arange(BATCH), at, 0] += 4.0
    rates = {}
    for window in (1, WINDOW):
        layer = DSA(WIDTH, 1, 8, k=K, window=window, d_idx=D_IDX, chunk=256)
        kept, visible = layer.attended(x)
        found = (kept[:, -1] == at.view(-1, 1)) & visible[:, -1]
        rates[window] = float(found.any(-1).float().mean())
    print(
        f"window needle: window=1 recall {rates[1]:.3f}, window={WINDOW} recall {rates[WINDOW]:.3f}"
    )
    assert rates[WINDOW] == 1.0
    assert rates[1] < MLI_MIN_OVERLAP


def _block_scores(csa: CSA, x: torch.Tensor) -> torch.Tensor:
    """The selector's own score, `[batch, seq, blocks]`, differentiable.

    `CSA.select` returns only the top-k blocks, and the objective below needs the
    full row; this is `csa.py`'s equation (eq. `s[t,c] = <q_t, k̄_c>/√d_k`) written
    out without the top-k, with HCA off.
    """
    batch, seq, _ = x.shape
    blocks = seq // CHUNK
    q, k, _ = csa.qkv(x).split(csa.d_kv, dim=-1)
    means = k.view(batch, blocks, CHUNK, csa.d_kv).mean(2)
    return torch.einsum("bsd,bnd->bsn", q, means) * csa.scale


def test_dsa_versus_block_selection_at_the_same_key_budget() -> None:
    """512 selected tokens either way, both trained 200 steps, both rates printed."""
    indexer, x, pos = _trained()
    dsa_rate = _recall(indexer, x, pos, K)

    torch.manual_seed(44)
    csa = CSA(WIDTH, 1, 8, chunk=CHUNK, topk=K // CHUNK)
    optimiser = torch.optim.Adam(csa.qkv.parameters(), lr=0.05)
    training = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        batch, needle = _task(BATCH, SEQ, WIDTH, training, window=WINDOW)
        loss = torch.nn.functional.cross_entropy(_block_scores(csa, batch)[:, -1], needle // CHUNK)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        kept = _block_scores(csa, x)[:, -1].topk(K // CHUNK, -1).indices
    csa_rate = float((kept == (pos // CHUNK).view(-1, 1)).any(-1).float().mean())
    print(
        f"\nsame budget ({K} keys): DSA {dsa_rate:.3f} vs block CSA {csa_rate:.3f} "
        f"(chance {K / SEQ:.3f}); CSA blocks of {CHUNK}, top-{K // CHUNK}"
    )
    assert dsa_rate > 0.5, "the DSA arm has to clear chance for this to be a comparison"
    if dsa_rate <= csa_rate:
        print("FINDING: block selection matched or beat per-token selection on this task")


def test_adjacent_layers_select_the_same_tokens_on_corpus_activations() -> None:
    """MLI's premise, measured on real corpus bytes, and refused when it fails.

    The bytes are the fixture corpus through an untrained embedding: what a fresh
    layer sees. Two layers initialised within noise of each other stand in for the
    adjacent trained layers the paper measured; the null is a different draw. The
    overlap falls with the noise (mean over ten draws, printed with its range), and
    reuse is allowed above the floor and REFUSED below it.
    """
    torch.manual_seed(45)
    raw = torch.frombuffer(bytearray(CORPUS.read_bytes()[:SEQ]), dtype=torch.uint8).long()
    x = torch.nn.Embedding(256, WIDTH)(raw).unsqueeze(0)
    layer = DSA(WIDTH, 1, 8, k=32, window=8, d_idx=D_IDX, chunk=256)
    mine = layer.indexer.select(x, 32)

    def perturbed(noise: float, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        other = DSA(WIDTH, 1, 8, k=32, window=8, d_idx=D_IDX, chunk=256)
        other.load_state_dict(layer.state_dict())
        with torch.no_grad():
            for weight in (other.indexer.weight, other.indexer.key.weight):
                weight.add_(noise * torch.randn_like(weight))
        return other.indexer.select(x, 32)

    drawn = {
        noise: [layer.indexer.overlap(mine, perturbed(noise, seed)) for seed in range(45, 55)]
        for noise in (0.0, 1e-4, 1e-2)
    }
    means = {noise: statistics.mean(rates) for noise, rates in drawn.items()}
    print(
        "\nadjacent-layer overlap on 4096 corpus tokens, mean of 10 draws: "
        + ", ".join(
            f"{noise:g}->{means[noise]:.2f} [{min(drawn[noise]):.2f}, {max(drawn[noise]):.2f}]"
            for noise in drawn
        )
        + f" (chance {32 / SEQ:.3f}, floor {MLI_MIN_OVERLAP})"
    )
    assert means[0.0] == 1.0
    assert means[1e-4] >= MLI_MIN_OVERLAP
    assert means[1e-2] < MLI_MIN_OVERLAP
    assert layer.indexer.reuse(mine, k=32, own=perturbed(1e-4, 45)).shape == mine.shape
    refused = min(
        (perturbed(1e-2, seed) for seed in range(45, 55)),
        key=lambda s: layer.indexer.overlap(mine, s),
    )
    assert layer.indexer.overlap(mine, refused) < MLI_MIN_OVERLAP
    with pytest.raises(LoudError, match="overlap"):
        layer.indexer.reuse(mine, k=32, own=refused)
