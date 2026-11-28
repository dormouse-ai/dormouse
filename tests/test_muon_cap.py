"""The 128 MiB Newton–Schulz cap: bounded intermediates, bitwise parity, 20 steps.

The cap is a schedule, not a new algorithm: a stacked shape group is cut along its
matrix axis, and the matrices in one bucket never mix (no cross-matrix GEMM, no
cross-matrix reduction). So every matrix sees the same Gram, the same quintic and
the same ColRow as before. The pre-cap out-of-place chain below is copied from the
shipped `_polar_many` and is the reference both for `torch.equal` and for the
20-step optimizer parity.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from dormouse_core.errors import LoudError

from dormouse.train.muon import (
    NS_CHUNK_BYTES,
    NS_COEFFS,
    NS_DTYPES,
    NS_STEPS,
    Muon,
    _matmul,
    _pieces,
    _polar_chain,
    _polar_many,
    _prescale,
)


def _reference(
    batch: list[torch.Tensor], steps: int = NS_STEPS, *, ns_dtype: str = "fp32"
) -> list[torch.Tensor]:
    """The exact pre-cap chain: one group per matrix, out-of-place foreach, no cap."""
    work_dtype = torch.float16 if ns_dtype == "fp16" else torch.float32
    a, b, c = NS_COEFFS
    prepared = [_prescale(x, work_dtype=work_dtype) for x in batch]
    works = [work for work, _ in prepared]
    for _ in range(steps):
        grams = [_matmul(work, work.transpose(-2, -1)) for work in works]
        squares = [_matmul(gram, gram) for gram in grams]
        bg = list(torch._foreach_mul(grams, b))
        cg = list(torch._foreach_mul(squares, c))
        polys = list(torch._foreach_add(bg, cg))
        projected = [_matmul(poly, work) for poly, work in zip(polys, works, strict=True)]
        scaled = list(torch._foreach_mul(works, a))
        works = list(torch._foreach_add(scaled, projected))
    return [
        (work.transpose(-2, -1).contiguous() if flag else work.contiguous()).to(torch.float32)
        for work, (_, flag) in zip(works, prepared, strict=True)
    ]


def _groups() -> list[torch.Tensor]:
    torch.manual_seed(31)
    return [
        torch.randn(5, 8, 16),
        torch.randn(3, 16, 16),
        torch.randn(2, 16, 8),  # tall: transpose arm
    ]


@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_uncapped_cap_is_the_shipped_chain_bitwise(ns_dtype: str) -> None:
    """`chunk_bytes=None` must be the exact program the cap replaced."""
    batch = _groups()
    got = _polar_many(batch, ns_dtype=ns_dtype, chunk_bytes=None)
    ref = _reference(batch, ns_dtype=ns_dtype)
    assert all(torch.equal(a, b) for a, b in zip(got, ref, strict=True))


@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_in_place_chain_is_bitwise_equal_to_the_pre_cap_chain(ns_dtype: str) -> None:
    """Reusing bg/cg/scaled buffers changes bytes alive, not a single result bit."""
    batch = _groups()
    got = _polar_many(batch, ns_dtype=ns_dtype, chunk_bytes=None)
    ref = _reference(batch, ns_dtype=ns_dtype)
    assert all(torch.equal(a, b) for a, b in zip(got, ref, strict=True))


@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_chunking_is_bitwise_equal_to_full(ns_dtype: str) -> None:
    """Three caps, including one smaller than a row, all give the same tensor."""
    batch = _groups()
    full = _polar_many(batch, ns_dtype=ns_dtype, chunk_bytes=None)
    for cap in (1 << 16, 16 * 4 * 4 * 2, 4):
        got = _polar_many(batch, ns_dtype=ns_dtype, chunk_bytes=cap)
        assert all(torch.equal(a, b) for a, b in zip(got, full, strict=True)), cap


def test_the_128_mib_cap_cuts_the_65_expert_bucket_at_21_matrices() -> None:
    """The refused 390 MiB `projected` becomes four ≤126 MiB pieces."""
    work = torch.zeros(65, 256, 6144)
    cap_elems = NS_CHUNK_BYTES // 4
    pieces = _pieces(work, cap_elems)
    assert len(pieces) == 4
    assert [p.shape[0] for p in pieces] == [21, 21, 21, 2]
    assert max(p.numel() * 4 for p in pieces) <= NS_CHUNK_BYTES
    assert len(_pieces(work, None)) == 1  # parity control


@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_every_chain_gemm_output_respects_the_cap(ns_dtype: str) -> None:
    """Instrument the chain's `bmm`: every `[s,l]` and `[s,s]` buffer obeys the cap.

    The prescale power iteration also makes an `[N,l,1]` vector (1.6 MiB on the
    live 65-bucket), which is not a chain intermediate and is far below the cap.
    """
    cap = 512
    batch = [torch.randn(9, 8, 16), torch.randn(4, 16, 8)]
    sizes: list[int] = []
    original = torch.bmm

    def measured(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = original(x, y)
        sizes.append(out.numel() * out.element_size())
        return out

    real_chain = _polar_chain

    def measuring(works: list[torch.Tensor], a: float, b: float, c: float, steps: int) -> None:
        with patch("dormouse.train.muon.torch.bmm", measured):
            real_chain(works, a, b, c, steps)

    with patch("dormouse.train.muon._polar_chain", measuring):
        _polar_many(batch, ns_dtype=ns_dtype, chunk_bytes=cap)
    assert sizes and max(sizes) <= cap


@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_twenty_optimizer_steps_are_bitwise_equal(ns_dtype: str) -> None:
    """Full step, 20 times, grouped and ungrouped, head split included."""
    torch.set_num_threads(1)

    def build() -> tuple[list[torch.nn.Parameter], Muon]:
        torch.manual_seed(41)
        params = [torch.nn.Parameter(torch.randn(32, 48)) for _ in range(4)]
        params += [torch.nn.Parameter(torch.randn(12, 16)) for _ in range(3)]
        for param in params:
            param.grad = torch.randn_like(param)
        opt = Muon(
            [
                {"params": params[:4], "head_split": None},
                {"params": params[4:], "head_split": (3, 6)},
            ],
            ns_dtype=ns_dtype,
            chunk_bytes=1 << 13,
        )
        return params, opt

    left, capped = build()
    right, full = build()
    full.chunk_bytes = None
    for _ in range(20):
        capped.step()
        full.step()
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    for a, b in zip(left, right, strict=True):
        assert torch.equal(capped.state[a]["momentum"], full.state[b]["momentum"])


@pytest.mark.parametrize("shape", [(64, 832), (832, 64), (256, 6144)])
@pytest.mark.parametrize("ns_dtype", NS_DTYPES)
def test_external_fp64_polar_anchor(shape: tuple[int, int], ns_dtype: str) -> None:
    """cos ≥ 0.9999 against numpy's fp64 SVD polar, on the capped chain."""
    torch.manual_seed(7)
    left, _, right = torch.linalg.svd(torch.randn(*shape), full_matrices=False)
    spectrum = torch.linspace(0.8, 1.2, min(shape))
    x = left @ torch.diag(spectrum) @ right
    itemsize = 2 if ns_dtype == "fp16" else 4
    cap = min(shape) * max(shape) * itemsize + 1  # a group of two splits into two
    got = _polar_many([torch.stack([x, x])], ns_dtype=ns_dtype, chunk_bytes=cap)[0][0]
    ref_np = np.linalg.svd(x.numpy().astype(np.float64), full_matrices=False)
    ref = ref_np[0] @ ref_np[2]
    flat = got.numpy().astype(np.float64).ravel()
    cos = float(flat @ ref.ravel() / (np.linalg.norm(flat) * np.linalg.norm(ref)))
    assert cos >= 0.9999, (shape, ns_dtype, cos)


def test_a_cap_smaller_than_one_element_refuses() -> None:
    with pytest.raises(LoudError, match="chunk cap"):
        _polar_many([torch.randn(4, 4)], chunk_bytes=1)
    with pytest.raises(LoudError, match="chunk cap"):
        _polar_many([torch.randn(4, 4)], ns_dtype="fp16", chunk_bytes=1)
