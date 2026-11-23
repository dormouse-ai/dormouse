"""ReWY backward for chunkwise Kimi Delta Attention (arXiv 2611.01940, MIT).

The forward recurrence is unchanged (Kimi Linear 2510.26692 eq. 1):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

Inside a chunk, with gamma_i = prod_{t<=i} alpha_t, the WY correction R solves

    M R = Z - E S,   M = I + strict_lower(E Y^T),
    E = beta gamma K,  Y = K / gamma,  Z = beta V,  Qg = gamma Q.

The conventional WY backward differentiates the forward auxiliaries W = A E and
A = M^{-1} separately.  It forms Wbar = -Rbar S^T, Abar = Rbar Z^T + Wbar E^T,
then pushes Abar through -A^T Abar A^T, plus Zbar = A^T Rbar and
Ebar_dir = A^T Wbar: seven local matrix products.  Appendix B.1 of the paper
shows A^T Abar A^T = (A^T Rbar) R^T, so all seven collapse to three:

    P        = A^T Rbar                (triangular solve, not a dense product)
    D        = -strict_lower(P R^T)
    Ebar_dir = -P S^T

Everything downstream (the output/state composition, the factor chain rules
and the reverse gate scan) is shared with the conventional graph.  ``backward_local``
exposes both pullbacks with the same interface so the matmul count and latency
can be compared on the same tensors; ``KDAReWY`` always uses the 3-product path.

On CUDA the three local products run in ONE Triton program per chunk: the unit
upper-triangular solve P = M^{-T} Rbar stays in registers, and the two
contractions D = -strict_lower(P R^T), Ebar_dir = -P S^T are tl.dot'd from the
same registers.  No global P intermediate, one launch instead of solve + two
matmuls + mask.  fp64 and CPU keep the eager path (tl.dot has no fp64 operands),
which is the same path the external complex-step anchor exercises.

Clean-room from the paper equations; no ReWY source code is vendored.
"""
# ruff: noqa: N803, B905, D102, E741  # clean-room port; paper's matrix symbols

from __future__ import annotations

import math
from typing import Any, cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.errors import LoudError

from . import mfu40 as mfu40_kernels

__all__ = [
    "BWD_COUNTER",
    "DEFAULT_CHUNK",
    "EXPANDED_PRODUCTS",
    "FWD_COUNTER",
    "MAX_SMEM",
    "REWY_PRODUCTS",
    "KDAReWY",
    "backward_local",
    "fused_available",
    "kda_rewy",
    "matmul_count",
    "set_fused",
    "smem_bytes",
]

FWD_COUNTER = "kda_fwd"
BWD_COUNTER = "kda_bwd"
DEFAULT_CHUNK = 64
MAX_SMEM = 101376  # bytes, sm_120
EXPANDED_PRODUCTS = 7
REWY_PRODUCTS = 3

_KERNELS: dict[str, Any] = {}
_MODE = "auto"  # "auto" | "on" | "off"; "off" is the measured eager baseline

LocalPullback = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# (Zbar, D = Lbar, Ebar_dir) with shapes [b,h,n,d], [b,h,n,n], [b,h,n,d].
Cache = dict[str, Any]


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
) -> None:
    """Validate the same kernel boundary as the reference KDA op."""
    if q.ndim != 4 or beta.ndim != 3:
        raise LoudError(
            "ReWY expects q/k/v/alpha [batch, heads, seq, dim] and beta [batch, heads, seq]",
            "transpose the projected tensors into the kernel layout before the call",
        )
    if (k.shape, v.shape, alpha.shape) != (q.shape,) * 3 or beta.shape != q.shape[:3]:
        raise LoudError(
            "ReWY inputs disagree on shape",
            f"q{q.shape} k{k.shape} v{v.shape} alpha{alpha.shape} beta{beta.shape}",
        )
    if any(t.dtype != q.dtype or t.device != q.device for t in (k, v, alpha, beta)):
        raise LoudError(
            "ReWY inputs disagree on dtype or device",
            "the triangular solve and the contractions need one dtype and one device",
        )
    if chunk < 1 or q.shape[-2] % chunk:
        raise LoudError(
            f"chunk={chunk} does not tile seq={q.shape[-2]}",
            f"pad the window or choose a chunk dividing {q.shape[-2]}",
        )


def _masks(chunk: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive (output read) and strict (correction system) causal triangles."""
    ones = torch.ones(chunk, chunk, dtype=torch.bool, device=device)
    inclusive = torch.tril(ones)
    return inclusive, torch.tril(inclusive, -1)


def _tri_solve(matrix: torch.Tensor, rhs: torch.Tensor, *, upper: bool) -> torch.Tensor:
    """Unit-triangular solve in a promoted work dtype, returned in rhs' dtype."""
    work = torch.promote_types(matrix.dtype, torch.float32)
    solved = torch.linalg.solve_triangular(
        matrix.to(work), rhs.to(work), upper=upper, unitriangular=True
    )
    return cast(torch.Tensor, solved.to(rhs.dtype))


def set_fused(mode: str) -> None:
    """Select the local schedule: ``auto`` (CUDA fp32/bf16), ``on``, ``off``."""
    if mode not in ("auto", "on", "off"):
        raise LoudError(f"unknown ReWY fused mode {mode!r}", "expected 'auto', 'on' or 'off'")
    global _MODE
    _MODE = mode


def fused_available() -> bool:
    """Whether the fused local schedule can run on this host."""
    return bool(torch.cuda.is_available() and _build() is not None)


def _build() -> dict[str, Any] | None:  # pragma: no cover - compiled on device
    """Build the one local-pullback kernel once; ``None`` without Triton."""
    if _KERNELS:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit  # type: ignore[untyped-decorator]
    def local_kernel(  # type: ignore[no-untyped-def]
        M,
        RB,
        R,
        S,
        PO,
        DO,
        EO,
        index,
        n: tl.constexpr,
        d: tl.constexpr,
        H: tl.constexpr,
        sm_b: tl.constexpr,
        sm_h: tl.constexpr,
        sr_b: tl.constexpr,
        sr_h: tl.constexpr,
        sc_b: tl.constexpr,
        sc_h: tl.constexpr,
        ss_b: tl.constexpr,
        ss_h: tl.constexpr,
        sp_b: tl.constexpr,
        sp_h: tl.constexpr,
        sd_b: tl.constexpr,
        sd_h: tl.constexpr,
        se_b: tl.constexpr,
        se_h: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        FP32: tl.constexpr,
    ):
        """P = M^{-T} Rbar, D = -strict_lower(P R^T), Ebar = -P S^T in one CTA.

        Backward substitution over the unit lower-triangular M keeps the whole
        P tile in registers; rows > i are solved, rows <= i are still zero, so
        one tl.sum per step gives the substitution dot without a triangular mask.
        The two contractions then read that same tile instead of a global P.
        """
        pid = tl.program_id(0)
        head = pid % H
        batch = pid // H
        rows = tl.arange(0, BN)
        cols = tl.arange(0, BD)
        rmask = rows < n
        cmask = cols < d
        base_m = batch * sm_b + head * sm_h + index * n * n
        base_r = batch * sr_b + head * sr_h
        base_c = batch * sc_b + head * sc_h + index * n * d
        base_s = batch * ss_b + head * ss_h
        base_p = batch * sp_b + head * sp_h
        base_d = batch * sd_b + head * sd_h
        base_e = batch * se_b + head * se_h

        p = tl.zeros((BN, BD), dtype=tl.float32)
        for i in range(n - 1, -1, -1):
            w = tl.load(M + base_m + i + rows * n, mask=rmask, other=0.0).to(tl.float32)
            w = tl.where(rows == i, w + 1.0, w)  # M is unit lower triangular
            contribution = tl.sum(w[:, None] * p, 0)
            rhs = tl.load(RB + base_r + i * d + cols, mask=cmask, other=0.0).to(tl.float32)
            p = tl.where(rows[:, None] == i, (rhs - contribution)[None, :], p)

        dtype = PO.dtype.element_ty
        ps = p.to(dtype)
        tl.store(
            PO + base_p + rows[:, None] * d + cols[None, :],
            ps,
            mask=rmask[:, None] & cmask[None, :],
        )

        rv = tl.load(
            R + base_c + rows[:, None] * d + cols[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
        rt = tl.trans(rv)
        if FP32:
            correction = tl.dot(ps, rt, out_dtype=tl.float32, input_precision="ieee")
        else:
            correction = tl.dot(ps, rt, out_dtype=tl.float32)
        strict = (rows[:, None] > rows[None, :]) & rmask[:, None] & rmask[None, :]
        tl.store(
            DO + base_d + rows[:, None] * n + rows[None, :],
            tl.where(strict, -correction, 0.0).to(DO.dtype.element_ty),
            mask=rmask[:, None] & rmask[None, :],
        )

        st = tl.load(
            S + base_s + cols[None, :] * d + cols[:, None],
            mask=cmask[:, None] & cmask[None, :],
            other=0.0,
        )
        if FP32:
            erase = tl.dot(ps, st, out_dtype=tl.float32, input_precision="ieee")
        else:
            erase = tl.dot(ps, st, out_dtype=tl.float32)
        tl.store(
            EO + base_e + rows[:, None] * d + cols[None, :],
            (-erase).to(EO.dtype.element_ty),
            mask=rmask[:, None] & cmask[None, :],
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def intra_kernel(  # type: ignore[no-untyped-def]
        O,
        R,
        DM,
        Y,
        QG,
        EH,
        BETA,
        UQ,
        UE,
        UY,
        n: tl.constexpr,
        d: tl.constexpr,
        H: tl.constexpr,
        so_b: tl.constexpr,
        so_h: tl.constexpr,
        sg_b: tl.constexpr,
        sg_h: tl.constexpr,
        sk_b: tl.constexpr,
        sk_h: tl.constexpr,
        sy_b: tl.constexpr,
        sy_h: tl.constexpr,
        sq_b: tl.constexpr,
        sq_h: tl.constexpr,
        se_b: tl.constexpr,
        se_h: tl.constexpr,
        sb_b: tl.constexpr,
        sb_h: tl.constexpr,
        ou_b: tl.constexpr,
        ou_h: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        BK: tl.constexpr,
        FP32: tl.constexpr,
    ):
        """Matrix-intra dots: U_q = H_q Y, U_e = D Y, U_y = H_q^T Q_g + D^T E.

        Only the four dot operands stay live; endpoint scaling and the
        feature-local chain rules remain in the caller.  Products accumulate
        in fp32 at the endpoint scale, so the fp64 oracle stays inside 1e-6.
        """
        pid = tl.program_id(0)
        head = pid % H
        batch = pid // H
        rows = tl.arange(0, BN)
        cols = tl.arange(0, BD)
        rm = rows < n
        cm = cols < d
        mask2 = rm[:, None] & cm[None, :]
        bo = batch * so_b + head * so_h
        bg = batch * sg_b + head * sg_h
        bk = batch * sk_b + head * sk_h
        by = batch * sy_b + head * sy_h
        bq = batch * sq_b + head * sq_h
        be = batch * se_b + head * se_h
        bb = batch * sb_b + head * sb_h

        # K-tiled H_q keeps the fp32 operands at BK slices: a full [64, 128]
        # pair plus the other live tiles would need 147 KiB on sm_120.
        hq = tl.zeros((BN, BN), dtype=tl.float32)
        for start in range(0, d, BK):
            ks = start + tl.arange(0, BK)
            km = ks < d
            o = tl.load(
                O + bo + rows[:, None] * d + ks[None, :],
                mask=rm[:, None] & km[None, :],
                other=0.0,
            )
            r = tl.load(
                R + bg + rows[:, None] * d + ks[None, :],
                mask=rm[:, None] & km[None, :],
                other=0.0,
            )
            if FP32:
                hq += tl.dot(o, tl.trans(r), out_dtype=tl.float32, input_precision="ieee")
            else:
                hq += tl.dot(o, tl.trans(r), out_dtype=tl.float32)
        hq = tl.where((rows[:, None] >= rows[None, :]) & rm[:, None] & rm[None, :], hq, 0.0)
        y = tl.load(Y + by + rows[:, None] * d + cols[None, :], mask=mask2, other=0.0)
        qg = tl.load(QG + bq + rows[:, None] * d + cols[None, :], mask=mask2, other=0.0)
        eh = tl.load(EH + be + rows[:, None] * d + cols[None, :], mask=mask2, other=0.0)
        beta = tl.load(BETA + bb + rows, mask=rm, other=0.0)
        dm = tl.load(
            DM + bk + rows[:, None] * n + rows[None, :],
            mask=rm[:, None] & rm[None, :],
            other=0.0,
        )

        if FP32:
            uq = tl.dot(hq, y, out_dtype=tl.float32, input_precision="ieee")
        else:
            uq = tl.dot(hq, y, out_dtype=tl.float32)
        ou = batch * ou_b + head * ou_h
        tl.store(UQ + ou + rows[:, None] * d + cols[None, :], uq, mask=mask2)
        if FP32:
            ue = tl.dot(dm, y, out_dtype=tl.float32, input_precision="ieee")
        else:
            ue = tl.dot(dm, y, out_dtype=tl.float32)
        tl.store(UE + ou + rows[:, None] * d + cols[None, :], ue, mask=mask2)
        if FP32:
            uy = tl.dot(tl.trans(hq), qg, out_dtype=tl.float32, input_precision="ieee")
            uy += tl.dot(
                tl.trans(dm), beta[:, None] * eh, out_dtype=tl.float32, input_precision="ieee"
            )
        else:
            uy = tl.dot(tl.trans(hq), qg, out_dtype=tl.float32)
            uy += tl.dot(tl.trans(dm), beta[:, None] * eh, out_dtype=tl.float32)
        tl.store(UY + ou + rows[:, None] * d + cols[None, :], uy, mask=mask2)

    _KERNELS.update({"local": local_kernel, "intra": intra_kernel, "tl": tl})
    return _KERNELS


def _use_fused(t: torch.Tensor) -> bool:
    """fp64 is the anchor's dtype and tl.dot has no fp64 operand: eager there."""
    built = _build()
    if _MODE == "off" or t.dtype == torch.float64 or not t.is_cuda:
        if _MODE == "on" and t.is_cuda and t.dtype != torch.float64 and built is None:
            raise LoudError(
                "ReWY fused schedule requested but Triton is unavailable",
                "install triton>=3.9 or call set_fused('off')",
            )
        return False
    if built is None:
        if _MODE == "on":
            raise LoudError(
                "ReWY fused schedule requested but Triton is unavailable",
                "install triton>=3.9 or call set_fused('off')",
            )
        return False
    return True


def _fused_pullback(cache: Cache, index: int, correction_bar: torch.Tensor) -> LocalPullback:
    """Launch the three local products as one Triton program per batch/head."""
    built = _build()
    assert built is not None
    system, correction = cache["systems"], cache["corrections"]
    state = cache["states_in"][index]
    batch, heads, _count, n, _ = system.shape
    d = correction.shape[-1]
    # Contiguous planes: empty_like on a strided chunk view would reserve the
    # whole [b, h, C, ...] footprint for each per-chunk temporary.
    p_out = torch.empty(
        correction[..., index, :, :].shape, dtype=correction.dtype, device=correction.device
    )
    d_out = torch.empty(system[..., index, :, :].shape, dtype=system.dtype, device=system.device)
    e_out = torch.empty(
        correction[..., index, :, :].shape, dtype=correction.dtype, device=correction.device
    )

    def block(size: int) -> int:
        return max(16, 1 << (size - 1).bit_length())

    if mfu40_kernels.enabled("KO6"):
        mfu40_kernels.count("ko6_rewy_fused")
    compiled = built["local"][(batch * heads,)](
        system,
        correction_bar,
        correction,
        state,
        p_out,
        d_out,
        e_out,
        index,
        n,
        d,
        heads,
        *system.stride()[:2],
        *correction_bar.stride()[:2],
        *correction.stride()[:2],
        *state.stride()[:2],
        *p_out.stride()[:2],
        *d_out.stride()[:2],
        *e_out.stride()[:2],
        BN=block(n),
        BD=block(d),
        FP32=(correction.dtype == torch.float32),
        num_warps=8,
        num_stages=1,
    )
    if compiled is not None:
        built["smem"] = int(compiled.metadata.shared)
        built["spills"] = int(getattr(compiled, "n_spills", 0))
    return p_out, d_out, e_out


def _fused_intra(
    cache: Cache, index: int, coefficient: torch.Tensor, o_bar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the three matrix-intra products into one Triton program per head."""
    built = _build()
    assert built is not None
    batch, heads, _count, n, width = cache["corrections"].shape

    def block(size: int) -> int:
        return max(16, 1 << (size - 1).bit_length())

    uq = torch.empty(o_bar.shape, dtype=o_bar.dtype, device=o_bar.device)
    ue = torch.empty(o_bar.shape, dtype=o_bar.dtype, device=o_bar.device)
    uy = torch.empty(o_bar.shape, dtype=o_bar.dtype, device=o_bar.device)
    inputs = (
        o_bar,
        cache["corrections"][..., index, :, :],
        coefficient,
        cache["divided"][..., index, :, :],
        cache["as_query"][..., index, :, :],
        cache["keyed"][..., index, :, :],
        cache["gates"][..., index, :],
    )
    strides: list[int] = []
    for tensor in inputs:
        strides.extend(tensor.stride()[:2])
    strides.extend(uq.stride()[:2])
    if mfu40_kernels.enabled("KO6"):
        mfu40_kernels.count("ko6_rewy_fused")
    compiled = built["intra"][(batch * heads,)](
        *inputs,
        uq,
        ue,
        uy,
        n,
        width,
        heads,
        *strides,
        BN=block(n),
        BD=block(width),
        BK=32,
        FP32=(o_bar.dtype == torch.float32),
        num_warps=8,
        num_stages=1,
    )
    if compiled is not None:
        built["smem"] = max(int(built.get("smem", 0)), int(compiled.metadata.shared))
        built["spills"] = max(int(built.get("spills", 0)), int(getattr(compiled, "n_spills", 0)))
    return uq, ue, uy


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, Cache]:
    """Batched chunk solve.  Identical arithmetic to the frozen KDA forward."""
    batch, heads, seq, width = q.shape
    count = seq // chunk

    def split(tile: torch.Tensor) -> torch.Tensor:
        return tile.reshape(batch, heads, count, chunk, width)

    keys, values, gates = split(k), split(v), beta.reshape(batch, heads, count, chunk)
    inclusive, strict = _masks(chunk, q.device)
    gamma = torch.cumprod(split(alpha), dim=-2)
    #: No value floor: below sqrt(tiny) the reciprocal and its 1/a^2 gradient
    #: overflow, so those Gamma terms are masked to zero instead.
    alive = gamma >= math.sqrt(torch.finfo(gamma.dtype).tiny)
    safe = torch.where(alive, gamma, torch.ones_like(gamma))
    divided = keys * safe.reciprocal() * alive  # Y = K / gamma
    keyed = gamma * keys  # Ehat = gamma K
    as_query = gamma * split(q)  # Qg = gamma Q
    divided_t = divided.transpose(-1, -2)
    gram = (keyed @ divided_t).masked_fill(~strict, 0.0)
    qk = (as_query @ divided_t).masked_fill(~inclusive, 0.0)
    carried = gamma[..., -1, :].unsqueeze(-2) * divided  # F = Y Lambda
    last = gamma[..., -1, :]  # Lambda diagonal

    state = q.new_zeros(batch, heads, width, width)
    out: list[torch.Tensor] = []
    # Batched cache planes: the fused kernel addresses a chunk by index instead
    # of receiving five strided per-chunk views (no stacking in the hot path).
    states_in = q.new_empty(count, batch, heads, width, width)
    corrections = q.new_empty(batch, heads, count, chunk, width)
    systems = q.new_empty(batch, heads, count, chunk, chunk)
    for index in range(count):
        states_in[index] = state
        step = gates[..., index, :, None]
        system = step * gram[..., index, :, :]
        rhs = step * (values[..., index, :, :] - keyed[..., index, :, :] @ state)
        correction = _tri_solve(system, rhs, upper=False)
        out.append(as_query[..., index, :, :] @ state + qk[..., index, :, :] @ correction)
        state = last[..., index, :].unsqueeze(-1) * state + torch.einsum(
            "bhtd,bhtv->bhdv", carried[..., index, :, :], correction
        )
        corrections[..., index, :, :] = correction
        systems[..., index, :, :] = system

    cache: Cache = {
        "gamma": gamma,
        "alive": alive,
        "divided": divided,
        "keyed": keyed,
        "as_query": as_query,
        "qk": qk,
        "carried": carried,
        "last": last,
        "states_in": states_in,
        "corrections": corrections,
        "systems": systems,
        "inclusive": inclusive,
        "strict": strict,
        "keys": keys,
        "values": values,
        "gates": gates,
        "queries": split(q),
    }
    return torch.cat(out, dim=-2), state, cache


def backward_local(
    cache: Cache,
    index: int,
    correction_bar: torch.Tensor,
    *,
    mode: str = "rewy",
) -> LocalPullback:
    """One chunk's local WY pullback: Zbar, Lbar and the direct erase gradient.

    ``mode="rewy"`` uses the 3-product residual graph (P, P R^T, P S^T);
    ``mode="expanded"`` materialises Wbar, Abar and the inverse adjoint for the
    conventional 7-product graph.  Both return the same three tensors; the
    benchmark calls both on identical tensors to compare count and latency.
    """
    if mode not in ("rewy", "expanded"):
        raise LoudError(f"unknown ReWY pullback mode {mode!r}", "expected 'rewy' or 'expanded'")

    state = cache["states_in"][index]
    correction = cache["corrections"][..., index, :, :]
    system = cache["systems"][..., index, :, :]
    keyed = cache["keyed"][..., index, :, :]
    value = cache["values"][..., index, :, :]
    gate = cache["gates"][..., index, :]
    strict = cache["strict"]

    if mode == "rewy":
        if _use_fused(system):
            sensitivity, coefficient_bar, erase_dir_bar = _fused_pullback(
                cache, index, correction_bar
            )
        else:
            # P = A^T Rbar = M^{-T} Rbar; a triangular solve, then two products.
            identity = torch.eye(system.shape[-1], dtype=system.dtype, device=system.device)
            matrix = identity + system
            sensitivity = _tri_solve(matrix.transpose(-1, -2), correction_bar, upper=True)
            coefficient_bar = -(sensitivity @ correction.transpose(-1, -2)).masked_fill(
                ~strict, 0.0
            )
            erase_dir_bar = -sensitivity @ state.transpose(-1, -2)
        value_bar = sensitivity
    else:
        # Expanded graph: seven products, kept for a like-for-like benchmark.
        identity = torch.eye(system.shape[-1], dtype=system.dtype, device=system.device)
        matrix = identity + system
        inverse = _tri_solve(matrix, identity, upper=False)
        z = gate.unsqueeze(-1) * value
        e = gate.unsqueeze(-1) * keyed
        w_bar = -correction_bar @ state.transpose(-1, -2)
        inverse_bar = correction_bar @ z.transpose(-1, -2) + w_bar @ e.transpose(-1, -2)
        inverse_pull = -(inverse.transpose(-1, -2) @ inverse_bar) @ inverse.transpose(-1, -2)
        sensitivity = inverse.transpose(-1, -2) @ correction_bar
        coefficient_bar = inverse_pull.masked_fill(~strict, 0.0)
        erase_dir_bar = inverse.transpose(-1, -2) @ w_bar
        value_bar = sensitivity

    return value_bar, coefficient_bar, erase_dir_bar


def _backward(
    cache: Cache,
    grad_out: torch.Tensor | None,
    grad_state: torch.Tensor | None,
    inputs: tuple[torch.Tensor, ...],
    *,
    mode: str = "rewy",
) -> tuple[torch.Tensor, ...]:
    """Complete reverse pass: local pullback, composition, chain rules, gate scan."""
    q, k, v, alpha, beta = inputs
    batch, heads, seq, width = q.shape
    count = cache["states_in"].shape[0]
    chunk = seq // count
    if grad_out is None:
        grad_out = torch.zeros_like(q)
    if grad_state is None:
        grad_state = q.new_zeros(batch, heads, width, width)

    gamma = cache["gamma"]
    alive = cache["alive"]
    divided = cache["divided"]
    keyed = cache["keyed"]
    as_query = cache["as_query"]
    inclusive = cache["inclusive"]
    state_bar_plus = grad_state

    chunked = (
        q.reshape(batch, heads, count, chunk, width),
        k.reshape(batch, heads, count, chunk, width),
        v.reshape(batch, heads, count, chunk, width),
        alpha.reshape(batch, heads, count, chunk, width),
        beta.reshape(batch, heads, count, chunk),
    )
    grads = [torch.zeros_like(t) for t in chunked]
    g_out = grad_out.reshape(batch, heads, count, chunk, width)

    for index in reversed(range(count)):
        state = cache["states_in"][index]
        correction = cache["corrections"][..., index, :, :]
        system = cache["systems"][..., index, :, :]
        qg = as_query[..., index, :, :]
        y = divided[..., index, :, :]
        ehat = keyed[..., index, :, :]
        g = gamma[..., index, :, :]
        safe = torch.where(alive[..., index, :, :], g, torch.ones_like(g))
        gamma_inv = safe.reciprocal() * alive[..., index, :, :]
        last = cache["last"][..., index, :]
        gate = cache["gates"][..., index, :]
        value = cache["values"][..., index, :, :]
        o_bar = g_out[..., index, :, :]

        # Rbar = Cq^T Obar + F Sbar_plus; F = Y Lambda = carried.
        cq_t_o = cache["qk"][..., index, :, :].transpose(-1, -2) @ o_bar
        f_state_bar = torch.einsum(
            "bhtd,bhdv->bhtv", cache["carried"][..., index, :, :], state_bar_plus
        )
        correction_bar = cq_t_o + f_state_bar

        # The 3-product residual pullback, interleaved with the reverse scan
        # because the scan's state term needs P from this very chunk.
        p, d, erase_dir_bar = backward_local(cache, index, correction_bar, mode=mode)

        # Output / state composition (Appendix B.2).
        f_bar = correction @ state_bar_plus.transpose(-1, -2)  # Fbar = R (Sbar+)^T
        state_bar = (
            qg.transpose(-1, -2) @ o_bar
            + last.unsqueeze(-1) * state_bar_plus
            - ehat.transpose(-1, -2) @ (gate.unsqueeze(-1) * p)
        )
        if mode == "rewy" and _use_fused(system):
            uq, ue, uy = _fused_intra(cache, index, d, o_bar)
        else:
            hq = (o_bar @ correction.transpose(-1, -2)).masked_fill(~inclusive, 0.0)
            uq = hq @ y
            ue = d @ y
            uy = hq.transpose(-1, -2) @ qg + d.transpose(-1, -2) @ (gate.unsqueeze(-1) * ehat)
        qg_bar = o_bar @ state.transpose(-1, -2) + uq
        e = gate.unsqueeze(-1) * ehat
        e_bar = erase_dir_bar + ue
        y_bar = uy + f_bar * last.unsqueeze(-2)
        grads[0][..., index, :, :] = g * qg_bar
        grads[1][..., index, :, :] = g * gate.unsqueeze(-1) * e_bar + gamma_inv * y_bar
        grads[2][..., index, :, :] = gate.unsqueeze(-1) * p
        grads[4][..., index, :] = (ehat * e_bar).sum(-1) + (value * p).sum(-1)
        gate_bar = qg * qg_bar + e * e_bar - y * y_bar
        endpoint = last * (
            (state * state_bar_plus).sum(-1) + torch.einsum("bhtd,bhtd->bhd", y, f_bar)
        )
        gate_bar[..., -1, :] += endpoint
        cumulative = gate_bar.flip(-2).cumsum(-2).flip(-2)
        grads[3][..., index, :, :] = cumulative / chunked[3][..., index, :, :]

        state_bar_plus = state_bar

    return tuple(g.reshape(t.shape) for g, t in zip(grads, inputs))


class KDAReWY(torch.autograd.Function):
    """The custom op: forward counts, 3-product backward, one grad per input."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        chunk: int,
        counters: Counters | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.chunk = chunk
        ctx.counters = counters
        ctx.tokens = q.shape[0] * q.shape[-2]
        out, state, cache = _forward(q, k, v, alpha, beta, chunk)
        ctx.cache = cache
        ctx.save_for_backward(q, k, v, alpha, beta)
        if counters is not None:
            counters.count(FWD_COUNTER, ctx.tokens)
        return out, state

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor | None, grad_state: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if ctx.counters is not None:
            ctx.counters.count(BWD_COUNTER, ctx.tokens)
        wanted = tuple(ctx.needs_input_grad[:5])
        if not any(wanted):
            return (None,) * 7
        result = _backward(ctx.cache, grad_out, grad_state, ctx.saved_tensors, mode="rewy")
        grads: list[torch.Tensor | None] = [
            grad if need else None for grad, need in zip(result, wanted)
        ]
        return (*grads, None, None)


def matmul_count(mode: str = "rewy") -> int:
    """Dense local WY matrix products: 7 for the expanded graph, 3 for ReWY."""
    return {"rewy": REWY_PRODUCTS, "expanded": EXPANDED_PRODUCTS}[mode]


def smem_bytes() -> int:
    """Compiled tile's shared memory; 0 on CPU/eager or before the first launch."""
    built = _build()
    return int(built.get("smem", 0)) if built is not None else 0


def kda_rewy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk: int = DEFAULT_CHUNK,
    counters: Counters | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`q/k/v/alpha [b,h,s,d]`, `beta [b,h,s]` → `(write, final state [b,h,d,d])`."""
    _check(q, k, v, alpha, beta, chunk)
    return cast(
        tuple[torch.Tensor, torch.Tensor], KDAReWY.apply(q, k, v, alpha, beta, chunk, counters)
    )
