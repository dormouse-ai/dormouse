"""Build the stack a spec names: check every coeffect, then allocate in one pass.

The order is the contract, and it is the order because each step exists to make the
next one safe:

    registry.pieces(spec)   refuse a conflicting pair, naming it, with nothing allocated
    build per layer         one attention arm per layer, one FFN arm (or one bank) for all
    quantize                ternary weights / int4 activations, if the slots ask
    declarations            the mechanisms the forward must count, for `Counters.check`

WHY THE BANK IS SHARED BY THE LAYERS. `ffn=moe` is one bank of `n_experts`, and every
layer routes into it. That is the arithmetic the stack is designed to: at
`d_model=832, d_ffn=6144, n_experts=64, topk=1, n_shared=1` the bank stores
`65·3·832·6144 ≈ 1.0B` parameters and runs `2·3·832·6144 ≈ 30.7M` per token. One bank
per layer would store twelve times that and run 368M per token, which is not the
"1B stored / 46M active" the model is priced at — so the sharing is stated here, in
one place, rather than discovered from a `param_report` that does not add up.

WHY A WIDTH CANNOT BE QUIET. Every arm is built from `StackSpec`'s widths and every
arm's own constructor validates them (`n_heads·head_dim == d_model`, `3·d·d_ffn`
against the GELU control for a mixture, `seq_len` divisible by CSA's chunk, the
pattern tiling `n_layers`); this module only decides which arm, and passes `write_std`
(the `1/√(2L)` residual-write scale from `blocks.Block`) so a prebuilt arm is drawn
exactly as the one `Block` would have built.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.plan import Plan
from torch import nn

from ..train.optim import MixOptimizer
from ..train.step import StepResult, train_step
from . import quant, registry
from .attn import PROJ_STD, Attention
from .attn_variants.csa import CSA
from .attn_variants.kda import KDA
from .blocks import Block
from .ffn_spec import FFNSpec
from .model import Transformer
from .moe import MoEFFN
from .spec import StackSpec

__concepts__ = ("Assembled", "FlatBank")

#: Bytes one stored parameter costs per step: weight + gradient + two moments.
STATE_BYTES = 16


class FlatBank(nn.Module):
    """The bank a `Block` can hand `[batch, seq, d]` to; the flattening is the adapter.

    `MoEFFN` routes positions and refuses a 3-D input on purpose (see its docstring),
    so the reshape belongs at the call site — here — and not inside the bank.
    """

    def __init__(self, bank: MoEFFN) -> None:
        """Hold the bank and re-publish the two widths a block prints."""
        super().__init__()
        self.bank = bank
        self.d_model = bank.d_model
        self.d_ffn = bank.d_ffn

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, s, d]` → `[b, s, d]`, routed per position."""
        batch, seq, width = x.shape
        out = self.bank(x.reshape(batch * seq, width), counters)
        return cast(torch.Tensor, out.reshape(batch, seq, width))


class Assembled:
    """A built stack, the pieces it is made of, and the bill it is priced at."""

    def __init__(
        self,
        spec: StackSpec,
        model: Transformer,
        pieces: tuple[registry.Piece, ...],
        bank: MoEFFN | None,
    ) -> None:
        """Hold the four things a caller needs: the model, the spec, the pieces, the bank."""
        self.spec = spec
        self.model = model
        self.pieces = pieces
        self.bank = bank

    # ------------------------------------------------------------------ the bill
    def report(self) -> dict[str, int]:
        """Return stored and active parameters, from the tree and the bank's own bill."""
        groups = self.model.param_report()
        stored = groups["total"]
        if self.bank is None:
            return {**groups, "stored": stored, "active": stored}
        bill = self.bank.param_report()
        active = stored - bill["stored"] + bill["active_with_router"]
        return {**groups, "stored": stored, "active": active}

    def mechanisms(self) -> dict[str, str]:
        """Every enabled mechanism mapped to the counter that proves it ran."""
        return {piece.mechanism: piece.counter for piece in self.pieces}

    # ------------------------------------------------------------------ the step
    def step(
        self,
        ids: torch.Tensor,
        *,
        optimizer: MixOptimizer,
        accum: int = 1,
    ) -> tuple[StepResult, Counters]:
        """One optimizer step, with the objective's counter declared where it is read.

        `ntp_loss` is counted after `train_step` and enabled last on purpose: the
        stack's own `Counters.check()` runs inside its forward, so a counter the step
        produces must not be enabled before the forward has had its chance to finish.
        """
        counters = Counters()
        result = train_step(self.model, ids, optimizer=optimizer, accum=accum, counters=counters)
        if not result.skipped:
            counters.count("ntp_loss")
        counters.enable("ntp", counter="ntp_loss", why="the step computes the NTP loss")
        counters.check()
        return result, counters


def assemble(
    spec: StackSpec,
    *,
    device: torch.device | str | None = None,
    smoke: bool = False,
) -> Assembled:
    """Build the stack `spec` names, refusing a contradiction before allocating it."""
    pieces = registry.pieces(spec)
    write_std = PROJ_STD / (2.0 * spec.n_layers) ** 0.5
    bank = _bank(spec, write_std, device, smoke) if spec.is_moe else None
    blocks = [
        Block(
            spec.d_model,
            spec.n_heads,
            spec.head_dim,
            spec.d_ffn,
            n_layers=spec.n_layers,
            rope=spec.pos == "rope",
            max_seq_len=spec.seq_len,
            ffn=FlatBank(bank) if bank is not None else _dense_ffn(spec, write_std, device),
            attn=_attention(spec, kind, write_std, device),
            device=device,
        )
        for kind in spec.pattern
    ]
    model = Transformer(
        vocab=spec.vocab,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        head_dim=spec.head_dim,
        d_ffn=spec.d_ffn,
        max_seq_len=spec.seq_len,
        rope=spec.pos == "rope",
        blocks=blocks,
        declarations=registry.declarations(pieces),
        device=device,
    )
    _quantize(model, spec)
    return Assembled(spec=spec, model=model, pieces=pieces, bank=bank)


def _attention(
    spec: StackSpec, kind: str, write_std: float, device: torch.device | str | None
) -> nn.Module:
    """Build one layer's attention arm; the arm's own constructor validates the widths."""
    if kind == "full":
        return Attention(
            spec.d_model,
            spec.n_heads,
            spec.head_dim,
            rope=True,
            max_seq_len=spec.seq_len,
            write_std=write_std,
            device=device,
        )
    if kind == "kda":
        return KDA(spec.d_model, spec.n_heads, spec.head_dim, write_std=write_std, device=device)
    return CSA(spec.d_model, spec.n_heads, spec.head_dim, write_std=write_std, device=device)


def _dense_ffn(spec: StackSpec, write_std: float, device: torch.device | str | None) -> nn.Module:
    """Build the one dense FFN arm the slot names (`gated` is the SwiGLU control)."""
    arm = "silu" if spec.ffn == "gated" else spec.ffn
    gelu = spec.d_ffn_gelu if arm in ("la", "moa") else None
    return FFNSpec(spec.d_model, spec.d_ffn, act=arm, d_ffn_gelu=gelu).build(
        write_std=write_std, device=device
    )


def _bank(
    spec: StackSpec,
    write_std: float,
    device: torch.device | str | None,
    smoke: bool,
) -> MoEFFN:
    """Build the one bank every layer routes into, at the expert arm the spec names."""
    arm = spec.expert_arm
    return MoEFFN(
        spec.d_model,
        spec.d_ffn,
        n_experts=spec.n_experts,
        topk=spec.topk,
        n_shared=spec.n_shared,
        act=arm,
        d_ffn_gelu=spec.d_ffn_gelu if arm in ("la", "moa") else None,
        write_std=write_std,
        #: `smoke` is the only caller allowed to lower PR #18's 10M-parameter floor,
        #: and it is said out loud in the report rather than quietly in a test.
        min_expert_params=1 if smoke else 10_000_000,
        device=device,
    )


def _quantize(model: Transformer, spec: StackSpec) -> None:
    """Apply the quantization slots, and wrap each block so both counters can move."""
    if spec.weight_qat == "ternary":
        quant.ternarize(model)
    if spec.weight_qat == "none" and spec.act_quant == "none":
        return
    wrapped = [
        quant.QuantBlock(
            block,
            matrices=sum(type(m) is quant.QuantLinear for m in block.modules()),
            act_int4=spec.act_quant == "int4",
        )
        for block in model.blocks
    ]
    model.blocks = nn.ModuleList(wrapped)


def plan_lines(spec: StackSpec) -> list[str]:
    """Everything `dorm plan` prints: the stack, the bill, the step, the mechanisms.

    The model is built on the `meta` device, so the numbers are measured from the
    modules that would be allocated rather than from a formula that can drift — and
    `dorm plan` on a 1B stack costs no memory.
    """
    assembled = assemble(spec, device="meta")
    report = assembled.report()
    steps = max(spec.tokens // (spec.batch * spec.seq_len), 1)
    tokens_per_step = spec.batch * spec.seq_len
    seconds = steps * 6 * report["active"] * tokens_per_step / (spec.throughput_tflops * 1e12)
    bits = 0.25 if spec.weight_qat == "ternary" else 4.0

    experts = (
        f" (experts: {spec.expert_arm}, {spec.n_experts} routed, topk {spec.topk})"
        if spec.is_moe
        else ""
    )
    lines = [
        f"stack: {spec.attn} {spec.attn_pattern or ''}".rstrip()
        + f" (layers: {', '.join(spec.pattern)}), pos={spec.pos}, unit={spec.unit}",
        f"ffn: {spec.ffn}{experts}"
        + f", optimizer={spec.optimizer}, objective={spec.objective}"
        + f", weight_qat={spec.weight_qat}, act_quant={spec.act_quant}",
        f"widths: d_model={spec.d_model} x {spec.n_layers} layers, "
        f"d_ffn={spec.d_ffn}, heads={spec.n_heads}x{spec.head_dim}, seq={spec.seq_len}",
        f"params: stored={report['stored']:,} active={report['active']:,} "
        f"({report['active'] / report['stored']:.1%} of stored), ffn={report['ffn']:,} "
        f"attention={report['attention']:,}",
        f"step: {tokens_per_step:,} tokens/step, {steps:,} steps for {spec.tokens:,} tokens; "
        f"flops/step={6 * report['active'] * tokens_per_step:,}",
        f"bytes/step: state={STATE_BYTES * report['stored']:,} (weights+grad+2 moments), "
        f"checkpoint={int(bits * report['stored']):,}",
    ]
    plan = Plan().add(
        "train",
        seconds,
        note=f"{spec.tokens:,} tokens at {spec.throughput_tflops} TFLOP/s sustained",
    )
    if spec.unit == "bpe":
        plan.warn("unit=bpe: the tokenizer that produces 32768-way ids is not in this repo")
    if spec.is_moe:
        plan.warn(
            f"the bank is shared by {spec.n_layers} layers: stored is "
            f"{report['stored']:,}, not {spec.n_layers} banks"
        )
    lines.extend(plan.render())
    lines.append(
        "mechanisms: "
        + ", ".join(dict.fromkeys(f"{p.mechanism}→{p.counter}" for p in assembled.pieces))
    )
    return lines
