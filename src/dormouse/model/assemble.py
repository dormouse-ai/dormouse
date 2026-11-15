"""Build the stack a spec names: check every coeffect, then allocate in one pass.

The order is the contract, and it is the order because each step exists to make the
next one safe:

    registry.pieces(spec)   refuse a conflicting pair, naming it, with nothing allocated
    build per layer         one attention arm per layer, one bank for every FFN slot
    quantize                ternary weights / int4 activations, if the slots ask
    declarations            the mechanisms the forward must count, for `Counters.check`

WHY THE BANK IS SHARED BY THE LAYERS. `ffn=moe` is one bank of `n_experts`, and every
layer routes into it. That is the arithmetic the stack is designed to: at
`d_model=832, d_ffn=6144, n_experts=64, topk=2, n_shared=1` the bank stores
`65·3·832·6144 ≈ 1.0B` parameters and runs `3·3·832·6144 ≈ 46.1M` per token. One bank
per layer would store twelve times that and run 553M per token, which is not the
"1B stored / 46M active" the model is priced at — so the sharing is stated here, in
one place, rather than discovered from a `param_report` that does not add up.

WHY A WIDTH CANNOT BE QUIET. Every arm is built from `StackSpec`'s widths and every
arm's own constructor validates them (`n_heads % n_kv_heads`, `3·d·d_ffn` against the
GELU control for a mixture, `seq_len` divisible by KDA's chunk, the period tiling
`n_layers`); this module only decides which arm, and passes `write_std` (the
`1/√(2L)` residual-write scale from `blocks.Block`) so a prebuilt arm is drawn
exactly as the one `Block` would have built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.plan import Phase, Plan
from torch import nn

from ..train.optim import MixOptimizer
from ..train.step import StepResult, train_step
from . import quant, registry
from .attn_variants.dsa import DSA
from .attn_variants.kda import KDA
from .blocks import PROJ_STD, Block
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
        """Return stored and active parameters, from the tree and the bank's own bill.

        Two active numbers, because they answer two questions. `active_ffn` is the
        bank's own active set (`topk + shared` experts): the number the MoE design is
        quoted at. `active` is every parameter a token touches — that set plus the
        attention layers, the norms and the token table — and it is what the step's
        FLOPs are `6·` of.
        """
        groups = self.model.param_report()
        stored = groups["total"]
        if self.bank is None:
            return {**groups, "stored": stored, "active": stored, "active_ffn": 0}
        bill = self.bank.param_report()
        return {
            **groups,
            "stored": stored,
            "active": stored - bill["stored"] + bill["active_with_router"],
            "active_ffn": bill["active_with_router"],
        }

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
    on_layer: Callable[[int, int, str], None] | None = None,
) -> Assembled:
    """Build the stack `spec` names, refusing a contradiction before allocating it.

    `on_layer(done, total, kind)` is called after each block is built so a caller can
    show a step counter while a 1B stack is priced; it is called after the block, so a
    dropped counter can never look like a built layer.
    """
    pieces = registry.pieces(spec)
    write_std = PROJ_STD / (2.0 * spec.n_layers) ** 0.5
    bank = _bank(spec, write_std, device, smoke)
    blocks: list[Block] = []
    for index, kind in enumerate(spec.pattern, start=1):
        blocks.append(
            Block(
                spec.d_model,
                spec.d_ffn,
                n_layers=spec.n_layers,
                ffn=FlatBank(bank),
                attn=_attention(spec, kind, write_std, device),
                device=device,
            )
        )
        if on_layer is not None:
            on_layer(index, len(spec.pattern), kind)
    model = Transformer(
        vocab=spec.vocab,
        d_model=spec.d_model,
        d_ffn=spec.d_ffn,
        max_seq_len=spec.seq_len,
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
    if kind == "kda":
        return KDA(
            spec.d_model,
            spec.n_heads,
            spec.head_dim,
            n_kv_heads=spec.n_kv_heads,
            c_lat=spec.c_lat,
            write_std=write_std,
            device=device,
        )
    return DSA(
        spec.d_model,
        spec.n_heads,
        spec.head_dim,
        n_kv_heads=spec.n_kv_heads,
        c_lat=spec.c_lat,
        k=spec.attn_k,
        window=spec.attn_window,
        write_std=write_std,
        device=device,
    )


def _bank(
    spec: StackSpec,
    write_std: float,
    device: torch.device | str | None,
    smoke: bool,
) -> MoEFFN:
    """Build the one bank every layer routes into, at the expert arm the spec names."""
    return MoEFFN(
        spec.d_model,
        spec.d_ffn,
        n_experts=spec.n_experts,
        topk=spec.topk,
        n_shared=spec.n_shared,
        act=spec.expert_arm,
        d_ffn_gelu=spec.d_ffn_gelu,
        expert_tsct_rank=spec.expert_tsct_rank,
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


def plan_data(
    spec: StackSpec,
    on_layer: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Everything `dorm plan` knows, as data: the stack, the bill, the step, the mechanisms.

    The model is built on the `meta` device, so the numbers are measured from the
    modules that would be allocated rather than from a formula that can drift — and
    `dorm plan` on a 1B stack costs no memory. This returns numbers, never prose: the
    CLI renders it for a terminal and `--json` prints it unchanged, so the two views
    cannot disagree.
    """
    assembled = assemble(spec, device="meta", on_layer=on_layer)
    report = assembled.report()
    tokens_per_step = spec.batch * spec.seq_len
    steps = max(spec.tokens // tokens_per_step, 1)
    seconds = steps * 6 * report["active"] * tokens_per_step / (spec.throughput_tflops * 1e12)
    bits = 0.25 if spec.weight_qat == "ternary" else 4.0
    attention_share = report["attention"] / report["active"] if report["active"] else 0.0
    plan = Plan().add(
        "train",
        seconds,
        note=f"{spec.tokens:,} tokens at {spec.throughput_tflops} TFLOP/s sustained",
    )
    if spec.unit == "bpe":
        plan.warn("unit=bpe: the tokenizer that produces 32768-way ids is not in this repo")
    plan.warn(
        f"the bank is shared by {spec.n_layers} layers: stored is "
        f"{report['stored']:,}, not {spec.n_layers} banks"
    )
    return {
        "stack": {
            "attn": spec.attn,
            "label": "hybrid 3:1 kda:dsa" if spec.attn == "hybrid" else spec.attn,
            "pattern": list(spec.pattern),
            "pos": "none",
            "unit": spec.unit,
        },
        "ffn": {
            "kind": "moe" if spec.is_moe else "gelu",
            "experts": spec.expert_arm,
            "routed": spec.n_experts,
            "shared": spec.n_shared,
            "topk": spec.topk,
            "tsct_rank": spec.expert_tsct_rank,
        },
        "slots": {
            "optimizer": spec.optimizer,
            "objective": spec.objective,
            "weight_qat": spec.weight_qat,
            "act_quant": spec.act_quant,
        },
        "widths": {
            "d_model": spec.d_model,
            "n_layers": spec.n_layers,
            "d_ffn": spec.d_ffn,
            "n_heads": spec.n_heads,
            "head_dim": spec.head_dim,
            "n_kv_heads": spec.n_kv_heads,
            "c_lat": spec.c_lat,
            "seq_len": spec.seq_len,
            "batch": spec.batch,
        },
        "params": {
            "stored": report["stored"],
            "active": report["active"],
            "active_ffn": report["active_ffn"],
            "ffn": report["ffn"],
            "attention": report["attention"],
            "attention_share": attention_share,
            "active_share": report["active"] / report["stored"] if report["stored"] else 0.0,
        },
        "step": {
            "tokens": spec.tokens,
            "tokens_per_step": tokens_per_step,
            "steps": steps,
            "flops_per_step": 6 * report["active"] * tokens_per_step,
            "flops_per_token": 6 * report["active"],
        },
        "bytes": {
            "state": STATE_BYTES * report["stored"],
            "checkpoint": int(bits * report["stored"]),
        },
        "plan": {
            "total_seconds": plan.seconds,
            "human": Phase("", plan.seconds).human(),
            "throughput_tflops": spec.throughput_tflops,
            "phases": [
                {
                    "name": phase.name,
                    "seconds": phase.seconds,
                    "human": phase.human(),
                    "note": phase.note,
                }
                for phase in plan.phases
            ],
        },
        "mechanisms": assembled.mechanisms(),
        "warnings": list(plan.warnings),
    }
