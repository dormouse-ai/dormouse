"""Build the stack a spec names: check every coeffect, then allocate in one pass.

The order is the contract, and it is the order because each step exists to make the
next one safe:

    registry.pieces(spec)   refuse a conflicting pair, naming it, with nothing allocated
    build per layer         one attention arm per layer, one bank for every FFN slot
    quantize                ternary weights / int4 activations, if the slots ask
    declarations            the mechanisms the forward must count, for `Counters.check`

WHY THE BANK IS SHARED BY THE LAYERS. `ffn=moe` is one bank of `n_experts`, and every
layer routes into it. That is the arithmetic the stack is designed to: at
`d_model=832, d_ffn=6144, n_experts=64, topk=2, n_shared=1, rank=256` the bank stores
`65·3·(832·256 + 256·6144) ≈ 348.6M` parameters and runs `3·3·(832·256 + 256·6144) ≈
16.1M` per token. One bank per layer would store twelve times that and run 193M per
token, which is not the "348.6M stored / 16.1M active" the model is priced at — so the
sharing is stated here, in one place, rather than discovered from a `param_report` that
does not add up.

WHY A WIDTH CANNOT BE QUIET. Every arm is built from `StackSpec`'s widths and every
arm's own constructor validates them (`n_heads % n_kv_heads`, `3·d·d_ffn` against the
GELU control for a mixture, `seq_len` divisible by KDA's chunk, the period tiling
`n_layers`); this module only decides which arm, and passes `write_std` (the
`1/√(2L)` residual-write scale, or plain `0.02` under `DORMOUSE_LNS=1`) so a prebuilt
arm is drawn exactly as the one `Block` would have built. `lns_scale` numbers the
blocks 1..L; deactivating it would give every layer the same factor.
"""

from __future__ import annotations

from typing import cast

import torch
from dormouse_core.counters import Counters
from dormouse_core.plan import Plan
from dormouse_kernels.engine.bank import bank_enabled
from torch import nn

from .. import economics
from ..train import compiled
from ..train.graph import StepGraph
from ..train.optim import MixOptimizer
from ..train.step import StepResult, train_step
from . import quant, registry, tsct
from .attn_variants.csa import CHUNK as CSA_CHUNK
from .attn_variants.csa import CSA, HCA_GROUP
from .attn_variants.csa import TOPK as CSA_TOPK
from .attn_variants.dsa import DSA
from .attn_variants.kda import KDA
from .blocks import Block, residual_write_std
from .ffn_spec import FFNSpec
from .model import Transformer
from .moe import MoEFFN
from .spec import StackSpec
from .tsct import TSCT_RANK

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
        #: One `Counters` for the life of the stack. A fresh one per step changes a Python
        #: object's identity between compiled calls, which makes dynamo guard and recompile
        #: the whole step every time — a compile loop reported as a compiled run.
        self.counters = Counters()

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
        graph: StepGraph | None = None,
    ) -> tuple[StepResult, Counters]:
        """One optimizer step, with the objective's counter declared where it is read.

        `graph` replaces the forward/backward half with a replay of a captured one
        (`train/graph.py`); the clip, the firewall and `optimizer.step()` are the same
        eager statements either way. `ntp_loss` is counted after the step and enabled
        last on purpose: the stack's own `Counters.check()` runs inside its forward, so a
        counter the step produces must not be enabled before that forward has finished.
        """
        counters = self.counters
        counters.counts.clear()
        counters.enabled.clear()
        counters.required.clear()
        if self.bank is not None and self.bank.ideal_bank:
            #: Declared before the forward so an ideal flag whose bank never ran is a
            #: MissingReport, not a step that quietly used the old path.
            counters.enable(
                "ideal_bank_fwd", counter="ideal_bank_fwd", why="the dense U·V bank ran"
            )
        if graph is not None:
            counters.enable(
                "graph",
                counter="graph_replays",
                why="the step must replay a captured graph, not run eager",
            )
            result = graph.run(ids, counters=counters)
        else:
            result = train_step(
                self.model,
                ids,
                optimizer=optimizer,
                accum=accum,
                #: A `Counters` is a Python object dynamo specialises on and its `count`
                #: calls are side effects it cannot trace: under compile the step counts
                #: what it can see from outside (graphs, replays) and the mechanism
                #: counters stay the eager path's own, where they can still add up.
                counters=None if compiled.enabled() else counters,
            )
        if self.bank is not None and self.bank.ideal_bank:
            #: Declared after backward, like the A4 backward receipts: the hook in
            #: `MoEFFN._ideal_forward` fires during autograd, not during the forward.
            counters.enable(
                "ideal_bank_bwd",
                counter="ideal_bank_bwd",
                why="the dense bank backpropagated",
            )
        if not result.skipped:
            counters.count("ntp_loss")
        counters.enable("ntp", counter="ntp_loss", why="the step computes the NTP loss")
        if self.spec.act_quant == "a4" and not compiled.enabled():
            #: The reload's backward is bound only after `train_step` returned: the
            #: model's own mid-forward `check()` cannot ask for a gradient that has
            #: not run yet.
            counters.enable(
                "act_a4_bwd",
                counter="a4_bwd",
                why="the int4 reload's backward must run, not just its forward",
            )
            if quant.codesave_on():
                counters.enable(
                    "act_a4_codes",
                    counter="a4_codes_saved",
                    why="the forward must save packed int4 codes, not a fp32 activation",
                )
            if quant.fused_on():
                #: Enabled after the backward, as with `a4_bwd`: a fused flag with no
                #: kernel receipt is a MissingReport, not an eager path under a new name.
                counters.enable(
                    "act_a4_fused_fwd",
                    counter="a4_fused_fwd",
                    why="the fused FWHT/quant/pack kernel must run, not the Python path",
                )
                counters.enable(
                    "act_a4_fused_bwd",
                    counter="a4_fused_bwd",
                    why="the fused unpack/STE kernel must run in the backward",
                )
        if self.spec.weight_qat == "ternary_native":
            #: Enabled here, after the backward, not in `Transformer.forward`: that
            #: forward checks the counters before autograd has produced any gradient.
            counters.enable(
                "tsct_ternary_bwd",
                counter="tsct_ternary_bwd",
                why="the ternary STE backward reached the fp32 shadows",
            )
            counters.count("tsct_ternary_bwd", tsct.take_ternary_bwd())
        if (
            self.bank is not None
            and quant.bank_codesave_on()
            and not (bank_enabled() and next(self.model.parameters()).is_cuda)
        ):
            #: The fused bank writes no `[rows, d_ffn]` slab — there is nothing to
            #: pack there, so an enabled counter would be a MissingReport. The anchor
            #: is the only arm this flag can prove, and the counters are read after
            #: the backward because `bank_a4_bwd` is produced there.
            fwd, bwd, codes = quant.take_bank_codesave_counters()
            for name, value, why in (
                ("bank_a4_fwd", fwd, "the bank packed its wide slabs into INT4 codes"),
                ("bank_a4_bwd", bwd, "the bank unpacked codes in the STE backward"),
                ("bank_a4_codes_saved", codes, "the saved bytes are codes, not bf16 slabs"),
            ):
                counters.enable(name, counter=name, why=why)
                counters.count(name, value)
        if compiled.enabled():
            #: Enabled and counted after the step, and zero is a hard error: a compile that
            #: built no graph ran the eager path under a compiled name.
            counters.enable(
                "compile",
                counter=compiled.COUNTER,
                why="torch_compile=true: the step must build a graph, not fall back to eager",
            )
            counters.count(compiled.COUNTER, compiled.unique_graphs())
        counters.check()
        return result, counters

    def capture(
        self,
        ids: torch.Tensor,
        *,
        optimizer: MixOptimizer,
        accum: int = 1,
        epilogue: bool = False,
    ) -> StepGraph:
        """Capture one step's forward/backward at `ids`'s shape; see `train/graph.py`."""
        return StepGraph(self.model, optimizer, ids, accum=accum, epilogue=epilogue)


def assemble(
    spec: StackSpec,
    *,
    device: torch.device | str | None = None,
    smoke: bool = False,
) -> Assembled:
    """Build the stack `spec` names, refusing a contradiction before allocating it."""
    if (
        quant.bank_codesave_on()
        and not quant.a4_config_on()
        and bank_enabled()
        and device is not None
        and torch.device(device).type == "cuda"
    ):
        from dormouse_core.errors import LoudError

        raise LoudError(
            "DORMOUSE_BANK_A4_CODESAVE=1 with the fused bank: it writes no "
            "[rows, d_ffn] slab to compress",
            "run with DORMOUSE_BANK=0 to measure the anchor, where the slab exists; "
            "a flag that packs nothing must not be reported as a saving",
        )
    pieces = registry.pieces(spec)
    write_std = residual_write_std(spec.n_layers)
    bank = None if spec.ffn == "dense" else _bank(spec, write_std, device, smoke)
    blocks = [
        Block(
            spec.d_model,
            spec.d_ffn,
            n_layers=spec.n_layers,
            #: The dense arm is one full-rank SwiGLU per layer; the bank is shared,
            #: but each block gets its own FlatBank adapter as the spec always did.
            ffn=(
                FFNSpec(spec.d_model, spec.d_ffn, tsct_rank=spec.ffn_tsct_rank)
                if bank is None
                else FlatBank(bank)
            ),
            attn=_attention(spec, kind, write_std, device),
            device=device,
            a4=spec.act_quant == "a4",
            layer_idx=layer_idx,
        )
        for layer_idx, kind in enumerate(spec.pattern, start=1)
    ]
    model = Transformer(
        vocab=spec.vocab,
        d_model=spec.d_model,
        d_ffn=spec.d_ffn,
        max_seq_len=spec.seq_len,
        blocks=blocks,
        declarations=registry.declarations(pieces),
        loop_passes=spec.loop_passes,
        loop_start_frac=spec.loop_start_frac,
        loop_end_frac=spec.loop_end_frac,
        loop_rank=spec.loop_rank,
        block_checkpoint=spec.block_checkpoint,
        chunked_loss=spec.chunked_loss,
        device=device,
    )
    _quantize(model, spec)
    if spec.torch_compile:
        compiled.checked(model)
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
    if kind in ("csa", "hca"):
        return CSA(
            spec.d_model,
            spec.n_heads,
            spec.head_dim,
            chunk=CSA_CHUNK,
            topk=CSA_TOPK,
            hca_group=1 if kind == "csa" else HCA_GROUP,
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
        ideal_bank=spec.ideal_bank,
        ideal_slots=spec.ideal_slots,
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
    if spec.weight_qat == "ternary_native":
        #: Swap the bank's factors BEFORE wrapping the blocks: the bank's stacked
        #: operands are built from whatever the experts hold at forward time.
        tsct.native_ternarize(model)
        quant.native_ternarize(model)
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


def plan_lines(
    spec: StackSpec,
    *,
    elasticity: float = economics.ELASTICITY,
    elasticity_from: str = "constant",
) -> list[str]:
    """Everything `dorm plan` prints: the stack, the bill, the step, the mechanisms.

    The model is built on the `meta` device, so the numbers are measured from the
    modules that would be allocated rather than from a formula that can drift — and
    `dorm plan` on a 1B stack costs no memory.
    """
    assembled = assemble(spec, device="meta")
    report = assembled.report()
    tokens_per_step = spec.batch * spec.seq_len
    data_per_step = spec.batch * (spec.seq_len + 1)
    steps = max(spec.tokens // data_per_step, 1)
    seconds = (
        steps
        * 6
        * report["active"]
        * tokens_per_step
        * spec.loop_factor
        / (spec.throughput_tflops * 1e12)
    )
    bits = 0.25 if spec.weight_qat in ("ternary", "ternary_native") else 4.0
    attention_share = report["attention"] / report["active"] if report["active"] else 0.0

    if spec.attn_pattern:
        name = f"hybrid pattern={list(spec.attn_pattern)}"
    elif spec.attn == "hybrid":
        name = "hybrid 3:1 kda:dsa"
    else:
        name = spec.attn
    lines = [
        f"stack: {name} (layers: {', '.join(spec.pattern)}), pos=none, unit={spec.unit}",
        (
            "ffn: dense (full-rank SwiGLU per layer)"
            if spec.ffn == "dense" and not spec.ffn_tsct_rank
            else (
                f"ffn: dense (TSCT rank {spec.ffn_tsct_rank} SwiGLU per layer)"
                if spec.ffn == "dense"
                else f"ffn: moe (experts: {spec.expert_arm}, {spec.n_experts} routed, topk "
                f"{spec.topk}, tsct rank {spec.expert_tsct_rank})"
            )
        )
        + f", optimizer={spec.optimizer}, objective={spec.objective}"
        + f", weight_qat={spec.weight_qat}, act_quant={spec.act_quant}",
        f"widths: d_model={spec.d_model} x {spec.n_layers} layers, "
        f"d_ffn={spec.d_ffn}, heads={spec.n_heads}x{spec.head_dim} "
        f"(kv={spec.n_kv_heads}, c_lat={spec.c_lat}), seq={spec.seq_len}",
        f"params: stored={report['stored']:,} active={report['active']:,} "
        f"(ffn active={report['active_ffn']:,}, {report['active'] / report['stored']:.1%} "
        f"of stored), ffn={report['ffn']:,} attention={report['attention']:,} "
        f"({attention_share:.1%} of the active set)",
        f"step: {tokens_per_step:,} tokens/step, {steps:,} steps for {spec.tokens:,} tokens; "
        f"passes={spec.loop_passes}, loop_window={spec.loop_window_fraction:.3f}, "
        f"flop_factor={spec.loop_factor:.3f}; "
        f"flops/step={6 * report['active'] * tokens_per_step * spec.loop_factor:,}",
        f"bytes/step: state={STATE_BYTES * report['stored']:,} (weights+grad+2 moments), "
        f"checkpoint={int(bits * report['stored']):,}",
    ]
    plan = Plan().add(
        "train",
        seconds,
        note=f"{spec.tokens:,} tokens at {spec.throughput_tflops} TFLOP/s sustained",
    )
    if spec.unit == "bpe":
        plan.warn(
            f"unit=bpe: the tokenizer that produces {spec.vocab:,}-way ids is "
            "data/tokenizer.py; a run must ship the tokenizer.json its manifest names"
        )
    if spec.ffn == "moe":
        plan.warn(
            f"the bank is shared by {spec.n_layers} layers: stored is "
            f"{report['stored']:,}, not {spec.n_layers} banks"
        )
    lines.extend(plan.render())
    lines.append(
        "mechanisms: "
        + ", ".join(dict.fromkeys(f"{p.mechanism}→{p.counter}" for p in assembled.pieces))
    )
    lines.append(f"economics: elasticity {elasticity:.4f} ({elasticity_from})")
    lines.append("  lever | speedup | loss | verdict")
    lines.extend(f"  {lever.row(elasticity)}" for lever in levers(spec))
    return lines


#: The lever slots a config sets, and the recipe's own value of each. Any other value is
#: a proposal, and a proposal is priced by `economics` rather than assumed.
RECIPE_LEVERS: dict[str, object] = {
    "weight_qat": "none",
    "act_quant": "none",
    "torch_compile": False,
    "expert_tsct_rank": TSCT_RANK,
}


def levers(spec: StackSpec) -> list[economics.Lever]:
    """Every lever slot this config sets: the recipe's own value costs nothing.

    A proposal with no measurement prints as `unmeasured` and is never `take`: the one
    thing this table must not do is wave through a lever nobody measured.
    """
    out: list[economics.Lever] = []
    for slot, recipe in RECIPE_LEVERS.items():
        name = f"{slot}={getattr(spec, slot)}"
        measured = economics.MEASURED.get(name)
        if measured is not None:
            out.append(economics.Lever(name, *measured))
        elif getattr(spec, slot) == recipe:
            out.append(economics.Lever(name, 1.0, 0.0))
        else:
            out.append(economics.Lever(name))
    return out
