"""How `dorm` looks: one rich renderer, and colour/progress off when asked.

Presentation is allowed to be pretty, not to be a second implementation. There is one
rendering path; the only switches are the ones a pipe needs — `NO_COLOR` and a
non-terminal stream produce the same layout with no escape sequences, and progress is
omitted entirely where nobody is watching.

Three rules decided here rather than in each command:

* `--json` is the machine interface, not a fallback: it prints the same data these
  renderers read, so the screen and the pipe cannot disagree.
* progress goes to stderr, so `dorm plan --json | jq` never reads a bar.
* a refusal is a panel with the escape on its own line, never a traceback.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as Bar
from rich.table import Table
from rich.text import Text

#: The three things a caller can build here: a verdict row, a step counter, the screen.
__concepts__ = ("Check", "Progress", "UI")

#: The three verdicts a check can carry; the glyph is what makes one visible at a glance.
GLYPHS = {"ok": "✓", "warn": "▲", "fail": "✗"}
STYLES = {"ok": "green", "warn": "yellow", "fail": "bold red"}

#: Where the sustained rate came from, in the words the plan's `assumed` row uses.
RATE_SOURCES = {
    "config": "rate set in the config",
    "journal": "rate recorded by the newest run",
    "profile": "rate measured 2026-11-15 (step profile)",
}


@dataclass(frozen=True)
class Check:
    """One row of `dorm doctor`: the verdict, what was checked, and what was found."""

    status: str
    name: str
    value: str


def console(*, stderr: bool = False) -> Console:
    """Build the one console a command writes through.

    `NO_COLOR` turns styling off, not rendering off: the tables and panels still line
    up, they just carry no escape sequences. A non-terminal stream gets the same
    treatment from rich itself, which is what makes `dorm doctor | less` readable.
    """
    no_color = bool(os.environ.get("NO_COLOR"))
    return Console(stderr=stderr, no_color=no_color, force_terminal=False if no_color else None)


def as_json(data: object) -> str:
    """Render the `--json` document: sorted keys, two-space indent, nothing else."""
    return json.dumps(data, indent=2, sort_keys=True)


def age(seconds: float) -> str:
    """Render a duration the way a person reads it, never more than two significant digits."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}min"
    if seconds < 129_600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86_400:.1f}d"


def doctor_checks(report: Mapping[str, Any]) -> list[Check]:
    """Turn the doctor's facts into verdicts: a missing tool is not a neutral fact.

    "ok" means a command could use it, "warn" means it is absent or unusable but the
    tool still runs on this machine, "fail" means a command that needs it will not run.
    """
    torch_version = report.get("torch")
    cuda = report.get("cuda")
    gpu_name = report.get("gpu_name")
    total = report.get("vram_total_mib")
    used = report.get("vram_used_mib")
    disk = report.get("disk_free_gb")
    sha = report.get("git_sha")

    if gpu_name is None:
        gpu = Check("warn", "gpu", "none found, cpu only")
    elif cuda is None:
        gpu = Check("warn", "gpu", f"{gpu_name} (torch has no CUDA)")
    else:
        gpu = Check("ok", "gpu", str(gpu_name))

    if isinstance(total, int) and total > 0:
        busy = isinstance(used, int) and used / total > 0.9
        vram = Check(
            "warn" if busy else "ok",
            "vram",
            f"{used if isinstance(used, int) else '?'} / {total} MiB used",
        )
    else:
        vram = Check("warn", "vram", "unknown")

    free = float(disk) if isinstance(disk, (int, float)) else 0.0
    disk_check = Check(
        "fail" if free < 5 else ("warn" if free < 50 else "ok"), "disk free", f"{free:.1f} GB"
    )

    return [
        Check("ok", "python", str(report.get("python", "?"))),
        Check("ok" if torch_version else "fail", "torch", str(torch_version or "not installed")),
        Check("ok" if cuda else "warn", "cuda", str(cuda or "cpu build")),
        gpu,
        vram,
        disk_check,
        Check("ok" if sha else "warn", "git", str(sha or "not a checkout")),
        Check("ok", "platform", str(report.get("platform", "?"))),
    ]


class Progress:
    """A bar for an operation that can take seconds; absent when nobody can see one.

    A pipe or a `NO_COLOR` terminal gets no bar at all: progress is a courtesy for a
    human watching, and a control sequence in a log is noise.
    """

    def __init__(self, total: int, description: str) -> None:
        """Fix the step count; the decision to render is made when the bar starts."""
        self.total = max(total, 1)
        self.description = description
        self.done = 0
        self._bar: Any = None
        self._task: Any = None

    def __enter__(self) -> Progress:
        """Start the bar when stderr is a terminal and colour was not refused."""
        if not sys.stderr.isatty() or os.environ.get("NO_COLOR"):
            return self
        self._bar = Bar(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console(stderr=True),
            transient=True,
        )
        self._bar.start()
        self._task = self._bar.add_task(self.description, total=self.total)
        return self

    def advance(self, metric: str = "") -> None:
        """Record one finished step, with the metric that step produced."""
        self.done += 1
        if self._bar is not None:
            suffix = f" · {metric}" if metric else ""
            self._bar.update(self._task, advance=1, description=f"{self.description}{suffix}")

    def __exit__(self, *exc: object) -> None:
        """Stop the bar; a failure still leaves the last step on screen."""
        if self._bar is not None:
            self._bar.stop()


class UI:
    """The screen: the doctor's verdicts, the plan, the runs, and the refusals."""

    def progress(self, total: int, description: str) -> Progress:
        """Return a progress counter for a long operation."""
        return Progress(total, description)

    def doctor(self, report: Mapping[str, Any]) -> None:
        """Print the doctor: a table of verdicts, the glyph first so a failure reads at a glance."""
        table = Table(
            title="dorm doctor",
            caption="what this machine can do",
            title_justify="left",
            caption_justify="left",
            header_style="bold dim",
        )
        table.add_column("", width=1)
        table.add_column("check")
        table.add_column("value")
        for check in doctor_checks(report):
            table.add_row(
                f"[{STYLES[check.status]}]{GLYPHS[check.status]}[/]",
                f"[{STYLES[check.status]}]{check.name}[/]",
                check.value,
            )
        console().print(table)

    def plan(self, data: Mapping[str, Any]) -> None:
        """Print the plan: the flagship screen — model, bill, mechanisms, warnings."""
        output = console()
        stack = data["stack"]
        ffn = data["ffn"]
        slots = data["slots"]
        widths = data["widths"]
        params = data["params"]
        step = data["step"]
        memory = data["bytes"]
        compute = data["compute"]
        timing = data["time"]

        summary = Table.grid(padding=(0, 2))
        summary.add_column(justify="right", style="bold dim")
        summary.add_column()
        summary.add_row(
            "model",
            f"[bold]{stack['label']}[/] · {widths['n_layers']} layers · "
            f"d_model {widths['d_model']} · heads {widths['n_heads']}x{widths['head_dim']} "
            f"(kv {widths['n_kv_heads']}, c_lat {widths['c_lat']}) · "
            f"d_ffn {widths['d_ffn']} · seq {widths['seq_len']}",
        )
        summary.add_row(
            "slots",
            f"{stack['unit']} ids · pos {stack['pos']} · {slots['optimizer']} · "
            f"{slots['objective']} · qat {slots['weight_qat']} · act {slots['act_quant']}",
        )
        summary.add_row(
            "ffn",
            f"{ffn['kind']} · {ffn['routed']} routed + {ffn['shared']} shared "
            f"{ffn['experts']} experts · top-{ffn['topk']} · tsct rank {ffn['tsct_rank']}",
        )
        summary.add_row("stored", f"[bold]{params['stored']:,}[/] parameters")
        summary.add_row(
            "active",
            f"[bold]{params['active']:,}[/] parameters "
            f"({params['active_share']:.1%} of stored) · ffn {params['active_ffn']:,} · "
            f"attention {params['attention']:,} ({params['attention_share']:.1%})",
        )
        summary.add_row(
            "step",
            f"{step['tokens_per_step']:,} tokens/step · {step['steps']:,} steps for "
            f"{step['tokens']:,} tokens",
        )
        summary.add_row(
            "compute",
            f"[bold]{compute['forward_flops_per_token'] / 1e6:.1f}[/] MFLOP/token forward "
            f"({compute['blocks_flops_per_token'] / 1e6:.1f} blocks + "
            f"{compute['head_flops_per_token'] / 1e6:.1f} head)",
        )
        summary.add_row(
            "train",
            f"[bold]{compute['train_flops_per_token'] / 1e6:.1f}[/] MFLOP/token fwd+bwd "
            f"(3x forward, the convention)",
        )
        summary.add_row(
            "memory",
            f"{memory['state'] / 1e9:.1f} GB/step state (weights+grad+2 moments) · "
            f"{memory['checkpoint'] / 1e9:.2f} GB checkpoint",
        )
        summary.add_row(
            "time",
            f"[bold]{timing['human']}[/] for {step['tokens']:,} tokens "
            f"({timing['data_gb']:.1f} GB) at {timing['sustained_tflops']} TFLOP/s",
        )
        summary.add_row(
            "assumed",
            f"{RATE_SOURCES[timing['sustained_source']]} · "
            f"{timing['bytes_per_token']} B/token · 3x forward for the backward",
        )
        title = "dorm plan"
        if "config" in data:
            title = f"dorm plan · {data['config']}"
        output.print(
            Panel(
                summary,
                title=title,
                subtitle="measured on the meta device: nothing allocated",
                subtitle_align="left",
                border_style="cyan",
                title_align="left",
            )
        )

        output.print(
            "\n[bold]mechanisms[/] [dim](every enabled one, with the counter that proves it ran)[/]"
        )
        mechanisms = Table(header_style="bold dim", box=None, padding=(0, 2))
        mechanisms.add_column("mechanism")
        mechanisms.add_column("counter")
        for name, counter in data["mechanisms"].items():
            mechanisms.add_row(name, counter)
        output.print(mechanisms)

        if data["warnings"]:
            output.print(
                Panel(
                    "\n".join(f"[yellow]![/] {warning}" for warning in data["warnings"]),
                    title="warnings",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

    def status(self, runs: Sequence[Mapping[str, Any]]) -> None:
        """Print the runs as a table read from the journals."""
        table = Table(
            title=f"dorm status · {len(runs)} run{'' if len(runs) == 1 else 's'}",
            title_justify="left",
            caption="read from the journal, not from a log",
            caption_justify="left",
            header_style="bold dim",
        )
        table.add_column("id")
        table.add_column("phase")
        table.add_column("step", justify="right")
        table.add_column("metric", justify="right")
        table.add_column("swaps", justify="right")
        table.add_column("age", justify="right")
        colors = {"running": "cyan", "ended": "green", "not started": "dim"}
        for run in runs:
            phase = str(run["phase"])
            table.add_row(
                str(run["id"]),
                f"[{colors.get(phase, 'white')}]{phase}[/]",
                f"{run['step']:,}",
                str(run["metric"] or "-"),
                str(run["segments"]),
                age(float(run["age_seconds"])),
            )
        console().print(table)

    def error(self, exc: Exception) -> None:
        """Print a refusal: what in bold, the escape on its own line, no traceback."""
        what = getattr(exc, "what", str(exc))
        escape = getattr(exc, "escape", None)
        body = Text()
        body.append(str(what), style="bold")
        if escape:
            body.append("\n")
            body.append(f"Escape: {escape}", style="cyan")
        console(stderr=True).print(
            Panel(body, title="refused", title_align="left", border_style="red")
        )
