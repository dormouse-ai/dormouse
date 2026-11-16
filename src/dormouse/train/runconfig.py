"""The run's config as a value: the spec, the mixture, the cadence, and the text that said it.

`StackSpec` is the architecture (`model/spec.py`), and it stays that: putting data paths on
it would make one value answer two questions and `assemble.plan_lines` would start printing
a cadence. What a *run* needs beyond the stack is six things, and they live here:

    [[source]]             path (relative to this file), weight, stage
    [[stage]]              name, until — the last stage may omit it
    eval = "path.bin"      the held-out stream BPB and the harness are measured on
    schedule = "cosine"    cosine | wsd | sf, routed by train/optim.py
    lr_muon, lr_adamw      the arms' peak rates, or the mode's own default
    warmup, eval_every, ckpt_every, eval_batches, harness, seed

Steps are `tokens // (batch x seq_len)` — the number `dorm plan` already prints — rather
than a second budget that can disagree with the first.

THE RECIPE IS AN IDENTITY, NOT A BUNDLE OF FIELDS. `canonical()` is the recipe as one
JSON value, `digest` is its sha256, and the run directory is the config's stem. Both halves
are needed and neither is decoration: the directory is what `--resume` has to *reach* when
the recipe has changed (a name derived from the hash turns "you changed lr_muon" into "that
run does not exist"), and the hash is what the journal records, so a run whose journal and
file disagree refuses by name instead of continuing under a recipe nobody ran.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from dormouse_core.errors import LoudError
from dormouse_core.journal import Journal

from ..data.mix import Source
from ..data.mix import Stage as MixStage
from ..model.spec import StackSpec
from .recipe import LoopConfig, Stage

__all__ = ["RUN_KEYS", "RunRecipe", "changed", "check_tokenizer", "fresh", "refuse_drift"]

#: The keys a run adds to the stack spec. Anything else in the file is a typo, by name.
RUN_KEYS = frozenset(
    {
        "source",
        "stage",
        "eval",
        "schedule",
        "lr_muon",
        "lr_adamw",
        "warmup",
        "eval_every",
        "ckpt_every",
        "eval_batches",
        "harness",
        "seed",
    }
)


def check_tokenizer(spec: StackSpec, sources: Sequence[Source], config: Path) -> None:
    """Refuse at config load — before any model is allocated — a tokenizer that disagrees.

    The token table and every data manifest must answer to one tokenizer file. The
    vocabulary is checked first (`cfg.vocab` against the tokenizer's real size), then
    each source's manifest hash against the file's. Both numbers are printed: a refusal
    that only says "mismatch" costs an hour to diagnose.
    """
    if spec.unit != "bpe":
        return
    from ..data.bin import Manifest
    from ..data.tokenizer import check_vocab, load, sha_file

    if not spec.tokenizer:
        raise LoudError(
            f"{config}: unit=bpe names no tokenizer",
            'set tokenizer = "tokenizer32k.json": the token table and every data '
            "manifest must answer to one tokenizer, and a run that cannot name it "
            "has no way to read its own ids back",
        )
    token_path = spec.tokenizer
    check_vocab(load(token_path), spec.vocab)
    expected = sha_file(token_path)
    for source in sources:
        actual = Manifest.read(Path(source.path)).tokenizer_sha
        if actual != expected:
            raise LoudError(
                f"{source.path}: the data's tokenizer is {actual or '(none)'}, "
                f"the run's {token_path} is {expected}",
                "tokenize the corpus with this run's tokenizer, or point the config at "
                "the tokenizer the data was written with; ids from two vocabularies are noise",
            )


@dataclass(frozen=True)
class RunRecipe:
    """One run as its config states it: the stack, the mixture, the cadence, the text."""

    spec: StackSpec
    sources: tuple[Source, ...]
    stages: tuple[MixStage, ...]
    eval_path: Path | None
    schedule: str
    lr_muon: float | None
    lr_adamw: float | None
    warmup: int | None
    eval_every: int
    ckpt_every: int
    eval_batches: int
    harness_on: bool
    seed: int
    path: Path
    text: str

    # ------------------------------------------------------------------ reading
    @classmethod
    def read(cls, path: str | Path) -> RunRecipe:
        """Parse and validate one run config, refusing an unknown key by name."""
        source = Path(path).resolve()
        try:
            text = source.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise LoudError(
                f"config not found: {source}",
                "pass a path to an existing .toml; `dorm plan` reads the same file",
            ) from None
        raw = tomllib.loads(text)
        stack_keys = {field.name for field in fields(StackSpec) if field.init}
        unknown = sorted(set(raw) - stack_keys - RUN_KEYS)
        if unknown:
            raise LoudError(
                f"{source}: unknown key(s) {', '.join(unknown)}",
                "a key no code path reads is a silent no-op; the run's keys are "
                f"{', '.join(sorted(RUN_KEYS))}, the rest are StackSpec's fields",
            )
        spec = cast(
            StackSpec,
            StackSpec.from_mapping(
                {key: value for key, value in raw.items() if key in stack_keys},
                source=str(source),
            ),
        )
        stages = cls._stages(raw.get("stage", ()), source)
        sources = cls._sources(raw.get("source", ()), source)
        check_tokenizer(spec, sources, source)
        if not stages:
            names = list(dict.fromkeys(item.stage for item in sources))
            if len(names) > 1:
                raise LoudError(
                    f"{source}: sources name {len(names)} stages "
                    f"({', '.join(names)}) and no [[stage]] table orders them",
                    'add one [[stage]] per name in run order: name = "bulk" until = N, '
                    "and the last one without until",
                )
            stages = (MixStage(names[0]),)
        held = raw.get("eval")
        return cls(
            spec=spec,
            sources=sources,
            stages=stages,
            eval_path=None if held is None else (source.parent / str(held)).resolve(),
            schedule=str(raw.get("schedule", "cosine")),
            lr_muon=_rate(raw, "lr_muon"),
            lr_adamw=_rate(raw, "lr_adamw"),
            warmup=_count(raw, "warmup"),
            eval_every=_count(raw, "eval_every", 0),
            ckpt_every=_count(raw, "ckpt_every", 0),
            eval_batches=_count(raw, "eval_batches", 8),
            harness_on=bool(raw.get("harness", True)),
            seed=_count(raw, "seed", 0),
            path=source,
            text=text,
        )

    @classmethod
    def from_canonical(cls, data: dict[str, Any], *, path: Path, text: str = "") -> RunRecipe:
        """Rebuild the recipe a run was written under, from its own `config.json`."""
        run: dict[str, Any] = data["run"]
        return cls(
            spec=cast(StackSpec, StackSpec.from_mapping(data["stack"])),
            sources=tuple(
                Source(Path(item["path"]), float(item["weight"]), str(item["stage"]))
                for item in run["sources"]
            ),
            stages=tuple(MixStage(str(item["name"]), item["until"]) for item in run["stages"]),
            eval_path=None if run["eval"] is None else Path(str(run["eval"])),
            schedule=str(run["schedule"]),
            lr_muon=run["lr_muon"],
            lr_adamw=run["lr_adamw"],
            warmup=run["warmup"],
            eval_every=int(run["eval_every"]),
            ckpt_every=int(run["ckpt_every"]),
            eval_batches=int(run["eval_batches"]),
            harness_on=bool(run["harness"]),
            seed=int(run["seed"]),
            path=path,
            text=text,
        )

    @staticmethod
    def _sources(tables: Any, config: Path) -> tuple[Source, ...]:
        """Read `[[source]]` tables into mixture sources, paths relative to the config."""
        if not tables:
            raise LoudError(
                f"{config}: no [[source]] table, so the run reads no data",
                'add one:\n  [[source]]\n  path = "data/corpus.bin"\n  weight = 1.0\n'
                "the path is relative to this config, and the stream needs its "
                "<path>.json manifest beside it (data/bin.py refuses one without)",
            )
        out = []
        for table in tables:
            unknown = sorted(set(table) - {"path", "weight", "stage"})
            if unknown or "path" not in table:
                raise LoudError(
                    f"{config}: a [[source]] has {', '.join(unknown) or 'no path'}",
                    "a source is path = ..., weight = ... (default 1.0), stage = ... "
                    "(default 'train')",
                )
            out.append(
                Source(
                    (config.parent / str(table["path"])).resolve(),
                    float(table.get("weight", 1.0)),
                    str(table.get("stage", "train")),
                )
            )
        return tuple(out)

    @staticmethod
    def _stages(tables: Any, config: Path) -> tuple[MixStage, ...]:
        """Read `[[stage]]` tables into the mixture's schedule."""
        for table in tables:
            unknown = sorted(set(table) - {"name", "until"})
            until = table.get("until")
            if unknown or "name" not in table or (until is not None and not isinstance(until, int)):
                raise LoudError(
                    f"{config}: a [[stage]] has {', '.join(unknown) or 'a bad field'}",
                    "a stage is name = ... and until = <int>; the last may omit until, "
                    "which means the end of the budget, and no stage may be a string",
                )
        return tuple(MixStage(str(table["name"]), table.get("until")) for table in tables)

    # ------------------------------------------------------------------ derived
    def canonical(self) -> dict[str, Any]:
        """Return the recipe as one JSON-able value: what a resume is compared against."""
        return {
            "stack": self.spec.as_dict(),
            "run": {
                "sources": [
                    {"path": str(item.path), "weight": item.weight, "stage": item.stage}
                    for item in self.sources
                ],
                "stages": [{"name": stage.name, "until": stage.until} for stage in self.stages],
                "eval": None if self.eval_path is None else str(self.eval_path),
                "schedule": self.schedule,
                "lr_muon": self.lr_muon,
                "lr_adamw": self.lr_adamw,
                "warmup": self.warmup,
                "eval_every": self.eval_every,
                "ckpt_every": self.ckpt_every,
                "eval_batches": self.eval_batches,
                "harness": self.harness_on,
                "seed": self.seed,
            },
        }

    @property
    def digest(self) -> str:
        """Return the recipe's identity: the same numbers, the same hash."""
        blob = json.dumps(self.canonical(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    @property
    def tokens_per_step(self) -> int:
        """Return what one step reads: the number the plan and the loop must agree on."""
        return self.spec.batch * self.spec.seq_len

    @property
    def steps(self) -> int:
        """Return the budget in steps — the plan's own arithmetic, not a second number."""
        return max(self.spec.tokens // self.tokens_per_step, 1)

    def directory(self, root: str | Path) -> Path:
        """Return `<root>/<config stem>`: the run is named after its config, not a timestamp.

        The recipe's hash is journaled rather than put in the name, because `--resume` has
        to reach the directory it is disagreeing with: a name that changes with the recipe
        turns "you changed lr_muon" into "that run does not exist".
        """
        return Path(root) / self.path.stem

    def loop(self, steps: int) -> LoopConfig:
        """Return the loop's recipe at `steps`, with the mixture's stages as its stages."""
        stages: list[Stage] = []
        start = 0
        for stage in self.stages:
            end = steps if stage.until is None else int(stage.until)
            stages.append(Stage(stage.name, start, end))
            start = end
        warmup = self.warmup if self.warmup is not None else max(1, min(1900, steps // 10))
        return LoopConfig(
            steps=steps,
            stages=tuple(stages),
            warmup_steps=warmup,
            tokens_per_step=self.tokens_per_step,
            eval_batches=self.eval_batches,
        )

    def probes(self, steps: int) -> tuple[int, int]:
        """Return how many held-out windows the run asks for, and where its last eval starts."""
        points = steps // self.eval_every if self.eval_every else 0
        return (points + 1) * self.eval_batches, points * self.eval_batches


def _count(raw: dict[str, Any], key: str, default: int | None = None) -> Any:
    """Return a non-negative count, or `default` when the key is absent."""
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LoudError(
            f"{key}={value!r} is not a count",
            "a count is a non-negative integer; leave the key out for the default",
        )
    return value


def _rate(raw: dict[str, Any], key: str) -> float | None:
    """Return a positive rate, or None when the config leaves the mode's default alone."""
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise LoudError(
            f"{key}={value!r} is not a rate",
            "a rate is a positive number; leave the key out to use the mode's default",
        )
    return float(value)


def fresh(recipe: RunRecipe, directory: Path) -> None:
    """Create the run directory and write the recipe it belongs to; refuse to overwrite."""
    journal = directory / "run.jsonl"
    if journal.exists() and Journal.open(journal).last("step") is not None:
        raise LoudError(
            f"{directory} already holds a run",
            "--resume it, or point --runs at another directory: a run directory is one "
            "journal, and a second run would append to the first",
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.toml").write_text(recipe.text, encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps(recipe.canonical(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def refuse_drift(recipe: RunRecipe, directory: Path) -> None:
    """Refuse a `--resume` whose config is not the one the run was started with."""
    stored = directory / "config.json"
    if not stored.exists():
        raise LoudError(
            f"no run at {directory}",
            "start it without --resume first: `--resume` continues a run whose journal "
            "and config are already on disk",
        )
    changed_keys = changed(json.loads(stored.read_text(encoding="utf-8")), recipe.canonical())
    journaled = Journal.open(directory / "run.jsonl").last("start") or {}
    if journaled.get("config_hash") not in (None, recipe.digest) and not changed_keys:
        changed_keys = ["config_hash"]
    if changed_keys:
        raise LoudError(
            f"--resume would change the config: {', '.join(changed_keys)}",
            f"resume {directory / 'config.toml'} instead; a run is its recipe, and a "
            "changed rate or cadence under the same journal is a different run with the "
            "same step number",
        )


def changed(before: Any, now: Any, prefix: str = "") -> list[str]:
    """Return the dotted paths at which two recipes differ; none when they are identical."""
    if isinstance(before, dict) and isinstance(now, dict):
        return [
            name
            for key in sorted(set(before) | set(now))
            for name in changed(before.get(key), now.get(key), f"{prefix}{key}.")
        ]
    if before != now:
        return [prefix.removesuffix(".")]
    return []
