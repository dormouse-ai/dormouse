"""The FFN config value: which arm builds, and the bill its width is quoted at.

A value object, not a behaviour: it validates once at construction and hands the
arm to `build()`. It exists because the three new fields (`ffn_act`, `ffn_dict`,
`ffn_moa_sides`) and the width rule are exactly the places a config can be
wrong, and a wrong one here would be a silent no-op in the model:

* the dictionary is refused for the plain arms (nothing would read it) and
  refused empty or unknown for the mixtures (a mixture of nothing, or of a name
  that does not exist, is not a mixture);
* `sides` is MoA's topology and is refused on the plain arms;
* `la`/`moa` must state the GELU control width and satisfy
  `3·d·d_ffn == 2·d·d_ffn_gelu`. Those arms add `K` and `d·K + 2K` parameters on
  top of the parameter-neutral gated width; without the identity a "gain" can be
  a 50% bigger model wearing the same name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from dormouse_core.errors import LoudError

from .activations import DEFAULT_DICTIONARY, resolve
from .ffn import FFN_SIDES, PROJ_STD, GatedFFN, LaFFN, MoaFFN, neutral_width

#: The arms a config may name. The first two are plain activations; the last
#: two mix the fixed dictionary (`activations.py`).
FFN_ACTS: tuple[str, ...] = ("silu", "situ", "la", "moa")

#: The arms that own a dictionary; the plain arms are one activation each.
MIXTURES: tuple[str, ...] = ("la", "moa")


@dataclass(frozen=True)
class FFNSpec:
    """Which FFN arm a stack builds, and the bill its width is quoted against.

    The fields are the config surface: `act` is `silu | situ | la | moa`,
    `dictionary` is the fixed set `la`/`moa` mix (refused for the plain arms —
    a dictionary that nothing reads is a silent no-op), and `sides` is MoA's
    topology. `d_ffn_gelu` is the GELU control's width: when stated, the bill
    identity `3·d·d_ffn == 2·d·d_ffn_gelu` must hold, and `la`/`moa` must state
    it, because those arms are additions to the parameter-neutral gated width
    and not a licence to grow the model.
    """

    d_model: int
    d_ffn: int
    act: str = "silu"
    dictionary: tuple[str, ...] | None = None
    sides: str | None = None
    d_ffn_gelu: int | None = None

    def __post_init__(self) -> None:
        """Validate the arm, freeze the dictionary, and check the bill."""
        if self.d_model < 1:
            raise LoudError(
                f"d_model={self.d_model} is not a width",
                "pass the residual width the FFN reads",
            )
        if self.d_ffn < 1:
            raise LoudError(
                f"d_ffn={self.d_ffn} is not a width",
                "use a positive width; 2048 is the parameter-neutral one at d_model=832",
            )
        if self.act not in FFN_ACTS:
            raise LoudError(
                f"ffn_act={self.act!r} is not an FFN arm",
                f"use one of: {', '.join(FFN_ACTS)}",
            )
        if self.sides is not None:
            if self.sides not in FFN_SIDES:
                raise LoudError(
                    f"ffn_moa_sides={self.sides!r} is not a topology",
                    f"use one of: {', '.join(FFN_SIDES)}",
                )
            if self.act != "moa":
                raise LoudError(
                    f"ffn_moa_sides={self.sides!r} was set with ffn_act={self.act!r}",
                    "sides is MoA's topology; the plain arms are one-sided",
                )
        if self.act in MIXTURES:
            names = DEFAULT_DICTIONARY if self.dictionary is None else self.dictionary
            object.__setattr__(self, "dictionary", resolve(names))
        elif self.dictionary is not None:
            raise LoudError(
                f"ffn_dict was set with ffn_act={self.act!r}",
                "the dictionary only feeds la/moa; name one of those, or drop ffn_dict",
            )
        if self.d_ffn_gelu is not None:
            gated, gelu = 3 * self.d_model * self.d_ffn, 2 * self.d_model * self.d_ffn_gelu
            if gated != gelu:
                raise LoudError(
                    f"d_ffn={self.d_ffn} bills 3·{self.d_model}·{self.d_ffn}={gated} "
                    f"against the GELU control's 2·{self.d_model}·{self.d_ffn_gelu}={gelu}",
                    f"pass d_ffn={neutral_width(self.d_ffn_gelu)} (the parameter-neutral width)",
                )
        elif self.act in MIXTURES:
            raise LoudError(
                f"ffn_act={self.act!r} needs the parameter-neutral width",
                "state d_ffn_gelu (3072 at d_ffn=2048) so 3·d·d_ffn == 2·d·d_ffn_gelu"
                " is checked, not assumed",
            )

    @property
    def entries(self) -> tuple[str, ...]:
        """Return the resolved dictionary; empty for the plain arms, which have none."""
        return self.dictionary or ()

    def added_parameters(self) -> int:
        """Return extra parameters per layer: `K` for la, `K + d·K + K` for moa."""
        k = len(self.entries)
        if self.act == "la":
            return k
        if self.act == "moa":
            return self.d_model * k + 2 * k
        return 0

    def build(
        self, *, write_std: float = PROJ_STD, device: torch.device | str | None = None
    ) -> GatedFFN | LaFFN | MoaFFN:
        """Materialise the arm this spec names."""
        if self.act == "la":
            return LaFFN(
                self.d_model,
                self.d_ffn,
                dictionary=self.entries,
                write_std=write_std,
                device=device,
            )
        if self.act == "moa":
            return MoaFFN(
                self.d_model,
                self.d_ffn,
                dictionary=self.entries,
                sides=self.sides or "bi",
                write_std=write_std,
                device=device,
            )
        return GatedFFN(self.d_model, self.d_ffn, act=self.act, write_std=write_std, device=device)
