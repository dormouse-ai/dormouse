"""The one refusal a proof can raise, with the way out attached."""

from __future__ import annotations

from dormouse_core.errors import LoudError


class ProofError(LoudError):
    """A mechanism's contract is missing, captured from our own run, or false."""
