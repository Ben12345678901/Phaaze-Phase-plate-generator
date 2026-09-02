"""Unquantized wrapped phase-only kinoform mode."""

from __future__ import annotations

from .base import PhaseMode


class ContinuousWrappedMode(PhaseMode):
    """An unconstrained phase-only map, not a guaranteed smooth relief."""

    canonical_name = "continuous_wrapped"

