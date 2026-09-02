"""Classic stochastic zero/pi random phase plate."""

from __future__ import annotations

import numpy as np

from .base import PhaseMode


class RandomBinaryMode(PhaseMode):
    """Generate a stochastic 0/π map without target-driven phase retrieval.

    The optional manufacturing projection controls its physical cell geometry.
    This is kept separate from an optimized two-level plate because a random
    phase plate and a GS-designed binary kinoform are not the same optic.
    """

    canonical_name = "random_binary"
    skip_phase_retrieval = True

    def __init__(self, *, pi_fraction: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pi_fraction = pi_fraction

    def initialize(self) -> np.ndarray:
        phase = (
            self.rng.random(self.pupil.shape) < self.pi_fraction
        ).astype(float) * np.pi
        return phase * self.pupil

