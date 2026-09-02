"""Phase-mode interface used by the common GS engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModeResult:
    wrapped_phase_rad: np.ndarray
    unwrapped_phase_rad: np.ndarray | None = None


class PhaseMode:
    """Stateful projection applied to every near-field GS update."""

    canonical_name = "base"
    requires_adiabatic_target = False
    skip_phase_retrieval = False

    def __init__(
        self,
        *,
        pupil: np.ndarray,
        pixel_size_m: float,
        rng: np.random.Generator,
    ) -> None:
        self.pupil = np.asarray(pupil, dtype=bool)
        self.pixel_size_m = float(pixel_size_m)
        self.rng = rng

    def initialize(self) -> np.ndarray:
        return (
            (np.pi / 2)
            * self.rng.integers(-2, 3, size=self.pupil.shape)
            * self.pupil
        )

    def initialize_from(self, wrapped_phase_rad: np.ndarray) -> np.ndarray:
        """Initialize state from an imported wrapped phase map."""
        phase = np.asarray(wrapped_phase_rad, dtype=float)
        if phase.shape != self.pupil.shape:
            raise ValueError("imported phase and mode pupil shapes must match")
        return np.angle(np.exp(1j * phase)) * self.pupil

    def project(
        self,
        wrapped_candidate_rad: np.ndarray,
        iteration: int,
        total_iterations: int,
    ) -> np.ndarray:
        return np.where(self.pupil, wrapped_candidate_rad, 0.0)

    def finalize(self, wrapped_phase_rad: np.ndarray) -> ModeResult:
        return ModeResult(
            wrapped_phase_rad=np.mod(wrapped_phase_rad, 2 * np.pi)
            * self.pupil
        )
