"""Distributed phase plate with retained surface statistics."""

from __future__ import annotations

import numpy as np

from .continuous_relief import ContinuousReliefMode
from .surface import filter_surface, lift_near_surface, normalize_surface_rms


class DistributedReliefMode(ContinuousReliefMode):
    """Continuous random relief re-correlated during every GS cycle.

    Lin, Kessler, and Lawrence (1995) describe a strictly continuous starting
    DPP followed by a small number of phase-retrieval cycles. This projector
    preserves the legacy generator's additional behavior: it re-imposes a
    requested autocorrelation scale and phase RMS after every cycle. It is
    therefore a project-specific constrained variant, not a line-for-line
    reproduction of that paper.
    """

    canonical_name = "distributed_relief"

    def __init__(
        self,
        *,
        phase_rms_rad: float,
        projection_strength: float,
        **kwargs,
    ) -> None:
        super().__init__(initial_rms_rad=phase_rms_rad, **kwargs)
        self.phase_rms_rad = phase_rms_rad
        self.projection_strength = projection_strength

    def project(
        self,
        wrapped_candidate_rad: np.ndarray,
        iteration: int,
        total_iterations: int,
    ) -> np.ndarray:
        if self.surface_rad is None:
            raise RuntimeError("mode must be initialized before projection")
        lifted = lift_near_surface(
            wrapped_candidate_rad, self.surface_rad
        )
        correlated = filter_surface(
            lifted, self.transfer, self.pupil, self.weights
        )
        correlated = normalize_surface_rms(
            correlated, self.pupil, self.phase_rms_rad
        )
        blended = (
            (1 - self.projection_strength) * lifted
            + self.projection_strength * correlated
        )
        self.surface_rad = normalize_surface_rms(
            blended, self.pupil, self.phase_rms_rad
        )
        return np.angle(np.exp(1j * self.surface_rad)) * self.pupil

