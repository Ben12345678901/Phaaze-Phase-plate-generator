"""Strictly continuous physical surface-relief mode."""

from __future__ import annotations

import numpy as np

from .base import ModeResult, PhaseMode
from .surface import (
    correlation_fwhm_to_sigma_px,
    filter_surface,
    gaussian_transfer,
    lift_near_surface,
    normalize_surface_rms,
    unwrap_surface_2d,
)


class ContinuousReliefMode(PhaseMode):
    """Track one continuous unwrapped phase sheet throughout retrieval.

    This mode is the modular replacement for the legacy "true continuous"
    option. The target constraint is introduced adiabatically, following the
    continuity-preserving design principle described by Dixit et al. (1996).
    """

    canonical_name = "continuous_relief"
    requires_adiabatic_target = True

    def __init__(
        self,
        *,
        correlation_length_m: float,
        initial_rms_rad: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        sigma_px = correlation_fwhm_to_sigma_px(
            correlation_length_m, self.pixel_size_m
        )
        self.transfer = gaussian_transfer(self.pupil.shape, sigma_px)
        self.weights = np.fft.ifft2(
            np.fft.fft2(self.pupil.astype(float)) * self.transfer
        ).real
        self.initial_rms_rad = initial_rms_rad
        self.surface_rad: np.ndarray | None = None

    def initialize(self) -> np.ndarray:
        seed = self.rng.standard_normal(self.pupil.shape)
        seed = filter_surface(
            seed, self.transfer, self.pupil, self.weights
        )
        self.surface_rad = normalize_surface_rms(
            seed, self.pupil, self.initial_rms_rad
        )
        return np.angle(np.exp(1j * self.surface_rad)) * self.pupil

    def initialize_from(self, wrapped_phase_rad: np.ndarray) -> np.ndarray:
        seed = unwrap_surface_2d(wrapped_phase_rad, self.pupil)
        seed = filter_surface(
            seed, self.transfer, self.pupil, self.weights
        )
        self.surface_rad = normalize_surface_rms(
            seed, self.pupil, self.initial_rms_rad
        )
        return np.angle(np.exp(1j * self.surface_rad)) * self.pupil

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
        lifted = np.where(self.pupil, lifted, 0.0)
        self.surface_rad = np.where(
            self.pupil, lifted - np.min(lifted[self.pupil]), 0.0
        )
        return np.angle(np.exp(1j * self.surface_rad)) * self.pupil

    def finalize(self, wrapped_phase_rad: np.ndarray) -> ModeResult:
        if self.surface_rad is None:
            raise RuntimeError("mode did not produce a continuous surface")
        return ModeResult(
            wrapped_phase_rad=np.mod(self.surface_rad, 2 * np.pi)
            * self.pupil,
            unwrapped_phase_rad=self.surface_rad.copy(),
        )
