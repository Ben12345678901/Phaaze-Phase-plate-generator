"""Incident beam profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .optics import bilinear_resample, centered_axis, make_pupil


BEAM_ALIASES = {
    "circular": "circular_gaussian",
    "circle": "circular_gaussian",
    "gaussian": "circular_gaussian",
    "circular_gaussian": "circular_gaussian",
    "square": "square_supergaussian",
    "square_supergaussian": "square_supergaussian",
    "square-supergaussian": "square_supergaussian",
    "array": "array",
    "file": "array",
}


def normalise_beam_name(name: str) -> str:
    try:
        return BEAM_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "beam profile must be circular_gaussian, square_supergaussian, "
            "or array"
        ) from exc


def build_input_amplitude(
    samples: int,
    plate_size_m: float,
    aperture_shape: str,
    *,
    profile: str,
    fill_factor: float,
    supergaussian_order: float,
    array_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a normalized field amplitude on the physical plate grid."""
    profile = normalise_beam_name(profile)
    dx_m = plate_size_m / samples
    axis = centered_axis(samples, dx_m)
    X_m, Y_m = np.meshgrid(axis, axis, indexing="xy")
    pupil = make_pupil(X_m, Y_m, plate_size_m, aperture_shape)
    width_m = fill_factor * plate_size_m / 2

    if profile == "array":
        if array_path is None:
            raise ValueError("beam.array_path is required for array profile")
        amplitude = np.asarray(np.load(array_path), dtype=float).squeeze()
        if amplitude.ndim != 2 or not np.isfinite(amplitude).all():
            raise ValueError("beam array must be a finite 2-D NumPy array")
        amplitude = bilinear_resample(amplitude, (samples, samples))
        amplitude = np.clip(amplitude, 0, None)
        peak = amplitude.max(initial=0)
        if peak <= 0:
            raise ValueError("beam array contains no positive amplitude")
        amplitude /= peak
    elif profile == "circular_gaussian":
        amplitude = np.exp(-(X_m**2 + Y_m**2) / width_m**2)
    else:
        exponent = (
            (np.abs(X_m) / width_m) ** (2 * supergaussian_order)
            + (np.abs(Y_m) / width_m) ** (2 * supergaussian_order)
        )
        amplitude = np.exp(-exponent)

    return amplitude * pupil, pupil, dx_m
