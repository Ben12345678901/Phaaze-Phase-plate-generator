"""Focal-plane intensity targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .optics import (
    bilinear_resample,
    normalise_intensity,
    resample_intensity_to_grid,
)


TARGET_ALIASES = {
    "supergaussian": "supergaussian",
    "super-gaussian": "supergaussian",
    "super_gaussian": "supergaussian",
    "sg": "supergaussian",
    "striped_supergaussian": "striped_supergaussian",
    "striped-supergaussian": "striped_supergaussian",
    "modulated_supergaussian": "striped_supergaussian",
    "tophat": "striped_tophat",
    "top-hat": "striped_tophat",
    "striped_tophat": "striped_tophat",
    "striped-top-hat": "striped_tophat",
    "array": "array",
    "file": "array",
}


def normalise_target_name(name: str) -> str:
    try:
        return TARGET_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "target profile must be supergaussian, striped_supergaussian, "
            "striped_tophat, or array"
        ) from exc


def _stripe_coordinate(
    X_m: np.ndarray, Y_m: np.ndarray, axis: str
) -> np.ndarray:
    return X_m if axis == "x" else Y_m


def build_target_intensity(
    X_m: np.ndarray,
    Y_m: np.ndarray,
    *,
    profile: str,
    diameter_m: float,
    supergaussian_order: float,
    stripes: int,
    modulation_depth: float,
    stripe_axis: str,
    array_path: Path | None = None,
    array_pixel_size_m: float | None = None,
) -> np.ndarray:
    """Build a peak-normalized intensity target on a physical focal grid."""
    profile = normalise_target_name(profile)
    if profile == "array":
        if array_path is None:
            raise ValueError("target.array_path is required for array profile")
        intensity = np.asarray(np.load(array_path), dtype=float).squeeze()
        if intensity.ndim != 2 or not np.isfinite(intensity).all():
            raise ValueError("target array must be a finite 2-D NumPy array")
        if array_pixel_size_m is None:
            intensity = bilinear_resample(intensity, X_m.shape)
        else:
            target_pitch_m = float(np.abs(X_m[0, 1] - X_m[0, 0]))
            intensity = resample_intensity_to_grid(
                intensity,
                X_m.shape,
                array_pixel_size_m,
                target_pitch_m,
            )
        return normalise_intensity(intensity)

    radius_m = diameter_m / 2
    radius = np.hypot(X_m, Y_m)
    if profile in {"supergaussian", "striped_supergaussian"}:
        intensity = np.exp(
            -(radius / radius_m) ** (2 * supergaussian_order)
        )
        if profile == "striped_supergaussian":
            coordinate = _stripe_coordinate(X_m, Y_m, stripe_axis)
            period_m = diameter_m / stripes
            modulation = 1 + modulation_depth * np.cos(
                2 * np.pi * coordinate / period_m
            )
            intensity *= modulation
        return normalise_intensity(intensity)

    # Exactly `stripes` bright bands and `stripes - 1` dark gaps are formed
    # across the diameter, then clipped by the circular top-hat.
    coordinate = _stripe_coordinate(X_m, Y_m, stripe_axis)
    total_bands = 2 * stripes - 1
    band_width_m = diameter_m / total_bands
    band_index = np.floor(
        (coordinate + radius_m) / band_width_m
    ).astype(int)
    band_index = np.clip(band_index, 0, total_bands - 1)
    bright = 1 + modulation_depth
    dark = 1 - modulation_depth
    levels = np.where(band_index % 2 == 0, bright, dark)
    square_mask = (
        (np.abs(X_m) <= radius_m) & (np.abs(Y_m) <= radius_m)
    )
    intensity = levels * square_mask * (radius <= radius_m)
    return normalise_intensity(intensity)


def build_constraint_mask(
    X_m: np.ndarray,
    Y_m: np.ndarray,
    target_diameter_m: float,
    dark_region_factor: float | None,
) -> np.ndarray:
    if dark_region_factor is None:
        return np.ones(X_m.shape, dtype=bool)
    half_width = dark_region_factor * target_diameter_m / 2
    return (np.abs(X_m) <= half_width) & (np.abs(Y_m) <= half_width)
