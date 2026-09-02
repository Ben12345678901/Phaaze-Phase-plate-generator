"""Projection from numerical phase pixels to physical plate geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes.multilevel import (
    normalise_binarization_method,
    quantize_phase,
)
from .optics import centered_axis


@dataclass
class ManufacturedPhase:
    wrapped_phase_rad: np.ndarray
    unwrapped_phase_rad: np.ndarray | None
    element_values_rad: np.ndarray
    element_labels: np.ndarray
    element_count: int
    realized_pitch_m: float


@dataclass
class ElementGrid:
    """Precomputed physical-element labels for repeated phase projection."""

    labels: np.ndarray
    valid_mask: np.ndarray
    element_count: int
    realized_pitch_m: float

    def project_wrapped(
        self,
        wrapped_phase_rad: np.ndarray,
        *,
        levels: int | None,
        binarization_method: str = "threshold",
        amplitude_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a cell-constant wrapped raster and its cell values."""
        wrapped = np.asarray(wrapped_phase_rad, dtype=float)
        method = normalise_binarization_method(binarization_method)
        if wrapped.shape != self.labels.shape:
            raise ValueError("phase and element-grid shapes must match")
        if (
            amplitude_weights is not None
            and np.asarray(amplitude_weights).shape != wrapped.shape
        ):
            raise ValueError(
                "amplitude weights and element-grid shapes must match"
            )
        if self.element_count == 0:
            return wrapped.copy(), np.array([], dtype=float)
        cell_values = _circular_cell_means(
            wrapped,
            self.labels,
            self.valid_mask,
            self.element_count,
            levels,
            method,
            amplitude_weights,
        )
        raster = np.zeros_like(wrapped)
        raster[self.valid_mask] = cell_values[
            self.labels[self.valid_mask]
        ]
        return raster, cell_values

    def project_unwrapped(
        self, unwrapped_phase_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an element-averaged unwrapped relief and cell values."""
        surface = np.asarray(unwrapped_phase_rad, dtype=float)
        if surface.shape != self.labels.shape:
            raise ValueError("surface and element-grid shapes must match")
        if self.element_count == 0:
            return surface.copy(), np.array([], dtype=float)
        sample_labels = self.labels[self.valid_mask]
        counts = np.bincount(
            sample_labels, minlength=self.element_count
        )
        cell_surface = (
            np.bincount(
                sample_labels,
                weights=surface[self.valid_mask],
                minlength=self.element_count,
            )
            / np.maximum(counts, 1)
        )
        cell_surface -= np.min(cell_surface)
        raster_surface = np.zeros_like(surface)
        raster_surface[self.valid_mask] = cell_surface[sample_labels]
        return raster_surface, cell_surface


def _square_labels(
    shape: tuple[int, int], elements_across: int
) -> tuple[np.ndarray, int]:
    edges = np.rint(
        np.linspace(0, shape[0], elements_across + 1)
    ).astype(int)
    labels = np.zeros(shape, dtype=np.int32)
    label = 0
    for row in range(elements_across):
        for col in range(elements_across):
            labels[
                edges[row] : edges[row + 1],
                edges[col] : edges[col + 1],
            ] = label
            label += 1
    return labels, label


def _round_hex(
    q_fractional: np.ndarray, r_fractional: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cube_x = np.asarray(q_fractional)
    cube_z = np.asarray(r_fractional)
    cube_y = -cube_x - cube_z
    rx, ry, rz = np.rint(cube_x), np.rint(cube_y), np.rint(cube_z)
    x_error = np.abs(rx - cube_x)
    y_error = np.abs(ry - cube_y)
    z_error = np.abs(rz - cube_z)
    correct_x = (x_error > y_error) & (x_error > z_error)
    correct_y = (~correct_x) & (y_error > z_error)
    correct_z = ~(correct_x | correct_y)
    rx[correct_x] = -ry[correct_x] - rz[correct_x]
    ry[correct_y] = -rx[correct_y] - rz[correct_y]
    rz[correct_z] = -rx[correct_z] - ry[correct_z]
    return rx.astype(np.int32), rz.astype(np.int32)


def _hex_labels(
    shape: tuple[int, int],
    pitch_pixels: float,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    axis = centered_axis(shape[0], 1.0)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    q_fractional = 2 * X / (np.sqrt(3) * pitch_pixels)
    r_fractional = (
        Y / pitch_pixels - X / (np.sqrt(3) * pitch_pixels)
    )
    q_cell, r_cell = _round_hex(q_fractional, r_fractional)
    axial = np.column_stack(
        (q_cell[valid_mask], r_cell[valid_mask])
    )
    _, inverse = np.unique(axial, axis=0, return_inverse=True)
    labels = np.full(shape, -1, dtype=np.int32)
    labels[valid_mask] = inverse
    return labels, int(inverse.max(initial=-1) + 1)


def _circular_cell_means(
    phase_rad: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    cell_count: int,
    levels: int | None,
    binarization_method: str,
    amplitude_weights: np.ndarray | None,
) -> np.ndarray:
    samples = phase_rad[valid_mask]
    sample_labels = labels[valid_mask]
    cosine = np.bincount(
        sample_labels,
        weights=np.cos(samples),
        minlength=cell_count,
    )
    if levels == 2 and binarization_method == "threshold":
        return np.where(cosine >= 0, 0.0, np.pi)
    sine = np.bincount(
        sample_labels,
        weights=np.sin(samples),
        minlength=cell_count,
    )
    means = np.mod(np.arctan2(sine, cosine), 2 * np.pi)
    if levels == 2:
        cell_mask = np.ones(cell_count, dtype=bool)
        if binarization_method == "median":
            balance_weights = np.bincount(
                sample_labels, minlength=cell_count
            ).astype(float)
        else:
            if amplitude_weights is None:
                raise ValueError(
                    "amplitude_weighted binarization requires incident "
                    "field-amplitude weights"
                )
            balance_weights = np.bincount(
                sample_labels,
                weights=np.asarray(amplitude_weights, dtype=float)[valid_mask],
                minlength=cell_count,
            )
        return quantize_phase(
            means,
            levels,
            binarization_method=binarization_method,
            valid_mask=cell_mask,
            amplitude_weights=balance_weights,
        )
    return quantize_phase(means, levels) if levels else means


def build_element_grid(
    shape: tuple[int, int],
    *,
    valid_mask: np.ndarray,
    plate_size_m: float,
    geometry: str,
    requested_pitch_m: float,
) -> ElementGrid:
    """Build reusable square/hexagonal labels for iterative projection."""
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("valid mask and element-grid shapes must match")
    if geometry == "none":
        return ElementGrid(
            labels=np.full(shape, -1, dtype=np.int32),
            valid_mask=mask,
            element_count=0,
            realized_pitch_m=0.0,
        )

    pixel_size_m = plate_size_m / shape[0]
    if geometry == "square":
        elements_across = int(
            np.clip(
                round(plate_size_m / requested_pitch_m),
                1,
                shape[0],
            )
        )
        realized_pitch_m = plate_size_m / elements_across
        labels, cell_count = _square_labels(shape, elements_across)
    elif geometry == "hexagonal":
        realized_pitch_m = max(requested_pitch_m, pixel_size_m)
        labels, cell_count = _hex_labels(
            shape,
            realized_pitch_m / pixel_size_m,
            mask,
        )
    else:
        raise ValueError("geometry must be none, square, or hexagonal")
    labels = np.where(mask, labels, -1).astype(np.int32, copy=False)
    return ElementGrid(
        labels=labels,
        valid_mask=mask,
        element_count=cell_count,
        realized_pitch_m=realized_pitch_m,
    )


def project_to_elements(
    wrapped_phase_rad: np.ndarray,
    *,
    unwrapped_phase_rad: np.ndarray | None,
    valid_mask: np.ndarray,
    plate_size_m: float,
    geometry: str,
    requested_pitch_m: float,
    levels: int | None,
    binarization_method: str = "threshold",
    amplitude_weights: np.ndarray | None = None,
) -> ManufacturedPhase:
    """Average a phase or continuous surface over square/hexagonal cells."""
    wrapped = np.asarray(wrapped_phase_rad, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    grid = build_element_grid(
        wrapped.shape,
        valid_mask=mask,
        plate_size_m=plate_size_m,
        geometry=geometry,
        requested_pitch_m=requested_pitch_m,
    )

    if unwrapped_phase_rad is None:
        raster, cell_values = grid.project_wrapped(
            wrapped,
            levels=levels,
            binarization_method=binarization_method,
            amplitude_weights=amplitude_weights,
        )
        return ManufacturedPhase(
            wrapped_phase_rad=raster,
            unwrapped_phase_rad=None,
            element_values_rad=cell_values,
            element_labels=grid.labels,
            element_count=grid.element_count,
            realized_pitch_m=grid.realized_pitch_m,
        )

    raster_surface, cell_surface = grid.project_unwrapped(
        unwrapped_phase_rad
    )
    return ManufacturedPhase(
        wrapped_phase_rad=np.mod(raster_surface, 2 * np.pi),
        unwrapped_phase_rad=raster_surface,
        element_values_rad=cell_surface,
        element_labels=grid.labels,
        element_count=grid.element_count,
        realized_pitch_m=grid.realized_pitch_m,
    )


def round_binary_boundaries(
    phase_rad: np.ndarray,
    sigma_px: float,
    *,
    valid_mask: np.ndarray | None = None,
    binarization_method: str = "threshold",
    amplitude_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Round binary raster boundaries while preserving π area fraction."""
    phase = np.asarray(phase_rad, dtype=float)
    method = normalise_binarization_method(binarization_method)
    if sigma_px <= 0:
        return np.mod(phase, 2 * np.pi)
    mask = (
        np.ones(phase.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != phase.shape:
        raise ValueError("binary rounding mask must match the phase shape")
    wrapped = np.mod(phase, 2 * np.pi)
    distance_zero = np.minimum(wrapped, 2 * np.pi - wrapped)
    pi_mask = (np.abs(wrapped - np.pi) < distance_zero) & mask
    pi_fraction = float(pi_mask[mask].mean())
    if pi_fraction <= 0 or pi_fraction >= 1:
        return pi_mask.astype(float) * np.pi
    fy = np.fft.fftfreq(phase.shape[0])[:, None]
    fx = np.fft.fftfreq(phase.shape[1])[None, :]
    transfer = np.exp(
        -2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2)
    )
    smoothed = np.fft.ifft2(
        np.fft.fft2(pi_mask.astype(float)) * transfer
    ).real
    if method != "threshold":
        return quantize_phase(
            smoothed,
            2,
            binarization_method=method,
            valid_mask=mask,
            amplitude_weights=amplitude_weights,
        )
    threshold = np.quantile(smoothed[mask], 1 - pi_fraction)
    return (smoothed >= threshold).astype(float) * np.pi * mask
