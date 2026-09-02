"""Common Gerchberg-Saxton engine for every phase representation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..modes.base import ModeResult, PhaseMode
from ..optics import (
    focal_metrics,
    forward_for_gs,
    inverse_for_gs,
    normalise_intensity,
    pad_center,
)

ProgressCallback = Callable[
    [int, int, dict[str, float]], None
]
PhaseProjector = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass
class GSResult:
    mode_result: ModeResult
    focal_field: np.ndarray
    focal_intensity: np.ndarray
    history: list[dict[str, float]]


def _warmup_mask(
    shape: tuple[int, int], cutoff_fraction: float
) -> np.ndarray:
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(X, Y)
    radius /= radius.max(initial=1)
    return radius <= cutoff_fraction


def _prepare_initial_phase(
    initial_phase_rad: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    phase = np.asarray(initial_phase_rad, dtype=float).squeeze()
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("initial phase must be a finite 2-D array")
    if phase.shape != target_shape:
        phase = pad_center(phase, target_shape)
    return np.angle(np.exp(1j * phase))


def _apply_phase_projector(
    phase_rad: np.ndarray,
    projector: PhaseProjector | None,
    expected_shape: tuple[int, int],
    iteration: int,
    total_iterations: int,
) -> np.ndarray:
    if projector is None:
        return phase_rad
    projected = np.asarray(
        projector(phase_rad, iteration, total_iterations), dtype=float
    )
    if projected.shape != expected_shape or not np.isfinite(projected).all():
        raise ValueError(
            "phase projector must return a finite array with the GS shape"
        )
    return projected


def gerchberg_saxton(
    input_amplitude: np.ndarray,
    target_intensity: np.ndarray,
    constraint_mask: np.ndarray,
    mode: PhaseMode,
    *,
    pixel_size_m: float,
    wavelength_m: float,
    focal_length_m: float,
    propagation_model: str,
    propagation_distance_m: float,
    iterations: int,
    warmup_fraction: float,
    warmup_cutoff_fraction: float,
    target_ramp: bool,
    initial_phase_rad: np.ndarray | None = None,
    phase_projector: PhaseProjector | None = None,
    progress_callback: ProgressCallback | None = None,
) -> GSResult:
    """Retrieve a phase while delegating representation to ``mode``.

    The two plane constraints are explicit: target *amplitude* is the square
    root of the requested target intensity, while the original input amplitude
    is restored before every forward propagation. This is the standard
    Gerchberg-Saxton alternating-projection structure.
    """
    amplitude = np.asarray(input_amplitude, dtype=float)
    target = normalise_intensity(target_intensity)
    mask = np.asarray(constraint_mask, dtype=bool)
    if amplitude.shape != target.shape or target.shape != mask.shape:
        raise ValueError(
            "input amplitude, target intensity, and constraint mask must match"
        )
    if np.any(amplitude < 0) or not np.isfinite(amplitude).all():
        raise ValueError("input amplitude must be finite and non-negative")
    if not mask.any():
        raise ValueError("constraint mask contains no constrained samples")

    phase = mode.initialize()
    if initial_phase_rad is not None:
        imported = _prepare_initial_phase(
            initial_phase_rad, amplitude.shape
        )
        phase = mode.initialize_from(imported)
    phase = _apply_phase_projector(
        phase,
        phase_projector,
        amplitude.shape,
        0,
        iterations,
    )

    if mode.skip_phase_retrieval:
        result = mode.finalize(phase)
        focal_field = forward_for_gs(
            amplitude * np.exp(1j * result.wrapped_phase_rad),
            pixel_size_m,
            wavelength_m,
            focal_length_m,
            propagation_model,
            propagation_distance_m,
        )
        return GSResult(
            mode_result=result,
            focal_field=focal_field,
            focal_intensity=np.abs(focal_field) ** 2,
            history=[],
        )

    target_amplitude = np.sqrt(target)
    warmup_iterations = int(round(iterations * warmup_fraction))
    warmup = _warmup_mask(
        target.shape, warmup_cutoff_fraction
    )
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        previous_phase = phase
        near_field = amplitude * np.exp(1j * phase)
        far_field = forward_for_gs(
            near_field,
            pixel_size_m,
            wavelength_m,
            focal_length_m,
            propagation_model,
            propagation_distance_m,
        )
        current_amplitude = np.abs(far_field)
        target_power = float(np.sum(target_amplitude[mask] ** 2))
        current_power = float(np.sum(current_amplitude[mask] ** 2))
        scale = (
            np.sqrt(current_power / target_power)
            if target_power > 0
            else 1.0
        )
        scaled_target = scale * target_amplitude

        if mode.requires_adiabatic_target and target_ramp:
            alpha = (iteration + 1) / iterations
            constrained = (
                (1 - alpha) * current_amplitude
                + alpha * scaled_target
            )
            next_amplitude = np.where(
                mask, constrained, current_amplitude
            )
        elif mask.all():
            next_amplitude = target_amplitude
        else:
            next_amplitude = np.where(
                mask, scaled_target, current_amplitude
            )

        constrained_far_field = next_amplitude * np.exp(
            1j * np.angle(far_field)
        )
        # The legacy hard central warm-up is retained for wrapped and stepped
        # modes. It is intentionally disabled for continuous reliefs because
        # hard zeros create phase singularities.
        if (
            iteration < warmup_iterations
            and not mode.requires_adiabatic_target
        ):
            constrained_far_field *= warmup

        candidate_field = inverse_for_gs(
            constrained_far_field,
            pixel_size_m,
            wavelength_m,
            focal_length_m,
            propagation_model,
            propagation_distance_m,
        )
        candidate_phase = _apply_phase_projector(
            np.angle(candidate_field),
            phase_projector,
            amplitude.shape,
            iteration,
            iterations,
        )
        phase = mode.project(
            candidate_phase, iteration, iterations
        )

        constrained_test = normalise_intensity(
            np.abs(far_field) ** 2
        )
        metrics = focal_metrics(
            constrained_test[mask], target[mask]
        )
        phase_step = np.angle(
            np.exp(1j * (phase - previous_phase))
        )
        iteration_record = {
            "iteration": float(iteration + 1),
            "pcc": metrics["pcc"],
            "relative_nrmse": metrics["relative_nrmse"],
            "phase_step_rms_rad": float(
                np.sqrt(np.mean(phase_step[mode.pupil] ** 2))
            ),
        }
        history.append(iteration_record)
        if progress_callback is not None:
            progress_callback(
                iteration + 1, iterations, iteration_record
            )

    mode_result = mode.finalize(phase)
    focal_field = forward_for_gs(
        amplitude * np.exp(1j * mode_result.wrapped_phase_rad),
        pixel_size_m,
        wavelength_m,
        focal_length_m,
        propagation_model,
        propagation_distance_m,
    )
    return GSResult(
        mode_result=mode_result,
        focal_field=focal_field,
        focal_intensity=np.abs(focal_field) ** 2,
        history=history,
    )
