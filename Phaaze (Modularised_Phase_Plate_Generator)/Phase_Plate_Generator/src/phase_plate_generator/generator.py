"""High-level input-deck-driven phase-plate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import gerchberg_saxton
from .beams import build_input_amplitude
from .config import ExperimentDeck
from .manufacturing import (
    ManufacturedPhase,
    build_element_grid,
    project_to_elements,
    round_binary_boundaries,
)
from .modes import build_mode, normalise_mode_name
from .modes.surface import (
    estimate_correlation_fwhm_m,
    phase_to_thickness_m,
)
from .optics import (
    centered_axis,
    focal_metrics,
    make_pupil,
    normalise_intensity,
    normalise_propagation_model,
    pad_center,
    propagate_phase_plate,
    resample_phase_to_shape,
    resolve_propagation_samples,
)
from .progress import TerminalProgress
from .targets import build_constraint_mask, build_target_intensity


@dataclass
class GenerationResult:
    input_amplitude: np.ndarray
    pre_manufacturing_phase_rad: np.ndarray
    pre_manufacturing_unwrapped_phase_rad: np.ndarray | None
    pre_manufacturing_thickness_m: np.ndarray
    wrapped_phase_rad: np.ndarray
    unwrapped_phase_rad: np.ndarray | None
    thickness_m: np.ndarray
    pre_manufacturing_focal_intensity: np.ndarray
    focal_intensity: np.ndarray
    target_intensity: np.ndarray
    focal_x_m: np.ndarray
    focal_y_m: np.ndarray
    pupil: np.ndarray
    manufacturing_element_labels: np.ndarray
    history: list[dict[str, float]]
    metadata: dict[str, Any]
    output_directory: Path | None


def _load_initial_phase(
    path: Path | None,
    *,
    plate_samples: int,
    propagation_samples: int,
) -> np.ndarray | None:
    if path is None:
        return None
    phase = np.asarray(np.load(path), dtype=float).squeeze()
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("algorithm.initial_phase_path must hold a finite 2-D array")
    if phase.shape == (propagation_samples, propagation_samples):
        return phase
    if phase.shape != (plate_samples, plate_samples):
        phase = resample_phase_to_shape(
            phase, (plate_samples, plate_samples)
        )
    return pad_center(
        phase, (propagation_samples, propagation_samples)
    )


def _crop_physical_plate(
    array: np.ndarray, plate_samples: int
) -> np.ndarray:
    start = (array.shape[0] - plate_samples) // 2
    return array[start : start + plate_samples, start : start + plate_samples]


def _pad_physical_plate(
    array: np.ndarray,
    target_shape: tuple[int, int],
    *,
    fill_value: float | int,
) -> np.ndarray:
    """Center a physical-plate array without changing its fill semantics."""
    source = np.asarray(array)
    if source.ndim != 2 or any(
        source_size > target_size
        for source_size, target_size in zip(source.shape, target_shape)
    ):
        raise ValueError("physical plate must be 2-D and fit the target shape")
    result = np.full(target_shape, fill_value, dtype=source.dtype)
    y0 = (target_shape[0] - source.shape[0]) // 2
    x0 = (target_shape[1] - source.shape[1]) // 2
    result[y0 : y0 + source.shape[0], x0 : x0 + source.shape[1]] = source
    return result


def _mode_levels(deck: ExperimentDeck, mode_name: str) -> int | None:
    if mode_name == "multilevel":
        return deck.phase_plate.levels
    if mode_name == "random_binary":
        return 2
    return None


def generate(
    deck: ExperimentDeck, *, save: bool | None = None
) -> GenerationResult:
    """Generate and optionally save a phase plate from a validated deck."""
    deck.validate()
    model = normalise_propagation_model(deck.optics.propagation_model)
    distance_m = (
        deck.optics.focal_length_m
        if model == "fraunhofer"
        else deck.observation_distance_m
    )
    sampling = resolve_propagation_samples(
        deck.grid.plate_pixels,
        deck.grid.plate_size_m,
        deck.optics.wavelength_m,
        distance_m,
        pad_factor=deck.grid.propagation_pad_factor,
        output_pixel_size_m=deck.grid.output_pixel_size_m,
        maximum_samples=deck.grid.maximum_propagation_samples,
    )
    plate_samples = deck.grid.plate_pixels
    propagation_samples = int(sampling["propagation_samples"])
    dx_plate_m = float(sampling["plate_pixel_size_m"])

    plate_amplitude, plate_pupil, _ = build_input_amplitude(
        plate_samples,
        deck.grid.plate_size_m,
        deck.grid.aperture_shape,
        profile=deck.beam.profile,
        fill_factor=deck.beam.fill_factor,
        supergaussian_order=deck.beam.supergaussian_order,
        array_path=deck.beam.array_path,
    )
    target_shape = (propagation_samples, propagation_samples)
    input_amplitude = pad_center(plate_amplitude, target_shape)
    pupil = pad_center(
        plate_pupil.astype(float), target_shape
    ).astype(bool)

    focal_axis_m = centered_axis(
        propagation_samples, float(sampling["output_pixel_size_m"])
    )
    Xf_m, Yf_m = np.meshgrid(
        focal_axis_m, focal_axis_m, indexing="xy"
    )
    target = build_target_intensity(
        Xf_m,
        Yf_m,
        profile=deck.target.profile,
        diameter_m=deck.target.diameter_m,
        supergaussian_order=deck.target.supergaussian_order,
        stripes=deck.target.stripes,
        modulation_depth=deck.target.modulation_depth,
        stripe_axis=deck.target.stripe_axis,
        array_path=deck.target.array_path,
        array_pixel_size_m=deck.target.array_pixel_size_m,
    )
    constraint_mask = build_constraint_mask(
        Xf_m,
        Yf_m,
        deck.target.diameter_m,
        deck.target.dark_region_factor,
    )

    default_feature_m = (
        deck.optics.wavelength_m
        * distance_m
        / deck.target.diameter_m
    )
    mode_name = normalise_mode_name(deck.phase_plate.mode)
    levels = _mode_levels(deck, mode_name)
    requested_pitch_m = (
        default_feature_m
        if deck.manufacturing.element_pitch_m is None
        else deck.manufacturing.element_pitch_m
    )
    element_grid_start_iteration = int(
        round(
            deck.algorithm.iterations
            * deck.manufacturing.element_grid_start_fraction
        )
    )
    rng = np.random.default_rng(deck.algorithm.random_seed)
    iteration_grid = None
    binary_group_labels = None
    if deck.manufacturing.iterate_on_element_grid:
        iteration_grid = build_element_grid(
            (plate_samples, plate_samples),
            valid_mask=plate_pupil,
            plate_size_m=deck.grid.plate_size_m,
            geometry=deck.manufacturing.element_geometry,
            requested_pitch_m=requested_pitch_m,
        )
        binary_group_labels = _pad_physical_plate(
            iteration_grid.labels,
            target_shape,
            fill_value=-1,
        )
    mode = build_mode(
        deck.phase_plate,
        pupil=pupil,
        pixel_size_m=dx_plate_m,
        rng=rng,
        default_correlation_length_m=default_feature_m,
        input_amplitude=input_amplitude,
        binary_group_labels=binary_group_labels,
        binary_group_start_iteration=element_grid_start_iteration,
    )
    initial_phase = _load_initial_phase(
        deck.algorithm.initial_phase_path,
        plate_samples=plate_samples,
        propagation_samples=propagation_samples,
    )
    progress = None
    if deck.algorithm.show_progress:
        progress_label = (
            f"{deck.manufacturing.element_geometry.capitalize()}-grid "
            "GS iterations"
            if deck.manufacturing.iterate_on_element_grid
            else "GS iterations"
        )
        progress = TerminalProgress(
            deck.algorithm.iterations,
            updates=deck.algorithm.progress_updates,
            label=progress_label,
        )
        if mode.skip_phase_retrieval:
            progress.skipped(
                f"{normalise_mode_name(deck.phase_plate.mode)} mode"
            )
        else:
            progress.start()

    phase_projector = None
    if iteration_grid is not None:
        def phase_projector(
            phase_rad: np.ndarray,
            iteration: int,
            total_iterations: int,
        ) -> np.ndarray:
            del total_iterations
            if iteration < element_grid_start_iteration:
                return phase_rad
            physical_phase = _crop_physical_plate(
                phase_rad, plate_samples
            )
            element_phase, _ = iteration_grid.project_wrapped(
                physical_phase,
                levels=None,
            )
            return pad_center(element_phase, target_shape)

    gs_result = gerchberg_saxton(
        input_amplitude,
        target,
        constraint_mask,
        mode,
        pixel_size_m=dx_plate_m,
        wavelength_m=deck.optics.wavelength_m,
        focal_length_m=deck.optics.focal_length_m,
        propagation_model=model,
        propagation_distance_m=distance_m,
        iterations=deck.algorithm.iterations,
        warmup_fraction=deck.algorithm.warmup_fraction,
        warmup_cutoff_fraction=(
            deck.algorithm.warmup_cutoff_fraction
        ),
        target_ramp=deck.algorithm.target_ramp,
        initial_phase_rad=initial_phase,
        phase_projector=phase_projector,
        progress_callback=(
            None if progress is None else progress.update
        ),
    )

    pre_manufacturing_plate = _crop_physical_plate(
        gs_result.mode_result.wrapped_phase_rad, plate_samples
    )
    unwrapped_plate = (
        None
        if gs_result.mode_result.unwrapped_phase_rad is None
        else _crop_physical_plate(
            gs_result.mode_result.unwrapped_phase_rad, plate_samples
        )
    )
    manufactured: ManufacturedPhase = project_to_elements(
        pre_manufacturing_plate,
        unwrapped_phase_rad=unwrapped_plate,
        valid_mask=plate_pupil,
        plate_size_m=deck.grid.plate_size_m,
        geometry=deck.manufacturing.element_geometry,
        requested_pitch_m=requested_pitch_m,
        levels=levels,
        binarization_method=deck.phase_plate.binarization_method,
        amplitude_weights=plate_amplitude,
    )
    if deck.manufacturing.boundary_rounding_sigma_px > 0:
        if levels != 2:
            raise ValueError(
                "manufacturing.boundary_rounding_sigma_px is only valid for "
                "binary plates"
            )
        manufactured.wrapped_phase_rad = round_binary_boundaries(
            manufactured.wrapped_phase_rad,
            deck.manufacturing.boundary_rounding_sigma_px,
            valid_mask=plate_pupil,
            binarization_method=deck.phase_plate.binarization_method,
            amplitude_weights=plate_amplitude,
        ) * plate_pupil

    # Every post-retrieval manufacturing operation is re-propagated. Metrics
    # therefore describe the actual exported optic rather than the fine-grid
    # design that preceded it.
    manufactured_field = pad_center(
        plate_amplitude
        * np.exp(1j * manufactured.wrapped_phase_rad),
        target_shape,
    )
    focal_intensity, Xf_m, Yf_m, _ = propagate_phase_plate(
        manufactured_field,
        dx_plate_m,
        deck.optics.wavelength_m,
        deck.optics.focal_length_m,
        model,
        distance_m,
    )
    focal_intensity = normalise_intensity(focal_intensity)
    pre_manufacturing_focal_intensity = normalise_intensity(
        gs_result.focal_intensity
    )
    pre_manufacturing_metrics = focal_metrics(
        pre_manufacturing_focal_intensity[constraint_mask],
        target[constraint_mask],
    )
    final_metrics = focal_metrics(
        focal_intensity[constraint_mask], target[constraint_mask]
    )
    thickness_source = (
        manufactured.wrapped_phase_rad
        if manufactured.unwrapped_phase_rad is None
        else manufactured.unwrapped_phase_rad
    )
    pre_manufacturing_thickness_source = (
        pre_manufacturing_plate
        if unwrapped_plate is None
        else unwrapped_plate
    )
    pre_manufacturing_thickness_m = phase_to_thickness_m(
        pre_manufacturing_thickness_source,
        deck.optics.wavelength_m,
        deck.optics.refractive_index,
    )
    thickness_m = phase_to_thickness_m(
        thickness_source,
        deck.optics.wavelength_m,
        deck.optics.refractive_index,
    )

    correlation_fwhm_m = None
    if manufactured.unwrapped_phase_rad is not None:
        correlation_fwhm_m = estimate_correlation_fwhm_m(
            manufactured.unwrapped_phase_rad,
            plate_pupil,
            dx_plate_m,
        )
    metadata: dict[str, Any] = {
        "package_version": "0.1.0",
        "facility": deck.facility.name,
        "phase_mode": mode_name,
        "propagation_model": model,
        "propagation_distance_m": distance_m,
        "reciprocal_feature_scale_m": default_feature_m,
        "sampling": sampling,
        "pre_manufacturing_metrics": pre_manufacturing_metrics,
        "metrics": final_metrics,
        "manufacturing": {
            "element_geometry": deck.manufacturing.element_geometry,
            "iterate_on_element_grid": (
                deck.manufacturing.iterate_on_element_grid
            ),
            "element_grid_start_fraction": (
                deck.manufacturing.element_grid_start_fraction
            ),
            "element_grid_start_iteration": (
                element_grid_start_iteration + 1
                if (
                    deck.manufacturing.iterate_on_element_grid
                    and element_grid_start_iteration
                    < deck.algorithm.iterations
                )
                else None
            ),
            "element_count": manufactured.element_count,
            "requested_pitch_m": requested_pitch_m,
            "realized_pitch_m": manufactured.realized_pitch_m,
            "levels": levels,
            "binarization_method": (
                mode.binarization_method
                if mode_name == "multilevel" and levels == 2
                else None
            ),
        },
        "surface": {
            "configured_correlation_length_m": (
                deck.phase_plate.correlation_length_m
            ),
            "measured_correlation_fwhm_m": correlation_fwhm_m,
            "phase_rms_rad": (
                float(
                    np.std(
                        manufactured.unwrapped_phase_rad[plate_pupil]
                    )
                )
                if manufactured.unwrapped_phase_rad is not None
                else None
            ),
            "thickness_range_m": float(
                np.ptp(thickness_m[plate_pupil])
            ),
        },
    }

    result = GenerationResult(
        input_amplitude=plate_amplitude,
        pre_manufacturing_phase_rad=pre_manufacturing_plate,
        pre_manufacturing_unwrapped_phase_rad=unwrapped_plate,
        pre_manufacturing_thickness_m=(
            pre_manufacturing_thickness_m
        ),
        wrapped_phase_rad=manufactured.wrapped_phase_rad,
        unwrapped_phase_rad=manufactured.unwrapped_phase_rad,
        thickness_m=thickness_m,
        pre_manufacturing_focal_intensity=(
            pre_manufacturing_focal_intensity
        ),
        focal_intensity=focal_intensity,
        target_intensity=target,
        focal_x_m=Xf_m,
        focal_y_m=Yf_m,
        pupil=plate_pupil,
        manufacturing_element_labels=manufactured.element_labels,
        history=gs_result.history,
        metadata=metadata,
        output_directory=None,
    )
    should_save = (
        deck.output.save_arrays or deck.output.save_metadata
        if save is None
        else save
    )
    if should_save:
        from .outputs import save_result

        result.output_directory = save_result(result, deck)
    return result
