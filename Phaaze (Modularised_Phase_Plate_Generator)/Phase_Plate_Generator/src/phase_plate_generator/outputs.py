"""Stable output artifact names for generated phase plates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import grid_phase_stem, ideal_kinoform_stem
from .config import ExperimentDeck
from .generator import GenerationResult
from .output_layout import ensure_run_folders


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_result(
    result: GenerationResult, deck: ExperimentDeck
) -> Path:
    output = Path(deck.output.directory).resolve()
    folders = ensure_run_folders(output)

    if deck.output.save_arrays:
        mode = str(result.metadata["phase_mode"])
        ideal_stem = ideal_kinoform_stem(mode)
        grid_stem = grid_phase_stem(
            mode, deck.manufacturing.element_geometry
        )

        np.save(folders.arrays / "Input_beam.npy", result.input_amplitude)
        np.save(
            folders.arrays / f"{ideal_stem}.npy",
            result.pre_manufacturing_phase_rad,
        )
        if result.pre_manufacturing_unwrapped_phase_rad is not None:
            np.save(
                folders.arrays
                / f"{ideal_stem}_unwrapped_phase_rad.npy",
                result.pre_manufacturing_unwrapped_phase_rad,
            )
        np.save(
            folders.arrays / f"{ideal_stem}_thickness_m.npy",
            result.pre_manufacturing_thickness_m,
        )
        np.save(
            folders.arrays / f"{ideal_stem}_focal_spot.npy",
            result.pre_manufacturing_focal_intensity,
        )

        if grid_stem is not None:
            np.save(
                folders.arrays / f"{grid_stem}.npy",
                result.wrapped_phase_rad,
            )
            if result.unwrapped_phase_rad is not None:
                np.save(
                    folders.arrays
                    / f"{grid_stem}_unwrapped_phase_rad.npy",
                    result.unwrapped_phase_rad,
                )
            np.save(
                folders.arrays / f"{grid_stem}_thickness_m.npy",
                result.thickness_m,
            )
            np.save(
                folders.arrays / f"{grid_stem}_focal_spot.npy",
                result.focal_intensity,
            )
            np.save(
                folders.arrays / f"{grid_stem}_element_labels.npy",
                result.manufacturing_element_labels,
            )
        np.save(
            folders.arrays / "Target_focal_spot_structure.npy",
            result.target_intensity,
        )
        np.save(
            folders.arrays / "Input_beam_pupil_mask.npy",
            result.pupil,
        )
        X_um = result.focal_x_m * 1e6
        Y_um = result.focal_y_m * 1e6
        np.savez(
            folders.arrays / "Focal_spot_coordinates_um.npz",
            X_um=X_um,
            Y_um=Y_um,
        )
        np.save(
            folders.arrays / "Focal_spot_extent_um.npy",
            np.asarray(
                [
                    np.min(X_um),
                    np.max(X_um),
                    np.min(Y_um),
                    np.max(Y_um),
                ],
                dtype=float,
            ),
        )

    if deck.output.save_metadata:
        (folders.json_files / "metadata.json").write_text(
            json.dumps(_json_safe(result.metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        (folders.json_files / "convergence.json").write_text(
            json.dumps(_json_safe(result.history), indent=2) + "\n",
            encoding="utf-8",
        )
        (folders.json_files / "resolved_input_deck.json").write_text(
            json.dumps(_json_safe(deck.as_dict()), indent=2) + "\n",
            encoding="utf-8",
        )
    return output
