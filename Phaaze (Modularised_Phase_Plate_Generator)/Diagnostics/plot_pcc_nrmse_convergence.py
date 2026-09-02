"""
Plot PCC and NRMSE convergence histories for the RPP, CPP and DPP.

The script reads each convergence.json file, extracts iteration, PCC and
NRMSE arrays, and plots:

    Left:  Pearson correlation coefficient versus iteration
    Right: Normalised root-mean-square error versus iteration

The JSON reader accepts several common key spellings, including:
    pcc, PCC, pearson, pearson_correlation
    nrmse, NRMSE, nmrse, relative_nrmse
    iteration, iterations, iter, step

Outputs
-------
pcc_nrmse_convergence_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER CONTROLS
# =============================================================================

RPP_FOCAL_SPOT = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox"
    r"\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)"
    r"\Validation\Omega_400um\Results\RPP"
    r"\omega_400um_2026-08-03_11-11-29\Arrays"
    r"\Hex_grid_multilevel_focal_spot.npy"
)

RPP_PCC_NMRSE = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox"
    r"\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)"
    r"\Validation\Omega_400um\Results\RPP"
    r"\omega_400um_2026-08-03_11-11-29\json_files"
    r"\convergence.json"
)

CPP_FOCAL_SPOT = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox"
    r"\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)"
    r"\Validation\Omega_400um\Results\CPP"
    r"\omega_400um_2026-08-03_11-11-29\Arrays"
    r"\Hex_grid_multilevel_focal_spot.npy"
)

CPP_PCC_NMRSE = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)\Validation\Omega_400um\Results\CPP\omega_400um_2026-08-03_11-23-17\json_files\convergence.json"
)

DPP_FOCAL_SPOT = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox"
    r"\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)"
    r"\Validation\Omega_400um\Results\DPP"
    r"\omega_400um_2026-08-03_11-11-29\Arrays"
    r"\Hex_grid_multilevel_focal_spot.npy"
)

DPP_PCC_NMRSE = Path(
    r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)\Validation\Omega_400um\Results\DPP\omega_400um_2026-08-03_11-26-23\json_files\convergence.json"
)

# The focal-spot paths are retained above for convenience, but this plot only
# needs the three convergence.json files.

SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_FILE = Path(__file__).with_name(
    "pcc_nrmse_convergence_comparison.png"
)

FIGURE_SIZE = (13, 5.5)
DPI = 300

AXIS_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
LINE_WIDTH = 2.0

# Set to None for automatic limits.
PCC_Y_LIMITS: tuple[float, float] | None = None
NRMSE_Y_LIMITS: tuple[float, float] | None = None


# =============================================================================
# JSON READING
# =============================================================================

ITERATION_KEYS = (
    "iteration",
    "iterations",
    "iter",
    "iters",
    "step",
    "steps",
    "iteration_number",
)

PCC_KEYS = (
    "pcc",
    "PCC",
    "pearson",
    "pearson_correlation",
    "pearson_correlation_coefficient",
    "correlation",
)

NRMSE_KEYS = (
    "nrmse",
    "NRMSE",
    "nmrse",
    "NMRSE",
    "relative_nrmse",
    "normalised_rmse",
    "normalized_rmse",
)


def load_json(path: Path) -> Any:
    """Load a JSON file and provide a clear error if the path is invalid."""
    if not path.exists():
        raise FileNotFoundError(f"Convergence file does not exist:\n{path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalise_key(key: str) -> str:
    """Normalise JSON keys for case-insensitive matching."""
    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_matching_key(mapping: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    """Return the first mapping key matching one of the supplied aliases."""
    normalised_mapping = {
        normalise_key(key): key
        for key in mapping
    }

    for alias in aliases:
        key = normalised_mapping.get(normalise_key(alias))
        if key is not None:
            return key

    return None


def as_1d_float_array(values: Any, label: str) -> np.ndarray:
    """Convert a JSON value into a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float).squeeze()

    if array.ndim == 0:
        array = array.reshape(1)

    if array.ndim != 1:
        raise ValueError(
            f"{label} must be one-dimensional, but found shape {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")

    return array


def extract_from_dictionary(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract convergence arrays from a dictionary-of-arrays structure."""
    pcc_key = find_matching_key(data, PCC_KEYS)
    nrmse_key = find_matching_key(data, NRMSE_KEYS)

    if pcc_key is None or nrmse_key is None:
        return None

    pcc = as_1d_float_array(data[pcc_key], pcc_key)
    nrmse = as_1d_float_array(data[nrmse_key], nrmse_key)

    iteration_key = find_matching_key(data, ITERATION_KEYS)
    if iteration_key is None:
        iterations = np.arange(len(pcc), dtype=int)
    else:
        iterations = as_1d_float_array(data[iteration_key], iteration_key)

    return iterations, pcc, nrmse


def extract_from_record_list(
    records: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract convergence arrays from a list of per-iteration dictionaries."""
    if not records or not all(isinstance(record, dict) for record in records):
        return None

    first_record = records[0]
    pcc_key = find_matching_key(first_record, PCC_KEYS)
    nrmse_key = find_matching_key(first_record, NRMSE_KEYS)
    iteration_key = find_matching_key(first_record, ITERATION_KEYS)

    if pcc_key is None or nrmse_key is None:
        return None

    pcc = np.asarray([record[pcc_key] for record in records], dtype=float)
    nrmse = np.asarray([record[nrmse_key] for record in records], dtype=float)

    if iteration_key is None:
        iterations = np.arange(len(records), dtype=int)
    else:
        iterations = np.asarray(
            [record[iteration_key] for record in records],
            dtype=float,
        )

    return iterations, pcc, nrmse


def recursively_extract(
    data: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Search nested JSON structures for convergence arrays."""
    if isinstance(data, dict):
        direct_result = extract_from_dictionary(data)
        if direct_result is not None:
            return direct_result

        for value in data.values():
            nested_result = recursively_extract(value)
            if nested_result is not None:
                return nested_result

    elif isinstance(data, list):
        record_result = extract_from_record_list(data)
        if record_result is not None:
            return record_result

        for value in data:
            nested_result = recursively_extract(value)
            if nested_result is not None:
                return nested_result

    return None


def load_convergence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load iteration, PCC and NRMSE histories from a convergence JSON file."""
    data = load_json(path)
    result = recursively_extract(data)

    if result is None:
        if isinstance(data, dict):
            available = ", ".join(map(str, data.keys()))
        else:
            available = f"top-level JSON type: {type(data).__name__}"

        raise KeyError(
            f"Could not find PCC and NRMSE data in:\n{path}\n"
            f"Available top-level content: {available}"
        )

    iterations, pcc, nrmse = result

    if not (len(iterations) == len(pcc) == len(nrmse)):
        raise ValueError(
            f"Length mismatch in {path.name}: "
            f"iterations={len(iterations)}, PCC={len(pcc)}, "
            f"NRMSE={len(nrmse)}"
        )

    if len(iterations) == 0:
        raise ValueError(f"No convergence samples were found in {path}")

    order = np.argsort(iterations)
    return iterations[order], pcc[order], nrmse[order]


# =============================================================================
# PLOTTING
# =============================================================================

def main() -> None:
    convergence_files = {
        "RPP": RPP_PCC_NMRSE,
        "CPP": CPP_PCC_NMRSE,
        "DPP": DPP_PCC_NMRSE,
    }

    histories = {
        label: load_convergence(path)
        for label, path in convergence_files.items()
    }

    fig, (ax_pcc, ax_nrmse) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        constrained_layout=True,
    )

    for label, (iterations, pcc, nrmse) in histories.items():
        ax_pcc.plot(
            iterations,
            pcc,
            linewidth=LINE_WIDTH,
            label=label,
        )
        ax_nrmse.plot(
            iterations,
            nrmse,
            linewidth=LINE_WIDTH,
            label=label,
        )

    ax_pcc.set_xlabel("Iteration", fontsize=AXIS_FONT_SIZE)
    ax_pcc.set_ylabel("Pearson correlation coefficient", fontsize=AXIS_FONT_SIZE)
    ax_pcc.set_title("PCC convergence", fontsize=AXIS_FONT_SIZE)
    ax_pcc.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax_pcc.grid(True, alpha=0.3)
    ax_pcc.legend(fontsize=LEGEND_FONT_SIZE)

    ax_nrmse.set_xlabel("Iteration", fontsize=AXIS_FONT_SIZE)
    ax_nrmse.set_ylabel("NRMSE", fontsize=AXIS_FONT_SIZE)
    ax_nrmse.set_title("NRMSE convergence", fontsize=AXIS_FONT_SIZE)
    ax_nrmse.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax_nrmse.grid(True, alpha=0.3)
    ax_nrmse.legend(fontsize=LEGEND_FONT_SIZE)

    if PCC_Y_LIMITS is not None:
        ax_pcc.set_ylim(*PCC_Y_LIMITS)

    if NRMSE_Y_LIMITS is not None:
        ax_nrmse.set_ylim(*NRMSE_Y_LIMITS)

    print("\nFinal convergence values:")
    for label, (_, pcc, nrmse) in histories.items():
        print(
            f"  {label}: "
            f"PCC = {pcc[-1]:.6f}, "
            f"NRMSE = {nrmse[-1]:.6f}, "
            f"iterations = {len(pcc)}"
        )

    if SAVE_FIGURE:
        fig.savefig(
            OUTPUT_FILE,
            dpi=DPI,
            bbox_inches="tight",
        )
        print(f"\nSaved figure to:\n  {OUTPUT_FILE}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
