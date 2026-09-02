"""GSI seven-stripe top-hat binary phase-plate input deck.

Edit ``INPUTS`` below and run this file directly from the IDE.
"""

from pathlib import Path
import sys

DECK_PATH = Path(__file__).resolve()
PROJECT_DIRECTORY = DECK_PATH.parents[2] / "Phase_Plate_Generator"
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from phase_plate_generator import deck_from_mapping, run_configured_deck


INPUTS = {
    "facility": {
        "name": "GSI",
        "description": (
            "GSI starter case: seven-stripe top-hat target and binary "
            "hexagonal plate."
        ),
    },
    "grid": {
        "plate_size_m": 0.11,
        "plate_pixels": 1024,
        "aperture_shape": "square",
        "propagation_pad_factor": 1.0,
        "output_pixel_size_m": 5e-6,
        "maximum_propagation_samples": 4096,
    },
    "optics": {
        "wavelength_m": 1064e-9,
        "focal_length_m": 1.5,
        "propagation_model": "fraunhofer",
        "propagation_distance_m": 1.5,
        "refractive_index": 1.5,
    },
    "beam": {
        "profile": "array",
        "fill_factor": 0.90,
        "supergaussian_order": 8.0,
        "array_path": r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\PHAAZE\Phaaze (Modularised_Phase_Plate_Generator)\Validation\scitech_GSI_2026\GSI_Near_field\GSI_near_field_intensity_1024.npy",
    },
    "target": {
        "profile": "striped_tophat",
        "diameter_m": 600e-6,
        "supergaussian_order": 5.2,
        "stripes": 7,
        "modulation_depth": 1.0,
        "stripe_axis": "y",
        "array_path": None,
        "array_pixel_size_m": None,
        "dark_region_factor": None,
    },
    "algorithm": {
        "name": "gerchberg_saxton",
        "iterations": 150,
        "random_seed": 42,
        "warmup_fraction": 0.25,
        "warmup_cutoff_fraction": 0.20,
        "target_ramp": True,
        "initial_phase_path": None,
        "show_progress": True,
        "progress_updates": 20,
    },
    "phase_plate": {
        "mode": "multilevel",
        "levels": 2,
        "quantization_ramp_fraction": 0.50,
        "binarization_method": "threshold",
        "correlation_length_m": None,
        "phase_rms_rad": 3.141592653589793,
        "distributed_projection_strength": 1.0,
        "random_pi_fraction": 0.50,
    },
    "manufacturing": {
        "element_geometry": "square",
        "element_pitch_m": None,
        "iterate_on_element_grid": False,
        "boundary_rounding_sigma_px": 0.0,
    },
    "output": {
        "directory": r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Modularised_Phase_Plate_Generator\Validation\scitech_GSI_2026\Generated_results",
        "save_arrays": True,
        "save_metadata": True,
        "save_plots": True,
        "show_plots": True,
        "plot_dpi": 160,
        "focal_plot_window_factor": 2.0,
    },
}


def main(
    *,
    outputs_root=None,
    started_at=None,
    show_plots=None,
    show_progress=None,
):
    """Construct and run this GSI example input deck."""
    deck = deck_from_mapping(INPUTS, source_path=DECK_PATH)
    if show_plots is not None:
        deck.output.show_plots = show_plots
    if show_progress is not None:
        deck.algorithm.show_progress = show_progress
    result = run_configured_deck(
        deck,
        DECK_PATH,
        outputs_root=outputs_root,
        started_at=started_at,
    )
    metrics = result.metadata["metrics"]
    print("\nPhase-plate generation complete.")
    print(f"Input deck: {DECK_PATH.name}")
    print(f"Output folder: {result.output_directory}")
    print(f"PCC: {metrics['pcc']:.6f}")
    print(f"Relative NRMSE: {metrics['relative_nrmse']:.6f}")
    return result


if __name__ == "__main__":
    main()
