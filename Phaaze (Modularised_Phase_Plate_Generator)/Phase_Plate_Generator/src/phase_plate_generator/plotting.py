"""Legacy-style diagnostic plots for completed phase-plate runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import comparison_plot_name
from .config import ExperimentDeck
from .generator import GenerationResult
from .output_layout import ensure_run_folders


def _plate_extent_mm(deck: ExperimentDeck) -> list[float]:
    half_width = 0.5 * deck.grid.plate_size_m * 1e3
    return [-half_width, half_width, -half_width, half_width]


def _focal_extent_um(result: GenerationResult) -> list[float]:
    return [
        float(np.min(result.focal_x_m)) * 1e6,
        float(np.max(result.focal_x_m)) * 1e6,
        float(np.min(result.focal_y_m)) * 1e6,
        float(np.max(result.focal_y_m)) * 1e6,
    ]


def _set_focal_window(axis, deck: ExperimentDeck) -> None:
    factor = deck.output.focal_plot_window_factor
    if factor is None:
        return
    half_width_um = (
        0.5 * factor * deck.target.diameter_m * 1e6
    )
    axis.set_xlim(-half_width_um, half_width_um)
    axis.set_ylim(-half_width_um, half_width_um)


def _save_figure(
    figure,
    output_directory: Path,
    filename: str,
    *,
    dpi: int,
) -> Path:
    destination = output_directory / filename
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    return destination


def _convergence_figure(result: GenerationResult, plt):
    if not result.history:
        return None
    iterations = np.asarray(
        [item["iteration"] for item in result.history]
    )
    pcc = np.asarray([item["pcc"] for item in result.history])
    nrmse = np.asarray(
        [item["relative_nrmse"] for item in result.history]
    )
    accuracy = (
        np.maximum(pcc, 0) * np.maximum(1 - nrmse, 0) * 100
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(13, 4.2), constrained_layout=True
    )
    axes[0].plot(iterations, pcc, lw=1.5)
    axes[0].set(
        title="PCC vs iteration",
        xlabel="Iteration",
        ylabel="PCC",
    )
    axes[1].plot(iterations, nrmse, lw=1.5)
    axes[1].set(
        title="Relative NRMSE vs iteration",
        xlabel="Iteration",
        ylabel="Relative NRMSE",
    )
    axes[2].plot(iterations, accuracy, lw=1.5)
    axes[2].set(
        title="Combined accuracy vs iteration",
        xlabel="Iteration",
        ylabel="Accuracy (%)",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
    manufacturing = result.metadata.get("manufacturing", {})
    grid_suffix = (
        f" on the {manufacturing['element_geometry']} element grid"
        if manufacturing.get("iterate_on_element_grid", False)
        else ""
    )
    figure.suptitle(
        f"Gerchberg-Saxton convergence{grid_suffix} "
        f"(final PCC={pcc[-1]:.3f}, NRMSE={nrmse[-1]:.3f})"
    )
    return figure


def _overview_figure(
    result: GenerationResult,
    deck: ExperimentDeck,
    plt,
):
    plate_extent = _plate_extent_mm(deck)
    focal_extent = _focal_extent_um(result)
    grid_iterations = deck.manufacturing.iterate_on_element_grid
    design_label = (
        f"{deck.manufacturing.element_geometry.title()}-grid-constrained"
        if grid_iterations
        else "Ideal fine-grid"
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(17, 10.5), constrained_layout=True
    )

    input_image = axes[0, 0].imshow(
        result.input_amplitude,
        cmap="gray",
        origin="lower",
        extent=plate_extent,
    )
    axes[0, 0].set_title("Input beam amplitude")
    axes[0, 0].set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(
        input_image, ax=axes[0, 0], label="Normalised amplitude"
    )

    target_image = axes[0, 1].imshow(
        result.target_intensity,
        cmap="gray",
        origin="lower",
        extent=focal_extent,
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Target focal intensity")
    axes[0, 1].set(xlabel="x (µm)", ylabel="y (µm)")
    _set_focal_window(axes[0, 1], deck)
    figure.colorbar(
        target_image, ax=axes[0, 1], label="Normalised intensity"
    )

    log_focal = np.log10(
        np.clip(
            result.pre_manufacturing_focal_intensity,
            1e-8,
            None,
        )
    )
    log_image = axes[0, 2].imshow(
        log_focal,
        cmap="viridis",
        origin="lower",
        extent=focal_extent,
        vmin=-6,
        vmax=0,
    )
    axes[0, 2].set_title(
        f"Log10 {design_label.lower()} focal intensity"
    )
    axes[0, 2].set(xlabel="x (µm)", ylabel="y (µm)")
    _set_focal_window(axes[0, 2], deck)
    figure.colorbar(
        log_image, ax=axes[0, 2], label="log10 normalised intensity"
    )

    focal_image = axes[1, 0].imshow(
        result.pre_manufacturing_focal_intensity,
        cmap="inferno",
        origin="lower",
        extent=focal_extent,
        vmin=0,
        vmax=1,
    )
    metrics = result.metadata["pre_manufacturing_metrics"]
    axes[1, 0].set_title(
        f"{design_label} focal intensity | "
        f"PCC={metrics['pcc']:.3f}, "
        f"NRMSE={metrics['relative_nrmse']:.3f}"
    )
    axes[1, 0].set(xlabel="x (µm)", ylabel="y (µm)")
    _set_focal_window(axes[1, 0], deck)
    figure.colorbar(
        focal_image, ax=axes[1, 0], label="Normalised intensity"
    )

    phase_image = axes[1, 1].imshow(
        result.pre_manufacturing_phase_rad,
        cmap="gray",
        origin="lower",
        extent=plate_extent,
        vmin=0,
        vmax=2 * np.pi,
    )
    axes[1, 1].set_title(f"{design_label} phase plate")
    axes[1, 1].set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(
        phase_image, ax=axes[1, 1], label="Phase (rad)"
    )

    thickness_image = axes[1, 2].imshow(
        result.pre_manufacturing_thickness_m * 1e6,
        cmap="gray",
        origin="lower",
        extent=plate_extent,
    )
    axes[1, 2].set_title(f"{design_label} material thickness")
    axes[1, 2].set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(
        thickness_image, ax=axes[1, 2], label="Thickness (µm)"
    )

    figure.suptitle(
        f"{deck.facility.name}: {design_label.lower()} "
        f"{deck.phase_plate.mode} solution | "
        f"{deck.optics.propagation_model.title()} propagation"
    )
    return figure


def _manufacturing_comparison_figure(
    result: GenerationResult,
    deck: ExperimentDeck,
    plt,
):
    geometry = deck.manufacturing.element_geometry
    if geometry == "none":
        return None
    plate_extent = _plate_extent_mm(deck)
    focal_extent = _focal_extent_um(result)
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 10.5), constrained_layout=True
    )

    phase_vmax = 2 * np.pi
    design_label = (
        f"{geometry.title()}-grid iteration"
        if deck.manufacturing.iterate_on_element_grid
        else "Fine-grid"
    )
    design_phase = axes[0, 0].imshow(
        result.pre_manufacturing_phase_rad,
        cmap="gray",
        origin="lower",
        extent=plate_extent,
        vmin=0,
        vmax=phase_vmax,
    )
    axes[0, 0].set_title(f"{design_label} phase design")
    axes[0, 0].set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(
        design_phase, ax=axes[0, 0], label="Phase (rad)"
    )

    manufactured_phase = axes[0, 1].imshow(
        result.wrapped_phase_rad,
        cmap="gray",
        origin="lower",
        extent=plate_extent,
        vmin=0,
        vmax=phase_vmax,
    )
    cell_count = result.metadata["manufacturing"]["element_count"]
    axes[0, 1].set_title(
        f"{geometry.title()} realization: {cell_count} cells"
    )
    axes[0, 1].set(xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(
        manufactured_phase, ax=axes[0, 1], label="Phase (rad)"
    )

    before_metrics = result.metadata["pre_manufacturing_metrics"]
    design_focal = axes[1, 0].imshow(
        result.pre_manufacturing_focal_intensity,
        cmap="inferno",
        origin="lower",
        extent=focal_extent,
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        f"{design_label} focal intensity | "
        f"PCC={before_metrics['pcc']:.3f}, "
        f"NRMSE={before_metrics['relative_nrmse']:.3f}"
    )
    axes[1, 0].set(xlabel="x (µm)", ylabel="y (µm)")
    _set_focal_window(axes[1, 0], deck)
    figure.colorbar(
        design_focal, ax=axes[1, 0], label="Normalised intensity"
    )

    after_metrics = result.metadata["metrics"]
    manufactured_focal = axes[1, 1].imshow(
        result.focal_intensity,
        cmap="inferno",
        origin="lower",
        extent=focal_extent,
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title(
        f"{geometry.title()} focal intensity | "
        f"PCC={after_metrics['pcc']:.3f}, "
        f"NRMSE={after_metrics['relative_nrmse']:.3f}"
    )
    axes[1, 1].set(xlabel="x (µm)", ylabel="y (µm)")
    _set_focal_window(axes[1, 1], deck)
    figure.colorbar(
        manufactured_focal,
        ax=axes[1, 1],
        label="Normalised intensity",
    )

    pitch_mm = (
        result.metadata["manufacturing"]["realized_pitch_m"] * 1e3
    )
    figure.suptitle(
        "Fine-grid design versus physical-element realization | "
        f"{geometry}, pitch={pitch_mm:.4g} mm"
    )
    return figure


def plot_generation_result(
    result: GenerationResult,
    deck: ExperimentDeck,
) -> list[Path]:
    """Save and/or show convergence, overview, and element-grid figures."""
    if not (deck.output.save_plots or deck.output.show_plots):
        return []
    try:
        if not deck.output.show_plots:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plot output requires Matplotlib. Install the project "
            "dependencies or set output.save_plots/show_plots to false."
        ) from exc

    output_directory = (
        Path(deck.output.directory).resolve()
        if result.output_directory is None
        else result.output_directory
    )
    folders = ensure_run_folders(output_directory)
    figures = [
        (
            "convergence.png",
            _convergence_figure(result, plt),
        ),
        (
            "results_overview.png",
            _overview_figure(result, deck, plt),
        ),
        (
            comparison_plot_name(
                str(result.metadata["phase_mode"]),
                deck.manufacturing.element_geometry,
            ),
            _manufacturing_comparison_figure(result, deck, plt),
        ),
    ]
    saved: list[Path] = []
    if deck.output.save_plots:
        for filename, figure in figures:
            if filename is not None and figure is not None:
                saved.append(
                    _save_figure(
                        figure,
                        folders.plots,
                        filename,
                        dpi=deck.output.plot_dpi,
                    )
                )
    if deck.output.show_plots:
        print(
            "Plots are ready. Close the figure windows to finish the run."
        )
        plt.show()
    else:
        for _, figure in figures:
            if figure is not None:
                plt.close(figure)
    return saved
