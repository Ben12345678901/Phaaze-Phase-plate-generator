"""Typed configuration for a phase-plate input deck.

All lengths are metres, phase is radians, and target data is intensity unless
the field name explicitly says otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FacilityConfig:
    name: str = "custom"
    description: str = ""


@dataclass
class GridConfig:
    plate_size_m: float = 0.11
    plate_pixels: int = 512
    aperture_shape: str = "square"
    propagation_pad_factor: float = 1.0
    output_pixel_size_m: float | None = None
    maximum_propagation_samples: int = 4096


@dataclass
class OpticsConfig:
    wavelength_m: float = 1064e-9
    focal_length_m: float = 1.5
    propagation_model: str = "fraunhofer"
    propagation_distance_m: float | None = None
    refractive_index: float = 1.5


@dataclass
class BeamConfig:
    profile: str = "square_supergaussian"
    fill_factor: float = 0.90
    supergaussian_order: float = 8.0
    array_path: Path | None = None


@dataclass
class TargetConfig:
    profile: str = "supergaussian"
    diameter_m: float = 600e-6
    supergaussian_order: float = 5.2
    stripes: int = 7
    modulation_depth: float = 1.0
    stripe_axis: str = "x"
    array_path: Path | None = None
    array_pixel_size_m: float | None = None
    dark_region_factor: float | None = None


@dataclass
class AlgorithmConfig:
    name: str = "gerchberg_saxton"
    iterations: int = 100
    random_seed: int = 42
    warmup_fraction: float = 0.25
    warmup_cutoff_fraction: float = 0.20
    target_ramp: bool = True
    initial_phase_path: Path | None = None
    show_progress: bool = False
    progress_updates: int = 20


@dataclass
class PhasePlateConfig:
    # Canonical names are documented in modes/__init__.py.
    mode: str = "continuous_wrapped"
    levels: int = 2
    quantization_ramp_fraction: float = 0.50
    binarization_method: str = "threshold"
    correlation_length_m: float | None = None
    phase_rms_rad: float = 3.141592653589793
    distributed_projection_strength: float = 1.0
    random_pi_fraction: float = 0.50


@dataclass
class ManufacturingConfig:
    element_geometry: str = "none"
    element_pitch_m: float | None = None
    iterate_on_element_grid: bool = False
    element_grid_start_fraction: float = 0.0
    boundary_rounding_sigma_px: float = 0.0


@dataclass
class OutputConfig:
    directory: Path = Path("outputs")
    save_arrays: bool = True
    save_metadata: bool = True
    save_plots: bool = False
    show_plots: bool = False
    plot_dpi: int = 160
    focal_plot_window_factor: float | None = 2.0


@dataclass
class ExperimentDeck:
    facility: FacilityConfig = field(default_factory=FacilityConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    phase_plate: PhasePlateConfig = field(default_factory=PhasePlateConfig)
    manufacturing: ManufacturingConfig = field(
        default_factory=ManufacturingConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    source_path: Path | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        result = asdict(self)
        result.pop("source_path", None)

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(result)

    @property
    def observation_distance_m(self) -> float:
        return (
            self.optics.focal_length_m
            if self.optics.propagation_distance_m is None
            else self.optics.propagation_distance_m
        )

    def validate(self) -> None:
        """Fail early when a deck is physically or numerically inconsistent."""
        from .modes import normalise_mode_name
        from .modes.multilevel import normalise_binarization_method
        from .optics import normalise_propagation_model
        from .targets import normalise_target_name

        mode_name = normalise_mode_name(self.phase_plate.mode)
        binarization_method = normalise_binarization_method(
            self.phase_plate.binarization_method
        )
        normalise_propagation_model(self.optics.propagation_model)
        normalise_target_name(self.target.profile)

        if self.grid.plate_pixels < 8:
            raise ValueError("grid.plate_pixels must be at least 8")
        if self.grid.aperture_shape not in {"square", "circle"}:
            raise ValueError(
                "grid.aperture_shape must be 'square' or 'circle'"
            )
        if self.grid.propagation_pad_factor < 1:
            raise ValueError(
                "grid.propagation_pad_factor must be at least 1"
            )
        if (
            self.grid.output_pixel_size_m is not None
            and self.grid.output_pixel_size_m <= 0
        ):
            raise ValueError(
                "grid.output_pixel_size_m must be positive or null"
            )
        if (
            self.grid.maximum_propagation_samples
            < self.grid.plate_pixels
        ):
            raise ValueError(
                "grid.maximum_propagation_samples cannot be smaller than "
                "grid.plate_pixels"
            )
        if not 0 < self.beam.fill_factor <= 1:
            raise ValueError("beam.fill_factor must lie in (0, 1]")
        if self.beam.supergaussian_order <= 0:
            raise ValueError(
                "beam.supergaussian_order must be positive"
            )
        if self.target.stripes < 1:
            raise ValueError("target.stripes must be at least 1")
        if not 0 <= self.target.modulation_depth <= 1:
            raise ValueError(
                "target.modulation_depth must lie in [0, 1]"
            )
        if self.target.stripe_axis not in {"x", "y"}:
            raise ValueError("target.stripe_axis must be 'x' or 'y'")
        if (
            self.target.array_pixel_size_m is not None
            and self.target.array_pixel_size_m <= 0
        ):
            raise ValueError(
                "target.array_pixel_size_m must be positive or null"
            )
        if (
            self.target.dark_region_factor is not None
            and self.target.dark_region_factor < 1
        ):
            raise ValueError(
                "target.dark_region_factor must be at least 1 or null"
            )
        if self.algorithm.name != "gerchberg_saxton":
            raise ValueError(
                "algorithm.name currently supports only 'gerchberg_saxton'"
            )
        if self.algorithm.iterations < 1:
            raise ValueError("algorithm.iterations must be positive")
        if self.algorithm.progress_updates < 1:
            raise ValueError(
                "algorithm.progress_updates must be positive"
            )
        if not 0 <= self.algorithm.warmup_fraction <= 1:
            raise ValueError(
                "algorithm.warmup_fraction must lie in [0, 1]"
            )
        if not 0 <= self.algorithm.warmup_cutoff_fraction <= 1:
            raise ValueError(
                "algorithm.warmup_cutoff_fraction must lie in [0, 1]"
            )
        if self.phase_plate.levels < 2:
            raise ValueError("phase_plate.levels must be at least 2")
        if not 0 <= self.phase_plate.quantization_ramp_fraction <= 1:
            raise ValueError(
                "phase_plate.quantization_ramp_fraction must lie in [0, 1]"
            )
        if binarization_method != "threshold" and not (
            mode_name == "multilevel" and self.phase_plate.levels == 2
        ):
            raise ValueError(
                "median and amplitude_weighted binarization require "
                "phase_plate.mode='multilevel' and phase_plate.levels=2"
            )
        if (
            self.phase_plate.correlation_length_m is not None
            and self.phase_plate.correlation_length_m <= 0
        ):
            raise ValueError(
                "phase_plate.correlation_length_m must be positive or null"
            )
        if self.phase_plate.phase_rms_rad <= 0:
            raise ValueError("phase_plate.phase_rms_rad must be positive")
        if not 0 <= self.phase_plate.distributed_projection_strength <= 1:
            raise ValueError(
                "phase_plate.distributed_projection_strength must lie in [0, 1]"
            )
        if not 0 < self.phase_plate.random_pi_fraction < 1:
            raise ValueError(
                "phase_plate.random_pi_fraction must lie in (0, 1)"
            )
        if self.manufacturing.element_geometry not in {
            "none",
            "square",
            "hexagonal",
        }:
            raise ValueError(
                "manufacturing.element_geometry must be 'none', 'square', "
                "or 'hexagonal'"
            )
        if (
            self.manufacturing.element_pitch_m is not None
            and self.manufacturing.element_pitch_m <= 0
        ):
            raise ValueError(
                "manufacturing.element_pitch_m must be positive or null"
            )
        if self.manufacturing.iterate_on_element_grid:
            if mode_name != "multilevel":
                raise ValueError(
                    "manufacturing.iterate_on_element_grid currently "
                    "requires phase_plate.mode='multilevel'"
                )
            if self.manufacturing.element_geometry == "none":
                raise ValueError(
                    "manufacturing.iterate_on_element_grid requires square "
                    "or hexagonal element geometry"
                )
        if not 0 <= self.manufacturing.element_grid_start_fraction <= 1:
            raise ValueError(
                "manufacturing.element_grid_start_fraction must lie in [0, 1]"
            )
        if self.manufacturing.boundary_rounding_sigma_px < 0:
            raise ValueError(
                "manufacturing.boundary_rounding_sigma_px cannot be negative"
            )
        if self.output.plot_dpi < 72:
            raise ValueError("output.plot_dpi must be at least 72")
        if (
            self.output.focal_plot_window_factor is not None
            and self.output.focal_plot_window_factor < 1
        ):
            raise ValueError(
                "output.focal_plot_window_factor must be at least 1 or null"
            )

        positive = {
            "grid.plate_size_m": self.grid.plate_size_m,
            "optics.wavelength_m": self.optics.wavelength_m,
            "optics.focal_length_m": self.optics.focal_length_m,
            "optics.refractive_index": self.optics.refractive_index,
            "target.diameter_m": self.target.diameter_m,
            "observation distance": self.observation_distance_m,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.optics.refractive_index <= 1:
            raise ValueError("optics.refractive_index must be greater than 1")

        for field_name, path in {
            "beam.array_path": self.beam.array_path,
            "target.array_path": self.target.array_path,
            "algorithm.initial_phase_path": self.algorithm.initial_phase_path,
        }.items():
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"{field_name} does not exist: {path}")
