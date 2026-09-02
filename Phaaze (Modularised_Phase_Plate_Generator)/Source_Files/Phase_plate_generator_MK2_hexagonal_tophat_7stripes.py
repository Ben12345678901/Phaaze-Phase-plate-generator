"""MK2 realistic phase-plate generator with square/hexagonal elements.

The Gerchberg-Saxton calculation first produces the useful fine-grid solution.
That phase is then projected onto the selected physical element lattice and
both plates are propagated with identical numerical
sampling.  The control panel selects either Fraunhofer focal-plane propagation
or one-FFT Fresnel propagation through the same focusing lens to a configurable
observation plane.  This isolates the degradation caused by element size
without confusing it with propagation or FFT resolution.  The realistic plate
can use either square elements or a true staggered hexagonal lattice.  Other
generator files are left untouched.

Workflow
--------
1. Build or load the incident amplitude and target intensity.
2. Retrieve a fine-grid phase with Gerchberg-Saxton iterations.
3. Quantise the phase when a stepped plate is requested.
4. Project the fine phase onto square or hexagonal physical elements.
5. Propagate ideal and realistic plates on the same numerical grid.
6. Save phase maps, focal intensities, geometry and propagation metadata.

User-editable values are collected in ``CONTROL_PANEL``. Lengths stored by the
program use SI metres, phase uses radians and intensity is in arbitrary units.
"""

from __future__ import annotations

import numpy as np
try:
    import cv2
except ImportError:  # Optional: only image-loading/legacy visual helpers need OpenCV.
    cv2 = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.patches import Polygon, Circle
except ImportError:  # Numerical/headless runs do not require Matplotlib.
    plt = None
    colors = None
    Polygon = Circle = None
import time
from dataclasses import dataclass
try:
    from scipy.interpolate import RectBivariateSpline
except ImportError:  # Only the optional smoothing path requires SciPy.
    RectBivariateSpline = None
import sys
from pathlib import Path

GENERATOR_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = GENERATOR_DIR.parent
if str(COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARISON_DIR))

from optics_common import (
    centered_axis,
    focal_metrics,
    load_2d_array,
    make_pupil,
    normalise_intensity,
    normalise_propagation_model,
    pad_center,
    propagate_phase_plate,
    round_binary_phase_boundaries,
    resample_intensity_to_grid,
    resample_phase_to_shape,
)


@dataclass
class RealisticGeneratorConfig:
    """All user-editable settings for the realistic comparison workflow."""

    # RUN BEHAVIOUR
    # Controls file saving, interactive figures and detailed debug output.
    save_outputs: bool
    show_plots: bool
    debug: bool

    # Target and phase representation
    desired_focal_spot_m: float
    target_profile: str
    top_hat_num_stripes: int
    phase_mode: str
    num_phase_steps: int

    # Optical system
    wavelength_m: float
    focal_length_m: float
    propagation_model: str
    propagation_distance_m: float
    refractive_index: float

    # Phase plate and input beam
    plate_size_m: float
    plate_pixels: int
    aperture_shape: str
    input_beam_shape: str
    beam_fill_factor: float
    square_beam_order: float
    element_shape: str

    # Gerchberg-Saxton controls
    max_iterations: int
    cut_fraction: float
    quantization_ramp_fraction: float
    optimisation_pad_pixels: int
    output_pixel_size_m: float | None
    maximum_propagation_samples: int

    # Output-plane constraint, plotting and common final propagation
    propagation_pad_factor: float
    target_dark_region_factor: float | None
    focal_plot_window_factor: float | None
    export_plate_pixels: int
    boundary_rounding_sigma_px: float

    # Optional inputs. Use None to generate the corresponding field/target.
    input_beam_path: Path | None
    initial_phase_path: Path | None
    target_intensity_path: Path | None

    # None selects a mode-specific folder beside this script.
    output_directory: Path | None


# =============================================================================
# MAIN CONTROL PANEL - edit values in this block
# =============================================================================
CONTROL_PANEL = RealisticGeneratorConfig(
    # RUN BEHAVIOUR
    # Controls file saving, interactive figures and detailed debug output.
    save_outputs=True,
    show_plots=True,
    debug=False,

    # FOCAL TARGET AND PHASE LEVELS
    # Controls the requested spot diameter and continuous/stepped phase mode.
    desired_focal_spot_m=600e-6,       # 600 um; use 150e-6 for the coarse case
    target_profile="tophat",   # "supergaussian" or "tophat"
    top_hat_num_stripes=7,            # used only when target_profile="tophat"
    phase_mode="quantized",           # "continuous" or "quantized"
    num_phase_steps=2,                # used only when phase_mode="quantized"

    # OPTICAL SYSTEM
    # Controls the laser, focusing optic and phase-to-thickness conversion.
    wavelength_m=1064e-9,
    focal_length_m=1.5,

    # "fraunhofer" evaluates the lens focal plane. "fresnel" propagates
    # through that lens to propagation_distance_m; z=f gives the same focal
    # intensity as Fraunhofer, while z!=f includes defocus/near-field effects.
    propagation_model="fraunhofer",
    propagation_distance_m=1.5,
    refractive_index=1.5,

    # PHASE PLATE, APERTURE AND INPUT BEAM
    # aperture_shape controls the clear phase-plate aperture.
    # input_beam_shape independently selects the generated beam profile.
    plate_size_m=0.11,
    plate_pixels=1024,
    aperture_shape="square",          # "circle" or "square"
    
    input_beam_shape="square",        # "circular" or "square"
    beam_fill_factor=0.90,             # 1/e amplitude radius/half-width
    square_beam_order=8.0,             # 2-4 soft; 8-12 nearly top-hat
    element_shape="hexagonal",        # "square" or "hexagonal"

    # FINE-GRID GERCHBERG-SAXTON SOLUTION
    # Controls convergence length, warm-up filtering and phase quantisation.
    max_iterations=150,
    cut_fraction=0.20,
    quantization_ramp_fraction=0.50,

    # FOURIER TRANSFORM, TARGET DARK REGION AND PLOT WINDOW
    # Lock the output-plane sampling as propagation distance changes. The code
    # automatically selects an FFT-friendly zero-padded size that meets or
    # slightly exceeds this resolution. Set None to restore distance-dependent
    # one-FFT sampling and use optimisation_pad_pixels directly.
    output_pixel_size_m=5e-6,
    maximum_propagation_samples=4096,

    # Manual GS padding, used only when output_pixel_size_m is None.
    optimisation_pad_pixels=0,

    # Manual final-propagation padding, also used only when the output sampling
    # lock is disabled.
    propagation_pad_factor=1,

    # None constrains the full Fourier plane to the target. A value such as
    # 2.0 constrains a square twice the requested spot width, leaving the field
    # outside it free. The dark margin on each side is (factor - 1)*spot/2.
    target_dark_region_factor=None,
    
    # Display only: 2.0 shows a window twice the requested spot width. None
    # shows the complete calculated Fourier plane and does not zoom the plots.
    focal_plot_window_factor=2.0,

    # FINAL PLATE EXPORT
    # Controls exported plate resolution and optional binary edge rounding.
    export_plate_pixels=1024,
    boundary_rounding_sigma_px=0.0,

    # OPTIONAL INPUT AND OUTPUT FILES
    # Use None for generated defaults, or provide paths to saved arrays/images.
    input_beam_path=None,
    initial_phase_path=None,
    target_intensity_path=None,        # e.g. COMPARISON_DIR / "SCITECH FOCAL SPOT.npy"
    output_directory=None,
)

TARGET_PROFILE_ALIASES = {
    "supergaussian": "supergaussian",
    "super-gaussian": "supergaussian",
    "super_gaussian": "supergaussian",
    "sg": "supergaussian",
    "tophat": "tophat",
    "top-hat": "tophat",
    "top_hat": "tophat",
    "flat": "tophat",
}


def normalise_target_profile(target_profile):
    """Return ``supergaussian`` or ``tophat``, accepting common aliases."""
    key = str(target_profile).strip().lower()
    try:
        return TARGET_PROFILE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "target_profile must be 'supergaussian' or 'tophat'"
        ) from exc


ELEMENT_SHAPE_ALIASES = {
    "square": "square",
    "squares": "square",
    "hex": "hexagonal",
    "hexagon": "hexagonal",
    "hexagons": "hexagonal",
    "hexagonal": "hexagonal",
}

INPUT_BEAM_SHAPE_ALIASES = {
    "circle": "circular",
    "circular": "circular",
    "round": "circular",
    "square": "square",
}


def normalise_input_beam_shape(input_beam_shape):
    """Return ``circular`` or ``square``, accepting common aliases."""
    key = str(input_beam_shape).strip().lower()
    try:
        return INPUT_BEAM_SHAPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "input_beam_shape must be 'circular' or 'square'"
        ) from exc


def build_generated_input_amplitude(
        samples, plate_size, fill_factor=0.90, aperture_shape="square",
        input_beam_shape="circular", square_beam_order=8.0):
    """Build a circular Gaussian or square super-Gaussian field amplitude.

    ``fill_factor`` sets the 1/e amplitude radius of the circular Gaussian or
    the 1/e amplitude half-width along x and y of the square super-Gaussian.
    The beam profile and clear-aperture geometry are controlled independently.
    """
    input_beam_shape = normalise_input_beam_shape(input_beam_shape)
    if samples < 1:
        raise ValueError("samples must be positive")
    if plate_size <= 0:
        raise ValueError("plate_size must be positive")
    if not 0 < fill_factor <= 1:
        raise ValueError("fill_factor must lie in (0, 1]")
    if square_beam_order <= 0:
        raise ValueError("square_beam_order must be positive")

    dx_plate = plate_size / samples
    axis = centered_axis(samples, dx_plate)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    width = fill_factor * plate_size / 2.0

    if input_beam_shape == "circular":
        amplitude = np.exp(-(X**2 + Y**2) / width**2)
    else:
        exponent = (
            (np.abs(X) / width) ** (2 * square_beam_order)
            + (np.abs(Y) / width) ** (2 * square_beam_order)
        )
        amplitude = np.exp(-exponent)

    pupil = make_pupil(X, Y, plate_size, aperture_shape)
    return amplitude * pupil, pupil, dx_plate


# =============================================================================
# Propagation models, output sampling and diffraction-regime guidance
# =============================================================================
def normalise_element_shape(element_shape):
    """Return ``square`` or ``hexagonal``, accepting short hex aliases."""
    key = str(element_shape).strip().lower()
    try:
        return ELEMENT_SHAPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "element_shape must be 'square' or 'hexagonal'"
        ) from exc


def propagation_plane_distance(
        propagation_model, focal_length_m, propagation_distance_m):
    """Return the distance which determines the output-plane FFT sampling."""
    model = normalise_propagation_model(propagation_model)
    return focal_length_m if model == "fraunhofer" else propagation_distance_m


def calculate_propagation_guide(
        wavelength_m, focal_length_m, plate_size_m, beam_fill_factor,
        desired_focal_spot_m, propagation_model, propagation_distance_m,
        input_beam_shape="circular"):
    """Calculate Rayleigh ranges and free-space diffraction-regime guides.

    Three Gaussian-beam scales are deliberately reported:

    * the incident beam's Rayleigh range at the plate;
    * the lens's diffraction-limited waist and Rayleigh range; and
    * a Gaussian-equivalent Rayleigh range based on half the requested shaped
      spot diameter.

    The last value is only an envelope guide because a striped phase-plate
    target is not a Gaussian mode.  The free-space Fresnel number is the more
    direct Fraunhofer/Fresnel criterion for propagation from the illuminated
    aperture.
    """
    positive_values = {
        "wavelength_m": wavelength_m,
        "focal_length_m": focal_length_m,
        "plate_size_m": plate_size_m,
        "beam_fill_factor": beam_fill_factor,
        "desired_focal_spot_m": desired_focal_spot_m,
        "propagation_distance_m": propagation_distance_m,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    model = normalise_propagation_model(propagation_model)
    input_beam_shape = normalise_input_beam_shape(input_beam_shape)
    observation_distance_m = propagation_plane_distance(
        model, focal_length_m, propagation_distance_m
    )
    input_beam_radius_m = beam_fill_factor * plate_size_m / 2
    clear_aperture_radius_m = plate_size_m / 2

    input_rayleigh_range_m = (
        np.pi * input_beam_radius_m**2 / wavelength_m
    )
    diffraction_limited_waist_m = (
        wavelength_m * focal_length_m / (np.pi * input_beam_radius_m)
    )
    diffraction_limited_rayleigh_range_m = (
        np.pi * diffraction_limited_waist_m**2 / wavelength_m
    )
    target_equivalent_waist_m = desired_focal_spot_m / 2
    target_equivalent_rayleigh_range_m = (
        np.pi * target_equivalent_waist_m**2 / wavelength_m
    )

    defocus_m = abs(observation_distance_m - focal_length_m)
    beam_fresnel_number = (
        input_beam_radius_m**2 / (wavelength_m * observation_distance_m)
    )
    aperture_fresnel_number = (
        clear_aperture_radius_m**2
        / (wavelength_m * observation_distance_m)
    )
    illuminated_diameter_m = 2 * input_beam_radius_m
    conventional_far_field_distance_m = (
        2 * illuminated_diameter_m**2 / wavelength_m
    )

    # This is a practical guide, not a hard boundary: N_F << 1 is Fraunhofer.
    if beam_fresnel_number < 0.1:
        free_space_regime = "Fraunhofer / far field"
    elif beam_fresnel_number <= 1:
        free_space_regime = "Fresnel-Fraunhofer transition"
    else:
        free_space_regime = "Fresnel / near field"

    return {
        "propagation_model": model,
        "input_beam_shape": input_beam_shape,
        "wavelength_m": float(wavelength_m),
        "focal_length_m": float(focal_length_m),
        "propagation_distance_m": float(observation_distance_m),
        "defocus_from_focal_plane_m": float(defocus_m),
        "input_beam_radius_m": float(input_beam_radius_m),
        "input_rayleigh_range_m": float(input_rayleigh_range_m),
        "diffraction_limited_waist_m": float(diffraction_limited_waist_m),
        "diffraction_limited_rayleigh_range_m": float(
            diffraction_limited_rayleigh_range_m
        ),
        "target_equivalent_waist_m": float(target_equivalent_waist_m),
        "target_equivalent_rayleigh_range_m": float(
            target_equivalent_rayleigh_range_m
        ),
        "defocus_in_diffraction_rayleigh_ranges": float(
            defocus_m / diffraction_limited_rayleigh_range_m
        ),
        "defocus_in_target_equivalent_rayleigh_ranges": float(
            defocus_m / target_equivalent_rayleigh_range_m
        ),
        "beam_fresnel_number": float(beam_fresnel_number),
        "aperture_fresnel_number": float(aperture_fresnel_number),
        "conventional_far_field_distance_m": float(
            conventional_far_field_distance_m
        ),
        "free_space_regime": free_space_regime,
    }


def print_propagation_guide(guide):
    """Print a concise interpretation of :func:`calculate_propagation_guide`."""
    print("\nRayleigh-length and diffraction-regime guide:")
    if guide.get("input_beam_shape", "circular") == "square":
        beam_width_label = "square-beam 1/e amplitude half-width"
        rayleigh_label = "square-beam Gaussian-equivalent Rayleigh range"
    else:
        beam_width_label = "circular Gaussian 1/e^2 intensity radius"
        rayleigh_label = "incident-beam Rayleigh range"
    print(
        f"  {beam_width_label} = "
        f"{guide['input_beam_radius_m']*1e3:.4f} mm"
    )
    print(
        f"  {rayleigh_label} = "
        f"{guide['input_rayleigh_range_m']:.6g} m"
    )
    print(
        "  diffraction-limited focused waist = "
        f"{guide['diffraction_limited_waist_m']*1e6:.4f} um"
    )
    print(
        "  diffraction-limited focused Rayleigh range = "
        f"{guide['diffraction_limited_rayleigh_range_m']*1e3:.4f} mm"
    )
    print(
        "  requested-spot Gaussian-equivalent Rayleigh range = "
        f"{guide['target_equivalent_rayleigh_range_m']:.6g} m "
        "(envelope guide only)"
    )
    print(
        "  observation-plane defocus |z-f| = "
        f"{guide['defocus_from_focal_plane_m']:.6g} m = "
        f"{guide['defocus_in_diffraction_rayleigh_ranges']:.3g} focused "
        "Rayleigh ranges"
    )
    print(
        "  free-space Fresnel number (illuminated radius) = "
        f"{guide['beam_fresnel_number']:.6g}"
    )
    print(f"  free-space regime guide = {guide['free_space_regime']}")
    print(
        "  conventional free-space far-field distance 2D^2/lambda = "
        f"{guide['conventional_far_field_distance_m']:.6g} m"
    )
    print(
        "  Note: the free-space criterion does not prevent a lens from forming "
        "a Fourier/Fraunhofer pattern at its back focal plane; in this model "
        "Fresnel z=f and Fraunhofer give the same focal intensity.\n"
    )


def _centered_fft2(field):
    """Centred two-dimensional FFT used by both propagation models."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def _centered_ifft2(field):
    """Inverse of :func:`_centered_fft2`, including NumPy's normalisation."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def propagation_axis(samples, dx_input_m, wavelength_m, distance_m):
    """Physical one-FFT output-plane coordinates, in metres."""
    frequency = np.fft.fftshift(np.fft.fftfreq(samples, d=dx_input_m))
    return wavelength_m * distance_m * frequency


def _has_fft_friendly_size(samples):
    """Return True when ``samples`` factors entirely into 2, 3 and 5."""
    remaining = int(samples)
    for factor in (2, 3, 5):
        while remaining > 1 and remaining % factor == 0:
            remaining //= factor
    return remaining == 1


def next_fft_friendly_size(minimum_samples, parity=0):
    """Return a 2/3/5-smooth size at least as large with the given parity."""
    samples = max(1, int(np.ceil(minimum_samples)))
    parity = int(parity) % 2
    if samples % 2 != parity:
        samples += 1
    while not _has_fft_friendly_size(samples):
        samples += 2
    return samples


def resolve_output_sampling(
        plate_samples, plate_size_m, wavelength_m, propagation_distance_m,
        requested_output_pixel_size_m=None, manual_total_samples=None,
        maximum_samples=None):
    """Resolve a padded grid and its resulting physical output sampling.

    For a one-FFT Fresnel/Fraunhofer transform,

        dx_out = wavelength * z / (N_fft * dx_plate).

    When a requested output pitch is supplied, ``N_fft`` is increased to an
    FFT-friendly value so ``dx_out`` is no larger than requested.  Matching the
    plate-grid parity guarantees symmetric integer padding on both sides.
    """
    if plate_samples < 1:
        raise ValueError("plate_samples must be positive")
    if min(plate_size_m, wavelength_m, propagation_distance_m) <= 0:
        raise ValueError("sampling dimensions and propagation distance must be positive")

    dx_plate_m = plate_size_m / plate_samples
    if requested_output_pixel_size_m is None:
        total_samples = (
            plate_samples if manual_total_samples is None
            else max(plate_samples, int(round(manual_total_samples)))
        )
    else:
        if requested_output_pixel_size_m <= 0:
            raise ValueError("output_pixel_size_m must be positive or None")
        minimum_samples = np.ceil(
            wavelength_m * propagation_distance_m
            / (dx_plate_m * requested_output_pixel_size_m)
        )
        total_samples = next_fft_friendly_size(
            max(plate_samples, minimum_samples), parity=plate_samples % 2
        )

    if maximum_samples is not None and total_samples > maximum_samples:
        raise ValueError(
            f"Fixed output sampling requires {total_samples} propagation "
            f"samples, exceeding maximum_propagation_samples={maximum_samples}. "
            "Increase output_pixel_size_m or the safety limit."
        )

    output_pixel_size_m = (
        wavelength_m * propagation_distance_m
        / (total_samples * dx_plate_m)
    )
    output_extent_m = total_samples * output_pixel_size_m
    symmetric_padding = (
        (total_samples - plate_samples) // 2
        if (total_samples - plate_samples) % 2 == 0 else None
    )
    return {
        "plate_samples": int(plate_samples),
        "propagation_samples": int(total_samples),
        "symmetric_padding_per_side": symmetric_padding,
        "plate_pixel_size_m": float(dx_plate_m),
        "requested_output_pixel_size_m": (
            np.nan if requested_output_pixel_size_m is None
            else float(requested_output_pixel_size_m)
        ),
        "output_pixel_size_m": float(output_pixel_size_m),
        "output_extent_m": float(output_extent_m),
    }


def _fresnel_phase_factors(
        shape, dx_input_m, wavelength_m, propagation_distance_m,
        focal_length_m):
    """Return the input, lens and output quadratic phases for Fresnel FFTs."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Fresnel propagation requires a square field, got {shape}")
    if min(dx_input_m, wavelength_m, propagation_distance_m, focal_length_m) <= 0:
        raise ValueError("Fresnel sampling and optical distances must be positive")

    samples = shape[0]
    k = 2 * np.pi / wavelength_m
    input_axis = centered_axis(samples, dx_input_m)
    input_radius_squared = (
        input_axis[None, :] ** 2 + input_axis[:, None] ** 2
    )
    input_quadratic = np.exp(
        1j * k * input_radius_squared / (2 * propagation_distance_m)
    )
    focusing_lens = np.exp(
        -1j * k * input_radius_squared / (2 * focal_length_m)
    )

    output_axis = propagation_axis(
        samples, dx_input_m, wavelength_m, propagation_distance_m
    )
    output_radius_squared = (
        output_axis[None, :] ** 2 + output_axis[:, None] ** 2
    )
    output_quadratic = np.exp(
        1j * k * output_radius_squared / (2 * propagation_distance_m)
    )
    return input_quadratic, focusing_lens, output_quadratic, output_axis


def propagate_for_gs(
        field, dx_input_m, wavelength_m, focal_length_m,
        propagation_model="fraunhofer", propagation_distance_m=None):
    """Forward propagation operator used inside Gerchberg-Saxton.

    Constant prefactors and integration-area factors are omitted because GS
    replaces the output amplitude on every iteration.  Quadratic phase factors
    are retained, since they are essential to a correct Fresnel inverse.
    """
    model = normalise_propagation_model(propagation_model)
    distance = (
        focal_length_m if propagation_distance_m is None
        else propagation_distance_m
    )
    if model == "fraunhofer":
        return _centered_fft2(field)

    input_quadratic, focusing_lens, output_quadratic, _ = (
        _fresnel_phase_factors(
            field.shape,
            dx_input_m,
            wavelength_m,
            distance,
            focal_length_m,
        )
    )
    return output_quadratic * _centered_fft2(
        field * focusing_lens * input_quadratic
    )


def backpropagate_for_gs(
        output_field, dx_input_m, wavelength_m, focal_length_m,
        propagation_model="fraunhofer", propagation_distance_m=None):
    """Exact discrete inverse of :func:`propagate_for_gs` up to round-off."""
    model = normalise_propagation_model(propagation_model)
    distance = (
        focal_length_m if propagation_distance_m is None
        else propagation_distance_m
    )
    if model == "fraunhofer":
        return _centered_ifft2(output_field)

    input_quadratic, focusing_lens, output_quadratic, _ = (
        _fresnel_phase_factors(
            output_field.shape,
            dx_input_m,
            wavelength_m,
            distance,
            focal_length_m,
        )
    )
    weighted_input = _centered_ifft2(
        output_field * np.conj(output_quadratic)
    )
    return weighted_input * np.conj(focusing_lens * input_quadratic)


# =============================================================================
# Reproducible run logging
# =============================================================================
import os, json, inspect, datetime, pathlib

def save_call(log_dir="runs", save_arrays=True, write_markdown=True,
              include_source=False):
    """
    Decorator that logs each call's inputs to disk and includes the function's
    docstring at the top of a Markdown run report. Also embeds docstring/signature
    inside the JSON for machine use.
    """
    os.makedirs(log_dir, exist_ok=True)

    def decorator(func):
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        src = None
        if include_source:
            try:
                src = inspect.getsource(func)
            except OSError:
                src = None

        def to_jsonable(val):
            # make values JSON serializable
            if isinstance(val, np.ndarray):
                return {"__ndarray__": True, "shape": list(val.shape), "dtype": str(val.dtype)}
            if isinstance(val, (np.generic,)):
                return val.item()
            if isinstance(val, pathlib.Path):
                return {"__path__": str(val)}
            try:
                json.dumps(val)
                return val
            except TypeError:
                return {"__repr__": repr(val)}

        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Collect metadata + arrays
            meta = {
                "_function": func.__name__,
                "_signature": str(sig),
                "_docstring": doc,
                "_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            arrays = {}
            for name, val in bound.arguments.items():
                if isinstance(val, np.ndarray) and save_arrays:
                    arrays[name] = val
                    meta[name] = to_jsonable(val)  # includes shape/dtype marker
                else:
                    meta[name] = to_jsonable(val)

            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            base  = os.path.join(log_dir, f"{func.__name__}_{stamp}")

            # Write JSON (includes docstring/signature)
            with open(base + ".json", "w") as f:
                json.dump(meta, f, indent=2)

            # Save arrays
            if arrays:
                np.savez(base + ".npz", **arrays)

            # Optional human-readable Markdown report with docstring at top
            if write_markdown:
                with open(base + ".md", "w", encoding="utf-8") as f:
                    f.write(f"# {func.__name__} run\n\n")
                    f.write(f"- Timestamp: {meta['_timestamp']}\n")
                    f.write(f"- Signature: `{func.__name__}{sig}`\n\n")
                    if doc:
                        f.write("## Docstring\n\n")
                        f.write("```text\n" + doc + "\n```\n\n")
                    f.write("## Inputs\n\n```json\n")
                    f.write(json.dumps({k:v for k,v in meta.items() if not k.startswith('_')},
                                       indent=2))
                    f.write("\n```\n\n")
                    if arrays:
                        f.write("## Arrays\n\n")
                        for n, a in arrays.items():
                            f.write(f"- **{n}**: shape={a.shape}, dtype={a.dtype}\n")
                    if include_source and src:
                        f.write("\n## Source\n\n```python\n" + src + "\n```\n")

            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Target-intensity and input-beam construction
# =============================================================================
def generate_focal_spot_modulated_supergaussian_x(
        pixels=1000,
        extent_m=0.2,
        w_m=25e-6,
        n=5,
        mod_amp=0.2,
        mod_period_m=40e-6,
        I_peak=1e14
    ):
    """
    Generate a 2D modulated super-Gaussian intensity map representing the focal spot,
    with a stripe modulation along the X direction.
    
    Parameters
    ----------
    pixels : int
        Number of pixels along each axis.
    extent_m : float
        Total width/height of the grid [m].
    w_m : float
        1/e^n radius of the Gaussian envelope [m].
    n : int
        Order of the super-Gaussian.
    mod_amp : float
        Modulation amplitude (relative).
    mod_period_m : float
        Spatial period of the cosine modulation [m].
    I_peak : float
        Peak intensity to scale to (same units as final I_modulated).
    
    Returns
    -------
    I_modulated : 2D ndarray
        Intensity map [same units as I_peak].
    x, y : 1D ndarrays
        Coordinate arrays in metres.
    """
    # create coordinate grid in metres
    x = np.linspace(-extent_m/2, extent_m/2, pixels)
    y = np.linspace(-extent_m/2, extent_m/2, pixels)
    X, Y = np.meshgrid(x, y)

    # super-Gaussian envelope
    R = np.sqrt(X**2 + Y**2)
    I_gauss = np.exp(-(R / w_m)**(2 * n))

    # stripe modulation along X only
    kx = 2 * np.pi / mod_period_m
    modulation = 1 + mod_amp * np.cos(kx * X)

    # combine and scale
    I_modulated = I_gauss * modulation
    I_modulated /= np.max(I_modulated)
    I_modulated *= I_peak

    return I_modulated, x, y


def generate_focal_spot_modulated_tophat_x(
        pixels=1000,
        extent_m=0.2,
        radius_m=25e-6,
        mod_amp=0.2,
        mod_period_m=40e-6,
        I_peak=1e14,
        num_stripes=7,
    ):
    """Generate a solid-striped circular top-hat focal-intensity target.

    ``num_stripes`` is the number of *bright visible stripes* inside the
    circular mask. The stripe pattern is first constructed across a square of
    side ``2*radius_m`` and then clipped by the circular mask. Equal-width dark
    gaps are inserted between neighbouring bright stripes, so ``num_stripes=7``
    gives seven bright bands and six dark gaps.

    ``mod_amp`` controls the dark-gap level relative to the bright stripes.
    With ``mod_amp=1`` the gaps are exactly zero intensity.
    """
    if pixels < 1:
        raise ValueError("pixels must be positive")
    if extent_m <= 0:
        raise ValueError("extent_m must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if mod_period_m <= 0:
        raise ValueError("mod_period_m must be positive")
    if not 0 <= mod_amp <= 1:
        raise ValueError("mod_amp must lie in [0, 1]")
    if I_peak < 0:
        raise ValueError("I_peak cannot be negative")

    num_stripes = int(num_stripes)
    if num_stripes < 1:
        raise ValueError("num_stripes must be at least 1")

    x = np.linspace(-extent_m / 2, extent_m / 2, pixels)
    y = np.linspace(-extent_m / 2, extent_m / 2, pixels)
    X, Y = np.meshgrid(x, y)

    # Build the stripe pattern on a square first, then mask it to a circle.
    stripe_square_mask = (
        (np.abs(X) <= radius_m) & (np.abs(Y) <= radius_m)
    )

    # Seven requested stripes means seven BRIGHT bands, not seven alternating
    # bright/dark bands. Therefore use 2*N-1 equal bands across the diameter:
    # bright, dark, bright, ..., dark, bright.
    total_bands = 2 * num_stripes - 1
    band_width_m = (2.0 * radius_m) / total_bands
    band_index = np.floor((X + radius_m) / band_width_m).astype(int)
    band_index = np.clip(band_index, 0, total_bands - 1)

    bright_level = 1.0 + mod_amp
    dark_level = 1.0 - mod_amp
    stripe_levels = np.where(band_index % 2 == 0, bright_level, dark_level)
    striped_square = stripe_levels * stripe_square_mask.astype(float)

    circular_mask = (np.hypot(X, Y) <= radius_m).astype(float)
    I_modulated = striped_square * circular_mask

    peak = np.max(I_modulated)
    if peak > 0:
        I_modulated = I_modulated / peak
    I_modulated *= I_peak

    return I_modulated, x, y

def generate_focal_spot_modulated_target_x(
        target_profile,
        pixels=1000,
        extent_m=0.2,
        radius_m=25e-6,
        supergaussian_order=5.2,
        mod_amp=0.2,
        mod_period_m=40e-6,
        I_peak=1e14,
        top_hat_num_stripes=7,
    ):
    """Generate the selected striped circular focal-intensity target."""
    target_profile = normalise_target_profile(target_profile)

    if target_profile == "supergaussian":
        return generate_focal_spot_modulated_supergaussian_x(
            pixels=pixels,
            extent_m=extent_m,
            w_m=radius_m,
            n=supergaussian_order,
            mod_amp=mod_amp,
            mod_period_m=mod_period_m,
            I_peak=I_peak,
        )

    return generate_focal_spot_modulated_tophat_x(
        pixels=pixels,
        extent_m=extent_m,
        radius_m=radius_m,
        mod_amp=mod_amp,
        mod_period_m=mod_period_m,
        I_peak=I_peak,
        num_stripes=top_hat_num_stripes,
    )


def generate_focal_spot_modulated_supergaussian(
        pixels=1000,
        extent_m=0.2,
        w_m=25e-6,
        n=5,
        mod_amp=0.2,
        mod_period_m=40e-6,
        I_peak=1e14
    ):
    """
    Generate a 2D modulated super-Gaussian intensity map representing the focal spot,
    with all distances in metres.
    
    Parameters
    ----------
    pixels : int
        Number of pixels along each axis.
    extent_m : float
        Total width/height of the grid [m].
    w_m : float
        1/e^n radius of the Gaussian envelope [m].
    n : int
        Order of the super-Gaussian.
    mod_amp : float
        Modulation amplitude (relative).
    mod_period_m : float
        Spatial period of the cosine modulation [m].
    I_peak : float
        Peak intensity to scale to (same units as final I_modulated).
    
    Returns
    -------
    I_modulated : 2D ndarray
        Intensity map [same units as I_peak].
    x, y : 1D ndarrays
        Coordinate arrays in metres.
    """
    # create coordinate grid in metres
    x = np.linspace(-extent_m/2, extent_m/2, pixels)
    y = np.linspace(-extent_m/2, extent_m/2, pixels)
    X, Y = np.meshgrid(x, y)

    # super-Gaussian envelope
    R = np.sqrt(X**2 + Y**2)
    I_gauss = np.exp(-(R / w_m)**(2 * n))

    # cosine modulation
    kx = 2 * np.pi / mod_period_m
    ky = 2 * np.pi / mod_period_m
    modulation = 1 + mod_amp * np.cos(kx * X) * np.cos(ky * Y)

    # combine and scale
    I_modulated = I_gauss * modulation
    I_modulated /= np.max(I_modulated)
    I_modulated *= I_peak

    return I_modulated, x, y


# Optional reconstruction of an input field from a specified focus.
def reconstruct_input_beam_from_focus(
        I_focus,
        x_um,
        y_um,
        wavelength_um=0.8,
        f_number=2.5,
        beam_diameter_um=20000,
        log_clip=1e-10
    ):
    """
    Back-propagate from the focal plane to reconstruct the input beam intensity.
    """
    pixels = I_focus.shape[0]
    focal_length_um = f_number * beam_diameter_um

    # Field amplitude at focus (flat phase assumption)
    E_focus = np.sqrt(I_focus)

    # Inverse FFT to input plane
    E_backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(E_focus)))

    # Create spatial axes for input plane
    dx_um = x_um[1] - x_um[0]
    fx = np.fft.fftfreq(pixels, d=dx_um)
    fx_shifted = np.fft.fftshift(fx)
    Xf, Yf = np.meshgrid(fx_shifted, fx_shifted)
    x_input_um = Xf * wavelength_um * focal_length_um
    y_input_um = Yf * wavelength_um * focal_length_um

    # Lens phase to remove (inverse of thin-lens transform)
    X_flat, Y_flat = np.meshgrid(x_um, y_um)
    lens_phase_inv = np.exp(+1j * np.pi * (X_flat**2 + Y_flat**2) / (wavelength_um * focal_length_um))
    E_input = E_backprop * lens_phase_inv

    # Input beam intensity (W/cm²)
    I_input = np.abs(E_input)**2

    # Plot log10 of input intensity
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(I_input + log_clip),
               extent=[x_input_um.min(), x_input_um.max(), y_input_um.min(), y_input_um.max()],
               cmap='inferno', origin='lower')
    plt.colorbar(label='log10(Intensity [W/cm²])')
    plt.title('Log-scaled Reconstructed Input Beam')
    plt.xlabel('x (µm)')
    plt.ylabel('y (µm)')
    plt.tight_layout()
    plt.show()

    return I_input, x_input_um, y_input_um


def load_input_beam(filepath, target_shape):
    """
    Load the input beam intensity profile or initial phase (e.g., from a wavefront sensor)
    from a NumPy array file.
    
    Parameters:
      filepath: Path to the NumPy file (e.g., 'input_beam.npy').
      target_shape: Desired shape (rows, cols) for the simulation grid.
      
    Returns:
      beam: 2D numpy array with normalized intensity values.
    """
    # Load the beam data from file
    beam = np.load(filepath)
    # If there are extra dimensions, remove them to obtain a 2D array
    if beam.ndim > 2:
        beam = np.squeeze(beam)
    
    # Convert to float64 for precision in subsequent computations
    beam = beam.astype(np.float64)
    
    # If the beam's shape doesn't match the target shape, resize it using OpenCV's interpolation
    if beam.shape != target_shape:
        import cv2  # cv2 is handy for resizing images/arrays
        beam = cv2.resize(beam, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return beam

def load_input_beam_png(filepath, target_shape):
    """
    Load the input beam intensity profile from an image file.
    
    Parameters:
      filepath: Path to the intensity image file.
      target_shape: Tuple specifying the desired shape (height, width).
      
    Returns:
      beam: 2D numpy array with normalized intensity values (ranging from 0 to 1).
    """
    # Read the image in grayscale mode
    beam = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if beam is None:
        raise ValueError(f"Could not read file: {filepath}")
    # Convert the image data to float64 (normalization can be applied if needed)
    beam = beam.astype(np.float64)
    
    # Resize the image if its shape does not match the target shape
    if beam.shape != target_shape:
        beam = cv2.resize(beam, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return beam

# =============================================================================
# Aperture masks and optional legacy post-processing helpers
# =============================================================================

def apply_circular_mask_beam(Z, x, z, diameter):
    """
    Apply a circular mask of given diameter (same units as x, z) to 2D array Z.
    Points outside the circle are set to zero.
    """
    X, Zg = np.meshgrid(x, z)
    center_x = x.mean()
    center_z = z.mean()
    radius = diameter / 2
    mask = (X - center_x)**2 + (Zg - center_z)**2 <= radius**2
    return Z * mask, mask


def apply_circular_mask(image):
    """
    Apply a circular mask to an image array, preserving values only within the circle.
    
    Parameters:
      image: 2D numpy array representing an image.
      
    Returns:
      mask: 2D numpy array with the same shape as 'image' where values outside the circle are zero.
    """
    # Initialize an array of zeros with the same shape as the image
    mask = np.zeros(image.shape)
    # Calculate the center of the image
    center = (image.shape[0] // 2, image.shape[1] // 2)
    # Determine the radius as half the minimum dimension of the image
    radius = min(image.shape) // 2
    # Generate a grid of indices for the image
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    # Create a boolean array that is True inside the circle and False outside
    mask_area = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
    # Copy the original image values to the mask only where the condition is True
    mask[mask_area] = image[mask_area]
    return mask

def apply_circular_mask_gpt(image, diameter=None, center=None, fill=0, as_bool=False, pixel_size=None):
    """
    Apply a circular mask to an image.

    Parameters
    ----------
    image : 2D np.ndarray
        The input image.
    diameter : float or None
        Circle diameter. If `pixel_size` is None, interpreted in **pixels**.
        If `pixel_size` is provided (units per pixel, e.g. meters/pixel), then
        `diameter` is interpreted in those **physical units**.
        If None, defaults to min(image.shape).
    center : (int, int) or None
        (row, col) center of the circle. Defaults to the image center.
    fill : scalar
        Value outside the circle (ignored if `as_bool=True`). Default 0.
    as_bool : bool
        If True, return the boolean mask instead of the masked image.
    pixel_size : float or None
        Physical size per pixel. If provided, `diameter` is taken in the same units.

    Returns
    -------
    masked : 2D np.ndarray
        If `as_bool=False`, the image with values outside the circle set to `fill`.
        If `as_bool=True`, a boolean mask (True inside the circle).
    """
    H, W = image.shape

    # Center
    if center is None:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = center

    # Diameter -> radius in pixels
    if diameter is None:
        radius_px = min(H, W) / 2.0
    else:
        radius_px = (diameter / pixel_size) if (pixel_size is not None) else float(diameter)
        radius_px *= 0.5

    # Build mask
    Y, X = np.ogrid[:H, :W]
    mask = (X - cx)**2 + (Y - cy)**2 <= radius_px**2

    if as_bool:
        return mask

    out = np.full_like(image, fill)
    out[mask] = image[mask]
    return out

# The following plotting and spline helpers are retained for optional visual
# inspection of continuous plates. They are not used by the main MK2 workflow.

def build_flat_top_hex_grid(plate_size: float, element_size: float):
    """
    Build a flat-topped hexagonal grid (with no overlaps) using direct geometry.
    The hexagons will be used to represent phase elements.
    
    Parameters:
      plate_size: Physical size of the phase plate.
      element_size: Horizontal corner-to-corner width of each hexagon.
    
    Returns:
      A NumPy array of shape (N_hex, 6, 2) where each hexagon is defined by its 6 vertices (x, y).
    """
    # Compute side length (distance from the hexagon center to a vertex)
    s = element_size / 2.0
    
    # Define the vertex angles for a flat-top hexagon (in degrees)
    angles_deg = [30, 90, 150, 210, 270, 330]
    angles_rad = np.radians(angles_deg)  # Convert angles to radians
    
    # Compute the spacing between hexagon centers
    dx = 1.745 * s          # Horizontal spacing
    dy = np.float16(np.sqrt(3 - 0.71) * s)  # Vertical spacing using an empirical adjustment factor
    offset_x = dx / 2.0       # Horizontal offset for every other (odd) row to create the staggered grid

    # Define the bounding box centered at (0,0)
    x_min, x_max = -plate_size / 2, plate_size / 2
    y_min, y_max = -plate_size / 2, plate_size / 2

    # Determine the number of rows and columns needed to cover the plate
    nrows = int(np.ceil((y_max - y_min) / dy)) + 2
    ncols = int(np.ceil((x_max - x_min) / dx)) + 2

    hex_list = []  # List to hold all hexagon vertex arrays
    for row in range(nrows):
        center_y = y_min + row * dy  # Y-coordinate of the hexagon center for this row
        for col in range(ncols):
            center_x = x_min + col * dx  # X-coordinate for the current column
            # For odd rows, shift the x-coordinate by the offset
            if row % 2 == 1:
                center_x += offset_x

            # Skip hexagons that fall outside the defined bounding box (with a margin)
            if center_x < x_min - dx or center_x > x_max + dx:
                continue
            if center_y < y_min - dy or center_y > y_max + dy:
                continue

            # Compute the six vertices for the hexagon
            hex_verts = []
            for theta in angles_rad:
                vx = center_x + s * np.cos(theta)
                vy = center_y + s * np.sin(theta)
                hex_verts.append((vx, vy))
            hex_list.append(hex_verts)

    # Convert the list of vertices into a NumPy array for further use
    return np.array(hex_list)  # Shape: (N_hex, 6, 2)

def plot_phase_map_with_hex_overlay(phase_map, plate_size, hex_array):
    """
    Plot the phase map (phase plate) with an overlay of the hexagonal grid.
    Also draws a circular boundary corresponding to the plate.
    
    Parameters:
      phase_map: 2D numpy array of phase values (radians).
      plate_size: Physical plate size (m).
      hex_array: Array of hexagon vertices (shape: (N_hex, 6, 2)).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (-plate_size/2, plate_size/2, -plate_size/2, plate_size/2)
    im = ax.imshow(phase_map, extent=extent, origin='upper', cmap='inferno')
    fig.colorbar(im, ax=ax, label='Phase (radians)')
    
    # Overlay each hexagon from the hex_array onto the phase map
    for hex_verts in hex_array:
        patch = Polygon(hex_verts, closed=True, edgecolor='k', facecolor='none', lw=1)
        ax.add_patch(patch)
    
    # Draw a circular boundary (red) to indicate the plate edge
    circle = Circle((0, 0), plate_size/2, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(circle)
    
    ax.set_xlim(-plate_size/2, plate_size/2)
    ax.set_ylim(-plate_size/2, plate_size/2)
    ax.set_aspect('equal')
    ax.set_title("Phase Map with Phase Element Overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.show()

def smooth_theta(theta, plate_size, order=4, smoothing=0):
    """
    Smooth the phase plate 'theta' using a bivariate spline interpolation.
    
    Parameters:
      theta: 2D numpy array of phase values (radians).
      plate_size: Physical size of the phase plate (m).
      order: Order of the spline (default 4th order).
      smoothing: Smoothing factor (set to 0 to interpolate exactly through points).
    
    Returns:
      theta_smooth: 2D numpy array of smoothed phase values evaluated on a finer grid.
    """
    nrows, ncols = theta.shape
    # Create coordinate arrays that span from -plate_size/2 to plate_size/2
    x = np.linspace(-plate_size / 2, plate_size / 2, ncols)
    y = np.linspace(-plate_size / 2, plate_size / 2, nrows)
    
    # Create the bivariate spline with specified order and smoothing.
    # Note: The first coordinate (y) comes first since the input array is (rows, cols)
    spline = RectBivariateSpline(y, x, theta, kx=order, ky=order, s=smoothing)
    
    # Retrieve and print the coefficients of the spline for inspection
    coeffs = spline.get_coeffs()
    print("Spline coefficients shape:", coeffs.shape)
    print("Spline coefficients:", coeffs)

    # Evaluate the spline on a finer grid to produce a smooth output
    new_grid_x = np.linspace(-plate_size / 2, plate_size / 2, ncols*10)
    new_grid_y = np.linspace(-plate_size / 2, plate_size / 2, nrows*10)
    theta_smooth = spline(new_grid_y, new_grid_x)
    return theta_smooth


def compute_focal_spot_with_smoothed_phase(theta_smoothed, plate_size, desired_focal_spot, 
                                           input_beam_filepath=None, file_type="npy", beam_fwhm=None):
    """
    Compute and visualize the focal spot after propagation through a smoothed phase plate.

    This function performs the following steps:
      1. Constructs a fine coordinate grid based on the resolution of theta_smoothed.
      2. Loads an input beam from file (if provided) and upscales it to match the fine grid. 
         If no beam file is provided, a Gaussian beam is generated using a FWHM.
         The FWHM is converted to the standard deviation sigma via:
             sigma = beam_fwhm / (2 * sqrt(2 * ln(2)))
         By default, beam_fwhm is set to the plate_size, ensuring the beam is comparable to 
         the physical extent of the phase plate.
      3. Applies the smoothed phase (theta_smoothed) to the beam.
      4. Propagates the field via a Fourier transform to compute the focal spot.
      5. Plots three panels:
           - The high-resolution input beam intensity.
           - The smoothed phase plate.
           - The focal spot intensity (with an extent based on desired_focal_spot).

    Parameters:
      theta_smoothed (np.ndarray): 2D array of the smoothed phase plate (radians).
      plate_size (float): Diameter of the phase plate (m).
      desired_focal_spot (float): Size of the desired focal spot (m) for plotting.
      input_beam_filepath (str, optional): Path to the stored input beam file.
      file_type (str): Type of file to load ("npy" or "image"). Default is "npy".
      beam_fwhm (float, optional): Full-width at half-maximum of the Gaussian beam (m). 
                                   If None, defaults to plate_size.

    Returns:
      np.ndarray: The computed focal intensity distribution.
    """

    # Set default beam FWHM if not provided.
    if beam_fwhm is None:
        beam_fwhm = plate_size  # Default: FWHM comparable to the plate size

    # Retrieve fine grid dimensions from the smoothed phase plate.
    nrows, ncols = theta_smoothed.shape

    # Construct a fine coordinate grid that spans the plate.
    x_fine = np.linspace(-plate_size / 2, plate_size / 2, ncols)
    y_fine = np.linspace(-plate_size / 2, plate_size / 2, nrows)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Load and upscale the input beam if a file is provided.
    if input_beam_filepath is not None:
        if file_type.lower() in ["npy", "numpy"]:
            native_beam = np.load(input_beam_filepath)
        elif file_type.lower() == "image":
            native_beam = cv2.imread(input_beam_filepath, cv2.IMREAD_GRAYSCALE)
            if native_beam is None:
                raise ValueError(f"Could not read file: {input_beam_filepath}")
        else:
            raise ValueError("Unsupported file_type. Use 'npy' or 'image'.")
        
        if native_beam.ndim > 2:
            native_beam = np.squeeze(native_beam)
        native_beam = native_beam.astype(np.float64)
        
        beam_high_res = cv2.resize(native_beam, (ncols, nrows), interpolation=cv2.INTER_LINEAR)
        beam_high_res = beam_high_res / np.max(beam_high_res)
        print("Input beam loaded and upscaled from shape {} to {}."
              .format(native_beam.shape, beam_high_res.shape))
    else:
        # Generate a Gaussian beam based on the provided FWHM.
        # Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2))).
        sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        beam_high_res = np.exp(- (X_fine**2 + Y_fine**2) / (2 * sigma**2))
        print("No input beam file provided. Generated a Gaussian beam with FWHM = {:.3e} m on the fine grid."
              .format(beam_fwhm))
    
    # Form the complex field by applying the smoothed phase.
    phase_field_smoothed = beam_high_res * np.exp(1j * theta_smoothed)
    
    # Propagate to the focal plane via Fourier transform.
    focal_field = np.fft.fftshift(np.fft.fft2(phase_field_smoothed))
    focal_intensity_smoothed = np.abs(focal_field)**2
    
    # Plot the three components.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the high-resolution input beam.
    im0 = axs[0].imshow(apply_circular_mask(beam_high_res), extent=[-plate_size/2, plate_size/2,
                                                 -plate_size/2, plate_size/2],
                          cmap='gray')
    axs[0].set_title("High-Resolution Input Beam (Gaussian)")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Normalized Intensity")
    
    # Plot the smoothed phase plate.
    im1 = axs[1].imshow(apply_circular_mask(theta_smoothed), extent=[-plate_size/2, plate_size/2,
                                                  -plate_size/2, plate_size/2],
                          cmap='hsv')
    axs[1].set_title("Smoothed Phase Plate")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label="Phase (radians)")
    
    # Plot the focal spot intensity.
    im2 = axs[2].imshow(np.log(apply_circular_mask(focal_intensity_smoothed)),
                          extent=[-desired_focal_spot/2, desired_focal_spot/2,
                                  -desired_focal_spot/2, desired_focal_spot/2],
                          cmap='inferno')
    axs[2].set_title("Focal Spot (Log Intensity)")
    axs[2].set_xlabel("x (m)")
    axs[2].set_ylabel("y (m)")
    fig.colorbar(im2, ax=axs[2], shrink=0.8, label="Log Intensity")
    
    plt.tight_layout()
    plt.show()
    
    #np.save("focal_intensity_smoothed.npy", focal_intensity_smoothed)
    return focal_intensity_smoothed

# =============================================================================
# Manufacturing constraints and square/hexagonal element projection
# =============================================================================

def make_radial_bands(shape, r_edges_px):
    """
    Build a radial band index map.

    Parameters
    ----------
    shape : (ny, nx)
        Shape of the phase plate array in pixels.
    r_edges_px : 1D array-like
        Radii (in pixels) that define the band boundaries:
        - band 0: r < r_edges_px[0]
        - band 1: r_edges_px[0] <= r < r_edges_px[1]
        - ...
        - band N: r >= r_edges_px[-1]
    """
    ny, nx = shape
    y = np.arange(ny) - (ny - 1) / 2.0
    x = np.arange(nx) - (nx - 1) / 2.0
    X, Y = np.meshgrid(x, y, indexing="xy")
    r_map = np.sqrt(X**2 + Y**2)

    r_edges_px = np.asarray(r_edges_px, dtype=float)
    band_map = np.digitize(r_map, r_edges_px, right=False)
    return band_map, r_map


def apply_radial_blocking(phase, band_map, block_sizes_px):
    """
    Enforce piecewise-constant phase in radial bands, with band-dependent
    'element size' (block size in pixels).
    """
    phase_out = np.array(phase, copy=True)
    ny, nx = phase.shape
    band_map = np.asarray(band_map, dtype=int)
    block_sizes_px = np.asarray(block_sizes_px, dtype=int)

    n_bands = band_map.max() + 1
    if block_sizes_px.shape[0] < n_bands:
        raise ValueError(
            f"Need block_sizes_px for {n_bands} bands, "
            f"only got {block_sizes_px.shape[0]}"
        )

    for b_idx in range(n_bands):
        bsize = int(block_sizes_px[b_idx])
        if bsize <= 1:
            continue  # no blocking in this band

        mask_band = (band_map == b_idx)
        if not mask_band.any():
            continue

        for y0 in range(0, ny, bsize):
            y1 = min(y0 + bsize, ny)
            for x0 in range(0, nx, bsize):
                x1 = min(x0 + bsize, nx)

                block_mask = mask_band[y0:y1, x0:x1]
                if not block_mask.any():
                    continue

                block_vals = phase_out[y0:y1, x0:x1][block_mask]
                complex_mean = np.mean(np.exp(1j * block_vals))
                mean_phase = np.angle(complex_mean)

                block = phase_out[y0:y1, x0:x1]
                block[block_mask] = mean_phase
                phase_out[y0:y1, x0:x1] = block

    return phase_out


def build_isotropic_manufacturing_filter(shape, sigma_px):
    """Return an isotropic Gaussian transfer function for a phase grid."""
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    return np.exp(-2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2))


def project_to_correlated_binary_phase(theta, transfer, pi_fraction):
    """Project a continuous phase onto organic correlated 0/pi regions.

    ``-cos(theta)`` is the preference score for pi rather than zero phase.  The
    score is filtered isotropically and thresholded while preserving a fixed
    pi-area fraction.  Unlike square blocking, this projection has no preferred
    boundary direction.
    """
    if not 0 < pi_fraction < 1:
        raise ValueError("pi_fraction must lie in (0, 1)")
    score = -np.cos(theta)
    correlated_score = np.fft.ifft2(
        np.fft.fft2(score) * transfer
    ).real
    threshold = np.quantile(correlated_score, 1.0 - pi_fraction)
    return (correlated_score >= threshold).astype(float) * np.pi


def project_to_correlated_continuous_phase(theta, transfer):
    """Apply an isotropic correlation constraint without quantising phase."""
    phasor = np.exp(1j * theta)
    correlated_phasor = np.fft.ifft2(
        np.fft.fft2(phasor) * transfer
    )
    return np.angle(correlated_phasor)


def quantize_phase(theta, num_steps):
    """Quantise wrapped phase to exactly ``num_steps`` levels in [0, 2*pi)."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 for a quantized plate")
    step_size = 2 * np.pi / num_steps
    quantized = np.round(np.mod(theta, 2 * np.pi) / step_size) * step_size
    quantized[quantized >= 2 * np.pi] = 0
    return quantized


def project_to_correlated_quantized_phase(theta, transfer, num_steps):
    """Apply isotropic phase correlation, then quantise to N phase levels."""
    return quantize_phase(
        project_to_correlated_continuous_phase(theta, transfer),
        num_steps,
    )


def compute_realistic_element_layout(
        plate_size, wavelength, focal_length, desired_focal_spot,
        plate_pixels):
    """Return an integer element layout that exactly tiles the square plate.

    The requested element size ``lambda*f/d_focus`` will almost never divide
    the plate exactly.  The nearest integer number of elements is therefore
    chosen and the realised element size is ``plate_size / count``.
    """
    if plate_size <= 0 or wavelength <= 0 or focal_length <= 0:
        raise ValueError("plate_size, wavelength and focal_length must be positive")
    if desired_focal_spot <= 0:
        raise ValueError("desired_focal_spot must be positive")
    if plate_pixels < 1:
        raise ValueError("plate_pixels must be positive")

    requested_element_size = wavelength * focal_length / desired_focal_spot
    number_elements = int(round(plate_size / requested_element_size))
    number_elements = int(np.clip(number_elements, 1, plate_pixels))
    actual_element_size = plate_size / number_elements
    actual_focal_spot = wavelength * focal_length / actual_element_size
    return (requested_element_size, number_elements,
            actual_element_size, actual_focal_spot)


def build_phase_aperture_mask(shape, plate_size, aperture_shape):
    """Return the physical square/circular aperture on a phase-map grid."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"phase-map shape must be square, got {shape}")
    pixel_size = plate_size / shape[0]
    axis = centered_axis(shape[0], pixel_size)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    return make_pupil(X, Y, plate_size, aperture_shape).astype(bool)


def project_phase_to_square_elements(
        phase, number_elements, num_steps=None, valid_mask=None):
    """Collapse a fine phase map into exactly ``number_elements`` per side.

    A circular phase mean is used so values close to zero and 2*pi average
    correctly.  The coarse element map is rasterised back onto the original
    fine grid for propagation; the fine pixels are then samples of the same
    physical element rather than independent phase degrees of freedom.
    """
    theta = np.asarray(phase, dtype=float)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"phase must be a square 2-D array, got {theta.shape}")
    if not 1 <= number_elements <= theta.shape[0]:
        raise ValueError("number_elements must lie between 1 and the plate pixels")
    if valid_mask is None:
        valid_mask = np.ones(theta.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != theta.shape:
            raise ValueError("valid_mask must have the same shape as phase")

    plate_pixels = theta.shape[0]
    edges = np.rint(
        np.linspace(0, plate_pixels, number_elements + 1)
    ).astype(int)
    element_phase = np.zeros((number_elements, number_elements), dtype=float)

    for row in range(number_elements):
        y0, y1 = edges[row], edges[row + 1]
        for col in range(number_elements):
            x0, x1 = edges[col], edges[col + 1]
            block_mask = valid_mask[y0:y1, x0:x1]
            if not block_mask.any():
                element_phase[row, col] = 0.0
                continue
            block = theta[y0:y1, x0:x1][block_mask]
            if num_steps == 2:
                # For a binary plate this is an unambiguous majority decision.
                element_phase[row, col] = (
                    0.0 if np.mean(np.cos(block)) >= 0 else np.pi
                )
            else:
                phasor = np.mean(np.exp(1j * block))
                element_phase[row, col] = np.mod(np.angle(phasor), 2 * np.pi)

    if num_steps is not None:
        element_phase = quantize_phase(element_phase, num_steps)

    phase_realistic = np.empty_like(theta)
    for row in range(number_elements):
        y0, y1 = edges[row], edges[row + 1]
        for col in range(number_elements):
            x0, x1 = edges[col], edges[col + 1]
            phase_realistic[y0:y1, x0:x1] = element_phase[row, col]

    phase_realistic[~valid_mask] = 0.0
    return np.mod(phase_realistic, 2 * np.pi), element_phase, edges


def _round_hex_axial_coordinates(q_fractional, r_fractional):
    """Round fractional flat-top axial coordinates to their nearest hex cell."""
    cube_x = np.asarray(q_fractional, dtype=float)
    cube_z = np.asarray(r_fractional, dtype=float)
    cube_y = -cube_x - cube_z

    rounded_x = np.rint(cube_x)
    rounded_y = np.rint(cube_y)
    rounded_z = np.rint(cube_z)
    x_error = np.abs(rounded_x - cube_x)
    y_error = np.abs(rounded_y - cube_y)
    z_error = np.abs(rounded_z - cube_z)

    correct_x = (x_error > y_error) & (x_error > z_error)
    correct_y = (~correct_x) & (y_error > z_error)
    correct_z = ~(correct_x | correct_y)
    rounded_x[correct_x] = -rounded_y[correct_x] - rounded_z[correct_x]
    rounded_y[correct_y] = -rounded_x[correct_y] - rounded_z[correct_y]
    rounded_z[correct_z] = -rounded_x[correct_z] - rounded_y[correct_z]
    return rounded_x.astype(np.int32), rounded_z.astype(np.int32)


def project_phase_to_hexagonal_elements(
        phase, element_pitch_pixels, num_steps=None, valid_mask=None):
    """Project a fine phase map onto a staggered flat-top hexagonal lattice.

    ``element_pitch_pixels`` is the nearest-neighbour centre pitch and therefore
    the flat-to-flat hexagon width.  The corner-to-corner width is
    ``2*element_pitch_pixels/sqrt(3)``.  Each fine pixel is assigned to its
    nearest lattice centre using cube-coordinate rounding.  A circular phase
    mean is then applied to every complete or aperture-clipped physical cell.

    Returns the rasterised phase, one phase per occupied cell, the matching
    axial ``(q, r)`` cell coordinates and the number of valid raster pixels in
    each cell.
    """
    theta = np.asarray(phase, dtype=float)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"phase must be a square 2-D array, got {theta.shape}")
    if element_pitch_pixels <= 0:
        raise ValueError("element_pitch_pixels must be positive")
    if valid_mask is None:
        valid_mask = np.ones(theta.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != theta.shape:
            raise ValueError("valid_mask must have the same shape as phase")
    if not valid_mask.any():
        raise ValueError("valid_mask contains no phase-plate pixels")

    # Flat-top axial coordinates for circumradius R=pitch/sqrt(3):
    # q = 2*x/(sqrt(3)*pitch), r = y/pitch - x/(sqrt(3)*pitch).
    axis_pixels = centered_axis(theta.shape[0], 1.0)
    X_pixels, Y_pixels = np.meshgrid(
        axis_pixels, axis_pixels, indexing="xy"
    )
    q_fractional = (
        2 * X_pixels / (np.sqrt(3) * element_pitch_pixels)
    )
    r_fractional = (
        Y_pixels / element_pitch_pixels
        - X_pixels / (np.sqrt(3) * element_pitch_pixels)
    )
    q_cell, r_cell = _round_hex_axial_coordinates(
        q_fractional, r_fractional
    )

    axial_samples = np.column_stack((
        q_cell[valid_mask], r_cell[valid_mask]
    ))
    axial_indices, inverse = np.unique(
        axial_samples, axis=0, return_inverse=True
    )
    phase_samples = theta[valid_mask]
    cell_pixel_counts = np.bincount(inverse, minlength=len(axial_indices))

    if num_steps == 2:
        cosine_score = np.bincount(
            inverse,
            weights=np.cos(phase_samples),
            minlength=len(axial_indices),
        )
        element_phase = np.where(cosine_score >= 0, 0.0, np.pi)
    else:
        cosine_sum = np.bincount(
            inverse,
            weights=np.cos(phase_samples),
            minlength=len(axial_indices),
        )
        sine_sum = np.bincount(
            inverse,
            weights=np.sin(phase_samples),
            minlength=len(axial_indices),
        )
        element_phase = np.mod(
            np.arctan2(sine_sum, cosine_sum), 2 * np.pi
        )

    if num_steps is not None:
        element_phase = quantize_phase(element_phase, num_steps)

    phase_realistic = np.zeros_like(theta)
    phase_realistic[valid_mask] = element_phase[inverse]
    return (
        np.mod(phase_realistic, 2 * np.pi),
        element_phase,
        axial_indices,
        cell_pixel_counts,
    )


def build_comparison_target(
        target_shape, target_pixel_size, wavelength, focal_length, plate_size,
        desired_focal_spot, target_intensity_file=None,
        reference_pupil_pixels=1024, reference_distance_m=None,
        target_profile="supergaussian", top_hat_num_stripes=7):
    """Build the same striped target on the comparison output-plane grid."""
    if target_intensity_file is not None:
        reference = load_2d_array(target_intensity_file)
        if reference_distance_m is None:
            reference_distance_m = focal_length
        reference_pixel_size = (
            wavelength * reference_distance_m * reference_pupil_pixels
            / (reference.shape[0] * plate_size)
        )
        target = resample_intensity_to_grid(
            reference, target_shape, reference_pixel_size, target_pixel_size
        )
    else:
        target_extent = target_shape[0] * target_pixel_size
        target, _, _ = generate_focal_spot_modulated_target_x(
            target_profile=target_profile,
            pixels=target_shape[0],
            extent_m=target_extent,
            radius_m=desired_focal_spot / 2,
            supergaussian_order=50.2,
            mod_amp=1.0,
            mod_period_m=desired_focal_spot / 7,
            I_peak=1.0,
            top_hat_num_stripes=top_hat_num_stripes,
        )
        target = np.rot90(target)
    return normalise_intensity(target)


def build_focal_constraint_mask(
        Xf, Yf, desired_focal_spot, dark_region_factor=None):
    """Return the Fourier-plane region in which the target is enforced.

    ``dark_region_factor`` is the total constrained width divided by the
    requested focal-spot width.  For example, 2.0 retains one half-spot-width
    of explicitly dark target on each side.  ``None`` retains the historical
    behaviour and constrains the entire calculated Fourier plane.
    """
    Xf_broadcast, Yf_broadcast = np.broadcast_arrays(
        np.asarray(Xf, dtype=float), np.asarray(Yf, dtype=float)
    )
    if dark_region_factor is None:
        return np.ones(Xf_broadcast.shape, dtype=bool)
    if dark_region_factor < 1:
        raise ValueError("dark_region_factor must be at least 1 or None")
    half_width = 0.5 * dark_region_factor * desired_focal_spot
    return ((np.abs(Xf_broadcast) <= half_width)
            & (np.abs(Yf_broadcast) <= half_width))


def focal_metrics_in_region(test, reference, constraint_mask):
    """Evaluate focal metrics only where the Fourier target is constrained."""
    test = np.asarray(test)
    reference = np.asarray(reference)
    mask = np.broadcast_to(np.asarray(constraint_mask, dtype=bool), test.shape)
    if reference.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: test {test.shape} versus reference {reference.shape}"
        )
    if not mask.any():
        raise ValueError("The focal constraint region contains no samples")
    if mask.all():
        return focal_metrics(test, reference)
    return focal_metrics(test[mask], reference[mask])


def focal_plot_half_width_um(desired_focal_spot, plot_window_factor):
    """Convert a total focal-plot width factor to a half-width in microns."""
    if plot_window_factor is None:
        return None
    if plot_window_factor < 1:
        raise ValueError("plot_window_factor must be at least 1 or None")
    return 0.5 * plot_window_factor * desired_focal_spot * 1e6


def compare_ideal_and_element_limited(
        theta_ideal, plate_size, wavelength, focal_length,
        desired_focal_spot, fill_factor=0.90, aperture_shape="square",
        input_beam_shape="circular", square_beam_order=8.0,
        target_profile="supergaussian", top_hat_num_stripes=7, element_shape="square",
        num_steps=None, reference_pad_factor=2,
        reference_pupil_pixels=1024, target_intensity_file=None,
        target_reference_distance_m=None,
        target_dark_region_factor=None, focal_plot_window_factor=2.0,
        propagation_model="fraunhofer", propagation_distance_m=None,
        output_pixel_size_m=None, maximum_propagation_samples=None,
        output_dir=None, save=True, plot=True):
    """Propagate the fine solution and its physical-element realisation.

    No second optimisation is performed: this comparison answers what happens
    when the successful fine solution is manufactured with the available
    element size.  Both branches use the same plate sampling, aperture, input
    beam, propagation model, observation distance and Fourier padding.
    """
    propagation_model = normalise_propagation_model(propagation_model)
    element_shape = normalise_element_shape(element_shape)
    input_beam_shape = normalise_input_beam_shape(input_beam_shape)
    target_profile = normalise_target_profile(target_profile)
    if propagation_distance_m is None:
        propagation_distance_m = focal_length
    propagation_scale_m = propagation_plane_distance(
        propagation_model, focal_length, propagation_distance_m
    )
    theta_ideal = np.mod(np.asarray(theta_ideal, dtype=float), 2 * np.pi)
    if theta_ideal.ndim != 2 or theta_ideal.shape[0] != theta_ideal.shape[1]:
        raise ValueError("theta_ideal must be a square 2-D phase map")
    if num_steps is not None:
        theta_ideal = quantize_phase(theta_ideal, num_steps)

    plate_pixels = theta_ideal.shape[0]
    comparison_sampling = resolve_output_sampling(
        plate_pixels,
        plate_size,
        wavelength,
        propagation_scale_m,
        requested_output_pixel_size_m=output_pixel_size_m,
        manual_total_samples=round(reference_pad_factor * plate_pixels),
        maximum_samples=maximum_propagation_samples,
    )
    reference_pad_factor = (
        comparison_sampling["propagation_samples"] / plate_pixels
    )
    phase_aperture = build_phase_aperture_mask(
        theta_ideal.shape, plate_size, aperture_shape
    )
    theta_ideal = np.where(phase_aperture, theta_ideal, 0.0)
    (requested_element_size, number_elements,
     actual_element_size, actual_focal_spot) = compute_realistic_element_layout(
        plate_size, wavelength, propagation_scale_m,
        desired_focal_spot, plate_pixels
    )
    element_pitch_pixels = plate_pixels / number_elements
    element_edges = np.array([], dtype=int)
    hex_axial_indices = np.empty((0, 2), dtype=np.int32)
    cell_pixel_counts = np.array([], dtype=int)
    if element_shape == "square":
        theta_realistic, element_phase, element_edges = (
            project_phase_to_square_elements(
                theta_ideal,
                number_elements,
                num_steps=num_steps,
                valid_mask=phase_aperture,
            )
        )
        element_cell_count = int(element_phase.size)
        element_area_m2 = actual_element_size**2
        hex_corner_to_corner_m = np.nan
    else:
        (theta_realistic, element_phase, hex_axial_indices,
         cell_pixel_counts) = project_phase_to_hexagonal_elements(
            theta_ideal,
            element_pitch_pixels,
            num_steps=num_steps,
            valid_mask=phase_aperture,
        )
        element_cell_count = int(element_phase.size)
        element_area_m2 = np.sqrt(3) * actual_element_size**2 / 2
        hex_corner_to_corner_m = 2 * actual_element_size / np.sqrt(3)

    input_amplitude, _, dx_plate = build_generated_input_amplitude(
        plate_pixels,
        plate_size,
        fill_factor=fill_factor,
        aperture_shape=aperture_shape,
        input_beam_shape=input_beam_shape,
        square_beam_order=square_beam_order,
    )
    ideal_field = input_amplitude * np.exp(1j * theta_ideal)
    realistic_field = input_amplitude * np.exp(1j * theta_realistic)
    focal_ideal, Xf, Yf, _ = propagate_phase_plate(
        ideal_field,
        dx_plate,
        wavelength,
        focal_length,
        propagation_model=propagation_model,
        propagation_distance_m=propagation_distance_m,
        pad_factor=reference_pad_factor,
    )
    focal_realistic, Xf_realistic, Yf_realistic, _ = propagate_phase_plate(
        realistic_field,
        dx_plate,
        wavelength,
        focal_length,
        propagation_model=propagation_model,
        propagation_distance_m=propagation_distance_m,
        pad_factor=reference_pad_factor,
    )
    if not (np.array_equal(Xf, Xf_realistic)
            and np.array_equal(Yf, Yf_realistic)):
        raise RuntimeError("Ideal and realistic propagation grids do not match")

    focal_pixel_size = float(Xf[0, 1] - Xf[0, 0])
    target = build_comparison_target(
        focal_ideal.shape,
        focal_pixel_size,
        wavelength,
        focal_length,
        plate_size,
        desired_focal_spot,
        target_intensity_file=target_intensity_file,
        reference_pupil_pixels=reference_pupil_pixels,
        reference_distance_m=target_reference_distance_m,
        target_profile=target_profile,
        top_hat_num_stripes=top_hat_num_stripes,
    )
    focal_constraint_mask = build_focal_constraint_mask(
        Xf, Yf, desired_focal_spot, target_dark_region_factor
    )
    metrics_ideal = focal_metrics_in_region(
        focal_ideal, target, focal_constraint_mask
    )
    metrics_realistic = focal_metrics_in_region(
        focal_realistic, target, focal_constraint_mask
    )

    print("\nIdeal versus physical-element comparison:")
    print(f"  propagation model = {propagation_model}")
    print(f"  observation distance = {propagation_scale_m:.6g} m")
    print(
        "  output sampling = "
        f"{comparison_sampling['output_pixel_size_m']*1e6:.6g} um/pixel "
        f"on {comparison_sampling['propagation_samples']} x "
        f"{comparison_sampling['propagation_samples']} samples"
    )
    print(f"  element shape = {element_shape}")
    print(f"  requested element pitch = {requested_element_size*1e3:.4f} mm")
    print(f"  nominal elements across plate = {number_elements}")
    print(f"  actual centre pitch = {actual_element_size*1e3:.4f} mm")
    print(f"  physical element cells = {element_cell_count}")
    if element_shape == "square":
        pixel_widths = np.diff(element_edges)
        print(
            f"  square side raster width = {pixel_widths.min()} to "
            f"{pixel_widths.max()} numerical pixels"
        )
    else:
        print(
            "  hexagon flat-to-flat width = "
            f"{actual_element_size*1e3:.4f} mm"
        )
        print(
            "  hexagon corner-to-corner width = "
            f"{hex_corner_to_corner_m*1e3:.4f} mm"
        )
    print(f"  element-limited focal scale = {actual_focal_spot*1e6:.4f} um")
    print(
        f"  ideal: PCC={metrics_ideal['pcc']:.6f}, "
        f"relative NRMSE={metrics_ideal['relative_nrmse']:.6f}"
    )
    print(
        f"  realistic: PCC={metrics_realistic['pcc']:.6f}, "
        f"relative NRMSE={metrics_realistic['relative_nrmse']:.6f}\n"
    )

    output_dir = (
        Path(output_dir) if output_dir is not None
        else GENERATOR_DIR / f"outputs_mk2_{element_shape}_{propagation_model}"
    )
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "phase_map_ideal.npy", theta_ideal)
        np.save(output_dir / "phase_elements.npy", element_phase)
        np.save(output_dir / "phase_map_element_limited.npy", theta_realistic)
        np.save(output_dir / "focal_spot_ideal.npy", focal_ideal)
        np.save(output_dir / "focal_spot_element_limited.npy", focal_realistic)
        np.save(output_dir / "ideal_focal_spot_comparison.npy", target)
        np.save(output_dir / "focal_constraint_mask_comparison.npy",
                focal_constraint_mask)
        if element_shape == "square":
            np.save(output_dir / "element_edges_pixels.npy", element_edges)
        else:
            np.save(
                output_dir / "hex_cell_axial_indices.npy",
                hex_axial_indices,
            )
            np.save(
                output_dir / "hex_cell_pixel_counts.npy",
                cell_pixel_counts,
            )
        np.savez(
            output_dir / "element_geometry.npz",
            element_shape=element_shape,
            element_pitch_m=actual_element_size,
            element_pitch_pixels=element_pitch_pixels,
            element_area_m2=element_area_m2,
            element_cell_count=element_cell_count,
            square_side_m=(
                actual_element_size if element_shape == "square" else np.nan
            ),
            hex_flat_to_flat_m=(
                actual_element_size
                if element_shape == "hexagonal" else np.nan
            ),
            hex_corner_to_corner_m=hex_corner_to_corner_m,
            square_edges_pixels=element_edges,
            hex_axial_indices=hex_axial_indices,
            cell_pixel_counts=cell_pixel_counts,
        )
        np.save(output_dir / "metrics_comparison.npy", np.array([
            [metrics_ideal["pcc"], metrics_ideal["relative_nrmse"]],
            [metrics_realistic["pcc"], metrics_realistic["relative_nrmse"]],
        ]))
        np.savez(
            output_dir / "element_parameters.npz",
            requested_element_size_m=requested_element_size,
            number_elements=number_elements,
            actual_element_size_m=actual_element_size,
            element_shape=element_shape,
            element_pitch_m=actual_element_size,
            element_pitch_pixels=element_pitch_pixels,
            element_area_m2=element_area_m2,
            element_cell_count=element_cell_count,
            hex_corner_to_corner_m=hex_corner_to_corner_m,
            actual_focal_spot_m=actual_focal_spot,
            plate_pixels=plate_pixels,
            plate_pixel_size_m=dx_plate,
            reference_pad_factor=reference_pad_factor,
            target_dark_region_factor=(
                np.nan if target_dark_region_factor is None
                else target_dark_region_factor
            ),
            focal_plot_window_factor=(
                np.nan if focal_plot_window_factor is None
                else focal_plot_window_factor
            ),
            phase_steps=(0 if num_steps is None else num_steps),
            input_beam_shape=input_beam_shape,
            beam_fill_factor=fill_factor,
            square_beam_order=square_beam_order,
            target_profile=target_profile,
            propagation_model=propagation_model,
            propagation_distance_m=propagation_scale_m,
            focal_length_m=focal_length,
            requested_output_pixel_size_m=(
                np.nan if output_pixel_size_m is None
                else output_pixel_size_m
            ),
            achieved_output_pixel_size_m=(
                comparison_sampling["output_pixel_size_m"]
            ),
            propagation_samples=(
                comparison_sampling["propagation_samples"]
            ),
        )

    if plot:
        if plt is None:
            raise ImportError("plot=True requires Matplotlib")
        phase_vmax = (
            2 * np.pi if num_steps is None
            else 2 * np.pi * (num_steps - 1) / num_steps
        )
        focal_ideal_n = normalise_intensity(focal_ideal)
        focal_realistic_n = normalise_intensity(focal_realistic)
        plate_extent_mm = [
            -plate_size * 0.5e3, plate_size * 0.5e3,
            -plate_size * 0.5e3, plate_size * 0.5e3,
        ]
        focal_extent_um = [
            Xf.min() * 1e6, Xf.max() * 1e6,
            Yf.min() * 1e6, Yf.max() * 1e6,
        ]
        display_half_width_um = focal_plot_half_width_um(
            desired_focal_spot, focal_plot_window_factor
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
        phase_images = []
        phase_images.append(axes[0, 0].imshow(
            theta_ideal, cmap="gray", origin="lower", extent=plate_extent_mm,
            vmin=0, vmax=phase_vmax,
        ))
        axes[0, 0].set_title("Ideal fine-grid phase")
        phase_images.append(axes[0, 1].imshow(
            theta_realistic, cmap="gray", origin="lower", extent=plate_extent_mm,
            vmin=0, vmax=phase_vmax,
        ))
        axes[0, 1].set_title(
            f"Realistic {element_shape} phase: "
            f"{element_cell_count} physical cells"
        )
        for axis, image in zip(axes[0], phase_images):
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            fig.colorbar(image, ax=axis, label="phase (rad)")

        focal_images = []
        focal_images.append(axes[1, 0].imshow(
            focal_ideal_n, cmap="inferno", origin="lower",
            extent=focal_extent_um, vmin=0, vmax=1,
        ))
        axes[1, 0].set_title(
            f"Ideal focal spot | PCC={metrics_ideal['pcc']:.3f}"
        )
        focal_images.append(axes[1, 1].imshow(
            focal_realistic_n, cmap="inferno", origin="lower",
            extent=focal_extent_um, vmin=0, vmax=1,
        ))
        axes[1, 1].set_title(
            f"Element-limited focal spot | PCC={metrics_realistic['pcc']:.3f}"
        )
        for axis, image in zip(axes[1], focal_images):
            axis.set_xlabel("x (um)")
            axis.set_ylabel("y (um)")
            if display_half_width_um is not None:
                axis.set_xlim(-display_half_width_um, display_half_width_um)
                axis.set_ylim(-display_half_width_um, display_half_width_um)
            fig.colorbar(image, ax=axis, label="normalised intensity")

        fig.suptitle(
            f"{propagation_model.title()} at z={propagation_scale_m:.4g} m; "
            f"requested spot {desired_focal_spot*1e6:.1f} um: fine solution "
            "versus physical-element realisation"
        )
        print("Close the ideal/realistic comparison figure to finish.")
        plt.show()

    return {
        "phase_ideal": theta_ideal,
        "phase_elements": element_phase,
        "phase_realistic": theta_realistic,
        "focal_ideal": focal_ideal,
        "focal_realistic": focal_realistic,
        "target": target,
        "focal_constraint_mask": focal_constraint_mask,
        "metrics_ideal": metrics_ideal,
        "metrics_realistic": metrics_realistic,
        "number_elements": number_elements,
        "actual_element_size_m": actual_element_size,
        "element_shape": element_shape,
        "element_cell_count": element_cell_count,
        "element_geometry": {
            "element_pitch_m": actual_element_size,
            "element_pitch_pixels": element_pitch_pixels,
            "element_area_m2": element_area_m2,
            "hex_corner_to_corner_m": hex_corner_to_corner_m,
            "hex_axial_indices": hex_axial_indices,
            "cell_pixel_counts": cell_pixel_counts,
        },
        "propagation_model": propagation_model,
        "propagation_distance_m": propagation_scale_m,
        "output_sampling": comparison_sampling,
    }

# =============================================================================
# Optional element-map diagnostics
# =============================================================================
def show_element_maps(band_map, block_sizes_px, dx_phys, plate_size):
    """
    Visualize pixels-per-element and physical element size (µm) across the plate.

    Parameters
    ----------
    band_map : 2D int array
        band_map[y, x] = band index.
    block_sizes_px : 1D int array
        block_sizes_px[b] = block size (pixels) for band b.
    dx_phys : float
        Pixel size in the plate plane [m/px].
    plate_size : float
        Physical plate diameter [m] (clear aperture).
    """
    band_map = np.asarray(band_map, dtype=int)
    block_sizes_px = np.asarray(block_sizes_px, dtype=int)

    ny, nx = band_map.shape
    total_diameter = dx_phys * nx  # full simulation grid (including padding)

    # Build coordinate grid for plotting
    x = np.linspace(-total_diameter / 2, total_diameter / 2, nx)
    y = np.linspace(-total_diameter / 2, total_diameter / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    # Make maps
    pixels_per_elem = block_sizes_px[band_map]
    elem_size_m = pixels_per_elem.astype(float) * dx_phys
    elem_size_um = elem_size_m * 1e6

    # Mask everything outside the clear aperture so rings are clear
    plate_radius = plate_size / 2.0
    mask_plate = R <= plate_radius

    pixels_per_elem_plot = np.where(mask_plate, pixels_per_elem, np.nan)
    elem_size_um_plot    = np.where(mask_plate, elem_size_um, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(
        pixels_per_elem_plot,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis"
    )
    axes[0].set_title("Pixels per element")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], label="pixels")

    im1 = axes[1].imshow(
        elem_size_um_plot,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis"
    )
    axes[1].set_title("Element size (µm)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[1], label="µm")

    # Optionally overlay the plate boundary for sanity
    for ax in axes:
        circ = Circle((0.0, 0.0), plate_radius, edgecolor="w", facecolor="none", lw=1.0)
        ax.add_patch(circ)

    plt.show()
def show_element_mesh_grid(band_map, block_sizes_px, dx_phys, plate_size):
    """
    Visualize the element grid:
      - left: pixels per element
      - right: element size (µm)
    and overlay a dynamic mesh of rectangles showing the element tiling
    for each radial band.

    Notes
    -----
    - Much faster than the pixel-wise version (loops in block-space).
    - Uses the center pixel of each block to decide which band it belongs to,
      so blocks from different bands do NOT overlap.
    """
    band_map = np.asarray(band_map, dtype=int)
    block_sizes_px = np.asarray(block_sizes_px, dtype=int)

    ny, nx = band_map.shape
    total_diameter = dx_phys * nx

    # Build coordinate grid for plotting
    x = np.linspace(-total_diameter / 2, total_diameter / 2, nx)
    y = np.linspace(-total_diameter / 2, total_diameter / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    plate_radius = plate_size / 2.0
    mask_plate = R <= plate_radius

    # Background maps: pixels per element & element size
    pixels_per_elem = block_sizes_px[band_map]
    elem_size_um = pixels_per_elem.astype(float) * dx_phys * 1e6

    pixels_per_elem_plot = np.where(mask_plate, pixels_per_elem, np.nan)
    elem_size_um_plot    = np.where(mask_plate, elem_size_um, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        pixels_per_elem_plot,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis"
    )
    axes[0].set_title("Pixels per element")
    fig.colorbar(im0, ax=axes[0], label="pixels")

    im1 = axes[1].imshow(
        elem_size_um_plot,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis"
    )
    axes[1].set_title("Element size (µm)")
    fig.colorbar(im1, ax=axes[1], label="µm")

    # --------------------------------------------------------------
    # Draw element rectangles band-by-band in BLOCK space
    # --------------------------------------------------------------
    unique_bands = np.unique(band_map)

    for ax in axes:
        for b in unique_bands:
            block = int(block_sizes_px[b])
            if block < 1:
                continue

            # Pixels belonging to this band AND inside plate
            mask_band = (band_map == b) & mask_plate
            if not np.any(mask_band):
                continue

            js, is_ = np.where(mask_band)
            j_min, j_max = js.min(), js.max()
            i_min, i_max = is_.min(), is_.max()

            # Scan in steps of 'block' over this band's bounding box
            for j0 in range(j_min, j_max + 1, block):
                for i0 in range(i_min, i_max + 1, block):
                    j1 = min(j0 + block, ny)
                    i1 = min(i0 + block, nx)

                    if j1 <= j0 or i1 <= i0:
                        continue

                    # Use the center pixel of this block to decide which band it belongs to
                    jc = (j0 + j1 - 1) // 2
                    ic = (i0 + i1 - 1) // 2

                    if not mask_plate[jc, ic]:
                        continue

                    band_here = band_map[jc, ic]
                    if band_here != b:
                        # This block's center belongs to another band → skip
                        continue

                    # Physical coords of the block boundaries
                    x_start = x[i0] - dx_phys / 2
                    x_end   = x[i1-1] + dx_phys / 2
                    y_start = y[j0] - dx_phys / 2
                    y_end   = y[j1-1] + dx_phys / 2

                    # Quick radius check with center point
                    cx = 0.5 * (x_start + x_end)
                    cy = 0.5 * (y_start + y_end)
                    if np.sqrt(cx**2 + cy**2) > plate_radius:
                        continue

                    rect = plt.Rectangle(
                        (x_start, y_start),
                        x_end - x_start,
                        y_end - y_start,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.5,
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

        # Plate boundary
        circ = Circle((0.0, 0.0), plate_radius,
                      edgecolor="red", facecolor="none", lw=2.0)
        ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    plt.show()


# =============================================================================
# Legacy diagnostic metrics and image-registration helpers
# =============================================================================

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

def image_pcc(I_test, I_ideal):
    # flatten into 1D arrays
    x = I_test.ravel().astype(float)
    y = I_ideal.ravel().astype(float)
    
    # subtract means
    x_mean = x - np.mean(x)
    y_mean = y - np.mean(y)
    
    # compute PCC
    r = np.sum(x_mean * y_mean) / np.sqrt(np.sum(x_mean**2) * np.sum(y_mean**2))
    return r

def nrmse(I_test, I_ideal, robust=True):
    """
    Compute Normalized RMSE between test and ideal images.
    
    Parameters
    ----------
    I_test, I_ideal : 2D arrays (same shape)
    robust : bool, optional
        If True, use 2–98 percentile range instead of (max - min).
    
    Returns
    -------
    float
        NRMSE value (0 = perfect match).
    """
    I_test = np.array(I_test, float)
    I_ideal = np.array(I_ideal, float)

    # RMSE
    diff = I_test - I_ideal
    rmse = np.sqrt(np.mean(diff**2))

    # Normalization range
    if robust:
        lo, hi = np.percentile(I_ideal, (2, 98))
        rng = hi - lo
    else:
        rng = I_ideal.max() - I_ideal.min()

    if rng < 1e-12:
        return np.inf
    return rmse / rng

def plot_nrmse_error(I_test, I_ideal, robust=True, *,
                     i=None, total=None, show_last_only=False,
                     save_path=None):
    """
    Compute NRMSE and (optionally) plot:
      - If show_last_only=True, a plot is shown only when i == total-1.
      - If show_last_only=False (default), the plot is always shown.
      - Optionally save the figure to `save_path` when plotting.

    Returns
    -------
    float : NRMSE value (0 = perfect match).
    """
    # --- compute NRMSE ---
    diff = I_test - I_ideal
    rmse = np.sqrt(np.mean(diff**2))
    if robust:
        lo, hi = np.percentile(I_ideal, (2, 98))
        rng = hi - lo
    else:
        rng = I_ideal.max() - I_ideal.min()
    nrmse = rmse / rng if rng > 1e-12 else np.inf

    # decide whether to plot
    should_plot = True
    if show_last_only:
        # Only plot if we know we're on the last iteration
        should_plot = (i is not None and total is not None and i == total - 1)

    if should_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        fig.suptitle(f"Iteration {i} — NRMSE = {nrmse:.3f}")

        im0 = axes[0].imshow(I_ideal, cmap="inferno")
        axes[0].set_title("Ideal focal spot")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(I_test, cmap="inferno")
        axes[1].set_title("Generated focal spot")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax = np.max(np.abs(diff)) if np.isfinite(diff).all() else None
        im2 = axes[2].imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[2].set_title(f"Error map\nNRMSE={nrmse:.3f}")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes: ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    return nrmse

def _centroid_and_sigmas(I):
    I = np.asarray(I, float)
    I[I < 0] = 0  # guard
    H, W = I.shape
    y = np.arange(H)
    x = np.arange(W)
    X, Y = np.meshgrid(x, y)
    S = I.sum()
    if S <= 0:
        # fallback: center of array, unit sigmas
        return (W/2, H/2), (1.0, 1.0)
    xbar = (I * X).sum() / S
    ybar = (I * Y).sum() / S
    sx2 = (I * (X - xbar)**2).sum() / S
    sy2 = (I * (Y - ybar)**2).sum() / S
    sx = np.sqrt(max(sx2, 1e-12))
    sy = np.sqrt(max(sy2, 1e-12))
    return (xbar, ybar), (sx, sy)

def rescale_to_match(src, ref, clip_percent=0.0):
    """
    Return src_rescaled so that its 'beam size' matches ref.
    Uses second moments (sx, sy) to estimate scale; cubic interpolation.
    """
    if zoom is None:
        raise ImportError("rescale_to_match requires SciPy")
    S = np.array(src, float)
    R = np.array(ref, float)

    # optional robust clipping to reduce hot-pixel influence
    if clip_percent and clip_percent > 0:
        lo_s, hi_s = np.percentile(S, (clip_percent, 100-clip_percent))
        lo_r, hi_r = np.percentile(R, (clip_percent, 100-clip_percent))
        S = np.clip(S, lo_s, hi_s)
        R = np.clip(R, lo_r, hi_r)

    (_, _), (sx_s, sy_s) = _centroid_and_sigmas(S)
    (_, _), (sx_r, sy_r) = _centroid_and_sigmas(R)

    # average x/y scale for isotropic zoom (keeps spot shape)
    scale = 0.5 * (sx_r / sx_s + sy_r / sy_s)
    scale = float(np.clip(scale, 1e-3, 1e3))

    # resample src with cubic interpolation (order=3)
    S_rescaled = zoom(S, zoom=scale, order=3, mode="constant")

    # center-crop/pad to ref shape
    H, W = R.shape
    h, w = S_rescaled.shape
    # pad if smaller
    pad_y = max(0, H - h); pad_x = max(0, W - w)
    if pad_y or pad_x:
        t = pad_y // 2; b = pad_y - t; l = pad_x // 2; r = pad_x - l
        S_rescaled = np.pad(S_rescaled, ((t, b), (l, r)), mode="constant")
        h, w = S_rescaled.shape
    # center-crop to match
    y0 = (h - H) // 2; x0 = (w - W) // 2
    S_rescaled = S_rescaled[y0:y0+H, x0:x0+W]

    return S_rescaled, scale

# =============================================================================
# Gerchberg-Saxton phase retrieval
# =============================================================================

def compute_phase_elements(plate_size: float, wavelength: float, focal_length: float, desired_focal_spot: float):
    """
    Compute the phase element size and grid parameters based on the desired focal spot size.
    
    [1]: Lewis, C. L. S., et al. "Use of a random phase plate as a KrF laser beam homogenizer for thin film deposition applications." Review of scientific instruments 70.4 (1999): 2116-2121.

    #updated to appropriately calculate the number of hexagonal phase elements

    Parameters:
      plate_size: Physical size of the phase plate in meters.
      wavelength: Laser wavelength in meters.
      focal_length: Focal length of the optical system in meters.
      desired_focal_spot: Desired focal spot size in meters.
    
    Returns:
      element_size: The computed "diameter" (horizontal corner-to-corner distance) of each flat-top hexagon (m).
      num_elements: Approximate number of phase elements along one side.
      element_area: Area of each phase element (m²).
      focal_spot_size: Computed focal spot size (m) based on element_size.
    """
    # Compute element size using the formula: element_size = (wavelength * focal_length) / desired_focal_spot
    element_size = (wavelength * focal_length / desired_focal_spot)
    print(f"Computed element size: {element_size:.4e} m (from desired focal spot: {desired_focal_spot:.4e} m)")
    
    # Determine how many elements can fit along one side of the phase plate
    num_elements = int(plate_size / element_size)
    if num_elements < 8:
        raise ValueError(
            f"Computed number of phase elements ({num_elements}) is too low. "
            f"Consider increasing plate_size (currently {plate_size} m) or "
            f"increasing desired_focal_spot (currently {desired_focal_spot} m) "
            f"to obtain a higher resolution grid."
        )
    
    # Calculate the area of each phase element assuming a square (for simplicity)
    element_area = element_size ** 2
    print(f"Area of each phase element: {element_area:.4e} m² with {num_elements} elements along each side.")

    # Compute the focal spot size from the element size
    focal_spot_size = wavelength * focal_length / element_size
    print(f"Computed focal spot size (from element size): {focal_spot_size:.4e} m")
    
    return element_size, num_elements, element_area, focal_spot_size


@save_call(log_dir=str(GENERATOR_DIR / "gs_runs_mk2_hexagonal"), write_markdown=True, include_source=False)

def gs_2d(n: int, amp: float, beam_fwhm: float, mod_amp: float, mod_freq: float,
          plate_size=0.2, max_iter=100, discrete=True, plot=False, pzp=False, nsteps=2, 
          wavelength=532e-9, focal_length=1.0,
          propagation_model="fraunhofer", propagation_distance_m=None,
          refractive_index=1.5, discretize=False,
          num_steps=10, desired_focal_spot=500e-6, focal_spot_size=500e-6, f_number=2.5, pad_amount=0,
          output_pixel_size_m=None, maximum_propagation_samples=None,
          cut_frac=0.25, block_sizes_px = [10,6,4],
          input_beam_file: str = None, file_type: str = "npy",
          input_theta_file = None, file_type_theta = None,
          Debug: bool = False, save: bool = False,
          num_rings: int = 3,
          block_min_px: int = 4, block_max_px: int = 12,
          element_sizes_um=None, ring_edges_frac=None,
          aperture_shape="square", target_intensity_file=None,
          reference_pupil_pixels=1024, reference_pad_factor=2,
          fill_factor=0.90, input_beam_shape="circular",
          square_beam_order=8.0, target_profile="supergaussian",
          top_hat_num_stripes=7,
          quantization_ramp_fraction=0.0,
          boundary_rounding_sigma_px=0.0, enforce_element_blocking=True,
          enforce_isotropic_manufacturing=False,
          manufacturing_feature_size_m=None,
          manufacturing_filter_divisor=4.0,
          manufacturing_pi_fraction=None,
          target_dark_region_factor=None,
          focal_plot_window_factor=2.0,
          output_dir=None)-> np.ndarray:
    
    """
    Gerchberg-Saxton (GS) algorithm for designing a phase plate.
    Iteratively retrieves phase information in the Fourier domain to converge on a phase profile.
    
    Parameters:
      n: Grid resolution for phase calculation.
      amp: Input beam amplitude (arbitrary units).
      std: Standard deviation for the Gaussian beam profile (m).
      mod_amp: Modulation amplitude.
      mod_freq: Modulation frequency.
      plate_size: Physical diameter of the phase plate (m).
      max_iter: Maximum number of iterations for the algorithm.
      discrete: Boolean flag to indicate if discretization should be applied.
      plot: Boolean flag to enable plotting during iterations.
      pzp, nsteps: Additional parameters (not actively used here).
      wavelength: Laser wavelength (m).
      focal_length: Focal length of the optical system (m).
      propagation_model: ``"fraunhofer"`` for the lens focal plane or
        ``"fresnel"`` for one-FFT propagation through the focusing lens.
      propagation_distance_m: Observation distance in Fresnel mode. Defaults
        to ``focal_length``, where Fresnel and Fraunhofer intensities agree.
      refractive_index: Refractive index of the phase plate material.
      discretize: Boolean flag for discretizing the phase values.
      num_steps: Number of discrete phase steps (if discretization is applied).
      desired_focal_spot: Desired focal spot size (m).
      focal_spot_size: Focal spot size used in simulation (m).
      pad_amount: Manual padding on each side when output_pixel_size_m is None.
      output_pixel_size_m: Requested fixed physical output-plane pitch. When
        supplied, FFT-friendly padding is selected automatically for each z.
      maximum_propagation_samples: Safety limit for an automatically padded
        propagation dimension.
      input_beam_file: Optional file path for the input beam intensity.
      file_type: Type of file for input beam ('npy' or 'image').
      input_theta_file: Optional file path for an initial phase (theta) distribution.
      file_type_theta: File type for the theta file.
      aperture_shape: Clear-aperture geometry; GSI uses ``"square"``.
      target_intensity_file: Optional SCITECH focal intensity ``.npy`` target.
      reference_pupil_pixels: Unpadded pupil resolution used for the reference.
      reference_pad_factor: FFT zero-padding factor used for the reference.
      fill_factor: 1/e amplitude radius/half-width as a fraction of the
        clear-aperture half-width.
      input_beam_shape: ``"circular"`` for the original circular Gaussian or
        ``"square"`` for a square super-Gaussian input amplitude.
      square_beam_order: Super-Gaussian order used by the square input beam.
        Larger values produce flatter sides and sharper corners.
      target_profile: ``"supergaussian"`` for the existing striped circular
        super-Gaussian target, or ``"tophat"`` for a hard-edged striped
        circular top-hat target.
      top_hat_num_stripes: Exact number of solid vertical stripes used inside
        the circular mask when ``target_profile="tophat"``.
      quantization_ramp_fraction: Final fraction of iterations over which the
        continuous GS phase is progressively projected to ``num_steps`` levels.
      boundary_rounding_sigma_px: Gaussian corner-rounding width on the final
        reference-resolution binary export. Zero disables this post-processing.
      enforce_element_blocking: If True, project the phase onto the legacy
        square element blocks. False leaves every numerical phase pixel free.
      enforce_isotropic_manufacturing: Apply an isotropic correlated binary
        phase projection during the quantisation ramp.
      manufacturing_feature_size_m: Required characteristic physical feature
        scale. For this plate it is lambda*f/d_focus, approximately 2.66 mm.
      manufacturing_filter_divisor: Converts the characteristic feature scale
        to Gaussian sigma. A value of 4 gives sigma=feature_size/(4*dx).
      manufacturing_pi_fraction: Area fraction assigned pi. If None, preserve
        the nearest-binary fraction of the supplied initial solution.
      target_dark_region_factor: Total constrained Fourier-plane width divided
        by the requested focal-spot width. None constrains the whole plane.
      focal_plot_window_factor: Displayed focal-plane width divided by the
        requested spot width. None displays the whole calculated plane.
      output_dir: Directory for generated arrays; defaults beside this script.
    
    Returns:
      theta_in: The final computed phase profile (in radians) as a 2D array.
    """

    if n < 8:
        raise ValueError("n must be at least 8")
    if pad_amount < 0:
        raise ValueError("pad_amount cannot be negative")
    if aperture_shape.lower() not in {"circle", "square"}:
        raise ValueError("aperture_shape must be 'circle' or 'square'")
    input_beam_shape = normalise_input_beam_shape(input_beam_shape)
    target_profile = normalise_target_profile(target_profile)
    if not 0 < fill_factor <= 1:
        raise ValueError("fill_factor must lie in (0, 1]")
    if square_beam_order <= 0:
        raise ValueError("square_beam_order must be positive")
    if int(top_hat_num_stripes) < 1:
        raise ValueError("top_hat_num_stripes must be at least 1")
    if not 0 <= quantization_ramp_fraction <= 1:
        raise ValueError("quantization_ramp_fraction must lie in [0, 1]")
    if discretize and num_steps < 2:
        raise ValueError("num_steps must be at least 2 when discretize=True")
    if target_dark_region_factor is not None and target_dark_region_factor < 1:
        raise ValueError(
            "target_dark_region_factor must be at least 1 or None"
        )
    if focal_plot_window_factor is not None and focal_plot_window_factor < 1:
        raise ValueError(
            "focal_plot_window_factor must be at least 1 or None"
        )
    if output_pixel_size_m is not None and output_pixel_size_m <= 0:
        raise ValueError("output_pixel_size_m must be positive or None")
    if maximum_propagation_samples is not None and maximum_propagation_samples < n:
        raise ValueError("maximum_propagation_samples cannot be smaller than n")
    propagation_model = normalise_propagation_model(propagation_model)
    if propagation_distance_m is None:
        propagation_distance_m = focal_length
    if propagation_distance_m <= 0:
        raise ValueError("propagation_distance_m must be positive")
    propagation_scale_m = propagation_plane_distance(
        propagation_model, focal_length, propagation_distance_m
    )
    print(
        f"Propagation: {propagation_model}, observation plane "
        f"z={propagation_scale_m:.6g} m"
    )
    propagation_guide = calculate_propagation_guide(
        wavelength,
        focal_length,
        plate_size,
        fill_factor,
        desired_focal_spot,
        propagation_model,
        propagation_distance_m,
        input_beam_shape,
    )
    print_propagation_guide(propagation_guide)

    # The unpadded n x n region is the physical 11 cm phase plate.  Padding
    # enlarges only the numerical field of view and must not change dx.
    num_elements = n
    extent = plate_size / num_elements
    native_sampling = resolve_output_sampling(
        num_elements,
        plate_size,
        wavelength,
        propagation_scale_m,
        requested_output_pixel_size_m=output_pixel_size_m,
        manual_total_samples=num_elements + 2 * pad_amount,
        maximum_samples=maximum_propagation_samples,
    )
    pad_amount = native_sampling["symmetric_padding_per_side"]
    if pad_amount is None:
        raise RuntimeError("GS propagation requires symmetric integer padding")
    reference_sampling = resolve_output_sampling(
        reference_pupil_pixels,
        plate_size,
        wavelength,
        propagation_scale_m,
        requested_output_pixel_size_m=output_pixel_size_m,
        manual_total_samples=round(
            reference_pupil_pixels * reference_pad_factor
        ),
        maximum_samples=maximum_propagation_samples,
    )
    reference_pad_factor = (
        reference_sampling["propagation_samples"] / reference_pupil_pixels
    )
    if output_pixel_size_m is None:
        print(
            "Output sampling lock: disabled; achieved "
            f"{native_sampling['output_pixel_size_m']*1e6:.6g} um/pixel"
        )
    else:
        print(
            "Output sampling lock: requested "
            f"{output_pixel_size_m*1e6:.6g} um/pixel, achieved "
            f"{native_sampling['output_pixel_size_m']*1e6:.6g} um/pixel "
            f"with {native_sampling['propagation_samples']} x "
            f"{native_sampling['propagation_samples']} samples "
            f"({pad_amount} zero-padding pixels per side)"
        )
    x_plate = centered_axis(num_elements, extent)
    X_plate, Y_plate = np.meshgrid(x_plate, x_plate, indexing="xy")

 
    # Initialize lists to store iteration data and convergence metrics
    i_arr = []
    convergence = []
    Pcc = []
    NRMSE = []
    accuracy = []

    # Define a threshold for the Pearson correlation coefficient (PCC) to break the loop i.e define convergence
    threshold_pcc = 0.90

    # Build the same field amplitude used by TestBed.py.  The old implementation
    # loaded an input file and then immediately overwrote it; that is corrected.
    if input_beam_file:
        print(f"Loading input beam amplitude from {input_beam_file} as a {file_type} file...")
        if file_type in {"npy", "numpy"}:
            input_beam = load_input_beam(input_beam_file, (num_elements, num_elements))
        elif file_type == "image":
            input_beam = load_input_beam_png(input_beam_file, (num_elements, num_elements))
        else:
            raise ValueError("Unsupported file_type. Use 'npy', 'numpy' or 'image'.")
        peak = np.max(input_beam)
        if peak <= 0:
            raise ValueError("The input beam must contain positive amplitude")
        input_beam = input_beam / peak
    else:
        input_beam, _, _ = build_generated_input_amplitude(
            num_elements,
            plate_size,
            fill_factor=fill_factor,
            aperture_shape=aperture_shape,
            input_beam_shape=input_beam_shape,
            square_beam_order=square_beam_order,
        )
        if input_beam_shape == "square":
            print(
                "Generated square super-Gaussian input beam: "
                f"order={square_beam_order:g}, fill={fill_factor:g}"
            )
        else:
            print(
                "Generated circular Gaussian input beam: "
                f"fill={fill_factor:g}"
            )

    plate_pupil = make_pupil(X_plate, Y_plate, plate_size, aperture_shape)
    input_beam = input_beam * plate_pupil
    input_beam_title = (
        "Loaded input beam"
        if input_beam_file
        else (
            f"Square super-Gaussian input beam (order {square_beam_order:g})"
            if input_beam_shape == "square"
            else "Circular Gaussian input beam"
        )
    )

    # Zero-padding improves focal-plane sampling without inventing additional
    # phase elements.  A factor of two matches the SCITECH testbed.
    pad_width = ((pad_amount, pad_amount), (pad_amount, pad_amount))
    input_beam = np.pad(input_beam, pad_width, mode='constant', constant_values=0)
    length = num_elements + 2 * pad_amount  # Adjust length to include padding
    total_diameter = extent * length
    x_full = centered_axis(length, extent)
    X_full, Y_full = np.meshgrid(x_full, x_full, indexing="xy")
    pupil = make_pupil(X_full, Y_full, plate_size, aperture_shape)
    input_beam = input_beam * pupil
    xy = np.stack((X_full, Y_full), axis=-1)

    if Debug:
        plt.imshow(input_beam)
        plt.show()

    # Output-plane sampling follows directly from the one-FFT propagation.
    focal_pixel_size = wavelength * propagation_scale_m / (length * extent)
    focal_axis = centered_axis(length, focal_pixel_size)
    X_target = focal_axis[None, :]
    Y_target = focal_axis[:, None]

    reference = None
    if target_intensity_file is not None:
        reference = load_2d_array(target_intensity_file)
        # TestBed's reference is made from an unpadded plate grid with dx=D/N.
        # Deriving the pitch from the actual saved shape also handles a future
        # change in its padding factor without silently changing focal scale.
        expected_reference_samples = int(round(
            reference_pupil_pixels * reference_pad_factor
        ))
        if reference.shape != (expected_reference_samples, expected_reference_samples):
            print(
                f"WARNING: expected a {expected_reference_samples} x "
                f"{expected_reference_samples} reference, got {reference.shape}. "
                "Using the saved shape to derive its focal pitch."
            )
        reference_pixel_size = (
            wavelength * focal_length * reference_pupil_pixels
            / (reference.shape[0] * plate_size)
        )
        ideal_focal_spot = resample_intensity_to_grid(
            reference,
            (length, length),
            reference_pixel_size,
            focal_pixel_size,
        )
        print(f"Loaded SCITECH target: {target_intensity_file}")
    else:
        focal_extent_m = length * focal_pixel_size

        ideal_focal_spot, _, _ = generate_focal_spot_modulated_target_x(
            target_profile=target_profile,
            pixels=length,
            extent_m=focal_extent_m,
            radius_m=desired_focal_spot / 2,
            supergaussian_order=5.2,
            mod_amp=1.0,
            mod_period_m=desired_focal_spot / 7,
            I_peak=1.0,
            top_hat_num_stripes=top_hat_num_stripes,
        )
        print(f"Generated focal target profile: {target_profile}")

        # Retains the original stripe orientation
        ideal_focal_spot = np.rot90(ideal_focal_spot)

    ideal_focal_spot = normalise_intensity(ideal_focal_spot)
    target_amplitude = np.sqrt(ideal_focal_spot)
    focal_constraint_mask = build_focal_constraint_mask(
        X_target,
        Y_target,
        desired_focal_spot,
        target_dark_region_factor,
    )
    if target_dark_region_factor is None:
        print("Propagation target constraint: full calculated plane")
    else:
        constrained_width = target_dark_region_factor * desired_focal_spot
        dark_margin = 0.5 * (target_dark_region_factor - 1) * desired_focal_spot
        print(
            "Propagation target constraint: "
            f"{constrained_width*1e6:.1f} um square "
            f"({dark_margin*1e6:.1f} um dark margin per side)"
        )
        if focal_constraint_mask.all():
            print(
                "  The requested constraint window covers the complete "
                "calculated Fourier plane, so no outer region is free."
            )
    shape = xy.shape

        # ---------------- Radial bands / element sizes (physical control) ----------------
    # "extent" here is physical pixel size in the clear aperture region [m/px]
    dx_phys = extent

    # Effective clear-aperture radius in pixels (central region with nonzero mask)
    R_clear_px = (plate_size / dx_phys) * 0.5  # ≈ N_grid/2

    # ---- Determine block_sizes_px (from element_sizes_um or px-based fallback) ----
    if element_sizes_um is not None:
        element_sizes_um = np.asarray(element_sizes_um, dtype=float)
        num_rings = element_sizes_um.size  # override num_rings from this

        element_sizes_m = element_sizes_um * 1e-6

        # Safety: clip any requested element smaller than pixel size
        if element_sizes_m.min() < dx_phys:
            print(
                f"WARNING: smallest requested element size "
                f"{element_sizes_m.min()*1e6:.1f} µm is below pixel size {dx_phys*1e6:.1f} µm. "
                "Clipping to pixel size."
            )
            element_sizes_m = np.maximum(element_sizes_m, dx_phys)

        block_sizes_px = np.round(element_sizes_m / dx_phys).astype(int)
        block_sizes_px[block_sizes_px < 1] = 1

    else:
        # No physical sizes given: fall back to automatic px-based bands
        if num_rings < 1:
            raise ValueError("num_rings must be >= 1")

        block_sizes_px = np.linspace(block_max_px, block_min_px, num_rings)
        block_sizes_px = np.round(block_sizes_px).astype(int)
        block_sizes_px[block_sizes_px < 1] = 1

    # ---- Determine ring boundaries in radius ----
    # If user supplies ring_widths_frac, treat them as fractional widths that sum to 1
    if ring_edges_frac is not None:
        ring_edges_frac = np.asarray(ring_edges_frac, dtype=float)
        if ring_edges_frac.ndim != 1:
            raise ValueError("ring_edges_frac must be a 1D array of ring widths.")
        if not np.isclose(np.sum(ring_edges_frac), 1.0, atol=1e-6):
            raise ValueError(
                f"ring_edges_frac must sum to 1.0, got sum={np.sum(ring_edges_frac)}"
            )

        num_rings = ring_edges_frac.size  # override num_rings from element sizes

        # Cumulative edges: [w0, w0+w1, w0+w1+w2, ...] but drop the final 1.0
        cum_edges = np.cumsum(ring_edges_frac)
        r_edges_frac_internal = cum_edges[:-1]  # last edge = 1.0 not needed

        r_edges_px = r_edges_frac_internal * R_clear_px

    else:
        # Old behaviour: equal separation if element_sizes_um defines num_rings
        if num_rings == 1:
            r_edges_px = np.array([], dtype=float)
        else:
            r_edges_px = np.linspace(0.0, R_clear_px, num_rings + 1)[1:-1]

    # Build band map on the full padded grid
    band_map, r_map = make_radial_bands(
        shape=(length, length),
        r_edges_px=r_edges_px
    )

    # Print a small summary so you KNOW what you used
    r_edges_full = np.concatenate([[0.0], r_edges_px, [R_clear_px]])
    print("\nRadial bands / element sizes:")
    for b in range(num_rings):
        r_in_px  = r_edges_full[b]
        r_out_px = r_edges_full[b+1]
        elem_size_m = block_sizes_px[b] * dx_phys
        print(
            f"  Band {b}: r in [{r_in_px*dx_phys*1e6:7.1f}, {r_out_px*dx_phys*1e6:7.1f}] um"
            f"  element about {elem_size_m*1e6:6.1f} um  ({block_sizes_px[b]} px)"
        )
    print()
    if not enforce_element_blocking:
        print(
            "Element blocking disabled: all numerical phase pixels are "
            "independent GS degrees of freedom.\n"
        )
    # ------------------------------------------------------------------------
    if Debug:
        #show_element_maps(band_map, block_sizes_px, dx_phys, plate_size)
        show_element_mesh_grid(band_map, block_sizes_px, dx_phys, plate_size)
    # ------------------------------------------------------------------------

    #####################################################################################################
    # POTENTIALLY NOT NEEDED
    ####################################################################################################
    extent_focal_spot = focal_spot_size / num_elements  # Spatial extent of each pixel in the focal spot
    total_diameter_focal_spot = extent_focal_spot * length  # Total diameter of the focal spot in arbitrary units
    ####################################################################################################

    input_beam = input_beam * pupil

    if Debug:
        plt.imshow(input_beam, extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2], cmap='gray')
        plt.show()

    # Optionally plot the input and ideal beam profiles
    if plot:
        if plt is None:
            raise ImportError("plot=True requires Matplotlib")
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        im1 = ax1.imshow(input_beam, extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2], cmap='gray')
        ax1.set_title(input_beam_title)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        cbar = fig1.colorbar(im1, ax=ax1)
        cbar.ax.set_title('$Wm^{-2}$')

        focal_extent_um = [focal_axis.min() * 1e6, focal_axis.max() * 1e6,
                           focal_axis.min() * 1e6, focal_axis.max() * 1e6]
        im2 = ax2.imshow(ideal_focal_spot, extent=focal_extent_um,
                         origin='lower', cmap='gray')
        ax2.set_title('SCITECH focal-intensity target')
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        display_half_width_um = focal_plot_half_width_um(
            desired_focal_spot, focal_plot_window_factor
        )
        if display_half_width_um is not None:
            ax2.set_xlim(-display_half_width_um, display_half_width_um)
            ax2.set_ylim(-display_half_width_um, display_half_width_um)
        cbar = fig1.colorbar(im2, ax=ax2)
        cbar.ax.set_title('$Wm^{-2}$')

        fig1.suptitle(
            f"Input beam and {propagation_model.title()}-plane target "
            f"at z={propagation_scale_m:.4g} m"
        )
        print("Close the input/target figure to start the GS iterations.")
        plt.show()

    # If an initial theta file is provided, load it; otherwise, generate a random phase distribution.
    if input_theta_file:
        print(f"Loading initial phase from {input_theta_file} as a {file_type_theta} file...")
        if file_type_theta in {"npy", "numpy"}:
            theta_in = load_2d_array(input_theta_file)
        elif file_type_theta == "image":
            theta_in = load_input_beam_png(
                input_theta_file, (num_elements, num_elements)
            )
        else:
            raise ValueError("Unsupported file_type_theta. Use 'npy', 'numpy' or 'image'.")
        if theta_in.shape != (length, length):
            if theta_in.shape != (num_elements, num_elements):
                theta_in = resample_phase_to_shape(
                    theta_in, (num_elements, num_elements)
                )
            # The loaded map represents the physical phase plate. Preserve its
            # scale and add numerical padding around it; do not stretch it over
            # the larger propagation grid.
            theta_in = pad_center(theta_in, (length, length))
    else:
        #lock the seed used to generate the initial phase 
        rng = np.random.default_rng(42)

        # Create a repeatable random phase distribution using multiples of pi/2.
        theta_in = (np.pi / 2) * rng.integers(-2, 3, size=(length, length))
    
    theta_initial = np.array(theta_in, copy=True)

    manufacturing_transfer = None
    manufacturing_sigma_px = np.nan
    if enforce_isotropic_manufacturing:
        if manufacturing_feature_size_m is None or manufacturing_feature_size_m <= 0:
            raise ValueError(
                "manufacturing_feature_size_m must be positive when the "
                "isotropic manufacturing constraint is enabled"
            )
        if manufacturing_filter_divisor <= 0:
            raise ValueError("manufacturing_filter_divisor must be positive")
        manufacturing_sigma_px = (
            manufacturing_feature_size_m
            / (manufacturing_filter_divisor * dx_phys)
        )
        manufacturing_transfer = build_isotropic_manufacturing_filter(
            theta_in.shape, manufacturing_sigma_px
        )
        if discretize and num_steps == 2:
            if manufacturing_pi_fraction is None:
                wrapped_initial = np.mod(theta_in, 2 * np.pi)
                manufacturing_pi_fraction = float(np.mean(
                    np.abs(wrapped_initial - np.pi)
                    < np.minimum(wrapped_initial, 2 * np.pi - wrapped_initial)
                ))
            if not 0 < manufacturing_pi_fraction < 1:
                raise ValueError("manufacturing_pi_fraction must lie in (0, 1)")
        print("Isotropic manufacturing constraint:")
        print(f"  requested feature scale = {manufacturing_feature_size_m*1e3:.4f} mm")
        print(f"  numerical pixel pitch = {dx_phys*1e6:.4f} um")
        print(f"  Gaussian sigma = {manufacturing_sigma_px:.3f} px")
        if discretize and num_steps == 2:
            print(f"  preserved pi-area fraction = {manufacturing_pi_fraction:.6f}")
        elif discretize:
            print(f"  phase mode = {num_steps}-level quantized")
        else:
            print("  phase mode = continuous")
        print()

    #theta_in = np.pad(theta_initial, pad_width, mode='constant', constant_values=0)
    output_dir = (
        Path(output_dir) if output_dir is not None
        else GENERATOR_DIR / f"outputs_mk2_fine_{propagation_model}"
    )
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "initial_theta_in.npy", theta_in)

    #this section is for testing the code 
    # Get the amplitude (electric field magnitude) of the input beam
    original_beam_electric = np.abs(input_beam)

    #inital elctric field after the phase plate in the intial step of the GS algorithim
    initial_beam_electric = original_beam_electric * np.exp(1j * theta_in) #previously squared 

    #difference in between the initial electric field and the electric field after the phase plate, high indicates it's functioning correctly
    electric_field_test = np.sum(np.sum(np.abs(original_beam_electric) - np.abs(initial_beam_electric)))
    # print(f"Electric field difference: {electric_field_test}")
    # need better tests

    # --- Hard low-pass warm-up (additions) ---
    K_warm = max(1, int(0.25 * max_iter))

    # Precompute a fixed radial low-pass mask in the focal (FFT) plane
    Nf = ideal_focal_spot.shape[0]            # assume square, same as beam_ft
    u  = np.fft.fftshift(np.fft.fftfreq(Nf, d=1.0))  # normalized freq grid
    U, V = np.meshgrid(u, u, indexing='xy')
    R_norm = np.sqrt(U*U + V*V)
    R_norm /= R_norm.max()                    # 0 at center, 1 at corners
    Mlp = (R_norm <= cut_frac).astype(float)  # 1 in low-freq disk, 0 elsewhere                # 0 at center, ~1 at corners

    # Main iterative loop for the Gerchberg-Saxton algorithm
    for i in range(max_iter):
        # Print progress every 5% of the iterations
        if i % max(1, max_iter // 20) == 0:
            print(f"GS algorithm: {int((i / max_iter) * 100)} % complete")

        # Apply the phase to field amplitude (not intensity).
        input_beam_electric = original_beam_electric * np.exp(1j * theta_in)

        # Propagate to the selected output plane. Fresnel mode retains both
        # quadratic phases and the thin-lens phase; global scaling is irrelevant
        # because the output amplitude is replaced below.
        beam_ft = propagate_for_gs(
            input_beam_electric,
            extent,
            wavelength,
            focal_length,
            propagation_model=propagation_model,
            propagation_distance_m=propagation_distance_m,
        )

        # Extract the phase information from the Fourier domain
        theta_out = np.angle(beam_ft)

        # The saved target is intensity, so GS constrains its square-root
        # amplitude. With a finite constraint window, the amplitude outside
        # that window is left free (a mixed-region amplitude-freedom update).
        if focal_constraint_mask.all():
            new_amplitude = target_amplitude
        else:
            current_amplitude = np.abs(beam_ft)
            target_power = np.sum(
                target_amplitude[focal_constraint_mask] ** 2
            )
            current_power = np.sum(
                current_amplitude[focal_constraint_mask] ** 2
            )
            target_scale = (
                np.sqrt(current_power / target_power)
                if target_power > 0 else 1.0
            )
            new_amplitude = np.where(
                focal_constraint_mask,
                target_scale * target_amplitude,
                current_amplitude,
            )
        new_beam_ft = new_amplitude * np.exp(1j * theta_out)

        # ==== ADDED: brutally zero out high-frequency modes for early iterations ====
        if i < K_warm:
            new_beam_ft = new_beam_ft * Mlp   # hard low-pass: remove high-freq content entirely
        # ===========================================================================

        # Apply the exact inverse of the selected propagation operator.
        new_beam_electric = backpropagate_for_gs(
            new_beam_ft,
            extent,
            wavelength,
            focal_length,
            propagation_model=propagation_model,
            propagation_distance_m=propagation_distance_m,
        )

        # Update theta_in with the phase from the inverse-transformed field
        theta_in = np.angle(new_beam_electric)

        # -------- Radial element-size blocking (variable 'cell size') --------
        # This enforces piecewise-constant phase over blocks whose size depends
        # on the radial band (centre → larger cells, outer → smaller cells).
        if enforce_element_blocking:
            theta_in = apply_radial_blocking(
                theta_in,
                band_map=band_map,
                block_sizes_px=block_sizes_px,
            )
        # ---------------------------------------------------------------------

        # Gradually introduce the requested manufacturing phase model.  The
        # continuous mode receives correlation only; quantized modes receive
        # correlation followed by their requested number of phase levels.
        quantization_alpha = 0.0
        needs_projection = discretize or enforce_isotropic_manufacturing
        if needs_projection and quantization_ramp_fraction > 0:
            ramp_iterations = max(1, int(round(max_iter * quantization_ramp_fraction)))
            ramp_start = max_iter - ramp_iterations
            quantization_alpha = float(np.clip(
                (i + 1 - ramp_start) / ramp_iterations, 0.0, 1.0
            ))
            if quantization_alpha > 0:
                if enforce_isotropic_manufacturing and not discretize:
                    theta_projected = project_to_correlated_continuous_phase(
                        theta_in, manufacturing_transfer
                    )
                elif enforce_isotropic_manufacturing and num_steps == 2:
                    theta_projected = project_to_correlated_binary_phase(
                        theta_in, manufacturing_transfer, manufacturing_pi_fraction
                    )
                elif enforce_isotropic_manufacturing:
                    theta_projected = project_to_correlated_quantized_phase(
                        theta_in, manufacturing_transfer, num_steps
                    )
                else:
                    theta_projected = quantize_phase(theta_in, num_steps)
                theta_in = np.angle(
                    (1 - quantization_alpha) * np.exp(1j * theta_in)
                    + quantization_alpha * np.exp(1j * theta_projected)
                )

        # Record iteration number and convergence metric (difference between current Fourier amplitude and ideal)
        i_arr.append(i)
    
        # Calculate the difference in phase between the initial and current phase within the loop, changes over each iteration indicates correct functionality 
        theta_diff_loop = np.sum(theta_initial) - np.sum(theta_in)
        #print(f"Phase difference: {theta_diff_loop:.2f} radians")

        '''
        update to do:

        calculate the focal spot for each iteration and compare that with the ideal focal spot
        easy to do as we have already calculated the focal spot later on

        '''

        # After you finalize the array you’ll propagate:

        phase_plate_electric = np.abs(input_beam) * np.exp(1j * theta_in)

        N = phase_plate_electric.shape[0]
        dx_grid = plate_size / num_elements  # if you’re on the “resampled plate” convention
        x = (np.arange(N) - N//2) * dx_grid
        X, Y = np.meshgrid(x, x, indexing='xy')
        R = np.sqrt(X**2 + Y**2)

        #calculate the focal spot for each iteration and compare that with the ideal focal spot
        D       = plate_size                    # clear aperture diameter [m] (plate size == aperture )
        f       = focal_length                   # focal length [m]
        dx_grid = plate_size / num_elements                 # sampling pitch in the plate plane [m]
        dy_grid = dx_grid                        # square sampling
        
        if Debug:
            print(dx_grid, dy_grid)

        # --- build centered coordinate grid (VERY IMPORTANT for the pupil centering) ---
        # x runs from -L/2 ... +L/2 with L = Total Diameter
        x = (np.arange((N) ) - (N ) //2) * dx_grid

        if Debug:
            print(x)

        X, Y = np.meshgrid(x, x, indexing='xy')   # X: cols (x), Y: rows (y)

        # Use the configured GSI pupil geometry.
        pupil = make_pupil(X, Y, D, aperture_shape)

        if Debug:
            plt.imshow(pupil, cmap='gray', extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
            plt.show()

        E_pupil = phase_plate_electric * pupil    # zero outside aperture
        if Debug:
            plt.imshow(np.abs(E_pupil), extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
            plt.show()

        # Propagate the updated phase through the selected model for metrics.
        E_fft = propagate_for_gs(
            E_pupil,
            dx_grid,
            wavelength,
            focal_length,
            propagation_model=propagation_model,
            propagation_distance_m=propagation_distance_m,
        )
        Xf = X_target
        Yf = Y_target

        # --- focal spot intensity ---
        focal_spot = np.abs(E_fft)**2

        normalised_spot = normalise_intensity(focal_spot)

        ideal_spot = ideal_focal_spot

        # Compare directly on the physical grid.  Spatial rescaling would hide
        # precisely the focal-size error that this test is intended to detect.
        iteration_metrics = focal_metrics_in_region(
            normalised_spot, ideal_spot, focal_constraint_mask
        )
        pcc = iteration_metrics["pcc"]
        nrmse_val = iteration_metrics["relative_nrmse"]
        Pcc.append(pcc)
        NRMSE.append(nrmse_val)
        
        acc = max(0.0, pcc) * max(0.0, 1.0 - nrmse_val)   # 0..1
        accuracy.append(acc*100)     

        metric_spot = normalised_spot[focal_constraint_mask]
        metric_target = ideal_spot[focal_constraint_mask]
        target_sum = np.sum(metric_target)
        convergence.append(
            np.sum(np.abs(metric_spot - metric_target)) / target_sum
            if target_sum > 0 else float("inf")
        )

        ramp_finished = quantization_ramp_fraction == 0 or quantization_alpha >= 1.0
        if pcc >= threshold_pcc and ramp_finished:
            break

    completed_iterations = min(len(Pcc), len(NRMSE))
    if plot:
        iters = np.arange(completed_iterations)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].plot(iters, np.asarray(Pcc), lw=1)
        axes[0].axhline(threshold_pcc, linestyle='--', alpha=0.7,
                        label=f"Threshold = {threshold_pcc:.2f}")
        axes[0].set(title="PCC vs Iteration", xlabel="Iteration", ylabel="PCC")
        axes[0].legend()
        axes[1].plot(iters, np.asarray(NRMSE), lw=1)
        axes[1].set(title="Relative NRMSE vs Iteration", xlabel="Iteration",
                    ylabel="Relative NRMSE")
        axes[2].plot(iters, np.asarray(accuracy), lw=1)
        axes[2].set(title="Accuracy vs Iteration", xlabel="Iteration",
                    ylabel="Accuracy (%)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"GS convergence - final PCC={Pcc[-1]:.3f}, "
            f"relative NRMSE={NRMSE[-1]:.3f}"
        )
        print("Close the convergence figure to show the final phase-plate results.")
        plt.show()


    #A difference greater than 0 indicates that the GS algorithim is functioning correctly
    theta_diff = np.sum(theta_initial) - np.sum(theta_in)   
    print(f"Phase difference: {theta_diff:.2f} radians")

    # Adjust phase values to be within the range [0, 2π)
    theta_in = np.where(theta_in < 0, theta_in + 2 * np.pi, theta_in)
    theta_in = np.where(theta_in >= 2 * np.pi, theta_in - 2 * np.pi, theta_in)

    # If discretization is requested, round the phase values to discrete steps
    if discretize:
        theta_in = quantize_phase(theta_in, num_steps)
        print(f"Phase plate discretized into {num_steps} steps.")
    else:
        print("Continuous phase plate retained (no phase-level quantisation).")

    theta_in[theta_in >= 2*np.pi] = 0  # Ensure 2pi values are set to zero

    print(f"Continuous convergence accuracy: {convergence[-1]*100:.2f} %")
    
    # Compute the material thickness required to achieve the phase shift (using the refractive index)
    thickness = theta_in * wavelength / (2 * np.pi * (refractive_index - 1))

    # Simulate the beam after it passes through the phase plate by applying the computed phase shift
    phase_plate_electric = np.abs(input_beam) * np.exp(1j * theta_in)

    electric_field = np.sum(np.abs(phase_plate_electric)) - np.sum(np.abs(original_beam_electric))
    #print(f"Electric field difference: {electric_field}")

    field_map = np.abs(phase_plate_electric) - np.abs(original_beam_electric)
    #print(f"Field map difference: {field_map}")
    
    electric_field_test_diff = electric_field - electric_field_test
    #print(f"Electric field difference: {electric_field_test_diff}")

    # Calculating the focal spot, currently producing a focal spot of 10x smaller

    # ——— Incorporate f‑number effects here ———
    N = phase_plate_electric.shape[0]

    print(N)
    #phase_plate_electric = phase_plate_electric[pad_amount:shape_phase_plate_electric[0] - pad_amount, pad_amount:shape_phase_plate_electric[1] - pad_amount]
    #N = phase_plate_electric.shape[0]

    if Debug:
        print(np.shape(phase_plate_electric))

    # After you finalize the array you’ll propagate:
    N = phase_plate_electric.shape[0]
    dx_grid = extent  # padding changes field of view, not plate sampling pitch
    x = (np.arange(N) - N//2) * dx_grid
    X, Y = np.meshgrid(x, x, indexing='xy')
    # 1) Check leakage outside the intended aperture
    outside = ~make_pupil(X, Y, plate_size, aperture_shape)
    leak_ratio = np.linalg.norm(phase_plate_electric[outside]) / np.linalg.norm(phase_plate_electric)
    print(f"Leak outside aperture (L2 ratio): {leak_ratio:.2e}")
    # If this is ~1e-12 to 1e-8, you’re effectively zero outside. If ~1e-3 or worse, enforce a pupil.

    # 2) Compute the effective illuminated area from the sampled mask.
    mask_bool = np.abs(phase_plate_electric) > (1e-12 * np.max(np.abs(phase_plate_electric)))
    A_eff = mask_bool.sum() * (dx_grid**2)
    print(f"Effective illuminated area = {A_eff:.6e} m^2 ({aperture_shape} pupil)")
    

    D       = plate_size                     # clear aperture diameter [m] (plate size == aperture )
    f       = focal_length                   # focal length [m]
    dx_grid = extent                         # sampling pitch in the plate plane [m]
    dy_grid = dx_grid                        # square sampling
    
    if Debug:
        print(dx_grid, dy_grid)

    # --- build centered coordinate grid (VERY IMPORTANT for the pupil centering) ---
    # x runs from -L/2 ... +L/2 with L = Total Diameter
    x = (np.arange((N) ) - (N ) //2) * dx_grid

    if Debug:
        print(x)

    X, Y = np.meshgrid(x, x, indexing='xy')   # X: cols (x), Y: rows (y)

    # --- configured clear aperture ---
    pupil = make_pupil(X, Y, D, aperture_shape)

    if Debug:
        plt.imshow(pupil, cmap='gray', extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
        plt.show()

    E_pupil = phase_plate_electric * pupil    # zero outside aperture
    if Debug:
        plt.imshow(np.abs(E_pupil), extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
        plt.show()

    # Physically scaled propagation of the final native-grid solution.
    focal_spot, Xf, Yf, E_fft = propagate_phase_plate(
        E_pupil,
        dx_grid,
        wavelength,
        focal_length,
        propagation_model=propagation_model,
        propagation_distance_m=propagation_distance_m,
        pad_factor=1,
    )
    native_metrics = focal_metrics_in_region(
        focal_spot, ideal_focal_spot, focal_constraint_mask
    )
    print(f"Native-grid PCC before export: {native_metrics['pcc']:.6f}")
    print(f"Native-grid relative NRMSE: {native_metrics['relative_nrmse']:.6f}")

    # Export only the physical plate, not the numerical zero-padding region.
    plate_slice = slice(pad_amount, pad_amount + num_elements)
    theta_plate = theta_in[plate_slice, plate_slice]
    theta_plate = np.where(plate_pupil, theta_plate, 0.0)
    thickness_plate = (
        theta_plate * wavelength / (2 * np.pi * (refractive_index - 1))
    )
    input_beam_plate = input_beam[plate_slice, plate_slice]
    theta_export_unrounded = resample_phase_to_shape(
        theta_plate, (reference_pupil_pixels, reference_pupil_pixels)
    )
    export_phase_aperture = build_phase_aperture_mask(
        theta_export_unrounded.shape, plate_size, aperture_shape
    )
    theta_export_unrounded = np.where(
        export_phase_aperture, theta_export_unrounded, 0.0
    )
    if boundary_rounding_sigma_px > 0:
        if not discretize or num_steps != 2:
            raise ValueError(
                "boundary_rounding_sigma_px is only valid for a binary plate"
            )
        theta_export = round_binary_phase_boundaries(
            theta_export_unrounded, boundary_rounding_sigma_px
        )
    else:
        theta_export = theta_export_unrounded
    # Store fabrication phase consistently in [0, 2*pi), including continuous
    # maps returned by wrapped complex-phase interpolation.
    theta_export = np.mod(theta_export, 2 * np.pi)
    theta_export = np.where(export_phase_aperture, theta_export, 0.0)
    thickness_export = (
        np.mod(theta_export, 2 * np.pi) * wavelength
        / (2 * np.pi * (refractive_index - 1))
    )
    export_amplitude, _, export_dx = build_generated_input_amplitude(
        reference_pupil_pixels,
        plate_size,
        fill_factor=fill_factor,
        aperture_shape=aperture_shape,
        input_beam_shape=input_beam_shape,
        square_beam_order=square_beam_order,
    )
    export_field = export_amplitude * np.exp(1j * theta_export)
    export_focal_spot, export_Xf, export_Yf, _ = propagate_phase_plate(
        export_field,
        export_dx,
        wavelength,
        focal_length,
        propagation_model=propagation_model,
        propagation_distance_m=propagation_distance_m,
        pad_factor=reference_pad_factor,
    )
    export_focal_pixel_size = float(export_Xf[0, 1] - export_Xf[0, 0])
    reference_grid_matches = (
        reference is not None
        and reference.shape == export_focal_spot.shape
        and np.isclose(
            reference_pixel_size, export_focal_pixel_size,
            rtol=1e-10, atol=0,
        )
    )
    if reference_grid_matches:
        export_target = reference
    else:
        export_target = resample_intensity_to_grid(
            ideal_focal_spot,
            export_focal_spot.shape,
            focal_pixel_size,
            export_focal_pixel_size,
        )
    export_constraint_mask = build_focal_constraint_mask(
        export_Xf,
        export_Yf,
        desired_focal_spot,
        target_dark_region_factor,
    )
    final_metrics = focal_metrics_in_region(
        export_focal_spot, export_target, export_constraint_mask
    )
    print(f"Final exported PCC: {final_metrics['pcc']:.6f}")
    print(f"Final exported relative NRMSE: {final_metrics['relative_nrmse']:.6f}")
    extent_save = [export_Xf.min()/1e-6, export_Xf.max()/1e-6,
                   export_Yf.min()/1e-6, export_Yf.max()/1e-6]

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        # phase_map.npy is the SCITECH-resolution deliverable.  Native arrays
        # retain the actual optimisation grid for reproducibility.
        np.save(output_dir / 'phase_map.npy', theta_export)
        np.save(output_dir / 'phase_map_unrounded.npy', theta_export_unrounded)
        np.save(output_dir / 'phase_map_native.npy', theta_plate)
        np.save(output_dir / 'thickness_map.npy', thickness_export)
        np.save(output_dir / 'thickness_map_native.npy', thickness_plate)
        np.save(output_dir / 'focal_spot.npy', export_focal_spot)
        np.save(output_dir / 'focal_spot_native.npy', focal_spot)
        np.save(output_dir / 'input_beam.npy', export_amplitude)
        np.save(output_dir / 'input_beam_native.npy', input_beam_plate)
        np.save(output_dir / 'ideal_focal_spot.npy', export_target)
        np.save(output_dir / 'ideal_focal_spot_native.npy', ideal_focal_spot)
        np.save(output_dir / 'focal_constraint_mask.npy', export_constraint_mask)
        np.save(output_dir / 'focal_constraint_mask_native.npy',
                focal_constraint_mask)
        np.save(output_dir / 'output_beam.npy', np.abs(export_field))
        np.save(output_dir / 'focal_spot_extent.npy', extent_save)
        np.save(output_dir / 'metrics.npy', np.array([
            final_metrics['pcc'], final_metrics['relative_nrmse']
        ]))
        np.save(output_dir / 'manufacturing_parameters.npy', np.array([
            manufacturing_feature_size_m if manufacturing_feature_size_m is not None else np.nan,
            manufacturing_sigma_px,
            dx_phys,
            manufacturing_pi_fraction if manufacturing_pi_fraction is not None else np.nan,
            float(num_steps if discretize else 0),
        ]))
        np.savez(
            output_dir / 'propagation_parameters.npz',
            **propagation_guide,
            beam_fill_factor=fill_factor,
            square_beam_order=square_beam_order,
            target_profile=target_profile,
            aperture_shape=aperture_shape,
            plate_pixel_size_m=extent,
            requested_output_pixel_size_m=(
                np.nan if output_pixel_size_m is None
                else output_pixel_size_m
            ),
            native_output_pixel_size_m=focal_pixel_size,
            native_propagation_samples=native_sampling["propagation_samples"],
            export_output_pixel_size_m=export_focal_pixel_size,
            export_propagation_samples=(
                reference_sampling["propagation_samples"]
            ),
        )
        print(f"Saved generated solution to {output_dir}")
    
    ################################################################
    #testing
    # center line profile
    if Debug:
        center = N//2
        Ix = np.abs(E_fft[center, :])**2
        Iy = np.abs(E_fft[:, center])**2

        def fwhm_1d(I, x):
            I = I / I.max()
            # indices above half max
            above = np.where(I >= 0.5)[0]
            return x[above[-1]] - x[above[0]] if above.size else np.nan

        # output-plane axes (meters)
        focal_axis_debug = np.asarray(Xf)[0, :]
        FWHM_x = fwhm_1d(Ix, focal_axis_debug)
        FWHM_y = fwhm_1d(Iy, focal_axis_debug)  # same spacing

        print("FWHM_x, FWHM_y [m]:", FWHM_x, FWHM_y)
    #################################################
    if Debug:
        plt.imshow(np.log(focal_spot),
                cmap='viridis', extent=[Xf.min()/1e-6, Xf.max()/1e-6, Yf.min()/1e-6, Yf.max()/1e-6])
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.title(
            f'{propagation_model.title()} output '
            f'(z={propagation_scale_m:.4g} m)'
        )
        debug_half_width_um = focal_plot_half_width_um(
            desired_focal_spot, focal_plot_window_factor
        )
        if debug_half_width_um is not None:
            plt.xlim(-debug_half_width_um, debug_half_width_um)
            plt.ylim(-debug_half_width_um, debug_half_width_um)
        plt.show()

    # Plot various intermediate and final results if requested
    if plot:
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        native_focal_extent_um = [
            Xf.min() * 1e6, Xf.max() * 1e6,
            Yf.min() * 1e6, Yf.max() * 1e6,
        ]
        display_half_width_um = focal_plot_half_width_um(
            desired_focal_spot, focal_plot_window_factor
        )

        im0 = axes[0, 0].imshow(np.abs(original_beam_electric), cmap='gray', 
                          extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
        axes[0, 0].set_title(input_beam_title)
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('y (m)')
        cbar = fig2.colorbar(im0, ax=axes[0, 0], label='$Wm^{-2}$')

        im1 = axes[0, 1].imshow(
            np.abs(ideal_focal_spot), cmap='gray', origin='lower',
            extent=native_focal_extent_um,
        )
        axes[0, 1].set_title('Ideal focal spot structure ')
        axes[0, 1].set_xlabel('x (um)')
        axes[0, 1].set_ylabel('y (um)')
        cbar = fig2.colorbar(im1, ax=axes[0, 1], label='normalized intensity')

        im2 = axes[0, 2].imshow(np.log(focal_spot),
               cmap='viridis', extent=native_focal_extent_um)
        axes[0, 2].set_title(
            f'log10 {propagation_model.title()} output after phase plate'
        )
        axes[0, 2].set_xlabel('x (um)')
        axes[0, 2].set_ylabel('y (um)')
        cbar = fig2.colorbar(im2, ax=axes[0, 2], label='$Wm^{-2}$')

        '''
        im2 = axes[0, 2].imshow(apply_circular_mask_gpt(np.abs(phase_plate_electric),plate_size,pixel_size=extent), cmap='gray', 
                          extent=[-total_diameter/2, total_diameter/2, -total_diameter/2, total_diameter/2])
        axes[0, 2].set_title('Output Beam')
        axes[0, 2].set_xlabel('x (m)')
        axes[0, 2].set_ylabel('y (m)')
        cbar = fig2.colorbar(im2, ax=axes[0, 2], label='$Wm^{-2}$')

        '''

        im3 = axes[1, 0].imshow((focal_spot),
               cmap='viridis', extent=native_focal_extent_um)
        axes[1, 0].set_title(
            f'{propagation_model.title()} output after phase plate'
        )
        axes[1, 0].set_xlabel('x (um)')
        axes[1, 0].set_ylabel('y (um)')
        cbar = fig2.colorbar(im3, ax=axes[1, 0], label='$Wm^{-2}$')

        if display_half_width_um is not None:
            for focal_axis_plot in (axes[0, 1], axes[0, 2], axes[1, 0]):
                focal_axis_plot.set_xlim(
                    -display_half_width_um, display_half_width_um
                )
                focal_axis_plot.set_ylim(
                    -display_half_width_um, display_half_width_um
                )

        im4 = axes[1, 1].imshow(theta_plate, cmap='gray',
                                 extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[1, 1].set_title('Generated Phase Plate')
        axes[1, 1].set_xlabel('x (m)')
        axes[1, 1].set_ylabel('y (m)')
        cbar = fig2.colorbar(im4, ax=axes[1, 1], label='Phase (radians)')

        im5 = axes[1, 2].imshow(thickness_plate * 1e6, cmap='gray',
                                 extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[1, 2].set_title('Material Thickness (m)')
        axes[1, 2].set_xlabel('x (m)')
        axes[1, 2].set_ylabel('y (m)')
        cbar = fig2.colorbar(im5, ax=axes[1, 2], label='um')

        fig2.suptitle(
            f"{propagation_model.title()} propagation to "
            f"z={propagation_scale_m:.4g} m"
        )
        plt.show()

    # Return the computed phase map (theta)
    return theta_plate, focal_spot, Xf, Yf

# =============================================================================
# Validated control-panel orchestration
# =============================================================================
def validate_control_panel(config):
    """Fail early with a clear message when a control-panel value is invalid."""
    normalise_propagation_model(config.propagation_model)
    normalise_element_shape(config.element_shape)
    normalise_target_profile(config.target_profile)
    if config.phase_mode not in {"continuous", "quantized"}:
        raise ValueError("phase_mode must be 'continuous' or 'quantized'")
    if config.phase_mode == "quantized" and config.num_phase_steps < 2:
        raise ValueError("num_phase_steps must be at least 2 in quantized mode")
    if config.plate_pixels < 8 or config.export_plate_pixels < 8:
        raise ValueError("plate pixel counts must be at least 8")
    if config.aperture_shape not in {"square", "circle"}:
        raise ValueError("aperture_shape must be 'square' or 'circle'")
    normalise_input_beam_shape(config.input_beam_shape)
    if not 0 < config.beam_fill_factor <= 1:
        raise ValueError("beam_fill_factor must lie in (0, 1]")
    if config.square_beam_order <= 0:
        raise ValueError("square_beam_order must be positive")
    if int(config.top_hat_num_stripes) < 1:
        raise ValueError("top_hat_num_stripes must be at least 1")
    if not 0 <= config.quantization_ramp_fraction <= 1:
        raise ValueError("quantization_ramp_fraction must lie in [0, 1]")
    if not 0 <= config.cut_fraction <= 1:
        raise ValueError("cut_fraction must lie in [0, 1]")
    if config.optimisation_pad_pixels < 0:
        raise ValueError("optimisation_pad_pixels cannot be negative")
    if config.propagation_pad_factor < 1:
        raise ValueError("propagation_pad_factor must be at least 1")
    if (config.output_pixel_size_m is not None
            and config.output_pixel_size_m <= 0):
        raise ValueError("output_pixel_size_m must be positive or None")
    if config.maximum_propagation_samples < max(
            config.plate_pixels, config.export_plate_pixels):
        raise ValueError(
            "maximum_propagation_samples cannot be smaller than either "
            "plate pixel count"
        )
    if (config.target_dark_region_factor is not None
            and config.target_dark_region_factor < 1):
        raise ValueError(
            "target_dark_region_factor must be at least 1 or None"
        )
    if (config.focal_plot_window_factor is not None
            and config.focal_plot_window_factor < 1):
        raise ValueError(
            "focal_plot_window_factor must be at least 1 or None"
        )
    if config.refractive_index <= 1:
        raise ValueError("refractive_index must be greater than 1")
    if config.boundary_rounding_sigma_px < 0:
        raise ValueError("boundary_rounding_sigma_px cannot be negative")
    if (config.boundary_rounding_sigma_px > 0
            and not (config.phase_mode == "quantized"
                     and config.num_phase_steps == 2)):
        raise ValueError(
            "boundary rounding is only available for a two-level plate"
        )
    positive_values = {
        "desired_focal_spot_m": config.desired_focal_spot_m,
        "wavelength_m": config.wavelength_m,
        "focal_length_m": config.focal_length_m,
        "propagation_distance_m": config.propagation_distance_m,
        "plate_size_m": config.plate_size_m,
        "max_iterations": config.max_iterations,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def infer_input_file_type(path):
    """Return the loader name expected by the legacy GS function."""
    if path is None:
        return None
    return "numpy" if Path(path).suffix.lower() == ".npy" else "image"


def default_output_directory(config):
    """Select a folder from element shape, propagation and phase modes."""
    if config.output_directory is not None:
        return Path(config.output_directory)
    propagation_model = normalise_propagation_model(config.propagation_model)
    element_shape = normalise_element_shape(config.element_shape)
    base_name = f"outputs_mk2_{element_shape}_{propagation_model}"
    if config.phase_mode == "continuous":
        return GENERATOR_DIR / f"{base_name}_continuous"
    if config.num_phase_steps == 2:
        return GENERATOR_DIR / base_name
    return GENERATOR_DIR / f"{base_name}_{config.num_phase_steps}_level"


def run_realistic_generator(config=CONTROL_PANEL):
    """Run the fine-grid design and realistic element comparison."""
    run_start = time.time()
    validate_control_panel(config)

    save = config.save_outputs
    Debug = config.debug
    plot = config.show_plots and plt is not None
    if config.show_plots and plt is None:
        print("Matplotlib is unavailable; running without interactive plots.")

    PHASE_MODE = config.phase_mode
    NUM_PHASE_STEPS = config.num_phase_steps
    discretize_phase = PHASE_MODE == "quantized"

    plate_size = config.plate_size_m
    wavelength = config.wavelength_m
    focal_length = config.focal_length_m
    propagation_model = normalise_propagation_model(config.propagation_model)
    element_shape = normalise_element_shape(config.element_shape)
    input_beam_shape = normalise_input_beam_shape(config.input_beam_shape)
    target_profile = normalise_target_profile(config.target_profile)
    propagation_distance = config.propagation_distance_m
    propagation_scale = propagation_plane_distance(
        propagation_model, focal_length, propagation_distance
    )
    refractive_index = config.refractive_index
    desired_focal_spot = config.desired_focal_spot_m
    n = config.plate_pixels
    max_iter = config.max_iterations
    cut_frac = config.cut_fraction
    pad_amount = config.optimisation_pad_pixels
    f_number = focal_length / plate_size
    propagation_guide = calculate_propagation_guide(
        wavelength,
        focal_length,
        plate_size,
        config.beam_fill_factor,
        desired_focal_spot,
        propagation_model,
        propagation_distance,
        input_beam_shape,
    )

    # These positional values are retained only for compatibility with the
    # legacy gs_2d signature; target and beam shape use explicit controls below.
    amp = 1.0
    beam_fwhm = config.beam_fill_factor * plate_size
    mod_amp = 1.0
    mod_freq = 1.0

    (requested_element_size, num_elements,
     actual_element_size, focal_spot_size) = compute_realistic_element_layout(
        plate_size, wavelength, propagation_scale, desired_focal_spot, n
    )
    element_size = actual_element_size
    print(
        f"Requested phase-element size = {requested_element_size*1e3:.4f} mm"
    )
    print(
        f"Realistic element shape = {element_shape}; nominal elements across "
        f"plate = {num_elements}, actual centre pitch = "
        f"{actual_element_size*1e3:.4f} mm"
    )
    print(f"Generated focal target = {target_profile}")
    if target_profile == "tophat":
        print(f"Top-hat stripe count = {int(config.top_hat_num_stripes)}")
    if input_beam_shape == "square":
        print(
            "Input beam = square super-Gaussian; "
            f"order={config.square_beam_order:g}, "
            f"fill={config.beam_fill_factor:g}"
        )
    else:
        print(
            "Input beam = circular Gaussian; "
            f"fill={config.beam_fill_factor:g}"
        )

    element_sizes_um = [actual_element_size * 1e6]
    ring_edges_frac = [1.0]

    N_elem = num_elements

    input_beam_filepath = config.input_beam_path
    input_theta_filepath = config.initial_phase_path
    target_intensity_filepath = config.target_intensity_path
    for label, path in {
        "input_beam_path": input_beam_filepath,
        "initial_phase_path": input_theta_filepath,
        "target_intensity_path": target_intensity_filepath,
    }.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    output_directory = default_output_directory(config)

    # Compute the phase plate using the Gerchberg-Saxton algorithm
    theta_in, focal_spot, Xf, Yf = gs_2d(n, amp, beam_fwhm, mod_amp, mod_freq,
                        plate_size=plate_size, max_iter=max_iter, plot=plot, wavelength=wavelength,
                        focal_length=focal_length,
                        propagation_model=propagation_model,
                        propagation_distance_m=propagation_distance,
                        refractive_index=refractive_index,
                        discretize=discretize_phase, num_steps=NUM_PHASE_STEPS,
                        desired_focal_spot=desired_focal_spot,
                        focal_spot_size=focal_spot_size, f_number=f_number,
                        pad_amount=pad_amount,
                        output_pixel_size_m=config.output_pixel_size_m,
                        maximum_propagation_samples=(
                            config.maximum_propagation_samples
                        ),
                        cut_frac=cut_frac,
                        input_beam_file=input_beam_filepath,
                        file_type=infer_input_file_type(input_beam_filepath),
                        input_theta_file=input_theta_filepath,
                        file_type_theta=infer_input_file_type(input_theta_filepath),
                        Debug=Debug, save=save, element_sizes_um=element_sizes_um,
                        ring_edges_frac=ring_edges_frac,
                        aperture_shape=config.aperture_shape,
                        target_intensity_file=target_intensity_filepath,
                        reference_pupil_pixels=config.export_plate_pixels,
                        reference_pad_factor=config.propagation_pad_factor,
                        fill_factor=config.beam_fill_factor,
                        input_beam_shape=input_beam_shape,
                        square_beam_order=config.square_beam_order,
                        target_profile=target_profile,
                        top_hat_num_stripes=config.top_hat_num_stripes,
                        quantization_ramp_fraction=config.quantization_ramp_fraction,
                        boundary_rounding_sigma_px=config.boundary_rounding_sigma_px,
                        enforce_element_blocking=False,
                        enforce_isotropic_manufacturing=False,
                        manufacturing_feature_size_m=None,
                        manufacturing_filter_divisor=4.0,
                        manufacturing_pi_fraction=None,
                        target_dark_region_factor=(
                            config.target_dark_region_factor
                        ),
                        focal_plot_window_factor=(
                            config.focal_plot_window_factor
                        ),
                        output_dir=output_directory)

    # Apply the physical element size once, after the successful fine-grid
    # solution, then propagate both plates on the same numerical grid.
    # Reuse the exact target exported by GS when saving. This keeps the two
    # reported metric sets tied to precisely the same target samples.
    comparison_target_file = target_intensity_filepath
    comparison_target_distance = focal_length
    saved_gs_target = output_directory / "ideal_focal_spot.npy"
    if comparison_target_file is None and save and saved_gs_target.exists():
        comparison_target_file = saved_gs_target
        comparison_target_distance = propagation_scale

    comparison = compare_ideal_and_element_limited(
        theta_in,
        plate_size=plate_size,
        wavelength=wavelength,
        focal_length=focal_length,
        desired_focal_spot=desired_focal_spot,
        fill_factor=config.beam_fill_factor,
        aperture_shape=config.aperture_shape,
        input_beam_shape=input_beam_shape,
        square_beam_order=config.square_beam_order,
        target_profile=target_profile,
        top_hat_num_stripes=config.top_hat_num_stripes,
        element_shape=element_shape,
        num_steps=(NUM_PHASE_STEPS if discretize_phase else None),
        reference_pad_factor=config.propagation_pad_factor,
        reference_pupil_pixels=config.export_plate_pixels,
        target_intensity_file=comparison_target_file,
        target_reference_distance_m=comparison_target_distance,
        target_dark_region_factor=config.target_dark_region_factor,
        focal_plot_window_factor=config.focal_plot_window_factor,
        propagation_model=propagation_model,
        propagation_distance_m=propagation_distance,
        output_pixel_size_m=config.output_pixel_size_m,
        maximum_propagation_samples=config.maximum_propagation_samples,
        output_dir=output_directory,
        save=save,
        plot=plot,
    )

    elapsed = time.time() - run_start
    print(f"Total time = {elapsed:.2f} seconds")
    return {
        "fine_phase": theta_in,
        "fine_focal_spot": focal_spot,
        "comparison": comparison,
        "output_directory": output_directory,
        "propagation_model": propagation_model,
        "propagation_distance_m": propagation_scale,
        "element_shape": element_shape,
        "input_beam_shape": input_beam_shape,
        "square_beam_order": config.square_beam_order,
        "target_profile": target_profile,
        "top_hat_num_stripes": int(config.top_hat_num_stripes),
        "propagation_guide": propagation_guide,
        "output_sampling": comparison["output_sampling"],
        "elapsed_seconds": elapsed,
    }

if __name__ == "__main__":
    run_realistic_generator(CONTROL_PANEL)