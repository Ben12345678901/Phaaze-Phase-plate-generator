"""
Compare centre-line profiles from random, continuous and distributed phase-plate
focal spots against the order-5.2 Super-Gaussian target using integrated-energy
rather than peak-intensity normalisation.

The script reports residual standard deviations, calculates one-dimensional
Fourier power spectra of the central x and y lineouts, and calculates the raw
radially averaged 2-D Fourier power spectra of all focal-spot images.

Layout
------
Left column:
    - Rows 1-2: Random phase plate focal spot
    - Rows 3-4: Continuous phase plate focal spot
    - Rows 5-6: Distributed phase plate focal spot

Right column:
    - Rows 1-2: RPP central x/y lineouts
    - Rows 3-4: CPP central x/y lineouts
    - Rows 5-6: DPP central x/y lineouts

Each focal-spot image spans two rows on the left-hand side. Focal images are
normalised to unit integrated energy, and each plotted lineout is normalised to
unit area. Fourier power is plotted directly without spectral normalisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import transforms


# =============================================================================
# USER CONTROLS
# =============================================================================

DESIRED_FOCAL_SPOT_FILE = Path(
    r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1"
    r"\PHASEP~1\PHASE_~2\COMPLE~1\CHAPTE~1\GENERA~1"
    r"\PHSEPL~1\OUTPUT~1\IDEAL_~1.NPY"
)

RPP_FOCAL_SPOT_FILE = Path(
    r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1"
    r"\PHASEP~1\PHASE_~2\COMPLE~1\CHAPTE~1\GENERA~1"
    r"\PHSEPL~1\OUTPUT~1\FO01D1~1.NPY"
)

CONTINUOUS_FOCAL_SPOT_FILE = Path(
    r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1\PHASEP~1\PHASE_~2\COMPLE~1\CHAPTE~1\GENERA~1\PHSEPL~1\CONTIN~1\OUTPUT~1\FO4C3A~1.NPY")

# Update this path if the distributed-phase-plate output folder has a
# different name on your computer.
DISTRIBUTED_FOCAL_SPOT_FILE = Path(
    r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1\PHASEP~1\PHASE_~2\COMPLE~1\CHAPTE~1\GENERA~1\PHSEPL~1\CONTIN~1\OUTPUT~2\FO4C3A~1.NPY"
)

COMMON_FOCAL_SPOT_EXTENT_FILE = Path(
    r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1\PHASEP~1\PHASE_~2\COMPLE~1\CHAPTE~1\GENERA~1\PHSEPL~1\OUTPUT~2\FO2C79~1.NPY"
)

DESIRED_EXTENT_FILE = COMMON_FOCAL_SPOT_EXTENT_FILE
RPP_EXTENT_FILE = COMMON_FOCAL_SPOT_EXTENT_FILE
CONTINUOUS_EXTENT_FILE = COMMON_FOCAL_SPOT_EXTENT_FILE
DISTRIBUTED_EXTENT_FILE = COMMON_FOCAL_SPOT_EXTENT_FILE

EXTENT_UNITS = "um"
PLOT_HALF_WIDTH_UM = 400.0

# The target used by the phase-plate generator.
SUPER_GAUSSIAN_ORDER = 5.2
SUPER_GAUSSIAN_LABEL = rf"Super-Gaussian, $n={SUPER_GAUSSIAN_ORDER:g}$"

# Residual statistics are evaluated where the target is at least this fraction
# of its peak. This prevents the large dark background from artificially
# reducing the reported deviation.
STATISTICS_TARGET_THRESHOLD = 0.01

# One-dimensional Fourier analysis of the centre lineouts.
# The Fourier plots are shown versus spatial wavelength and restricted to the
# wavelength band below.
APPLY_HANN_WINDOW = True
POWER_SPECTRUM_LOG_X = True

# Keep only wavelengths in this range for plotting and mode reporting.
MIN_FOURIER_WAVELENGTH_UM = 4.0
MAX_FOURIER_WAVELENGTH_UM = 100.0

# Display slightly beyond the two thresholds to keep the threshold markers
# visible without leaving excessive white space.
# Start the displayed wavelength axis 0.5 um below the lower threshold
# and end a little above the upper threshold to reduce excess white space.
FOURIER_PLOT_MIN_UM = 3.5
FOURIER_PLOT_MAX_UM = 115.0

# Spectrum and threshold line appearance.
SPECTRUM_LINE_STYLE = "--"
THRESHOLD_LINE_STYLE = "--"
THRESHOLD_LINE_WIDTH = 1.3
THRESHOLD_LINE_COLOR = "0.35"

# Number of strongest modes printed in the terminal for each Fourier plot.
NUMBER_OF_REPORTED_MODES = 5
POWER_SPECTRUM_BINS = 350

# Radial-spectrum presentation and comparison controls.
RADIAL_FIGURE_SIZE = (9.5, 8.0)
RADIAL_AXIS_LABEL_FONT_SIZE = 15
RADIAL_TICK_FONT_SIZE = 13
RADIAL_TITLE_FONT_SIZE = 15
RADIAL_LEGEND_FONT_SIZE = 11
RADIAL_LEGEND_SHIFT_LEFT_MM = 110.0

# The dominant RPP mode is detected from a lightly smoothed copy of the raw
# radial spectrum. The reported powers are always taken from the unsmoothed,
# unnormalised spectra. Use an odd integer; set to 1 to disable smoothing.
RADIAL_PEAK_SMOOTHING_BINS = 7

# Number of common log-spaced wavelength samples used when comparing each
# plate's radial spectral shape against the Super-Gaussian target.
RADIAL_SHAPE_COMPARISON_SAMPLES = 800


FIGURE_SIZE = (13, 20)
IMAGE_CMAP = "viridis"
LINE_WIDTH = 2.2

# Reduced font sizes
BASE_FONT_SIZE = 11
TITLE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9

# Line colours used for the phase-plate profiles and image markers.
X_LINE_COLOR = "red"
Y_LINE_COLOR = "black"

# Opposite high-contrast colours used for the desired focal-spot profiles.
DESIRED_X_COLOR = "black"
DESIRED_Y_COLOR = "red"

SAVE_FIGURE = True
OUTPUT_FIGURE = Path("focal_spot_lineouts_with_statistics.png")

SAVE_POWER_SPECTRUM_FIGURE = True
POWER_SPECTRUM_FIGURE = Path("focal_spot_lineout_wavelength_power_spectrum_xy_thresholds.png")

SAVE_RADIAL_POWER_SPECTRUM_FIGURE = True
RADIAL_POWER_SPECTRUM_FIGURE = Path("focal_spot_radial_wavelength_power_spectrum_raw_thresholds.png")

SHOW_FIGURE = True


# =============================================================================
# DATA HANDLING
# =============================================================================

def load_focal_spot(path: Path) -> np.ndarray:
    """Load a finite two-dimensional focal-intensity array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Focal-spot file does not exist:\n{path}")

    array = np.asarray(np.load(path), dtype=float).squeeze()
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D focal spot at {path}, but found shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Array contains NaN or infinite values:\n{path}")

    return np.clip(array, 0.0, None)


def load_extent_um(path: Path, units: str = "um") -> np.ndarray:
    """Load [xmin, xmax, ymin, ymax] and return the values in microns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Focal-spot extent file does not exist:\n{path}")

    extent = np.asarray(np.load(path), dtype=float).squeeze()
    if extent.shape != (4,):
        raise ValueError(
            "Expected the extent file to contain four values "
            f"[xmin, xmax, ymin, ymax], but found shape {extent.shape}."
        )
    if not np.isfinite(extent).all():
        raise ValueError(f"Extent file contains NaN or infinite values:\n{path}")

    units_key = units.strip().lower().replace("µ", "u")
    if units_key in {"um", "micron", "microns", "micrometre", "micrometres"}:
        extent_um = extent
    elif units_key in {"m", "metre", "metres"}:
        extent_um = extent * 1e6
    else:
        raise ValueError("EXTENT_UNITS must be either 'um' or 'm'.")

    xmin, xmax, ymin, ymax = extent_um
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            "Extent limits must increase in the order [xmin, xmax, ymin, ymax]."
        )

    return extent_um


def energy_normalise_image(
    intensity: np.ndarray,
    extent_um: np.ndarray,
) -> np.ndarray:
    """Normalise a 2-D focal intensity so its integrated energy is unity.

    The integral is evaluated on the physical focal-plane grid, so the
    normalisation is independent of pixel count and sampling pitch.
    """
    image = np.clip(np.asarray(intensity, dtype=float), 0.0, None)
    x_um, y_um = coordinate_axes_um(image.shape, extent_um)

    energy = np.trapezoid(
        np.trapezoid(image, x=x_um, axis=1),
        x=y_um,
        axis=0,
    )
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Cannot normalise an image with zero integrated energy.")

    return image / energy


def coordinate_axes_um(shape: tuple[int, int], extent_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build x and y sample-centre coordinates from a saved focal extent."""
    ny, nx = shape
    xmin, xmax, ymin, ymax = extent_um
    x_um = np.linspace(xmin, xmax, nx)
    y_um = np.linspace(ymin, ymax, ny)
    return x_um, y_um


def image_edge_extent_um(shape: tuple[int, int], extent_um: np.ndarray) -> list[float]:
    """Convert sample-centre limits into pixel-edge limits for imshow."""
    x_um, y_um = coordinate_axes_um(shape, extent_um)
    dx_um = x_um[1] - x_um[0] if x_um.size > 1 else 1.0
    dy_um = y_um[1] - y_um[0] if y_um.size > 1 else 1.0

    return [
        x_um[0] - 0.5 * dx_um,
        x_um[-1] + 0.5 * dx_um,
        y_um[0] - 0.5 * dy_um,
        y_um[-1] + 0.5 * dy_um,
    ]


def extract_central_lineouts(
    intensity: np.ndarray,
    extent_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the central horizontal and vertical physical lineouts."""
    ny, nx = intensity.shape
    centre_y = ny // 2
    centre_x = nx // 2

    x_um, y_um = coordinate_axes_um(intensity.shape, extent_um)
    x_lineout = intensity[centre_y, :]
    y_lineout = intensity[:, centre_x]

    return (
        x_um,
        x_lineout,
        y_um,
        y_lineout,
        float(y_um[centre_y]),
        float(x_um[centre_x]),
    )


def area_normalise_lineout(
    coordinate_um: np.ndarray,
    lineout: np.ndarray,
) -> np.ndarray:
    """Normalise a 1-D lineout so the physical area under it is unity."""
    coordinate = np.asarray(coordinate_um, dtype=float)
    profile = np.clip(np.asarray(lineout, dtype=float), 0.0, None)

    valid = np.isfinite(coordinate) & np.isfinite(profile)
    if np.count_nonzero(valid) < 2:
        return profile

    area = np.trapezoid(profile[valid], x=coordinate[valid])
    if not np.isfinite(area) or area <= 0:
        return profile

    return profile / area


def interpolate_reference_lineout(
    reference_axis: np.ndarray,
    reference_lineout: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    """Interpolate a desired profile onto another physical focal-plane axis."""
    return np.interp(
        target_axis,
        reference_axis,
        reference_lineout,
        left=np.nan,
        right=np.nan,
    )



def resample_image_to_grid(
    source: np.ndarray,
    source_extent_um: np.ndarray,
    target_shape: tuple[int, int],
    target_extent_um: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a regular 2-D image onto another physical grid."""
    source = np.asarray(source, dtype=float)
    source_extent_um = np.asarray(source_extent_um, dtype=float)
    target_extent_um = np.asarray(target_extent_um, dtype=float)

    if (
        source.shape == target_shape
        and np.allclose(source_extent_um, target_extent_um, rtol=0.0, atol=1e-9)
    ):
        return source.copy()

    source_x_um, source_y_um = coordinate_axes_um(
        source.shape, source_extent_um
    )
    target_x_um, target_y_um = coordinate_axes_um(
        target_shape, target_extent_um
    )

    x_index = (
        (target_x_um - source_x_um[0])
        / (source_x_um[-1] - source_x_um[0])
        * (source.shape[1] - 1)
    )
    y_index = (
        (target_y_um - source_y_um[0])
        / (source_y_um[-1] - source_y_um[0])
        * (source.shape[0] - 1)
    )
    X_index, Y_index = np.meshgrid(x_index, y_index, indexing="xy")

    valid = (
        (X_index >= 0)
        & (X_index <= source.shape[1] - 1)
        & (Y_index >= 0)
        & (Y_index <= source.shape[0] - 1)
    )

    x0 = np.clip(np.floor(X_index).astype(int), 0, source.shape[1] - 1)
    y0 = np.clip(np.floor(Y_index).astype(int), 0, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = X_index - x0
    wy = Y_index - y0

    result = (
        (1 - wx) * (1 - wy) * source[y0, x0]
        + wx * (1 - wy) * source[y0, x1]
        + (1 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )
    return np.where(valid, result, np.nan)


def residual_standard_deviation(
    test: np.ndarray,
    reference: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """Return the population standard deviation of test-reference.

    Both arrays should already use the intended normalisation. Only samples
    where the reference is at least ``threshold`` of its peak are retained.
    """
    test = np.asarray(test, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if test.shape != reference.shape:
        raise ValueError(
            f"Residual-statistics shape mismatch: {test.shape} and "
            f"{reference.shape}"
        )

    reference_peak = np.nanmax(reference)
    if not np.isfinite(reference_peak) or reference_peak <= 0:
        return float("nan")

    valid = (
        np.isfinite(test)
        & np.isfinite(reference)
        & (reference >= threshold * reference_peak)
    )
    if not np.any(valid):
        return float("nan")

    residual = test[valid] - reference[valid]
    return float(np.std(residual, ddof=0))


def crop_image_to_window(
    image: np.ndarray,
    extent_um: np.ndarray,
    half_width_um: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a 2-D focal spot to a centred physical window."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array")

    x_um, y_um = coordinate_axes_um(image.shape, extent_um)

    x_valid = np.isfinite(x_um)
    y_valid = np.isfinite(y_um)
    if half_width_um is not None:
        x_valid &= np.abs(x_um) <= half_width_um
        y_valid &= np.abs(y_um) <= half_width_um

    if not np.any(x_valid) or not np.any(y_valid):
        raise ValueError("No image samples remain after applying the crop window.")

    cropped = image[np.ix_(y_valid, x_valid)]
    return cropped, x_um[x_valid], y_um[y_valid]


def radially_averaged_power_spectrum_2d(
    image: np.ndarray,
    extent_um: np.ndarray,
    *,
    half_width_um: float | None = None,
    bins: int = 350,
    apply_hann_window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw radially averaged 2-D Fourier power versus wavelength.

    The image mean is removed before the FFT so the DC term does not dominate.
    The returned power is not normalised.
    """
    image_crop, x_um, y_um = crop_image_to_window(
        image,
        extent_um,
        half_width_um,
    )

    if image_crop.shape[0] < 4 or image_crop.shape[1] < 4:
        raise ValueError("The cropped image is too small for a 2-D Fourier analysis.")

    dx_um = float(np.mean(np.diff(x_um)))
    dy_um = float(np.mean(np.diff(y_um)))
    if not np.isfinite(dx_um) or not np.isfinite(dy_um) or dx_um <= 0 or dy_um <= 0:
        raise ValueError("The cropped image sampling must be positive.")

    signal = np.nan_to_num(
        image_crop,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    signal = signal - np.mean(signal)

    if apply_hann_window:
        window = np.outer(np.hanning(signal.shape[0]), np.hanning(signal.shape[1]))
    else:
        window = np.ones_like(signal)

    window_energy = np.sum(window**2)
    if window_energy <= 0:
        raise ValueError("The 2-D Fourier window has zero energy.")

    spectrum = np.fft.fftshift(np.fft.fft2(signal * window))
    power_2d = dx_um * dy_um * np.abs(spectrum) ** 2 / window_energy

    fx_per_um = np.fft.fftshift(np.fft.fftfreq(signal.shape[1], d=dx_um))
    fy_per_um = np.fft.fftshift(np.fft.fftfreq(signal.shape[0], d=dy_um))
    FX_per_um, FY_per_um = np.meshgrid(fx_per_um, fy_per_um, indexing="xy")
    radial_frequency_per_um = np.hypot(FX_per_um, FY_per_um)

    if bins < 8:
        raise ValueError("bins must be at least 8 for the radial spectrum.")

    bin_edges = np.linspace(0.0, radial_frequency_per_um.max(), bins + 1)
    bin_index = np.digitize(radial_frequency_per_um.ravel(), bin_edges) - 1
    valid_bin = (bin_index >= 0) & (bin_index < bins)

    sum_power = np.bincount(
        bin_index[valid_bin],
        weights=power_2d.ravel()[valid_bin],
        minlength=bins,
    )
    count_power = np.bincount(
        bin_index[valid_bin],
        minlength=bins,
    )

    radial_power = np.divide(
        sum_power,
        count_power,
        out=np.zeros_like(sum_power, dtype=float),
        where=count_power > 0,
    )
    radial_frequency_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    valid = (
        np.isfinite(radial_power)
        & np.isfinite(radial_frequency_centres)
        & (radial_frequency_centres > 0)
        & (radial_power >= 0)
    )
    radial_frequency_centres = radial_frequency_centres[valid]
    radial_power = radial_power[valid]

    if radial_frequency_centres.size == 0:
        raise ValueError("No positive radial frequencies were found.")

    spatial_wavelength_um = 1.0 / radial_frequency_centres
    order = np.argsort(spatial_wavelength_um)
    return spatial_wavelength_um[order], radial_power[order]


def crop_lineout_to_window(
    coordinate_um: np.ndarray,
    lineout: np.ndarray,
    half_width_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a one-dimensional lineout to a centred physical window."""
    coordinate_um = np.asarray(coordinate_um, dtype=float)
    lineout = np.asarray(lineout, dtype=float)

    if coordinate_um.ndim != 1 or lineout.ndim != 1:
        raise ValueError("The Fourier inputs must be one-dimensional.")
    if coordinate_um.size != lineout.size:
        raise ValueError("Coordinate and lineout arrays must have equal length.")

    valid = np.isfinite(coordinate_um) & np.isfinite(lineout)
    if half_width_um is not None:
        valid &= np.abs(coordinate_um) <= half_width_um

    coordinate_crop = coordinate_um[valid]
    lineout_crop = lineout[valid]

    if coordinate_crop.size < 4:
        raise ValueError(
            "The Fourier-analysis lineout contains fewer than four samples."
        )

    return coordinate_crop, lineout_crop


def one_dimensional_power_spectrum(
    coordinate_um: np.ndarray,
    lineout: np.ndarray,
    *,
    half_width_um: float | None = None,
    apply_hann_window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided Fourier power versus spatial wavelength in microns.

    The lineout mean is removed before the FFT so the zero-frequency/DC term
    does not dominate. A Hann window can be applied to reduce leakage caused by
    the finite analysis window.

    Spatial frequency is calculated in cycles per micron and converted to
    spatial wavelength using

        wavelength = 1 / spatial_frequency.

    The zero-frequency component is excluded because it corresponds to an
    infinite spatial wavelength.
    """
    coordinate_crop, lineout_crop = crop_lineout_to_window(
        coordinate_um,
        lineout,
        half_width_um,
    )

    spacing_um = float(np.mean(np.diff(coordinate_crop)))
    if not np.isfinite(spacing_um) or spacing_um <= 0:
        raise ValueError("The lineout coordinate spacing must be positive.")

    spacing_variation = np.max(
        np.abs(np.diff(coordinate_crop) - spacing_um)
    )
    if spacing_variation > 1e-6 * spacing_um:
        raise ValueError(
            "The lineout coordinate grid must be uniformly spaced for the FFT."
        )

    signal = np.nan_to_num(
        lineout_crop,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    signal = signal - np.mean(signal)

    if apply_hann_window:
        window = np.hanning(signal.size)
    else:
        window = np.ones(signal.size)

    windowed_signal = signal * window
    fourier_amplitude = np.fft.rfft(windowed_signal)
    spatial_frequency_per_um = np.fft.rfftfreq(
        signal.size,
        d=spacing_um,
    )

    # Periodogram-like scaling makes spectra with slightly different sample
    # counts and pitches more directly comparable.
    window_energy = np.sum(window**2)
    if window_energy <= 0:
        raise ValueError("The Fourier window has zero energy.")

    power = (
        spacing_um
        * np.abs(fourier_amplitude) ** 2
        / window_energy
    )

    # Convert the two-sided energy represented by rFFT into one-sided power.
    if power.size > 2:
        if signal.size % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0

    valid = (
        np.isfinite(power)
        & np.isfinite(spatial_frequency_per_um)
        & (spatial_frequency_per_um > 0)
        & (power >= 0)
    )
    spatial_frequency_per_um = spatial_frequency_per_um[valid]
    power = power[valid]

    if spatial_frequency_per_um.size == 0:
        raise ValueError("No non-zero one-dimensional frequencies were found.")

    spatial_wavelength_um = 1.0 / spatial_frequency_per_um

    # Return the wavelength axis in ascending order.
    order = np.argsort(spatial_wavelength_um)
    return spatial_wavelength_um[order], power[order]


def normalise_spectrum_pair(
    first_power: np.ndarray,
    second_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise two spectra to their shared maximum power."""
    shared_peak = np.nanmax(
        np.concatenate([
            np.asarray(first_power, dtype=float),
            np.asarray(second_power, dtype=float),
        ])
    )

    if not np.isfinite(shared_peak) or shared_peak <= 0:
        return first_power, second_power

    return first_power / shared_peak, second_power / shared_peak


def normalise_spectrum(power: np.ndarray) -> np.ndarray:
    """Normalise one spectrum to its peak."""
    power = np.asarray(power, dtype=float)
    peak = np.nanmax(power)

    if not np.isfinite(peak) or peak <= 0:
        return power

    return power / peak


def restrict_wavelength_range(
    spatial_wavelength_um: np.ndarray,
    power: np.ndarray,
    minimum_wavelength_um: float,
    maximum_wavelength_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only Fourier wavelengths inside a requested interval."""
    wavelength = np.asarray(spatial_wavelength_um, dtype=float)
    power = np.asarray(power, dtype=float)

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(power)
        & (wavelength >= minimum_wavelength_um)
        & (wavelength <= maximum_wavelength_um)
    )

    wavelength = wavelength[valid]
    power = power[valid]

    if wavelength.size == 0:
        raise ValueError(
            "No Fourier samples remain after applying the wavelength cut."
        )

    return wavelength, power


def strongest_residual_modes(
    spatial_wavelength_um: np.ndarray,
    residual_power: np.ndarray,
    number_modes: int,
) -> list[tuple[float, float]]:
    """Return the strongest separated local maxima in a residual spectrum."""
    wavelength = np.asarray(spatial_wavelength_um, dtype=float)
    power = np.asarray(residual_power, dtype=float)

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(power)
        & (wavelength > 0)
        & (power >= 0)
    )
    wavelength = wavelength[valid]
    power = power[valid]

    if wavelength.size < 3 or number_modes < 1:
        return []

    local_maximum = np.zeros(power.size, dtype=bool)
    local_maximum[1:-1] = (
        (power[1:-1] > power[:-2])
        & (power[1:-1] >= power[2:])
    )

    candidate_indices = np.flatnonzero(local_maximum)
    if candidate_indices.size == 0:
        candidate_indices = np.arange(power.size)

    candidate_indices = candidate_indices[
        np.argsort(power[candidate_indices])[::-1]
    ]

    selected: list[tuple[float, float]] = []
    for index in candidate_indices:
        candidate_wavelength = float(wavelength[index])

        # Avoid reporting several neighbouring FFT bins as separate modes.
        sufficiently_separated = all(
            max(
                candidate_wavelength / selected_wavelength,
                selected_wavelength / candidate_wavelength,
            ) >= 1.15
            for selected_wavelength, _ in selected
        )
        if not sufficiently_separated:
            continue

        selected.append((
            candidate_wavelength,
            float(power[index]),
        ))
        if len(selected) >= number_modes:
            break

    return selected



def strongest_radial_peak(
    spatial_wavelength_um: np.ndarray,
    power: np.ndarray,
    smoothing_bins: int = 7,
) -> tuple[float, float, int]:
    """Return the strongest interior local maximum of a radial spectrum.

    Peak selection uses a short moving-average copy of the spectrum to avoid
    choosing a single noisy radial bin. The returned wavelength and power are
    taken from the original unsmoothed, unnormalised arrays.
    """
    wavelength = np.asarray(spatial_wavelength_um, dtype=float)
    raw_power = np.asarray(power, dtype=float)

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(raw_power)
        & (wavelength > 0)
        & (raw_power >= 0)
    )
    wavelength = wavelength[valid]
    raw_power = raw_power[valid]

    if wavelength.size < 3:
        raise ValueError("At least three radial-spectrum samples are required.")

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    raw_power = raw_power[order]

    smoothing_bins = max(1, int(smoothing_bins))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1
    smoothing_bins = min(smoothing_bins, wavelength.size)
    if smoothing_bins % 2 == 0:
        smoothing_bins -= 1

    if smoothing_bins > 1:
        kernel = np.ones(smoothing_bins, dtype=float) / smoothing_bins
        smoothed_power = np.convolve(raw_power, kernel, mode="same")
    else:
        smoothed_power = raw_power.copy()

    edge_bins = max(1, smoothing_bins // 2)
    local_maximum = np.zeros(smoothed_power.size, dtype=bool)
    local_maximum[1:-1] = (
        (smoothed_power[1:-1] > smoothed_power[:-2])
        & (smoothed_power[1:-1] >= smoothed_power[2:])
    )
    local_maximum[:edge_bins] = False
    local_maximum[-edge_bins:] = False

    candidates = np.flatnonzero(local_maximum)
    if candidates.size:
        coarse_peak_index = int(
            candidates[np.argmax(smoothed_power[candidates])]
        )
    else:
        interior = slice(edge_bins, wavelength.size - edge_bins)
        if wavelength.size <= 2 * edge_bins:
            coarse_peak_index = int(np.argmax(smoothed_power))
        else:
            coarse_peak_index = int(
                edge_bins + np.argmax(smoothed_power[interior])
            )

    # Refine the smoothed selection to the strongest raw bin nearby. This keeps
    # peak detection stable without reporting a point displaced from the actual
    # maximum by the moving-average kernel.
    refinement_radius = max(1, smoothing_bins // 2)
    lower_index = max(0, coarse_peak_index - refinement_radius)
    upper_index = min(
        wavelength.size,
        coarse_peak_index + refinement_radius + 1,
    )
    peak_index = int(
        lower_index + np.argmax(raw_power[lower_index:upper_index])
    )

    return (
        float(wavelength[peak_index]),
        float(raw_power[peak_index]),
        peak_index,
    )


def power_at_wavelength(
    spatial_wavelength_um: np.ndarray,
    power: np.ndarray,
    wavelength_um: float,
) -> float:
    """Linearly interpolate raw spectral power at a requested wavelength."""
    wavelength = np.asarray(spatial_wavelength_um, dtype=float)
    power = np.asarray(power, dtype=float)
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(power)
        & (wavelength > 0)
        & (power >= 0)
    )
    wavelength = wavelength[valid]
    power = power[valid]
    if wavelength.size < 2:
        return float("nan")

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    power = power[order]
    if wavelength_um < wavelength[0] or wavelength_um > wavelength[-1]:
        return float("nan")
    return float(np.interp(wavelength_um, wavelength, power))


def compare_radial_spectral_shape(
    test_wavelength_um: np.ndarray,
    test_power: np.ndarray,
    reference_wavelength_um: np.ndarray,
    reference_power: np.ndarray,
    samples: int = 800,
) -> dict[str, float]:
    """Compare radial spectral *shape* against a reference spectrum.

    Both spectra are interpolated onto the same logarithmic wavelength grid and
    area-normalised only inside this diagnostic calculation. The Fourier powers
    plotted elsewhere remain raw and unnormalised.

    ``shape_difference_percent`` is the total-variation distance between the two
    unit-area spectra. It ranges from 0 percent for identical shapes to 100
    percent for completely non-overlapping shapes. Lower is better.
    """
    test_wavelength = np.asarray(test_wavelength_um, dtype=float)
    test_power = np.asarray(test_power, dtype=float)
    reference_wavelength = np.asarray(reference_wavelength_um, dtype=float)
    reference_power = np.asarray(reference_power, dtype=float)

    test_valid = (
        np.isfinite(test_wavelength)
        & np.isfinite(test_power)
        & (test_wavelength > 0)
        & (test_power >= 0)
    )
    reference_valid = (
        np.isfinite(reference_wavelength)
        & np.isfinite(reference_power)
        & (reference_wavelength > 0)
        & (reference_power >= 0)
    )
    test_wavelength = test_wavelength[test_valid]
    test_power = test_power[test_valid]
    reference_wavelength = reference_wavelength[reference_valid]
    reference_power = reference_power[reference_valid]

    if test_wavelength.size < 2 or reference_wavelength.size < 2:
        raise ValueError("Both spectra need at least two finite samples.")

    test_order = np.argsort(test_wavelength)
    reference_order = np.argsort(reference_wavelength)
    test_wavelength = test_wavelength[test_order]
    test_power = test_power[test_order]
    reference_wavelength = reference_wavelength[reference_order]
    reference_power = reference_power[reference_order]

    lower = max(test_wavelength[0], reference_wavelength[0])
    upper = min(test_wavelength[-1], reference_wavelength[-1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("The spectra do not share a wavelength interval.")

    samples = max(32, int(samples))
    common_wavelength = np.geomspace(lower, upper, samples)
    log_wavelength = np.log(common_wavelength)
    test_common = np.interp(common_wavelength, test_wavelength, test_power)
    reference_common = np.interp(
        common_wavelength,
        reference_wavelength,
        reference_power,
    )
    test_common = np.clip(test_common, 0.0, None)
    reference_common = np.clip(reference_common, 0.0, None)

    test_area = np.trapezoid(test_common, x=log_wavelength)
    reference_area = np.trapezoid(reference_common, x=log_wavelength)
    if test_area <= 0 or reference_area <= 0:
        raise ValueError("Cannot compare a radial spectrum with zero area.")

    test_shape = test_common / test_area
    reference_shape = reference_common / reference_area

    shape_difference = 0.5 * np.trapezoid(
        np.abs(test_shape - reference_shape),
        x=log_wavelength,
    )
    reference_norm = np.linalg.norm(reference_shape)
    shape_nrmse = (
        float(np.linalg.norm(test_shape - reference_shape) / reference_norm)
        if reference_norm > 0
        else float("nan")
    )

    test_centred = test_shape - np.mean(test_shape)
    reference_centred = reference_shape - np.mean(reference_shape)
    correlation_denominator = (
        np.linalg.norm(test_centred) * np.linalg.norm(reference_centred)
    )
    shape_correlation = (
        float(np.dot(test_centred, reference_centred) / correlation_denominator)
        if correlation_denominator > 0
        else float("nan")
    )

    return {
        "shape_difference_percent": float(100.0 * shape_difference),
        "shape_nrmse": shape_nrmse,
        "shape_correlation": shape_correlation,
        "comparison_min_wavelength_um": float(lower),
        "comparison_max_wavelength_um": float(upper),
    }


def add_fourier_threshold_markers(axis) -> None:
    """Mark the accepted spatial-wavelength interval on a Fourier plot."""
    axis.axvline(
        MIN_FOURIER_WAVELENGTH_UM,
        color=THRESHOLD_LINE_COLOR,
        linestyle=THRESHOLD_LINE_STYLE,
        linewidth=THRESHOLD_LINE_WIDTH,
    )
    axis.axvline(
        MAX_FOURIER_WAVELENGTH_UM,
        color=THRESHOLD_LINE_COLOR,
        linestyle=THRESHOLD_LINE_STYLE,
        linewidth=THRESHOLD_LINE_WIDTH,
    )
    axis.set_xlim(FOURIER_PLOT_MIN_UM, FOURIER_PLOT_MAX_UM)

    text_transform = transforms.blended_transform_factory(
        axis.transData,
        axis.transAxes,
    )
    axis.text(
        MIN_FOURIER_WAVELENGTH_UM,
        0.98,
        rf"{MIN_FOURIER_WAVELENGTH_UM:g} $\mu$m",
        transform=text_transform,
        ha="center",
        va="top",
        fontsize=TICK_FONT_SIZE,
        color=THRESHOLD_LINE_COLOR,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        clip_on=False,
    )
    axis.text(
        MAX_FOURIER_WAVELENGTH_UM,
        0.98,
        rf"{MAX_FOURIER_WAVELENGTH_UM:g} $\mu$m",
        transform=text_transform,
        ha="center",
        va="top",
        fontsize=TICK_FONT_SIZE,
        color=THRESHOLD_LINE_COLOR,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        clip_on=False,
    )


# =============================================================================
# PLOTTING
# =============================================================================

def style_lineout_axis(axis):
    """Apply common formatting to all lineout axes."""
    # Equal-area lineouts are not constrained to a 0--1 peak range.
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axis.tick_params(labelsize=TICK_FONT_SIZE)
    if PLOT_HALF_WIDTH_UM is not None:
        axis.set_xlim(-PLOT_HALF_WIDTH_UM, PLOT_HALF_WIDTH_UM)


def plot_focal_spot_comparison() -> None:
    """Compare RPP, CPP and DPP focal spots with the Super-Gaussian target."""

    # -------------------------------------------------------------------------
    # Load extents first, then normalise every image to unit integrated energy.
    # This preserves meaningful differences in peak intensity while putting all
    # focal spots on the same total-energy basis. Fourier power remains raw.
    # -------------------------------------------------------------------------
    desired_extent_um = load_extent_um(DESIRED_EXTENT_FILE, EXTENT_UNITS)

    plates = {
        "RPP": {
            "name": "Random phase plate",
            "path": RPP_FOCAL_SPOT_FILE,
            "extent": load_extent_um(RPP_EXTENT_FILE, EXTENT_UNITS),
        },
        "CPP": {
            "name": "Continuous phase plate",
            "path": CONTINUOUS_FOCAL_SPOT_FILE,
            "extent": load_extent_um(CONTINUOUS_EXTENT_FILE, EXTENT_UNITS),
        },
        "DPP": {
            "name": "Distributed phase plate",
            "path": DISTRIBUTED_FOCAL_SPOT_FILE,
            "extent": load_extent_um(DISTRIBUTED_EXTENT_FILE, EXTENT_UNITS),
        },
    }

    desired = energy_normalise_image(
        load_focal_spot(DESIRED_FOCAL_SPOT_FILE),
        desired_extent_um,
    )
    for plate in plates.values():
        plate["image"] = energy_normalise_image(
            load_focal_spot(plate["path"]),
            plate["extent"],
        )
    (
        desired_x_um,
        desired_x_lineout,
        desired_y_um,
        desired_y_lineout,
        _,
        _,
    ) = extract_central_lineouts(desired, desired_extent_um)
    desired_x_lineout = area_normalise_lineout(desired_x_um, desired_x_lineout)
    desired_y_lineout = area_normalise_lineout(desired_y_um, desired_y_lineout)

    # Prepare the target on each image grid, extract central lineouts and
    # calculate equal-energy residual statistics.
    for plate in plates.values():
        image = plate["image"]
        extent = plate["extent"]

        plate["target_2d"] = resample_image_to_grid(
            desired,
            desired_extent_um,
            image.shape,
            extent,
        )

        (
            plate["x_um"],
            x_lineout,
            plate["y_um"],
            y_lineout,
            plate["centre_y_um"],
            plate["centre_x_um"],
        ) = extract_central_lineouts(image, extent)

        plate["x_lineout"] = area_normalise_lineout(plate["x_um"], x_lineout)
        plate["y_lineout"] = area_normalise_lineout(plate["y_um"], y_lineout)
        plate["target_x"] = interpolate_reference_lineout(
            desired_x_um,
            desired_x_lineout,
            plate["x_um"],
        )
        plate["target_y"] = interpolate_reference_lineout(
            desired_y_um,
            desired_y_lineout,
            plate["y_um"],
        )

        plate["x_sigma"] = residual_standard_deviation(
            plate["x_lineout"],
            plate["target_x"],
            STATISTICS_TARGET_THRESHOLD,
        )
        plate["y_sigma"] = residual_standard_deviation(
            plate["y_lineout"],
            plate["target_y"],
            STATISTICS_TARGET_THRESHOLD,
        )
        plate["sigma_2d"] = residual_standard_deviation(
            image,
            plate["target_2d"],
            STATISTICS_TARGET_THRESHOLD,
        )

    print(
        "\nResidual standard deviation from "
        f"Super-Gaussian n={SUPER_GAUSSIAN_ORDER:g}"
    )
    print(
        "  evaluated where target >= "
        f"{100 * STATISTICS_TARGET_THRESHOLD:.1f}% of peak"
    )
    for abbreviation, plate in plates.items():
        print(
            f"  {abbreviation}: "
            f"x={plate['x_sigma']:.5f}, "
            f"y={plate['y_sigma']:.5f}, "
            f"2-D={plate['sigma_2d']:.5f}"
        )
    print()

    plt.rcParams.update({
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    })

    # -------------------------------------------------------------------------
    # Focal-spot and centre-lineout comparison.
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = GridSpec(
        6,
        2,
        figure=fig,
        width_ratios=[1.1, 1.9],
        height_ratios=[1, 1, 1, 1, 1, 1],
    )

    image_axes = {
        "RPP": fig.add_subplot(gs[0:2, 0]),
        "CPP": fig.add_subplot(gs[2:4, 0]),
        "DPP": fig.add_subplot(gs[4:6, 0]),
    }
    lineout_axes = {
        "RPP": (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])),
        "CPP": (fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 1])),
        "DPP": (fig.add_subplot(gs[4, 1]), fig.add_subplot(gs[5, 1])),
    }

    image_vmax = max(
        float(np.nanmax(plate["image"])) for plate in plates.values()
    )

    for abbreviation, plate in plates.items():
        axis = image_axes[abbreviation]
        plotted_image = axis.imshow(
            plate["image"],
            origin="lower",
            cmap=IMAGE_CMAP,
            extent=image_edge_extent_um(
                plate["image"].shape,
                plate["extent"],
            ),
            vmin=0.0,
            vmax=image_vmax,
            interpolation="none",
        )
        axis.axhline(
            plate["centre_y_um"],
            color=X_LINE_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="x-lineout",
        )
        axis.axvline(
            plate["centre_x_um"],
            color=Y_LINE_COLOR,
            linestyle=":",
            linewidth=1.8,
            label="y-lineout",
        )
        axis.set_title(
            rf"{plate['name']} focal spot | "
            rf"$\sigma_{{res,2D}}={plate['sigma_2d']:.3e}$",
            fontsize=TITLE_FONT_SIZE,
        )
        axis.set_xlabel(r"$x$ ($\mu$m)", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.set_ylabel(r"$y$ ($\mu$m)", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
        axis.set_aspect("equal")

        if PLOT_HALF_WIDTH_UM is not None:
            axis.set_xlim(-PLOT_HALF_WIDTH_UM, PLOT_HALF_WIDTH_UM)
            axis.set_ylim(-PLOT_HALF_WIDTH_UM, PLOT_HALF_WIDTH_UM)

        axis.legend(
            loc="upper right",
            framealpha=0.9,
            fontsize=LEGEND_FONT_SIZE,
        )
        colour_bar = fig.colorbar(
            plotted_image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        colour_bar.set_label(
            r"Energy-normalised intensity ($\mu$m$^{-2}$)",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        colour_bar.ax.tick_params(labelsize=TICK_FONT_SIZE)

        ax_x, ax_y = lineout_axes[abbreviation]

        ax_x.plot(
            plate["x_um"],
            plate["x_lineout"],
            linewidth=LINE_WIDTH,
            color=X_LINE_COLOR,
            label=plate["name"],
        )
        ax_x.plot(
            plate["x_um"],
            plate["target_x"],
            linewidth=LINE_WIDTH,
            linestyle="--",
            color=DESIRED_X_COLOR,
            label=SUPER_GAUSSIAN_LABEL,
        )
        ax_x.set_title(
            rf"{abbreviation} central x-lineout | "
            rf"$\sigma_{{res}}={plate['x_sigma']:.3e}$",
            fontsize=TITLE_FONT_SIZE,
        )
        ax_x.set_xlabel(r"$x$ ($\mu$m)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_x.set_ylabel("")

        ax_y.plot(
            plate["y_um"],
            plate["y_lineout"],
            linewidth=LINE_WIDTH,
            color=Y_LINE_COLOR,
            label=plate["name"],
        )
        ax_y.plot(
            plate["y_um"],
            plate["target_y"],
            linewidth=LINE_WIDTH,
            linestyle="--",
            color=DESIRED_Y_COLOR,
            label=SUPER_GAUSSIAN_LABEL,
        )
        ax_y.set_title(
            rf"{abbreviation} central y-lineout | "
            rf"$\sigma_{{res}}={plate['y_sigma']:.3e}$",
            fontsize=TITLE_FONT_SIZE,
        )
        ax_y.set_xlabel(r"$y$ ($\mu$m)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_y.set_ylabel("")

        style_lineout_axis(ax_x)
        style_lineout_axis(ax_y)

    # -------------------------------------------------------------------------
    # One-dimensional Fourier spectra of the centre lineouts.
    # These are plotted directly without power normalisation.
    # -------------------------------------------------------------------------
    spectrum_fig, spectrum_axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        constrained_layout=True,
    )

    spectrum_colours = {
        "RPP": "red",
        "CPP": "black",
        "DPP": "green",
    }
    residual_modes = {}

    for abbreviation, plate in plates.items():
        for direction, spectrum_axis in zip(("x", "y"), spectrum_axes):
            wavelength_um, power = one_dimensional_power_spectrum(
                plate[f"{direction}_um"],
                plate[f"{direction}_lineout"],
                half_width_um=PLOT_HALF_WIDTH_UM,
                apply_hann_window=APPLY_HANN_WINDOW,
            )
            wavelength_um, power = restrict_wavelength_range(
                wavelength_um,
                power,
                MIN_FOURIER_WAVELENGTH_UM,
                MAX_FOURIER_WAVELENGTH_UM,
            )
            plate[f"{direction}_spectrum_wavelength_um"] = wavelength_um
            plate[f"{direction}_spectrum_power"] = power

            spectrum_axis.plot(
                wavelength_um,
                power,
                linewidth=LINE_WIDTH,
                color=spectrum_colours[abbreviation],
                linestyle=SPECTRUM_LINE_STYLE,
                label=f"{abbreviation} {direction}-lineout",
            )

            residual_wavelength_um, residual_power = (
                one_dimensional_power_spectrum(
                    plate[f"{direction}_um"],
                    plate[f"{direction}_lineout"]
                    - plate[f"target_{direction}"],
                    half_width_um=PLOT_HALF_WIDTH_UM,
                    apply_hann_window=APPLY_HANN_WINDOW,
                )
            )
            residual_wavelength_um, residual_power = restrict_wavelength_range(
                residual_wavelength_um,
                residual_power,
                MIN_FOURIER_WAVELENGTH_UM,
                MAX_FOURIER_WAVELENGTH_UM,
            )
            residual_modes[(abbreviation, direction)] = strongest_residual_modes(
                residual_wavelength_um,
                residual_power,
                NUMBER_OF_REPORTED_MODES,
            )

    # Calculate the target spectrum once on its own physical grid.
    for direction, spectrum_axis, coordinate_um, lineout in (
        ("x", spectrum_axes[0], desired_x_um, desired_x_lineout),
        ("y", spectrum_axes[1], desired_y_um, desired_y_lineout),
    ):
        wavelength_um, power = one_dimensional_power_spectrum(
            coordinate_um,
            lineout,
            half_width_um=PLOT_HALF_WIDTH_UM,
            apply_hann_window=APPLY_HANN_WINDOW,
        )
        wavelength_um, power = restrict_wavelength_range(
            wavelength_um,
            power,
            MIN_FOURIER_WAVELENGTH_UM,
            MAX_FOURIER_WAVELENGTH_UM,
        )
        spectrum_axis.plot(
            wavelength_um,
            power,
            linewidth=LINE_WIDTH,
            linestyle=SPECTRUM_LINE_STYLE,
            color="blue",
            label=SUPER_GAUSSIAN_LABEL,
        )

    for direction, spectrum_axis in zip(("x", "y"), spectrum_axes):
        if POWER_SPECTRUM_LOG_X:
            spectrum_axis.set_xscale("log")
        spectrum_axis.set_xlabel(
            r"Spatial wavelength, $\lambda_s$ ($\mu$m)",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        spectrum_axis.set_ylabel(
            "1D Fourier power",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        spectrum_axis.set_title(
            f"Fourier Power Spectrum: Focal spot {direction}-lineout",
            fontsize=TITLE_FONT_SIZE,
        )
        spectrum_axis.grid(alpha=0.25, which="both")
        spectrum_axis.tick_params(labelsize=TICK_FONT_SIZE)
        add_fourier_threshold_markers(spectrum_axis)
        spectrum_axis.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right")

    spectrum_fig.suptitle(
        "One-dimensional Fourier power spectra of the centre lineouts",
        fontsize=TITLE_FONT_SIZE + 1,
    )

    print(
        "Fourier wavelength range used: "
        f"{MIN_FOURIER_WAVELENGTH_UM:.1f} to "
        f"{MAX_FOURIER_WAVELENGTH_UM:.1f} um"
    )
    print("Dominant residual spatial wavelengths:")
    for abbreviation in plates:
        for direction in ("x", "y"):
            modes = residual_modes[(abbreviation, direction)]
            mode_text = (
                ", ".join(
                    f"{wavelength_um:.2f} um"
                    for wavelength_um, _ in modes
                )
                if modes
                else "none found"
            )
            print(f"  {abbreviation} {direction}-lineout residual: {mode_text}")

    if SAVE_POWER_SPECTRUM_FIGURE:
        POWER_SPECTRUM_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        spectrum_fig.savefig(
            POWER_SPECTRUM_FIGURE,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved 1-D Fourier power-spectrum figure to:\n"
            f"{POWER_SPECTRUM_FIGURE.resolve()}"
        )

    # -------------------------------------------------------------------------
    # Radially averaged 2-D Fourier power spectra.
    # The upper panel shows raw, unnormalised power. The lower panel compares
    # spectral shape with the Super-Gaussian using a separate unit-area metric.
    # -------------------------------------------------------------------------
    radial_fig, (radial_axis, radial_comparison_axis) = plt.subplots(
        2,
        1,
        figsize=RADIAL_FIGURE_SIZE,
        gridspec_kw={"height_ratios": [3.4, 1.25]},
        constrained_layout=True,
    )

    for abbreviation, plate in plates.items():
        wavelength_um, power = radially_averaged_power_spectrum_2d(
            plate["image"],
            plate["extent"],
            half_width_um=PLOT_HALF_WIDTH_UM,
            bins=POWER_SPECTRUM_BINS,
            apply_hann_window=APPLY_HANN_WINDOW,
        )
        wavelength_um, power = restrict_wavelength_range(
            wavelength_um,
            power,
            MIN_FOURIER_WAVELENGTH_UM,
            MAX_FOURIER_WAVELENGTH_UM,
        )
        plate["radial_wavelength_um"] = wavelength_um
        plate["radial_power"] = power
        radial_axis.plot(
            wavelength_um,
            power,
            linewidth=LINE_WIDTH,
            color=spectrum_colours[abbreviation],
            linestyle=SPECTRUM_LINE_STYLE,
            label=f"{abbreviation}",
        )

    sg_radial_wavelength_um, sg_radial_power = (
        radially_averaged_power_spectrum_2d(
            desired,
            desired_extent_um,
            half_width_um=PLOT_HALF_WIDTH_UM,
            bins=POWER_SPECTRUM_BINS,
            apply_hann_window=APPLY_HANN_WINDOW,
        )
    )
    sg_radial_wavelength_um, sg_radial_power = restrict_wavelength_range(
        sg_radial_wavelength_um,
        sg_radial_power,
        MIN_FOURIER_WAVELENGTH_UM,
        MAX_FOURIER_WAVELENGTH_UM,
    )
    radial_axis.plot(
        sg_radial_wavelength_um,
        sg_radial_power,
        linewidth=LINE_WIDTH,
        linestyle=SPECTRUM_LINE_STYLE,
        color="blue",
        label=SUPER_GAUSSIAN_LABEL,
    )

    # Identify the strongest interior RPP radial mode. Selection is stabilised
    # using a short moving average, but all reported/interpolated powers remain
    # the original raw, unnormalised values.
    rpp_peak_wavelength_um, rpp_peak_power, _ = strongest_radial_peak(
        plates["RPP"]["radial_wavelength_um"],
        plates["RPP"]["radial_power"],
        smoothing_bins=RADIAL_PEAK_SMOOTHING_BINS,
    )

    mode_power = {
        abbreviation: power_at_wavelength(
            plate["radial_wavelength_um"],
            plate["radial_power"],
            rpp_peak_wavelength_um,
        )
        for abbreviation, plate in plates.items()
    }
    mode_power["SG"] = power_at_wavelength(
        sg_radial_wavelength_um,
        sg_radial_power,
        rpp_peak_wavelength_um,
    )

    mode_reduction_percent = {}
    mode_change_db = {}
    for abbreviation in ("CPP", "DPP"):
        comparison_power = mode_power[abbreviation]
        if rpp_peak_power > 0 and np.isfinite(comparison_power):
            mode_reduction_percent[abbreviation] = float(
                100.0 * (1.0 - comparison_power / rpp_peak_power)
            )
            mode_change_db[abbreviation] = float(
                10.0 * np.log10(
                    max(comparison_power, np.finfo(float).tiny)
                    / rpp_peak_power
                )
            )
        else:
            mode_reduction_percent[abbreviation] = float("nan")
            mode_change_db[abbreviation] = float("nan")

    # Compare every phase plate with the Super-Gaussian radial spectral shape.
    # This metric normalises only inside the comparison function; it does not
    # alter the raw spectra displayed in the upper panel.
    radial_shape_metrics = {
        abbreviation: compare_radial_spectral_shape(
            plate["radial_wavelength_um"],
            plate["radial_power"],
            sg_radial_wavelength_um,
            sg_radial_power,
            samples=RADIAL_SHAPE_COMPARISON_SAMPLES,
        )
        for abbreviation, plate in plates.items()
    }
    best_plate_abbreviation = min(
        radial_shape_metrics,
        key=lambda abbreviation: radial_shape_metrics[abbreviation][
            "shape_difference_percent"
        ],
    )

    # Mark the selected RPP mode and each spectrum's power at that wavelength.
    radial_axis.axvline(
        rpp_peak_wavelength_um,
        color="0.35",
        linestyle=":",
        linewidth=1.5,
        label=(
            rf"Strongest RPP mode: "
            rf"$\lambda_s={rpp_peak_wavelength_um:.2f}\,\mu$m"
        ),
    )
    for abbreviation, plate in plates.items():
        radial_axis.scatter(
            rpp_peak_wavelength_um,
            mode_power[abbreviation],
            color=spectrum_colours[abbreviation],
            s=42,
            zorder=5,
        )
    radial_axis.scatter(
        rpp_peak_wavelength_um,
        mode_power["SG"],
        color="blue",
        s=42,
        zorder=5,
    )

    if POWER_SPECTRUM_LOG_X:
        radial_axis.set_xscale("log")

    add_fourier_threshold_markers(radial_axis)
    radial_axis.set_xlabel(
        r"Spatial wavelength, $\lambda_s$ ($\mu$m)",
        fontsize=RADIAL_AXIS_LABEL_FONT_SIZE,
    )
    radial_axis.set_ylabel(
        "Radially averaged 2D Fourier power",
        fontsize=RADIAL_AXIS_LABEL_FONT_SIZE,
    )
    radial_axis.set_title(
        "Radially averaged 2-D Fourier power spectrum",
        fontsize=RADIAL_TITLE_FONT_SIZE,
    )
    radial_axis.grid(alpha=0.25, which="both")
    radial_axis.tick_params(labelsize=RADIAL_TICK_FONT_SIZE)

    radial_legend_offset = transforms.ScaledTranslation(
        -RADIAL_LEGEND_SHIFT_LEFT_MM / 25.4,
        0.0,
        radial_fig.dpi_scale_trans,
    )
    radial_axis.legend(
        fontsize=RADIAL_LEGEND_FONT_SIZE,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=radial_axis.transAxes + radial_legend_offset,
    )

    # Lower panel: direct spectral-structure comparison with the target.
    comparison_labels = list(plates.keys())
    comparison_values = [
        radial_shape_metrics[abbreviation]["shape_difference_percent"]
        for abbreviation in comparison_labels
    ]
    comparison_colours = [
        spectrum_colours[abbreviation]
        for abbreviation in comparison_labels
    ]
    comparison_bars = radial_comparison_axis.bar(
        comparison_labels,
        comparison_values,
        color=comparison_colours,
        alpha=0.82,
    )
    radial_comparison_axis.set_ylabel(
        "Spectral-shape\ndifference (%)",
        fontsize=RADIAL_AXIS_LABEL_FONT_SIZE,
    )
    radial_comparison_axis.set_title(
        "Radial spectral structure relative to the Super-Gaussian "
        "— lower is better",
        fontsize=RADIAL_TITLE_FONT_SIZE,
    )
    radial_comparison_axis.tick_params(labelsize=RADIAL_TICK_FONT_SIZE)
    radial_comparison_axis.grid(axis="y", alpha=0.25)
    radial_comparison_axis.set_ylim(
        0.0,
        max(comparison_values) * 1.32 if max(comparison_values) > 0 else 1.0,
    )

    for abbreviation, bar, value in zip(
        comparison_labels,
        comparison_bars,
        comparison_values,
    ):
        label = f"{value:.2f}%"
        if abbreviation == best_plate_abbreviation:
            label += "\nBest"
            bar.set_linewidth(2.0)
            bar.set_edgecolor("0.15")
        radial_comparison_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=RADIAL_TICK_FONT_SIZE,
        )

    mode_summary = (
        rf"Strongest RPP mode: $\lambda_s={rpp_peak_wavelength_um:.2f}\,\mu$m"
        "    |    "
        f"CPP reduction: {mode_reduction_percent['CPP']:.2f}% "
        f"({mode_change_db['CPP']:.2f} dB)"
        "    |    "
        f"DPP reduction: {mode_reduction_percent['DPP']:.2f}% "
        f"({mode_change_db['DPP']:.2f} dB)"
    )
    radial_comparison_axis.set_xlabel(
        mode_summary,
        fontsize=RADIAL_LEGEND_FONT_SIZE,
        labelpad=8,
    )

    print("\nRadially averaged 2-D spectrum analysis:")
    print(
        "  strongest RPP mode = "
        f"{rpp_peak_wavelength_um:.4f} um, "
        f"power={rpp_peak_power:.6e}"
    )
    print(
        "  CPP at the RPP mode: "
        f"power={mode_power['CPP']:.6e}, "
        f"reduction={mode_reduction_percent['CPP']:.3f}%, "
        f"change={mode_change_db['CPP']:.3f} dB"
    )
    print(
        "  DPP at the RPP mode: "
        f"power={mode_power['DPP']:.6e}, "
        f"reduction={mode_reduction_percent['DPP']:.3f}%, "
        f"change={mode_change_db['DPP']:.3f} dB"
    )
    print(
        "  Super-Gaussian power at the RPP mode = "
        f"{mode_power['SG']:.6e}"
    )
    print("  Spectral-shape comparison with the Super-Gaussian:")
    for abbreviation in comparison_labels:
        metrics = radial_shape_metrics[abbreviation]
        print(
            f"    {abbreviation}: "
            f"difference={metrics['shape_difference_percent']:.3f}%, "
            f"NRMSE={metrics['shape_nrmse']:.5f}, "
            f"correlation={metrics['shape_correlation']:.5f}"
        )
    print(
        "  Best radial spectral match to the Super-Gaussian = "
        f"{best_plate_abbreviation} "
        f"({plates[best_plate_abbreviation]['name']})\n"
    )

    if SAVE_RADIAL_POWER_SPECTRUM_FIGURE:
        RADIAL_POWER_SPECTRUM_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        radial_fig.savefig(
            RADIAL_POWER_SPECTRUM_FIGURE,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved raw radial 2-D Fourier power-spectrum figure to:\n"
            f"{RADIAL_POWER_SPECTRUM_FIGURE.resolve()}"
        )

    if SAVE_FIGURE:
        OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
        print(f"Saved figure to:\n{OUTPUT_FIGURE.resolve()}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
        plt.close(spectrum_fig)
        plt.close(radial_fig)


if __name__ == "__main__":
    plot_focal_spot_comparison()