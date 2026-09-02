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
MIN_FOURIER_WAVELENGTH_UM = 10.0
MAX_FOURIER_WAVELENGTH_UM = 100

# Display slightly beyond the two thresholds to keep the threshold markers
# visible without leaving excessive white space.
# Start the displayed wavelength axis 0.5 um below the lower threshold
# and end a little above the upper threshold to reduce excess white space.
FOURIER_PLOT_MIN_UM = 9
FOURIER_PLOT_MAX_UM = 110.0

# Spectrum and threshold line appearance.
SPECTRUM_LINE_STYLE = "--"
THRESHOLD_LINE_STYLE = "--"
THRESHOLD_LINE_WIDTH = 1.3
THRESHOLD_LINE_COLOR = "0.35"

# Number of strongest modes printed in the terminal for each Fourier plot.
NUMBER_OF_REPORTED_MODES = 5
POWER_SPECTRUM_BINS = 350


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
    # The returned and plotted power is not normalised.
    # -------------------------------------------------------------------------
    radial_fig, radial_axis = plt.subplots(
        1,
        1,
        figsize=(8, 5.5),
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
        radial_axis.plot(
            wavelength_um,
            power,
            linewidth=LINE_WIDTH,
            color=spectrum_colours[abbreviation],
            linestyle=SPECTRUM_LINE_STYLE,
            label=f"{abbreviation} radial 2-D spectrum",
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

    if POWER_SPECTRUM_LOG_X:
        radial_axis.set_xscale("log")

    add_fourier_threshold_markers(radial_axis)
    radial_axis.set_xlabel(
        r"Spatial wavelength, $\lambda_s$ ($\mu$m)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    radial_axis.set_ylabel(
        "Radially averaged 2D Fourier power",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    radial_axis.set_title(
        "Radially averaged 2-D Fourier power spectrum",
        fontsize=TITLE_FONT_SIZE,
    )
    radial_axis.grid(alpha=0.25, which="both")
    radial_axis.legend(fontsize=LEGEND_FONT_SIZE)
    radial_axis.tick_params(labelsize=TICK_FONT_SIZE)

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
