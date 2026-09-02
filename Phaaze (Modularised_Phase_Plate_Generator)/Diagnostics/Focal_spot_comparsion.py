"""
Publication-style comparison of 2 or 3 focal-spot intensity maps.

Comparison structure
--------------------
TARGET is always the reference.
GENERATED is compared with TARGET.
SECONDARY is optional and, when present, is also compared with TARGET.
GENERATED and SECONDARY are never compared with each other.

All focal spots are independently peak-normalised before comparison:

    I_n(x, y) = I(x, y) / max(I)

Physical coordinates are taken directly from a saved focal-spot extent file
[xmin, xmax, ymin, ymax] in microns whenever available. These are treated as
the first/last sample-centre coordinates produced by the propagation grid.

Main outputs
------------
1. focal_spot_comparison.png / .pdf
   Normalised focal spots plus horizontal/vertical target-centred lineouts,
   with optional fitted super-Gaussian envelopes and fitted order n.
2. focal_spot_power_spectra.png / .pdf
   1-D lineout power spectra plotted against spatial wavelength.
3. focal_spot_ssim_maps.png / .pdf
   Local SSIM maps for each target comparison.
4. focal_spot_metrics.csv
   Pairwise scalar metrics.
5. focal_spot_report.txt
   Human-readable metric definitions, equations, sampling notes, and results.

Required packages
-----------------
numpy, scipy, matplotlib, scikit-image

Metric definitions and references
---------------------------------
Peak normalisation
    I_n = I / I_max.

Lineout residual standard deviation
    d_i = T_i - R_i
    sigma_d = sqrt( sum_i (d_i - mean(d))^2 / (N - 1) )

Pearson correlation coefficient (PCC)
    r = sum[(x-m_x)(y-m_y)] /
        sqrt(sum[(x-m_x)^2] sum[(y-m_y)^2])

    Reference: scipy.stats.pearsonr documentation, which states this equation.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

Structural Similarity Index (SSIM)
    SSIM(x,y) = [(2 mu_x mu_y + C1)(2 sigma_xy + C2)] /
                [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]

    Reference: Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P.
    "Image quality assessment: From error visibility to structural similarity."
    IEEE Trans. Image Process. 13, 600-612 (2004).
    DOI: 10.1109/TIP.2003.819861

    This script uses gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1.0, matching the settings
    recommended by scikit-image for the Wang et al. implementation.

Normalised Mutual Information (NMI, reported on 0-1 scale)
    NMI_Studholme(A,B) = [H(A) + H(B)] / H(A,B)
    NMI_01 = NMI_Studholme - 1
    H(X) = -sum p(x) log p(x)

    Reference: Studholme, C., Hill, D. L. G. & Hawkes, D. J.
    "An overlap invariant entropy measure of 3D medical image alignment."
    Pattern Recognition 32, 71-86 (1999).
    DOI: 10.1016/S0031-3203(98)00091-0

    scikit-image returns the Studholme convention, approximately 1 for
    unrelated intensities and 2 for perfectly dependent intensities. This
    script subtracts 1 before reporting, giving the more intuitive 0-1 scale:
    0 = unrelated and 1 = perfectly dependent.

Relative NRMSE
    NRMSE = ||R - T||_2 / ||R||_2

Mean absolute error
    MAE = (1/N) sum |T_i - R_i|

Centroid
    x_c = sum(I_i x_i) / sum(I_i),  y_c = sum(I_i y_i) / sum(I_i)

Enclosed-energy diameter D_p
    Find r_p such that sum_{r_i <= r_p} I_i = p sum_i I_i, then D_p = 2 r_p.

1-D Fourier power spectrum
    X_k = sum_{n=0}^{N-1} x_n exp(-2 pi i k n / N)
    P_k = |X_k|^2
    spatial wavelength Lambda_k = 1 / f_k

    References: NumPy rfft/rfftfreq documentation.
    https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
    https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html

Important sampling note
-----------------------
A sampled lineout with spacing dx can only represent spatial wavelengths down
approximately to the Nyquist wavelength 2*dx. Therefore a requested plotting
range of 1-100 um does NOT imply that 1 um structure is resolved. For example,
10 um/pixel focal-plane sampling only supports wavelengths >= 20 um.

The FFT comparison subtracts the lineout mean and applies a Hann window before
calculating power. This removes the dominant DC offset and reduces leakage from
the finite lineout window. Power is then normalised by the maximum within the
requested wavelength band so the comparison emphasises structural content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import shift as ndi_shift
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from skimage.metrics import normalized_mutual_information, structural_similarity
from skimage.registration import phase_cross_correlation


# =============================================================================
# USER CONTROL PANEL
# =============================================================================

SCRIPT_VERSION = "2026-07-27 simplified plotting v6"


@dataclass(frozen=True)
class SpotSpec:
    path: Path
    label: str
    pixel_size_um: float | None = None
    # Optional 4-value [xmin, xmax, ymin, ymax] file in microns. When present,
    # it defines the physical x/y coordinate axes directly and takes precedence
    # over pixel_size_um. If None, common extent filenames are searched beside
    # the focal-spot array before falling back to pixel_size_um/default sampling.
    extent_file: Path | None = None


# Replace these paths with your files.
TARGET = SpotSpec(
    path=Path(r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)\Validation\Vulcan_near_field\Results\Vulcan_near_field_2026-07-27_22-58-20\Arrays\Target_focal_spot_structure.npy"),
    label="Target",
    pixel_size_um=None,
    extent_file=Path(r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)\Validation\Vulcan_near_field\Results\Vulcan_near_field_2026-07-27_22-58-20\Arrays\Focal_spot_extent_um.npy"),
)

GENERATED = SpotSpec(
    path=Path(r"C:\Users\benny\OneDrive\DOCUME~1\Desktop\python\PHYSIC~1\PHASEP~1\PHAAZE~1\VALIDA~1\VULCAN~1\Results\VULCAN~1\Arrays\IDEAL_~3.NPY"),
    label="Generated",
    pixel_size_um=None,
    extent_file=Path(r"C:\Users\benny\OneDrive\Documents\Desktop\python\Physics toolbox\Phase plate generator\Phaaze (Modularised_Phase_Plate_Generator)\Validation\Vulcan_near_field\Results\Vulcan_near_field_2026-07-27_22-58-20\Arrays\Focal_spot_extent_um.npy"),
)

# Set SECONDARY = None for a two-spot comparison.
SECONDARY: SpotSpec | None = None
# Example:
# SECONDARY = SpotSpec(
#     path=Path(r"SECONDARY_FOCAL_SPOT.npy"),
#     label="Secondary",
#     pixel_size_um=None,
# )

# Used only when neither SpotSpec.pixel_size_um nor an extent file is available.
# Your current phase-plate generator commonly uses 10 um focal-plane pixels.
DEFAULT_PIXEL_SIZE_UM = 10.0

# Metrics are calculated in this square region centred on the TARGET centroid.
# None uses the full target field of view.
ANALYSIS_HALF_WIDTH_UM: float | None = 400.0

# Exact lineout location is the TARGET centroid. Increasing this to 3, 5, ...
# averages neighbouring rows/columns and can suppress single-pixel noise.
LINEOUT_AVERAGE_WIDTH_PX = 1

# Optional super-Gaussian envelope fit to the target-centred x/y lineouts.
# Model: I = B + A exp[-(|x-x0|/w)^(2n)]
FIT_SUPERGAUSSIAN = True
# None uses ANALYSIS_HALF_WIDTH_UM; set a value to choose a dedicated fit window.
SUPERGAUSSIAN_FIT_HALF_WIDTH_UM: float | None = None
SUPERGAUSSIAN_MIN_ORDER = 0.25
SUPERGAUSSIAN_MAX_ORDER = 50.0

# Optional sub-pixel translation before structure/error metrics.
# False is recommended when pointing/registration error is part of performance.
REGISTER_TO_TARGET = False
UPSAMPLE_REGISTRATION = 20

# Histogram granularity used by NMI.
NMI_BINS = 100

# Spatial-wavelength band requested for Fourier comparison.
FFT_WAVELENGTH_MIN_UM = 1.0
FFT_WAVELENGTH_MAX_UM = 100.0

# Save vector + raster versions suitable for papers.
OUTPUT_DIR = Path("focal_spot_comparison_outputs")
SAVE_PNG = True
SAVE_PDF = False
SHOW_PLOTS = True
DPI = 400

# Visual crop for focal-spot panels. None uses ANALYSIS_HALF_WIDTH_UM.
PLOT_HALF_WIDTH_UM: float | None = None

# =============================================================================
# PLOT CONTROLS - edit these first when changing figure appearance
# =============================================================================

# Images
IMAGE_CMAP = "viridis"
IMAGE_VMIN = 0.0
IMAGE_VMAX = 1.0

# Main lineouts
X_LINE_COLOUR = "red"
Y_LINE_COLOUR = "black"
X_FIT_COLOUR = "black"
Y_FIT_COLOUR = "red"
LINE_WIDTH = 1.5
FIT_LINE_WIDTH = 1.3
FIT_LINESTYLE = "--"
GRID_ALPHA = 0.20

# Fourier spectra: colour identifies which focal spot is being shown.
TARGET_COLOUR = "red"
GENERATED_COLOUR = "black"
SECONDARY_COLOUR = "blue"
FOURIER_LINESTYLE = "--"
FOURIER_LINE_WIDTH = 1.8

# Figure sizes
MAIN_ROW_HEIGHT_IN = 4.2
MAIN_FIGURE_WIDTH_IN = 12.5
FOURIER_FIGSIZE = (10.5, 7.0)
SSIM_PANEL_SIZE_IN = 3.5

# Main comparison layout
# Positive value moves the image colour bars LEFT, towards the focal-spot image.
COLOURBAR_SHIFT_LEFT_MM = 10.0
# Keep the horizontal axis on the upper x-lineout. The extra vertical spacing
# below prevents its x-label from colliding with the y-lineout title.
HIDE_TOP_LINEOUT_X_AXIS = False

# Vertical spacing controls for the main comparison figure.
# Increase LINEOUT_VERTICAL_HSPACE if x/y labels or titles get too close.
LINEOUT_VERTICAL_HSPACE = 0.80
SPOT_ROW_HSPACE = 0.38



# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpotData:
    label: str
    intensity: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    extent_um: tuple[float, float, float, float]
    extent_source: str

    @property
    def dx_um(self) -> float:
        return float(np.median(np.diff(self.x_um))) if self.x_um.size > 1 else float("nan")

    @property
    def dy_um(self) -> float:
        return float(np.median(np.diff(self.y_um))) if self.y_um.size > 1 else float("nan")

    @property
    def pixel_size_um(self) -> float:
        """Backward-compatible isotropic pitch (mean of dx and dy)."""
        return float(0.5 * (self.dx_um + self.dy_um))


@dataclass
class SuperGaussianFit:
    success: bool
    background: float = float("nan")
    amplitude: float = float("nan")
    centre_um: float = float("nan")
    width_1e_um: float = float("nan")
    order: float = float("nan")
    fwhm_um: float = float("nan")
    r_squared: float = float("nan")


@dataclass
class PairResult:
    label: str
    test_image: np.ndarray
    pcc: float
    ssim: float
    nmi: float
    nrmse: float
    mae: float
    centroid_shift_um: float
    centroid_dx_um: float
    centroid_dy_um: float
    lineout_std_x: float
    lineout_std_y: float
    lineout_rmse_x: float
    lineout_rmse_y: float
    fwhm_x_um: float
    fwhm_y_um: float
    fwhm_x_difference_um: float
    fwhm_y_difference_um: float
    d50_um: float
    d80_um: float
    d50_difference_um: float
    d80_difference_um: float
    spectral_log_pcc_x: float
    spectral_log_pcc_y: float
    supergaussian_x: SuperGaussianFit | None
    supergaussian_y: SuperGaussianFit | None
    applied_registration_dx_um: float
    applied_registration_dy_um: float
    ssim_map: np.ndarray


# =============================================================================
# STYLE
# =============================================================================

def apply_publication_style() -> None:
    """Use Matplotlib defaults and keep only export-specific settings."""
    plt.rcdefaults()
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


# =============================================================================
# LOADING, NORMALISATION, GRID HANDLING
# =============================================================================

def load_2d_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Focal-spot file not found: {path}")
    arr = np.asarray(np.load(path), dtype=float).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D focal spot at {path}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Focal spot contains NaN or inf: {path}")
    return arr


def peak_normalise(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=float), 0.0, None)
    peak = float(np.max(image))
    if peak <= 0:
        raise ValueError("Intensity map has no positive values and cannot be normalised")
    return image / peak


def _resolve_extent_path(spec: SpotSpec) -> Path | None:
    """Return an explicit or automatically discovered focal-plane extent file."""
    if spec.extent_file is not None:
        if not spec.extent_file.exists():
            raise FileNotFoundError(
                f"Configured extent file not found for {spec.label}: {spec.extent_file}"
            )
        return spec.extent_file

    candidates = [
        "Focal_spot_extent_um.npy",
        "focal_spot_extent_um.npy",
        "focal_spot_extent.npy",
        "Focal_spot_extent.npy",
    ]
    for name in candidates:
        candidate = spec.path.parent / name
        if candidate.exists():
            return candidate
    return None


def load_extent_um(path: Path) -> tuple[float, float, float, float]:
    """Load [xmin, xmax, ymin, ymax] coordinates in microns."""
    extent = np.asarray(np.load(path), dtype=float).ravel()
    if extent.size != 4 or not np.isfinite(extent).all():
        raise ValueError(
            f"Extent file must contain four finite values "
            f"[xmin, xmax, ymin, ymax], got {extent} from {path}"
        )
    xmin, xmax, ymin, ymax = map(float, extent)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid physical extent in {path}: {extent}")
    return xmin, xmax, ymin, ymax


def centered_axis_um(samples: int, pixel_size_um: float) -> np.ndarray:
    return (np.arange(samples) - samples // 2) * pixel_size_um


def build_physical_axes_um(
    spec: SpotSpec,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], str]:
    """Build the real focal-plane axes, preferring the saved physical extent."""
    ny, nx = shape
    extent_path = _resolve_extent_path(spec)
    if extent_path is not None:
        xmin, xmax, ymin, ymax = load_extent_um(extent_path)
        # The generator saves Xf.min()/Xf.max() and Yf.min()/Yf.max(), i.e.
        # coordinates of the first and last sample centres. Preserve those
        # coordinates exactly rather than forcing the grid to be symmetric.
        x_um = np.linspace(xmin, xmax, nx)
        y_um = np.linspace(ymin, ymax, ny)
        return x_um, y_um, (xmin, xmax, ymin, ymax), str(extent_path)

    pixel = spec.pixel_size_um
    if pixel is None:
        warnings.warn(
            f"No focal-plane extent or pixel size found for {spec.label}; using "
            f"DEFAULT_PIXEL_SIZE_UM={DEFAULT_PIXEL_SIZE_UM:g} um."
        )
        pixel = DEFAULT_PIXEL_SIZE_UM
    if pixel <= 0:
        raise ValueError(f"pixel_size_um must be positive for {spec.label}")

    x_um = centered_axis_um(nx, float(pixel))
    y_um = centered_axis_um(ny, float(pixel))
    extent = (float(x_um[0]), float(x_um[-1]), float(y_um[0]), float(y_um[-1]))
    source = "SpotSpec.pixel_size_um" if spec.pixel_size_um is not None else "DEFAULT_PIXEL_SIZE_UM"
    return x_um, y_um, extent, source


def load_spot(spec: SpotSpec) -> SpotData:
    raw = load_2d_npy(spec.path)
    image = peak_normalise(raw)
    x_um, y_um, extent_um, extent_source = build_physical_axes_um(spec, image.shape)
    return SpotData(
        label=spec.label,
        intensity=image,
        x_um=x_um,
        y_um=y_um,
        extent_um=extent_um,
        extent_source=extent_source,
    )


def image_extent_edges_um(spot: SpotData) -> list[float]:
    """Return imshow edge coordinates from sample-centre x/y axes."""
    dx = spot.dx_um
    dy = spot.dy_um
    return [
        float(spot.x_um[0] - 0.5 * dx),
        float(spot.x_um[-1] + 0.5 * dx),
        float(spot.y_um[0] - 0.5 * dy),
        float(spot.y_um[-1] + 0.5 * dy),
    ]

def resample_to_target_grid(source: SpotData, target: SpotData) -> np.ndarray:
    """Interpolate a centred source image onto the target physical grid."""
    if (
        source.intensity.shape == target.intensity.shape
        and np.allclose(source.x_um, target.x_um, rtol=1e-12, atol=1e-12)
        and np.allclose(source.y_um, target.y_um, rtol=1e-12, atol=1e-12)
    ):
        return source.intensity.copy()

    interpolator = RegularGridInterpolator(
        (source.y_um, source.x_um),
        source.intensity,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    X, Y = np.meshgrid(target.x_um, target.y_um, indexing="xy")
    points = np.column_stack((Y.ravel(), X.ravel()))
    out = interpolator(points).reshape(target.intensity.shape)
    return peak_normalise(out)


# =============================================================================
# BASIC METRICS
# =============================================================================

def weighted_centroid(image: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float]:
    I = np.clip(np.asarray(image, float), 0, None)
    total = float(I.sum())
    if total <= 0:
        return float("nan"), float("nan")
    X, Y = np.meshgrid(x_um, y_um, indexing="xy")
    return float((I * X).sum() / total), float((I * Y).sum() / total)


def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def build_analysis_mask(
    target: SpotData,
    target_centroid: tuple[float, float],
    half_width_um: float | None,
) -> np.ndarray:
    if half_width_um is None:
        return np.ones(target.intensity.shape, dtype=bool)
    if half_width_um <= 0:
        raise ValueError("ANALYSIS_HALF_WIDTH_UM must be positive or None")
    xc, yc = target_centroid
    X, Y = np.meshgrid(target.x_um, target.y_um, indexing="xy")
    return (np.abs(X - xc) <= half_width_um) & (np.abs(Y - yc) <= half_width_um)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def relative_nrmse(reference: np.ndarray, test: np.ndarray) -> float:
    ref = np.asarray(reference, float)
    tst = np.asarray(test, float)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(ref - tst) / denom) if denom > 0 else float("nan")


def mean_absolute_error(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(reference) - np.asarray(test))))


def nmi_zero_to_one(reference: np.ndarray, test: np.ndarray) -> float:
    """Return Studholme NMI shifted from its native 1-2 range to 0-1."""
    raw = float(normalized_mutual_information(reference, test, bins=NMI_BINS))
    return float(np.clip(raw - 1.0, 0.0, 1.0))


def maybe_register(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Return optionally registered test and applied (dy, dx) pixel shift."""
    if not REGISTER_TO_TARGET:
        return test.copy(), (0.0, 0.0)

    shift_px, _, _ = phase_cross_correlation(
        reference,
        test,
        upsample_factor=UPSAMPLE_REGISTRATION,
        normalization=None,
    )
    aligned = ndi_shift(test, shift=shift_px, order=3, mode="constant", cval=0.0)
    aligned = np.clip(aligned, 0.0, None)
    return peak_normalise(aligned), (float(shift_px[0]), float(shift_px[1]))


# =============================================================================
# LINEOUTS AND SPOT-SIZE METRICS
# =============================================================================

def averaged_lineouts(
    image: np.ndarray,
    row: int,
    col: int,
    width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    if width_px < 1 or width_px % 2 == 0:
        raise ValueError("LINEOUT_AVERAGE_WIDTH_PX must be a positive odd integer")
    half = width_px // 2
    r0, r1 = max(0, row - half), min(image.shape[0], row + half + 1)
    c0, c1 = max(0, col - half), min(image.shape[1], col + half + 1)
    x_line = np.mean(image[r0:r1, :], axis=0)
    y_line = np.mean(image[:, c0:c1], axis=1)
    return x_line, y_line


def residual_std(reference: np.ndarray, test: np.ndarray) -> float:
    d = np.asarray(test, float) - np.asarray(reference, float)
    return float(np.std(d, ddof=1)) if d.size > 1 else 0.0


def rmse(reference: np.ndarray, test: np.ndarray) -> float:
    d = np.asarray(test, float) - np.asarray(reference, float)
    return float(np.sqrt(np.mean(d**2)))


def fwhm_um(line: np.ndarray, axis_um: np.ndarray) -> float:
    y = np.asarray(line, float)
    if y.size < 2 or np.max(y) <= 0:
        return float("nan")
    y = y / np.max(y)
    above = np.flatnonzero(y >= 0.5)
    if above.size == 0:
        return float("nan")

    left = int(above[0])
    right = int(above[-1])

    def crossing(i0: int, i1: int) -> float:
        x0, x1 = axis_um[i0], axis_um[i1]
        y0, y1 = y[i0], y[i1]
        if np.isclose(y1, y0):
            return float(0.5 * (x0 + x1))
        return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))

    if left > 0:
        x_left = crossing(left - 1, left)
    else:
        x_left = float(axis_um[left])

    if right < y.size - 1:
        x_right = crossing(right, right + 1)
    else:
        x_right = float(axis_um[right])

    return float(x_right - x_left)


def enclosed_energy_diameter_um(
    image: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    fraction: float,
    centre: tuple[float, float] | None = None,
) -> float:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    I = np.clip(np.asarray(image, float), 0, None)
    total = float(I.sum())
    if total <= 0:
        return float("nan")
    if centre is None:
        centre = weighted_centroid(I, x_um, y_um)
    xc, yc = centre
    X, Y = np.meshgrid(x_um, y_um, indexing="xy")
    radius = np.hypot(X - xc, Y - yc).ravel()
    weights = I.ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(weights[order])
    idx = int(np.searchsorted(cumulative, fraction * cumulative[-1]))
    idx = min(idx, len(order) - 1)
    return float(2.0 * radius[order[idx]])


# =============================================================================
# OPTIONAL SUPER-GAUSSIAN LINEOUT FIT
# =============================================================================

def supergaussian_1d(
    x_um: np.ndarray,
    background: float,
    amplitude: float,
    centre_um: float,
    width_1e_um: float,
    order: float,
) -> np.ndarray:
    """Intensity super-Gaussian: B + A exp[-(|x-x0|/w)^(2n)]."""
    x = np.asarray(x_um, dtype=float)
    width = max(float(width_1e_um), np.finfo(float).eps)
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = np.power(np.abs((x - centre_um) / width), 2.0 * order)
    return background + amplitude * np.exp(-exponent)


def fit_supergaussian_lineout(
    line: np.ndarray,
    axis_um: np.ndarray,
    centre_hint_um: float,
) -> SuperGaussianFit | None:
    """Fit the broad lineout envelope and return order n plus goodness of fit."""
    if not FIT_SUPERGAUSSIAN:
        return None

    y_all = np.asarray(line, dtype=float)
    x_all = np.asarray(axis_um, dtype=float)
    if y_all.size < 7 or np.max(y_all) <= 0:
        return SuperGaussianFit(False)
    y_all = np.clip(y_all / np.max(y_all), 0.0, None)

    fit_half = SUPERGAUSSIAN_FIT_HALF_WIDTH_UM
    if fit_half is None:
        fit_half = ANALYSIS_HALF_WIDTH_UM
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    if fit_half is not None:
        if fit_half <= 0:
            raise ValueError("SUPERGAUSSIAN_FIT_HALF_WIDTH_UM must be positive or None")
        mask &= np.abs(x_all - centre_hint_um) <= fit_half

    x = x_all[mask]
    y = y_all[mask]
    if x.size < 7:
        return SuperGaussianFit(False)

    spacing = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    background0 = float(np.clip(np.percentile(y, 5), 0.0, 0.5))
    amplitude0 = float(max(np.max(y) - background0, 0.1))
    centre0 = float(x[np.argmax(y)])
    rough_fwhm = fwhm_um(y, x)
    n0 = 4.0
    if np.isfinite(rough_fwhm) and rough_fwhm > 0:
        width0 = rough_fwhm / (2.0 * np.log(2.0) ** (1.0 / (2.0 * n0)))
    else:
        width0 = max(span / 4.0, abs(spacing))

    lower = [0.0, 0.0, float(x.min()), max(abs(spacing) * 0.25, 1e-9), SUPERGAUSSIAN_MIN_ORDER]
    upper = [1.0, 2.0, float(x.max()), max(span * 2.0, abs(spacing)), SUPERGAUSSIAN_MAX_ORDER]
    p0 = [background0, amplitude0, centre0, float(np.clip(width0, lower[3] * 1.01, upper[3] * 0.99)), n0]

    try:
        popt, _ = curve_fit(
            supergaussian_1d,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=40000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return SuperGaussianFit(False)

    background, amplitude, centre, width, order = map(float, popt)
    y_fit = supergaussian_1d(x, *popt)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    fitted_fwhm = float(2.0 * width * np.log(2.0) ** (1.0 / (2.0 * order)))
    return SuperGaussianFit(
        True,
        background=background,
        amplitude=amplitude,
        centre_um=centre,
        width_1e_um=width,
        order=order,
        fwhm_um=fitted_fwhm,
        r_squared=r2,
    )


def supergaussian_label(label: str, fit: SuperGaussianFit | None) -> str:
    if fit is None or not fit.success or not np.isfinite(fit.order):
        return label
    return f"{label} ($n={fit.order:.2f}$)"


# =============================================================================
# FOURIER / POWER-SPECTRUM COMPARISON
# =============================================================================

def lineout_power_spectrum(
    line: np.ndarray,
    dx_um: float,
    wavelength_min_um: float,
    wavelength_max_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength [um] and normalised one-sided Hann-windowed power."""
    y = np.asarray(line, float)
    if y.size < 4:
        return np.array([]), np.array([])

    y = y - np.mean(y)
    window = np.hanning(y.size)
    transformed = np.fft.rfft(y * window)
    power = np.abs(transformed) ** 2
    freq = np.fft.rfftfreq(y.size, d=dx_um)  # cycles per um

    valid_freq = freq > 0
    freq = freq[valid_freq]
    power = power[valid_freq]
    wavelength = 1.0 / freq

    band = (
        (wavelength >= wavelength_min_um)
        & (wavelength <= wavelength_max_um)
        & np.isfinite(power)
    )
    wavelength = wavelength[band]
    power = power[band]

    if power.size and np.max(power) > 0:
        power = power / np.max(power)

    order = np.argsort(wavelength)
    return wavelength[order], power[order]


def spectral_log_correlation(
    reference_line: np.ndarray,
    test_line: np.ndarray,
    dx_um: float,
) -> float:
    wr, pr = lineout_power_spectrum(
        reference_line, dx_um, FFT_WAVELENGTH_MIN_UM, FFT_WAVELENGTH_MAX_UM
    )
    wt, pt = lineout_power_spectrum(
        test_line, dx_um, FFT_WAVELENGTH_MIN_UM, FFT_WAVELENGTH_MAX_UM
    )
    if wr.size < 3 or wt.size < 3:
        return float("nan")

    lo = max(float(wr.min()), float(wt.min()))
    hi = min(float(wr.max()), float(wt.max()))
    if hi <= lo:
        return float("nan")
    common = np.geomspace(lo, hi, 250)
    pr_i = np.interp(common, wr, pr)
    pt_i = np.interp(common, wt, pt)
    eps = 1e-12
    return pcc(np.log10(pr_i + eps), np.log10(pt_i + eps))


# =============================================================================
# PAIRWISE COMPARISON
# =============================================================================

def compare_pair(
    target: SpotData,
    test_label: str,
    test_on_target_grid: np.ndarray,
    mask: np.ndarray,
    target_centroid: tuple[float, float],
    row: int,
    col: int,
    target_line_x: np.ndarray,
    target_line_y: np.ndarray,
    target_supergaussian_x: SuperGaussianFit | None,
    target_supergaussian_y: SuperGaussianFit | None,
    target_fwhm_x: float,
    target_fwhm_y: float,
    target_d50: float,
    target_d80: float,
) -> PairResult:
    registered, applied_shift_px = maybe_register(target.intensity, test_on_target_grid)

    ref_roi = target.intensity[mask]
    tst_roi = registered[mask]

    # SSIM is a local-window metric; crop a rectangular ROI rather than pass a
    # flattened mask. This preserves image neighbourhoods.
    yy, xx = np.where(mask)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    ref_crop = target.intensity[y0:y1, x0:x1]
    tst_crop = registered[y0:y1, x0:x1]

    # SSIM requires enough samples for the Gaussian window. Fall back to a
    # smaller odd window if a user chooses an extremely small ROI.
    min_dim = min(ref_crop.shape)
    if min_dim < 3:
        raise ValueError("Analysis ROI is too small for SSIM")
    if min_dim < 11:
        win = min_dim if min_dim % 2 == 1 else min_dim - 1
        ssim_val, ssim_map_crop = structural_similarity(
            ref_crop, tst_crop, data_range=1.0, win_size=win,
            gaussian_weights=False, full=True,
        )
    else:
        ssim_val, ssim_map_crop = structural_similarity(
            ref_crop,
            tst_crop,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            full=True,
        )

    ssim_map = np.full_like(target.intensity, np.nan, dtype=float)
    ssim_map[y0:y1, x0:x1] = ssim_map_crop

    test_centroid = weighted_centroid(registered, target.x_um, target.y_um)
    dx_centroid = test_centroid[0] - target_centroid[0]
    dy_centroid = test_centroid[1] - target_centroid[1]

    test_line_x, test_line_y = averaged_lineouts(
        registered, row, col, LINEOUT_AVERAGE_WIDTH_PX
    )
    test_supergaussian_x = fit_supergaussian_lineout(
        test_line_x, target.x_um, target_centroid[0]
    )
    test_supergaussian_y = fit_supergaussian_lineout(
        test_line_y, target.y_um, target_centroid[1]
    )
    test_fwhm_x = fwhm_um(test_line_x, target.x_um)
    test_fwhm_y = fwhm_um(test_line_y, target.y_um)
    test_d50 = enclosed_energy_diameter_um(
        registered, target.x_um, target.y_um, 0.50, centre=test_centroid
    )
    test_d80 = enclosed_energy_diameter_um(
        registered, target.x_um, target.y_um, 0.80, centre=test_centroid
    )

    return PairResult(
        label=test_label,
        test_image=registered,
        pcc=pcc(ref_roi, tst_roi),
        ssim=float(ssim_val),
        nmi=nmi_zero_to_one(ref_roi, tst_roi),
        nrmse=relative_nrmse(ref_roi, tst_roi),
        mae=mean_absolute_error(ref_roi, tst_roi),
        centroid_shift_um=float(np.hypot(dx_centroid, dy_centroid)),
        centroid_dx_um=float(dx_centroid),
        centroid_dy_um=float(dy_centroid),
        lineout_std_x=residual_std(target_line_x, test_line_x),
        lineout_std_y=residual_std(target_line_y, test_line_y),
        lineout_rmse_x=rmse(target_line_x, test_line_x),
        lineout_rmse_y=rmse(target_line_y, test_line_y),
        fwhm_x_um=test_fwhm_x,
        fwhm_y_um=test_fwhm_y,
        fwhm_x_difference_um=test_fwhm_x - target_fwhm_x,
        fwhm_y_difference_um=test_fwhm_y - target_fwhm_y,
        d50_um=test_d50,
        d80_um=test_d80,
        d50_difference_um=test_d50 - target_d50,
        d80_difference_um=test_d80 - target_d80,
        spectral_log_pcc_x=spectral_log_correlation(
            target_line_x, test_line_x, target.dx_um
        ),
        spectral_log_pcc_y=spectral_log_correlation(
            target_line_y, test_line_y, target.dy_um
        ),
        supergaussian_x=test_supergaussian_x,
        supergaussian_y=test_supergaussian_y,
        applied_registration_dx_um=applied_shift_px[1] * target.dx_um,
        applied_registration_dy_um=applied_shift_px[0] * target.dy_um,
        ssim_map=ssim_map,
    )


# =============================================================================
# PLOTTING
# =============================================================================

def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure using the common output controls above."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")


def _plot_half_width() -> float | None:
    """Return the display half-width used by all spatial plots."""
    return PLOT_HALF_WIDTH_UM if PLOT_HALF_WIDTH_UM is not None else ANALYSIS_HALF_WIDTH_UM


def _spot_fourier_colour(index: int) -> str:
    """Return the colour used for target/generated/secondary Fourier curves."""
    colours = [TARGET_COLOUR, GENERATED_COLOUR, SECONDARY_COLOUR]
    if index < len(colours):
        return colours[index]
    return f"C{index}"


def _plot_lineout_panel(
    ax: plt.Axes,
    axis_um: np.ndarray,
    line: np.ndarray,
    fit: SuperGaussianFit | None,
    *,
    direction: str,
    spot_label: str,
    residual_std_value: float | None,
    centre_um: float,
    show_x_axis: bool = True,
) -> None:
    """Plot one lineout. Colours are controlled in PLOT CONTROLS above."""
    is_x = direction.lower() == "x"
    line_colour = X_LINE_COLOUR if is_x else Y_LINE_COLOUR
    fit_colour = X_FIT_COLOUR if is_x else Y_FIT_COLOUR

    ax.plot(
        axis_um,
        line,
        color=line_colour,
        lw=LINE_WIDTH,
        label=spot_label,
    )

    if fit is not None and fit.success:
        fitted = supergaussian_1d(
            axis_um,
            fit.background,
            fit.amplitude,
            fit.centre_um,
            fit.width_1e_um,
            fit.order,
        )
        ax.plot(
            axis_um,
            fitted,
            color=fit_colour,
            lw=FIT_LINE_WIDTH,
            ls=FIT_LINESTYLE,
            label=rf"Super-Gaussian, $n={fit.order:.2f}$",
        )

    title = f"{spot_label} {direction}-lineout"
    if residual_std_value is not None and np.isfinite(residual_std_value):
        title += rf" | $\sigma_{{\rm res}}={residual_std_value:.3f}$"
    ax.set_title(title)

    if show_x_axis:
        ax.set_xlabel(rf"${direction}$ ($\mu$m)")
    else:
        # The upper lineout shares the same physical horizontal scale as the
        # lower panel, so removing its x-axis avoids duplicated/overlapping text.
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax.set_ylabel(r"$I/I_{\max}$")
    ax.set_ylim(-0.03, 1.05)

    plot_half = _plot_half_width()
    if plot_half is not None:
        ax.set_xlim(centre_um - plot_half, centre_um + plot_half)

    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="upper right")


def plot_spots_and_lineouts(
    target: SpotData,
    results: list[PairResult],
    target_centroid: tuple[float, float],
    row: int,
    col: int,
    target_supergaussian_x: SuperGaussianFit | None,
    target_supergaussian_y: SuperGaussianFit | None,
) -> None:
    """Plot one row per focal spot: image left, x/y lineouts right."""
    target_x, target_y = averaged_lineouts(
        target.intensity, row, col, LINEOUT_AVERAGE_WIDTH_PX
    )

    entries = [
        {
            "label": target.label,
            "image": target.intensity,
            "xline": target_x,
            "yline": target_y,
            "fit_x": target_supergaussian_x,
            "fit_y": target_supergaussian_y,
            "std_x": None,
            "std_y": None,
        }
    ]

    for result in results:
        x_line, y_line = averaged_lineouts(
            result.test_image, row, col, LINEOUT_AVERAGE_WIDTH_PX
        )
        entries.append(
            {
                "label": result.label,
                "image": result.test_image,
                "xline": x_line,
                "yline": y_line,
                "fit_x": result.supergaussian_x,
                "fit_y": result.supergaussian_y,
                "std_x": result.lineout_std_x,
                "std_y": result.lineout_std_y,
            }
        )

    n_spots = len(entries)
    fig = plt.figure(
        figsize=(MAIN_FIGURE_WIDTH_IN, MAIN_ROW_HEIGHT_IN * n_spots)
    )

    # One simple 3-column layout per row:
    # focal spot | colour bar | x/y lineouts
    grid = GridSpec(
        n_spots,
        3,
        figure=fig,
        width_ratios=[1.0, 0.04, 2.1],
        hspace=SPOT_ROW_HSPACE,
        wspace=0.28,
    )

    xc, yc = target_centroid
    plot_half = _plot_half_width()
    extent = image_extent_edges_um(target)
    colourbar_axes: list[plt.Axes] = []

    for row_index, entry in enumerate(entries):
        # ----- focal spot -----
        ax_image = fig.add_subplot(grid[row_index, 0])
        image_handle = ax_image.imshow(
            entry["image"],
            origin="lower",
            cmap=IMAGE_CMAP,
            vmin=IMAGE_VMIN,
            vmax=IMAGE_VMAX,
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )

        ax_image.axhline(
            target.y_um[row],
            color=X_LINE_COLOUR,
            ls="--",
            lw=1.1,
            label="x-lineout",
        )
        ax_image.axvline(
            target.x_um[col],
            color=Y_LINE_COLOUR,
            ls=":",
            lw=1.2,
            label="y-lineout",
        )

        ax_image.set_title(f"{entry['label']} focal spot")
        ax_image.set_xlabel(r"$x$ ($\mu$m)")
        ax_image.set_ylabel(r"$y$ ($\mu$m)")
        ax_image.legend(loc="upper right")

        if plot_half is not None:
            ax_image.set_xlim(xc - plot_half, xc + plot_half)
            ax_image.set_ylim(yc - plot_half, yc + plot_half)

        # ----- colour bar -----
        cax = fig.add_subplot(grid[row_index, 1])
        colourbar_axes.append(cax)
        cbar = fig.colorbar(image_handle, cax=cax)
        cbar.set_label(r"$I/I_{\max}$")
        cbar.set_ticks([0.0, 0.5, 1.0])

        # ----- lineouts -----
        lineout_grid = grid[row_index, 2].subgridspec(2, 1, hspace=LINEOUT_VERTICAL_HSPACE)
        ax_x = fig.add_subplot(lineout_grid[0, 0])
        ax_y = fig.add_subplot(lineout_grid[1, 0])

        _plot_lineout_panel(
            ax_x,
            target.x_um,
            entry["xline"],
            entry["fit_x"],
            direction="x",
            spot_label=entry["label"],
            residual_std_value=entry["std_x"],
            centre_um=xc,
            show_x_axis=not HIDE_TOP_LINEOUT_X_AXIS,
        )
        _plot_lineout_panel(
            ax_y,
            target.y_um,
            entry["yline"],
            entry["fit_y"],
            direction="y",
            spot_label=entry["label"],
            residual_std_value=entry["std_y"],
            centre_um=yc,
            show_x_axis=True,
        )

    # Let Matplotlib resolve the normal layout first, then make the small
    # deliberate colour-bar adjustment in physical units. 3 mm = 3/25.4 inch.
    fig.tight_layout()
    shift_fraction = (COLOURBAR_SHIFT_LEFT_MM / 25.4) / fig.get_figwidth()
    for cax in colourbar_axes:
        pos = cax.get_position()
        cax.set_position([
            pos.x0 - shift_fraction,
            pos.y0,
            pos.width,
            pos.height,
        ])
    save_figure(fig, "focal_spot_comparison")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_power_spectra(
    target: SpotData,
    results: list[PairResult],
    row: int,
    col: int,
) -> None:
    """Plot x and y Fourier spectra; colour identifies each focal spot."""
    target_x, target_y = averaged_lineouts(
        target.intensity, row, col, LINEOUT_AVERAGE_WIDTH_PX
    )

    entries = [(target.label, target_x, target_y)]
    for result in results:
        x_line, y_line = averaged_lineouts(
            result.test_image, row, col, LINEOUT_AVERAGE_WIDTH_PX
        )
        entries.append((result.label, x_line, y_line))

    fig, axes = plt.subplots(2, 1, figsize=FOURIER_FIGSIZE, sharex=True)

    for ax, direction, spacing_um in [
        (axes[0], "x", target.dx_um),
        (axes[1], "y", target.dy_um),
    ]:
        for index, (label, x_line, y_line) in enumerate(entries):
            line = x_line if direction == "x" else y_line
            wavelength, power = lineout_power_spectrum(
                line,
                spacing_um,
                FFT_WAVELENGTH_MIN_UM,
                FFT_WAVELENGTH_MAX_UM,
            )
            ax.plot(
                wavelength,
                power,
                color=_spot_fourier_colour(index),
                ls=FOURIER_LINESTYLE,
                lw=FOURIER_LINE_WIDTH,
                label=label,
            )

        nyquist = 2.0 * spacing_um
        ax.axvline(nyquist, color="0.45", ls=":", lw=1.0)
        ax.text(
            nyquist,
            0.97,
            rf"Nyquist = {nyquist:.1f} $\mu$m",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            color="0.35",
        )

        ax.set_xscale("log")
        ax.set_xlim(FFT_WAVELENGTH_MIN_UM, FFT_WAVELENGTH_MAX_UM)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("Normalised power")
        ax.set_title(f"{direction}-lineout power spectrum")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc="upper right")

    axes[1].set_xlabel(r"Spatial wavelength, $\lambda_s$ ($\mu$m)")

    fig.tight_layout()
    save_figure(fig, "focal_spot_power_spectra")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_ssim_maps(
    target: SpotData,
    results: list[PairResult],
    target_centroid: tuple[float, float],
) -> None:
    """Plot local SSIM maps with minimal decoration."""
    if not results:
        return

    n_maps = len(results)
    fig, axes = plt.subplots(
        1,
        n_maps,
        figsize=(SSIM_PANEL_SIZE_IN * n_maps, SSIM_PANEL_SIZE_IN),
        squeeze=False,
    )
    axes = axes.ravel()

    xc, yc = target_centroid
    plot_half = _plot_half_width()
    extent = image_extent_edges_um(target)

    image_handle = None
    for ax, result in zip(axes, results):
        image_handle = ax.imshow(
            result.ssim_map,
            origin="lower",
            cmap=IMAGE_CMAP,
            vmin=-1.0,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )

        # Keep only the information that is useful on the figure.
        ax.set_title(f"{result.label} | SSIM = {result.ssim:.3f}")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")

        if plot_half is not None:
            ax.set_xlim(xc - plot_half, xc + plot_half)
            ax.set_ylim(yc - plot_half, yc + plot_half)

    # One compact shared colour bar. No figure-wide title or panel letters.
    fig.subplots_adjust(right=0.88, wspace=0.28)
    cax = fig.add_axes([0.90, 0.16, 0.025, 0.70])
    cbar = fig.colorbar(image_handle, cax=cax)
    cbar.set_label("Local SSIM")

    save_figure(fig, "focal_spot_ssim_maps")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# REPORTING
# =============================================================================

def write_metrics_csv(
    results: list[PairResult],
    target_fwhm_x: float,
    target_fwhm_y: float,
    target_d50: float,
    target_d80: float,
    target_supergaussian_x: SuperGaussianFit | None,
    target_supergaussian_y: SuperGaussianFit | None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "focal_spot_metrics.csv"
    fields = [
        "comparison",
        "pcc",
        "ssim",
        "nmi_0_to_1",
        "relative_nrmse",
        "mae",
        "centroid_shift_um",
        "centroid_dx_um",
        "centroid_dy_um",
        "lineout_residual_std_x",
        "lineout_residual_std_y",
        "lineout_rmse_x",
        "lineout_rmse_y",
        "target_fwhm_x_um",
        "target_fwhm_y_um",
        "test_fwhm_x_um",
        "test_fwhm_y_um",
        "fwhm_x_difference_um",
        "fwhm_y_difference_um",
        "target_d50_um",
        "target_d80_um",
        "test_d50_um",
        "test_d80_um",
        "d50_difference_um",
        "d80_difference_um",
        "spectral_log_pcc_x",
        "spectral_log_pcc_y",
        "target_supergaussian_order_x",
        "target_supergaussian_order_y",
        "test_supergaussian_order_x",
        "test_supergaussian_order_y",
        "delta_supergaussian_order_x",
        "delta_supergaussian_order_y",
        "target_supergaussian_width_1e_x_um",
        "target_supergaussian_width_1e_y_um",
        "test_supergaussian_width_1e_x_um",
        "test_supergaussian_width_1e_y_um",
        "target_supergaussian_fit_fwhm_x_um",
        "target_supergaussian_fit_fwhm_y_um",
        "test_supergaussian_fit_fwhm_x_um",
        "test_supergaussian_fit_fwhm_y_um",
        "target_supergaussian_r_squared_x",
        "target_supergaussian_r_squared_y",
        "test_supergaussian_r_squared_x",
        "test_supergaussian_r_squared_y",
        "applied_registration_dx_um",
        "applied_registration_dy_um",
    ]
    def fit_value(fit: SuperGaussianFit | None, name: str) -> float:
        if fit is None or not fit.success:
            return float("nan")
        return float(getattr(fit, name))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "comparison": f"{r.label} vs {TARGET.label}",
                "pcc": r.pcc,
                "ssim": r.ssim,
                "nmi_0_to_1": r.nmi,
                "relative_nrmse": r.nrmse,
                "mae": r.mae,
                "centroid_shift_um": r.centroid_shift_um,
                "centroid_dx_um": r.centroid_dx_um,
                "centroid_dy_um": r.centroid_dy_um,
                "lineout_residual_std_x": r.lineout_std_x,
                "lineout_residual_std_y": r.lineout_std_y,
                "lineout_rmse_x": r.lineout_rmse_x,
                "lineout_rmse_y": r.lineout_rmse_y,
                "target_fwhm_x_um": target_fwhm_x,
                "target_fwhm_y_um": target_fwhm_y,
                "test_fwhm_x_um": r.fwhm_x_um,
                "test_fwhm_y_um": r.fwhm_y_um,
                "fwhm_x_difference_um": r.fwhm_x_difference_um,
                "fwhm_y_difference_um": r.fwhm_y_difference_um,
                "target_d50_um": target_d50,
                "target_d80_um": target_d80,
                "test_d50_um": r.d50_um,
                "test_d80_um": r.d80_um,
                "d50_difference_um": r.d50_difference_um,
                "d80_difference_um": r.d80_difference_um,
                "spectral_log_pcc_x": r.spectral_log_pcc_x,
                "spectral_log_pcc_y": r.spectral_log_pcc_y,
                "target_supergaussian_order_x": fit_value(target_supergaussian_x, "order"),
                "target_supergaussian_order_y": fit_value(target_supergaussian_y, "order"),
                "test_supergaussian_order_x": fit_value(r.supergaussian_x, "order"),
                "test_supergaussian_order_y": fit_value(r.supergaussian_y, "order"),
                "delta_supergaussian_order_x": fit_value(r.supergaussian_x, "order") - fit_value(target_supergaussian_x, "order"),
                "delta_supergaussian_order_y": fit_value(r.supergaussian_y, "order") - fit_value(target_supergaussian_y, "order"),
                "target_supergaussian_width_1e_x_um": fit_value(target_supergaussian_x, "width_1e_um"),
                "target_supergaussian_width_1e_y_um": fit_value(target_supergaussian_y, "width_1e_um"),
                "test_supergaussian_width_1e_x_um": fit_value(r.supergaussian_x, "width_1e_um"),
                "test_supergaussian_width_1e_y_um": fit_value(r.supergaussian_y, "width_1e_um"),
                "target_supergaussian_fit_fwhm_x_um": fit_value(target_supergaussian_x, "fwhm_um"),
                "target_supergaussian_fit_fwhm_y_um": fit_value(target_supergaussian_y, "fwhm_um"),
                "test_supergaussian_fit_fwhm_x_um": fit_value(r.supergaussian_x, "fwhm_um"),
                "test_supergaussian_fit_fwhm_y_um": fit_value(r.supergaussian_y, "fwhm_um"),
                "target_supergaussian_r_squared_x": fit_value(target_supergaussian_x, "r_squared"),
                "target_supergaussian_r_squared_y": fit_value(target_supergaussian_y, "r_squared"),
                "test_supergaussian_r_squared_x": fit_value(r.supergaussian_x, "r_squared"),
                "test_supergaussian_r_squared_y": fit_value(r.supergaussian_y, "r_squared"),
                "applied_registration_dx_um": r.applied_registration_dx_um,
                "applied_registration_dy_um": r.applied_registration_dy_um,
            })


def format_metric(value: float, decimals: int = 5) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{decimals}f}"


def write_report(
    target: SpotData,
    results: list[PairResult],
    target_centroid: tuple[float, float],
    target_fwhm_x: float,
    target_fwhm_y: float,
    target_d50: float,
    target_d80: float,
    target_supergaussian_x: SuperGaussianFit | None,
    target_supergaussian_y: SuperGaussianFit | None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "focal_spot_report.txt"
    nyquist_x = 2.0 * target.dx_um
    nyquist_y = 2.0 * target.dy_um
    xmin, xmax, ymin, ymax = target.extent_um
    span_x = xmax - xmin
    span_y = ymax - ymin

    lines: list[str] = []
    lines.append("FOCAL-SPOT COMPARISON REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Comparison rule: every test spot is compared only with TARGET.")
    lines.append("All images are independently peak-normalised to I/I_max before analysis.")
    lines.append(f"Registration before metrics: {REGISTER_TO_TARGET}")
    lines.append(f"Target extent source: {target.extent_source}")
    lines.append(
        "Target saved extent [xmin, xmax, ymin, ymax]: "
        f"[{xmin:.6g}, {xmax:.6g}, {ymin:.6g}, {ymax:.6g}] um"
    )
    lines.append(f"Target sample-centre span: {span_x:.6g} x {span_y:.6g} um")
    lines.append(f"Target sampling dx, dy: {target.dx_um:.9g}, {target.dy_um:.9g} um/pixel")
    lines.append(f"Nyquist wavelength x, y: {nyquist_x:.9g}, {nyquist_y:.9g} um")
    lines.append(
        f"Requested Fourier wavelength plot: {FFT_WAVELENGTH_MIN_UM:g}-"
        f"{FFT_WAVELENGTH_MAX_UM:g} um"
    )
    if nyquist_x > FFT_WAVELENGTH_MIN_UM:
        lines.append(
            "WARNING (x): the lower part of the Fourier band is below Nyquist; "
            f"x-structure below {nyquist_x:.6g} um is not resolved."
        )
    if nyquist_y > FFT_WAVELENGTH_MIN_UM:
        lines.append(
            "WARNING (y): the lower part of the Fourier band is below Nyquist; "
            f"y-structure below {nyquist_y:.6g} um is not resolved."
        )
    lines.append("")
    lines.append("Target geometry")
    lines.append("-" * 72)
    lines.append(f"Centroid x, y: {target_centroid[0]:.6g}, {target_centroid[1]:.6g} um")
    lines.append(f"FWHM x: {target_fwhm_x:.6g} um")
    lines.append(f"FWHM y: {target_fwhm_y:.6g} um")
    lines.append(f"D50: {target_d50:.6g} um")
    lines.append(f"D80: {target_d80:.6g} um")
    if FIT_SUPERGAUSSIAN:
        for axis_name, fit in (("x", target_supergaussian_x), ("y", target_supergaussian_y)):
            if fit is not None and fit.success:
                lines.append(
                    f"Super-Gaussian {axis_name}: n={fit.order:.6g}, "
                    f"w_1/e={fit.width_1e_um:.6g} um, fit FWHM={fit.fwhm_um:.6g} um, "
                    f"R^2={fit.r_squared:.6g}"
                )
            else:
                lines.append(f"Super-Gaussian {axis_name}: fit unavailable")
    lines.append("")

    for r in results:
        lines.append(f"{r.label} vs {target.label}")
        lines.append("-" * 72)
        lines.append(f"PCC                         = {format_metric(r.pcc)}")
        lines.append(f"SSIM                        = {format_metric(r.ssim)}")
        lines.append(f"NMI (0-1 scale)             = {format_metric(r.nmi)}")
        lines.append(f"Relative NRMSE              = {format_metric(r.nrmse)}")
        lines.append(f"MAE                         = {format_metric(r.mae)}")
        lines.append(f"Centroid shift              = {r.centroid_shift_um:.6g} um")
        lines.append(f"  dx, dy                    = {r.centroid_dx_um:.6g}, {r.centroid_dy_um:.6g} um")
        lines.append(f"Lineout residual std x      = {format_metric(r.lineout_std_x)}")
        lines.append(f"Lineout residual std y      = {format_metric(r.lineout_std_y)}")
        lines.append(f"Lineout RMSE x              = {format_metric(r.lineout_rmse_x)}")
        lines.append(f"Lineout RMSE y              = {format_metric(r.lineout_rmse_y)}")
        lines.append(f"FWHM x                      = {r.fwhm_x_um:.6g} um")
        lines.append(f"FWHM y                      = {r.fwhm_y_um:.6g} um")
        lines.append(f"Delta FWHM x                = {r.fwhm_x_difference_um:.6g} um")
        lines.append(f"Delta FWHM y                = {r.fwhm_y_difference_um:.6g} um")
        lines.append(f"D50 / D80                   = {r.d50_um:.6g} / {r.d80_um:.6g} um")
        lines.append(f"Delta D50 / D80             = {r.d50_difference_um:.6g} / {r.d80_difference_um:.6g} um")
        lines.append(f"FFT log-power PCC x         = {format_metric(r.spectral_log_pcc_x)}")
        lines.append(f"FFT log-power PCC y         = {format_metric(r.spectral_log_pcc_y)}")
        if FIT_SUPERGAUSSIAN:
            for axis_name, fit, target_fit in (
                ("x", r.supergaussian_x, target_supergaussian_x),
                ("y", r.supergaussian_y, target_supergaussian_y),
            ):
                if fit is not None and fit.success:
                    delta_n = (
                        fit.order - target_fit.order
                        if target_fit is not None and target_fit.success else float("nan")
                    )
                    lines.append(
                        f"Super-Gaussian {axis_name}: n={fit.order:.6g}, "
                        f"Delta n={delta_n:.6g}, w_1/e={fit.width_1e_um:.6g} um, "
                        f"fit FWHM={fit.fwhm_um:.6g} um, R^2={fit.r_squared:.6g}"
                    )
                else:
                    lines.append(f"Super-Gaussian {axis_name}: fit unavailable")
        if REGISTER_TO_TARGET:
            lines.append(
                "Applied registration dx, dy  = "
                f"{r.applied_registration_dx_um:.6g}, {r.applied_registration_dy_um:.6g} um"
            )
        lines.append("")

    lines.append("Metric equations and references")
    lines.append("=" * 72)
    lines.append("PCC: r = sum[(x-mx)(y-my)] / sqrt(sum[(x-mx)^2] sum[(y-my)^2])")
    lines.append("Reference: scipy.stats.pearsonr documentation.")
    lines.append("https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html")
    lines.append("")
    lines.append("SSIM: [(2 mux muy + C1)(2 sigmaxy + C2)] / [(mux^2+muy^2+C1)(sigmax^2+sigmay^2+C2)]")
    lines.append("Wang et al., IEEE TIP 13, 600-612 (2004), DOI 10.1109/TIP.2003.819861")
    lines.append("")
    lines.append("NMI_01 = [H(A)+H(B)] / H(A,B) - 1, H(X)=-sum p(x)log p(x)")
    lines.append("Studholme et al., Pattern Recognition 32, 71-86 (1999), DOI 10.1016/S0031-3203(98)00091-0")
    lines.append("The native 1-2 Studholme/scikit-image score is shifted by -1: 0=unrelated, 1=perfect dependence.")
    lines.append("")
    lines.append("Relative NRMSE: ||R-T||_2 / ||R||_2")
    lines.append("MAE: mean(|R-T|)")
    lines.append("Lineout residual std: sample standard deviation of T-R")
    lines.append("")
    lines.append("Super-Gaussian: I(x)=B+A exp[-(|x-x0|/w)^(2n)]")
    lines.append("Fit FWHM = 2 w (ln 2)^(1/(2n)); R^2 reports envelope-fit quality.")
    lines.append("")
    lines.append("DFT: X_k=sum_n x_n exp(-2*pi*i*k*n/N); P_k=|X_k|^2; Lambda=1/f")
    lines.append("NumPy FFT documentation: numpy.fft.rfft and numpy.fft.rfftfreq")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: list[PairResult], target: SpotData) -> None:
    print("\nFocal-spot comparison")
    print("=" * 78)
    print(
        f"Target: {target.label} | dx={target.dx_um:g} um/pixel, "
        f"dy={target.dy_um:g} um/pixel"
    )
    print(f"Extent [xmin, xmax, ymin, ymax] = {target.extent_um} um")
    print(
        f"Fourier Nyquist wavelength: x={2*target.dx_um:g} um, "
        f"y={2*target.dy_um:g} um"
    )
    for r in results:
        print(
            f"{r.label:>16s} vs target | "
            f"PCC={r.pcc: .4f} | SSIM={r.ssim: .4f} | NMI={r.nmi: .4f} | "
            f"NRMSE={r.nrmse: .4f} | MAE={r.mae: .4f}"
        )
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"Running focal-spot comparison: {SCRIPT_VERSION}")
    apply_publication_style()

    target = load_spot(TARGET)
    tests = [load_spot(GENERATED)]
    if SECONDARY is not None:
        tests.append(load_spot(SECONDARY))

    target_centroid = weighted_centroid(target.intensity, target.x_um, target.y_um)
    row = nearest_index(target.y_um, target_centroid[1])
    col = nearest_index(target.x_um, target_centroid[0])

    mask = build_analysis_mask(target, target_centroid, ANALYSIS_HALF_WIDTH_UM)
    target_line_x, target_line_y = averaged_lineouts(
        target.intensity, row, col, LINEOUT_AVERAGE_WIDTH_PX
    )
    target_supergaussian_x = fit_supergaussian_lineout(
        target_line_x, target.x_um, target_centroid[0]
    )
    target_supergaussian_y = fit_supergaussian_lineout(
        target_line_y, target.y_um, target_centroid[1]
    )
    target_fwhm_x = fwhm_um(target_line_x, target.x_um)
    target_fwhm_y = fwhm_um(target_line_y, target.y_um)
    target_d50 = enclosed_energy_diameter_um(
        target.intensity, target.x_um, target.y_um, 0.50, centre=target_centroid
    )
    target_d80 = enclosed_energy_diameter_um(
        target.intensity, target.x_um, target.y_um, 0.80, centre=target_centroid
    )

    results: list[PairResult] = []
    for spot in tests:
        test_grid = resample_to_target_grid(spot, target)
        results.append(compare_pair(
            target=target,
            test_label=spot.label,
            test_on_target_grid=test_grid,
            mask=mask,
            target_centroid=target_centroid,
            row=row,
            col=col,
            target_line_x=target_line_x,
            target_line_y=target_line_y,
            target_supergaussian_x=target_supergaussian_x,
            target_supergaussian_y=target_supergaussian_y,
            target_fwhm_x=target_fwhm_x,
            target_fwhm_y=target_fwhm_y,
            target_d50=target_d50,
            target_d80=target_d80,
        ))

    plot_spots_and_lineouts(
        target, results, target_centroid, row, col,
        target_supergaussian_x, target_supergaussian_y,
    )
    plot_power_spectra(target, results, row, col)
    plot_ssim_maps(target, results, target_centroid)
    write_metrics_csv(
        results, target_fwhm_x, target_fwhm_y, target_d50, target_d80,
        target_supergaussian_x, target_supergaussian_y,
    )
    write_report(
        target, results, target_centroid,
        target_fwhm_x, target_fwhm_y, target_d50, target_d80,
        target_supergaussian_x, target_supergaussian_y,
    )
    print_summary(results, target)


if __name__ == "__main__":
    main()
