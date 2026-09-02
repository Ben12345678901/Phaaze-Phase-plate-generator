"""Operations on real, unwrapped continuous-relief phase surfaces."""

from __future__ import annotations

import numpy as np


def gaussian_transfer(
    shape: tuple[int, int], sigma_px: float
) -> np.ndarray:
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    return np.exp(
        -2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2)
    )


def correlation_fwhm_to_sigma_px(
    correlation_length_m: float, pixel_size_m: float
) -> float:
    """Convert surface autocorrelation FWHM to Gaussian kernel sigma."""
    if correlation_length_m <= 0 or pixel_size_m <= 0:
        raise ValueError("correlation length and pixel size must be positive")
    return (
        correlation_length_m
        / (4 * np.sqrt(np.log(2)) * pixel_size_m)
    )


def filter_surface(
    surface: np.ndarray,
    transfer: np.ndarray,
    valid_mask: np.ndarray,
    normalization_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized Gaussian convolution inside a clear aperture."""
    values = np.asarray(surface, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.shape != transfer.shape or values.shape != mask.shape:
        raise ValueError("surface, transfer, and mask shapes must match")
    numerator = np.fft.ifft2(
        np.fft.fft2(np.where(mask, values, 0.0)) * transfer
    ).real
    if normalization_weights is None:
        normalization_weights = np.fft.ifft2(
            np.fft.fft2(mask.astype(float)) * transfer
        ).real
    filtered = numerator / np.maximum(normalization_weights, 1e-12)
    filtered -= np.min(filtered[mask])
    return np.where(mask, filtered, 0.0)


def lift_near_surface(
    wrapped_phase_rad: np.ndarray, reference_surface_rad: np.ndarray
) -> np.ndarray:
    """Select the 2π branch nearest the previous real surface."""
    principal_difference = np.angle(
        np.exp(1j * (wrapped_phase_rad - reference_surface_rad))
    )
    return reference_surface_rad + principal_difference


def unwrap_surface_2d(
    wrapped_phase_rad: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Least-squares isotropic unwrap using a Fourier Poisson solve."""
    phase = np.asarray(wrapped_phase_rad, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if phase.ndim != 2 or phase.shape != mask.shape or not mask.any():
        raise ValueError("wrapped phase and a nonempty mask must match")

    gradient_x = np.zeros_like(phase)
    gradient_y = np.zeros_like(phase)
    x_valid = mask[:, 1:] & mask[:, :-1]
    y_valid = mask[1:, :] & mask[:-1, :]
    wrapped_dx = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    wrapped_dy = np.angle(np.exp(1j * np.diff(phase, axis=0)))
    gradient_x[:, :-1][x_valid] = wrapped_dx[x_valid]
    gradient_y[:-1, :][y_valid] = wrapped_dy[y_valid]
    divergence = (
        gradient_x
        - np.roll(gradient_x, 1, axis=1)
        + gradient_y
        - np.roll(gradient_y, 1, axis=0)
    )
    ny, nx = phase.shape
    ky = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    laplacian = 2 * np.cos(kx) + 2 * np.cos(ky) - 4
    laplacian[0, 0] = 1
    spectrum = np.fft.fft2(divergence) / laplacian
    spectrum[0, 0] = 0
    unwrapped = np.fft.ifft2(spectrum).real
    unwrapped -= np.min(unwrapped[mask])
    return np.where(mask, unwrapped, 0.0)


def normalize_surface_rms(
    surface_rad: np.ndarray,
    valid_mask: np.ndarray,
    required_rms_rad: float,
) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    centered = surface_rad[mask] - np.mean(surface_rad[mask])
    rms = float(np.sqrt(np.mean(centered**2)))
    if rms <= np.finfo(float).eps:
        raise RuntimeError("continuous surface has zero RMS")
    normalized = np.where(
        mask,
        (
            surface_rad - np.mean(surface_rad[mask])
        )
        * (required_rms_rad / rms),
        0.0,
    )
    return np.where(mask, normalized - np.min(normalized[mask]), 0.0)


def phase_to_thickness_m(
    unwrapped_phase_rad: np.ndarray,
    wavelength_m: float,
    refractive_index: float,
) -> np.ndarray:
    if wavelength_m <= 0 or refractive_index <= 1:
        raise ValueError(
            "wavelength must be positive and refractive index greater than 1"
        )
    return (
        np.asarray(unwrapped_phase_rad)
        * wavelength_m
        / (2 * np.pi * (refractive_index - 1))
    )


def estimate_correlation_fwhm_m(
    surface_rad: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size_m: float,
) -> float:
    """Estimate the mean axial autocorrelation FWHM."""
    surface = np.asarray(surface_rad, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    centered = np.where(
        mask, surface - np.mean(surface[mask]), 0.0
    )
    spectrum = np.fft.fft2(centered)
    mask_spectrum = np.fft.fft2(mask.astype(float))
    covariance = np.fft.ifft2(spectrum * np.conj(spectrum)).real
    pairs = np.fft.ifft2(
        mask_spectrum * np.conj(mask_spectrum)
    ).real
    covariance /= np.maximum(pairs, 1)
    if covariance[0, 0] <= 0:
        return float("nan")
    maximum_lag = max(1, min(surface.shape) // 4)
    lags = np.arange(maximum_lag + 1)
    profile = np.mean(
        np.stack(
            [
                covariance[0, lags],
                covariance[0, (-lags) % surface.shape[1]],
                covariance[lags, 0],
                covariance[(-lags) % surface.shape[0], 0],
            ]
        ),
        axis=0,
    )
    profile /= profile[0]
    below = np.flatnonzero(profile <= 0.5)
    if below.size == 0:
        return float("nan")
    upper = int(below[0])
    if upper == 0:
        half_width = 0.0
    else:
        lower = upper - 1
        denominator = profile[lower] - profile[upper]
        fraction = (
            (profile[lower] - 0.5) / denominator
            if denominator > 0
            else 0.0
        )
        half_width = lower + fraction
    return float(2 * half_width * pixel_size_m)
