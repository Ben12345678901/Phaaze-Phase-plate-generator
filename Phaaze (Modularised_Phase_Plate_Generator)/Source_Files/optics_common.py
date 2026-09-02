"""Shared numerical conventions for the GSI/SCITECH phase-plate comparison.

The generator and the testbed both import this module so that aperture geometry,
plate-plane sampling and Fraunhofer coordinates cannot silently drift apart.
Only NumPy is required; plotting remains in the calling scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


PROPAGATION_MODEL_ALIASES = {
    "fraunhofer": "fraunhofer",
    "franhouffer": "fraunhofer",
    "franhoufer": "fraunhofer",
    "fraunhoffer": "fraunhofer",
    "fresnel": "fresnel",
    "fresenel": "fresnel",
}


def normalise_propagation_model(model: str) -> str:
    """Return ``fraunhofer`` or ``fresnel``, accepting common misspellings."""
    key = str(model).strip().lower()
    try:
        return PROPAGATION_MODEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "propagation model must be 'fraunhofer' or 'fresnel'"
        ) from exc


def centered_axis(samples: int, pixel_size_m: float) -> np.ndarray:
    """Return the FFT-centred coordinate of every sample, in metres."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    return (np.arange(samples) - samples // 2) * pixel_size_m


def make_pupil(X: np.ndarray, Y: np.ndarray, plate_size_m: float,
               shape: str = "square") -> np.ndarray:
    """Return a boolean clear-aperture mask.

    ``plate_size_m`` is the diameter for a circle and the side length for a
    square.  GSI uses the square option for its 11 cm by 11 cm beam.
    """
    shape = shape.lower()
    half = plate_size_m / 2.0
    if shape == "circle":
        return X**2 + Y**2 <= half**2
    if shape == "square":
        return (np.abs(X) <= half) & (np.abs(Y) <= half)
    raise ValueError("shape must be 'circle' or 'square'")


def build_input_amplitude(
    samples: int,
    plate_size_m: float,
    *,
    fill_factor: float = 0.90,
    aperture_shape: str = "square",
    super_gaussian_order: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the input field amplitude used by both generator and testbed.

    The default is the Gaussian amplitude already used by ``TestBed.py``.  A
    super-Gaussian can be requested without changing the aperture definition.
    """
    if not 0 < fill_factor <= 1:
        raise ValueError("fill_factor must lie in (0, 1]")

    dx_m = plate_size_m / samples
    x_m = centered_axis(samples, dx_m)
    X, Y = np.meshgrid(x_m, x_m, indexing="xy")
    w0_m = fill_factor * plate_size_m / 2.0

    if super_gaussian_order is None:
        amplitude = np.exp(-(X**2 + Y**2) / w0_m**2)
    else:
        order = float(super_gaussian_order)
        if order <= 0:
            raise ValueError("super_gaussian_order must be positive")
        if aperture_shape.lower() == "square":
            exponent = ((np.abs(X) / w0_m) ** (2 * order)
                        + (np.abs(Y) / w0_m) ** (2 * order))
        else:
            exponent = (np.hypot(X, Y) / w0_m) ** (2 * order)
        amplitude = np.exp(-exponent)

    pupil = make_pupil(X, Y, plate_size_m, aperture_shape)
    return amplitude * pupil, pupil, dx_m


def pad_center(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to ``target_shape``."""
    if array.ndim != 2:
        raise ValueError(f"array must be 2-D, got {array.shape}")
    target_h, target_w = target_shape
    source_h, source_w = array.shape

    y0 = max(0, (source_h - target_h) // 2)
    x0 = max(0, (source_w - target_w) // 2)
    cropped = array[y0:y0 + min(source_h, target_h),
                    x0:x0 + min(source_w, target_w)]

    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]
    top = pad_h // 2
    left = pad_w // 2
    return np.pad(cropped, ((top, pad_h - top), (left, pad_w - left)))


def resample_phase_to_shape(phase_rad: np.ndarray,
                            target_shape: tuple[int, int]) -> np.ndarray:
    """Resize wrapped phase without interpolating across the 0/2-pi seam."""
    phase = np.asarray(phase_rad, dtype=float)
    wrapped_levels = np.unique(np.round(np.mod(phase, 2 * np.pi), decimals=10))
    if wrapped_levels.size <= 16:
        # Preserve discrete manufactured phase levels and sharp cell edges.
        y = np.rint(np.linspace(0, phase.shape[0] - 1, target_shape[0])).astype(int)
        x = np.rint(np.linspace(0, phase.shape[1] - 1, target_shape[1])).astype(int)
        return np.angle(np.exp(1j * phase[np.ix_(y, x)]))

    field = np.exp(1j * phase)
    real = bilinear_resample(field.real, target_shape)
    imag = bilinear_resample(field.imag, target_shape)
    return np.angle(real + 1j * imag)


def round_binary_phase_boundaries(
    phase_rad: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """Round raster-cell corners while preserving a binary plate's fill fraction.

    A periodic Gaussian convolution is applied to the 0/pi mask and the result
    is thresholded at a quantile which preserves the original pi-area fraction.
    This changes boundary geometry only; it does not introduce new phase levels.
    The final plate must still be re-propagated after this manufacturing step.
    """
    phase = np.asarray(phase_rad, dtype=float)
    if phase.ndim != 2:
        raise ValueError(f"phase_rad must be 2-D, got {phase.shape}")
    if sigma_px <= 0:
        return np.angle(np.exp(1j * phase))

    wrapped = np.mod(phase, 2 * np.pi)
    distance_to_zero = np.minimum(wrapped, 2 * np.pi - wrapped)
    distance_to_pi = np.abs(wrapped - np.pi)
    pi_mask = distance_to_pi < distance_to_zero
    pi_fraction = float(pi_mask.mean())
    if pi_fraction <= 0 or pi_fraction >= 1:
        return pi_mask.astype(float) * np.pi

    fy = np.fft.fftfreq(phase.shape[0])[:, None]
    fx = np.fft.fftfreq(phase.shape[1])[None, :]
    transfer = np.exp(-2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2))
    smoothed = np.fft.ifft2(np.fft.fft2(pi_mask.astype(float)) * transfer).real
    threshold = np.quantile(smoothed, 1.0 - pi_fraction)
    rounded_mask = smoothed >= threshold
    return rounded_mask.astype(float) * np.pi


def bilinear_resample(array: np.ndarray,
                      target_shape: tuple[int, int]) -> np.ndarray:
    """Dependency-free bilinear resize for a two-dimensional array."""
    source = np.asarray(array)
    if source.ndim != 2:
        raise ValueError(f"array must be 2-D, got {source.shape}")
    target_h, target_w = target_shape
    if source.shape == target_shape:
        return source.copy()

    y = np.linspace(0, source.shape[0] - 1, target_h)
    x = np.linspace(0, source.shape[1] - 1, target_w)
    X, Y = np.meshgrid(x, y, indexing="xy")
    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = X - x0
    wy = Y - y0

    return ((1 - wx) * (1 - wy) * source[y0, x0]
            + wx * (1 - wy) * source[y0, x1]
            + (1 - wx) * wy * source[y1, x0]
            + wx * wy * source[y1, x1])


def resample_intensity_to_grid(
    intensity: np.ndarray,
    target_shape: tuple[int, int],
    source_pixel_size_m: float,
    target_pixel_size_m: float,
) -> np.ndarray:
    """Resample a centred focal intensity onto another physical grid.

    Zero is used beyond the source field of view.  When the pitches agree this
    reduces to an exact centre crop/pad, avoiding unnecessary interpolation of
    the SCITECH speckle pattern.
    """
    source = np.asarray(intensity, dtype=float)
    if source.ndim != 2:
        raise ValueError(f"intensity must be 2-D, got {source.shape}")
    if np.isclose(source_pixel_size_m, target_pixel_size_m, rtol=1e-10, atol=0):
        return pad_center(source, target_shape)

    target_y = centered_axis(target_shape[0], target_pixel_size_m)
    target_x = centered_axis(target_shape[1], target_pixel_size_m)
    source_y_index = target_y / source_pixel_size_m + source.shape[0] // 2
    source_x_index = target_x / source_pixel_size_m + source.shape[1] // 2
    X, Y = np.meshgrid(source_x_index, source_y_index, indexing="xy")

    valid = ((X >= 0) & (X <= source.shape[1] - 1)
             & (Y >= 0) & (Y <= source.shape[0] - 1))
    x0 = np.clip(np.floor(X).astype(int), 0, source.shape[1] - 1)
    y0 = np.clip(np.floor(Y).astype(int), 0, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = X - x0
    wy = Y - y0
    result = ((1 - wx) * (1 - wy) * source[y0, x0]
              + wx * (1 - wy) * source[y0, x1]
              + (1 - wx) * wy * source[y1, x0]
              + wx * wy * source[y1, x1])
    return np.where(valid, result, 0.0)


def fraunhofer_focal_spot(
    plate_field: np.ndarray,
    *,
    dx_plate_m: float,
    focal_length_m: float,
    wavelength_m: float,
    pad_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a sampled plate field to the focal plane with a centred FFT."""
    field = np.asarray(plate_field)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"plate_field must be square and 2-D, got {field.shape}")
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1")

    target_n = max(field.shape[0], int(round(pad_factor * field.shape[0])))
    padded = pad_center(field, (target_n, target_n))
    focal_field = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(padded))
    ) * dx_plate_m**2 / (1j * wavelength_m * focal_length_m)

    frequency = np.fft.fftshift(np.fft.fftfreq(target_n, d=dx_plate_m))
    focal_axis_m = wavelength_m * focal_length_m * frequency
    return (np.abs(focal_field) ** 2,
            focal_axis_m[None, :], focal_axis_m[:, None], focal_field)


def propagation_axis(
    samples: int,
    dx_input_m: float,
    wavelength_m: float,
    distance_m: float,
) -> np.ndarray:
    """Return the one-FFT Fresnel output coordinate axis, in metres."""
    if min(samples, dx_input_m, wavelength_m, distance_m) <= 0:
        raise ValueError("Fresnel sampling parameters must be positive")
    spatial_frequency_m_inv = np.fft.fftshift(
        np.fft.fftfreq(samples, d=dx_input_m)
    )
    return wavelength_m * distance_m * spatial_frequency_m_inv


def _centered_fft2(field: np.ndarray) -> np.ndarray:
    """Return a centred two-dimensional Fourier transform."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def _fresnel_phase_factors(
    shape: tuple[int, int],
    dx_input_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    focal_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the input, thin-lens and output quadratic phase factors."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Fresnel propagation needs a square field, got {shape}")
    if min(
        dx_input_m,
        wavelength_m,
        propagation_distance_m,
        focal_length_m,
    ) <= 0:
        raise ValueError("Fresnel sampling and optical distances must be positive")

    samples = shape[0]
    wave_number_m_inv = 2 * np.pi / wavelength_m
    input_axis_m = centered_axis(samples, dx_input_m)
    input_radius_squared_m2 = (
        input_axis_m[None, :] ** 2 + input_axis_m[:, None] ** 2
    )
    input_quadratic = np.exp(
        1j
        * wave_number_m_inv
        * input_radius_squared_m2
        / (2 * propagation_distance_m)
    )
    focusing_lens = np.exp(
        -1j
        * wave_number_m_inv
        * input_radius_squared_m2
        / (2 * focal_length_m)
    )

    output_axis_m = propagation_axis(
        samples, dx_input_m, wavelength_m, propagation_distance_m
    )
    output_radius_squared_m2 = (
        output_axis_m[None, :] ** 2 + output_axis_m[:, None] ** 2
    )
    output_quadratic = np.exp(
        1j
        * wave_number_m_inv
        * output_radius_squared_m2
        / (2 * propagation_distance_m)
    )
    return input_quadratic, focusing_lens, output_quadratic, output_axis_m


def propagate_phase_plate(
    plate_field: np.ndarray,
    dx_plate_m: float,
    wavelength_m: float,
    focal_length_m: float,
    propagation_model: str = "fraunhofer",
    propagation_distance_m: float | None = None,
    pad_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a phase-plate field with Fraunhofer or thin-lens Fresnel optics.

    Fraunhofer mode evaluates the back focal plane of a lens. Fresnel mode
    applies that same thin lens and propagates a distance
    ``propagation_distance_m`` using the one-FFT Fresnel integral. At
    ``propagation_distance_m == focal_length_m`` both models produce the same
    intensity, apart from floating-point round-off and an overall phase.
    """
    field = np.asarray(plate_field)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"plate_field must be square and 2-D, got {field.shape}")
    if min(dx_plate_m, wavelength_m, focal_length_m) <= 0:
        raise ValueError("sampling, wavelength and focal length must be positive")
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1")

    model = normalise_propagation_model(propagation_model)
    if model == "fraunhofer":
        return fraunhofer_focal_spot(
            field,
            dx_plate_m=dx_plate_m,
            focal_length_m=focal_length_m,
            wavelength_m=wavelength_m,
            pad_factor=pad_factor,
        )

    distance_m = (
        focal_length_m
        if propagation_distance_m is None
        else float(propagation_distance_m)
    )
    if distance_m <= 0:
        raise ValueError("propagation_distance_m must be positive")

    target_samples = max(
        field.shape[0], int(round(pad_factor * field.shape[0]))
    )
    padded_field = pad_center(field, (target_samples, target_samples))
    input_phase, lens_phase, output_phase, output_axis_m = (
        _fresnel_phase_factors(
            padded_field.shape,
            dx_plate_m,
            wavelength_m,
            distance_m,
            focal_length_m,
        )
    )
    spectrum = (
        _centered_fft2(padded_field * lens_phase * input_phase)
        * dx_plate_m**2
    )
    wave_number_m_inv = 2 * np.pi / wavelength_m
    prefactor = (
        np.exp(1j * wave_number_m_inv * distance_m)
        / (1j * wavelength_m * distance_m)
    )
    output_field = prefactor * output_phase * spectrum
    return (
        np.abs(output_field) ** 2,
        output_axis_m[None, :],
        output_axis_m[:, None],
        output_field,
    )


def load_2d_array(path: str | Path) -> np.ndarray:
    """Load and validate a finite two-dimensional NumPy array."""
    array = np.asarray(np.load(Path(path)), dtype=float).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array at {path}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Array at {path} contains NaN or infinite values")
    return array


def normalise_intensity(intensity: np.ndarray) -> np.ndarray:
    """Peak-normalise a non-negative intensity map."""
    array = np.clip(np.asarray(intensity, dtype=float), 0, None)
    peak = array.max(initial=0.0)
    return array / peak if peak > 0 else array


def focal_metrics(test: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return scale-preserving PCC and relative RMS error."""
    test_n = normalise_intensity(test)
    ref_n = normalise_intensity(reference)
    if test_n.shape != ref_n.shape:
        raise ValueError(f"Shape mismatch: {test_n.shape} versus {ref_n.shape}")

    x = test_n.ravel() - test_n.mean()
    y = ref_n.ravel() - ref_n.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    pcc = float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")
    ref_norm = np.linalg.norm(ref_n)
    relative_nrmse = (float(np.linalg.norm(test_n - ref_n) / ref_norm)
                      if ref_norm > 0 else float("inf"))
    return {"pcc": pcc, "relative_nrmse": relative_nrmse}


def enclosed_energy_diameter(
    intensity: np.ndarray,
    Xf_m: np.ndarray,
    Yf_m: np.ndarray,
    fraction: float,
) -> float:
    """Return the centred circular diameter containing an energy fraction."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0, 1)")
    image = np.clip(np.asarray(intensity, dtype=float), 0, None)
    radius = np.hypot(np.asarray(Xf_m), np.asarray(Yf_m))
    radius = np.broadcast_to(radius, image.shape)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(image.ravel()[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return float("nan")
    index = np.searchsorted(cumulative, fraction * cumulative[-1])
    return float(2.0 * radius.ravel()[order[index]])
