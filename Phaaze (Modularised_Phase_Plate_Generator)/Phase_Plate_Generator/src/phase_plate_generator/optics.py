"""Shared grids and scalar diffraction operators."""

from __future__ import annotations

import numpy as np


PROPAGATION_ALIASES = {
    "fraunhofer": "fraunhofer",
    "franhouffer": "fraunhofer",
    "franhoufer": "fraunhofer",
    "fraunhoffer": "fraunhofer",
    "fresnel": "fresnel",
    "fresenel": "fresnel",
}


def normalise_propagation_model(model: str) -> str:
    try:
        return PROPAGATION_ALIASES[str(model).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "propagation model must be 'fraunhofer' or 'fresnel'"
        ) from exc


def centered_axis(samples: int, pixel_size_m: float) -> np.ndarray:
    if samples < 1 or pixel_size_m <= 0:
        raise ValueError("samples and pixel_size_m must be positive")
    return (np.arange(samples) - samples // 2) * pixel_size_m


def make_pupil(
    X_m: np.ndarray,
    Y_m: np.ndarray,
    plate_size_m: float,
    shape: str,
) -> np.ndarray:
    half = plate_size_m / 2
    if shape == "circle":
        return X_m**2 + Y_m**2 <= half**2
    if shape == "square":
        return (np.abs(X_m) <= half) & (np.abs(Y_m) <= half)
    raise ValueError("aperture shape must be 'circle' or 'square'")


def pad_center(
    array: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(array)
    if source.ndim != 2:
        raise ValueError(f"array must be 2-D, got {source.shape}")
    target_h, target_w = target_shape
    source_h, source_w = source.shape
    y0 = max(0, (source_h - target_h) // 2)
    x0 = max(0, (source_w - target_w) // 2)
    cropped = source[
        y0 : y0 + min(source_h, target_h),
        x0 : x0 + min(source_w, target_w),
    ]
    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]
    top = pad_h // 2
    left = pad_w // 2
    return np.pad(cropped, ((top, pad_h - top), (left, pad_w - left)))


def bilinear_resample(
    array: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Dependency-free bilinear resize for a real or complex 2-D array."""
    source = np.asarray(array)
    if source.ndim != 2:
        raise ValueError(f"array must be 2-D, got {source.shape}")
    if source.shape == target_shape:
        return source.copy()
    target_y = np.linspace(0, source.shape[0] - 1, target_shape[0])
    target_x = np.linspace(0, source.shape[1] - 1, target_shape[1])
    X, Y = np.meshgrid(target_x, target_y, indexing="xy")
    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = X - x0
    wy = Y - y0
    return (
        (1 - wx) * (1 - wy) * source[y0, x0]
        + wx * (1 - wy) * source[y0, x1]
        + (1 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )


def resample_phase_to_shape(
    phase_rad: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Resize phase by interpolating its phasor across the 0/2π seam."""
    phase = np.asarray(phase_rad, dtype=float)
    levels = np.unique(np.round(np.mod(phase, 2 * np.pi), 10))
    if levels.size <= 16:
        y = np.rint(
            np.linspace(0, phase.shape[0] - 1, target_shape[0])
        ).astype(int)
        x = np.rint(
            np.linspace(0, phase.shape[1] - 1, target_shape[1])
        ).astype(int)
        return np.angle(np.exp(1j * phase[np.ix_(y, x)]))
    phasor = np.exp(1j * phase)
    return np.angle(
        bilinear_resample(phasor.real, target_shape)
        + 1j * bilinear_resample(phasor.imag, target_shape)
    )


def resample_intensity_to_grid(
    intensity: np.ndarray,
    target_shape: tuple[int, int],
    source_pixel_size_m: float,
    target_pixel_size_m: float,
) -> np.ndarray:
    """Resample a centered physical intensity map onto a target grid."""
    source = np.asarray(intensity, dtype=float)
    if source.ndim != 2:
        raise ValueError(f"intensity must be 2-D, got {source.shape}")
    if np.isclose(source_pixel_size_m, target_pixel_size_m):
        return pad_center(source, target_shape)
    target_y = centered_axis(target_shape[0], target_pixel_size_m)
    target_x = centered_axis(target_shape[1], target_pixel_size_m)
    source_y = target_y / source_pixel_size_m + source.shape[0] // 2
    source_x = target_x / source_pixel_size_m + source.shape[1] // 2
    X, Y = np.meshgrid(source_x, source_y, indexing="xy")
    valid = (
        (X >= 0)
        & (X <= source.shape[1] - 1)
        & (Y >= 0)
        & (Y <= source.shape[0] - 1)
    )
    x0 = np.clip(np.floor(X).astype(int), 0, source.shape[1] - 1)
    y0 = np.clip(np.floor(Y).astype(int), 0, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = X - x0
    wy = Y - y0
    result = (
        (1 - wx) * (1 - wy) * source[y0, x0]
        + wx * (1 - wy) * source[y0, x1]
        + (1 - wx) * wy * source[y1, x0]
        + wx * wy * source[y1, x1]
    )
    return np.where(valid, result, 0.0)


def _centered_fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def _centered_ifft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def propagation_axis(
    samples: int,
    dx_input_m: float,
    wavelength_m: float,
    distance_m: float,
) -> np.ndarray:
    frequency = np.fft.fftshift(np.fft.fftfreq(samples, d=dx_input_m))
    return wavelength_m * distance_m * frequency


def _fresnel_phase_factors(
    shape: tuple[int, int],
    dx_input_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    focal_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Fresnel propagation needs a square field, got {shape}")
    if min(
        dx_input_m,
        wavelength_m,
        propagation_distance_m,
        focal_length_m,
    ) <= 0:
        raise ValueError("Fresnel sampling and distances must be positive")

    samples = shape[0]
    wave_number = 2 * np.pi / wavelength_m
    input_axis = centered_axis(samples, dx_input_m)
    input_radius_squared = (
        input_axis[None, :] ** 2 + input_axis[:, None] ** 2
    )
    input_quadratic = np.exp(
        1j
        * wave_number
        * input_radius_squared
        / (2 * propagation_distance_m)
    )
    focusing_lens = np.exp(
        -1j * wave_number * input_radius_squared / (2 * focal_length_m)
    )
    output_axis = propagation_axis(
        samples, dx_input_m, wavelength_m, propagation_distance_m
    )
    output_radius_squared = (
        output_axis[None, :] ** 2 + output_axis[:, None] ** 2
    )
    output_quadratic = np.exp(
        1j
        * wave_number
        * output_radius_squared
        / (2 * propagation_distance_m)
    )
    return input_quadratic, focusing_lens, output_quadratic, output_axis


def forward_for_gs(
    field: np.ndarray,
    dx_input_m: float,
    wavelength_m: float,
    focal_length_m: float,
    propagation_model: str = "fraunhofer",
    propagation_distance_m: float | None = None,
) -> np.ndarray:
    """Forward operator without irrelevant scalar prefactors."""
    model = normalise_propagation_model(propagation_model)
    distance = (
        focal_length_m
        if propagation_distance_m is None
        else propagation_distance_m
    )
    if model == "fraunhofer":
        return _centered_fft2(field)
    input_phase, lens_phase, output_phase, _ = _fresnel_phase_factors(
        field.shape,
        dx_input_m,
        wavelength_m,
        distance,
        focal_length_m,
    )
    return output_phase * _centered_fft2(
        field * lens_phase * input_phase
    )


def inverse_for_gs(
    output_field: np.ndarray,
    dx_input_m: float,
    wavelength_m: float,
    focal_length_m: float,
    propagation_model: str = "fraunhofer",
    propagation_distance_m: float | None = None,
) -> np.ndarray:
    """Exact discrete inverse of :func:`forward_for_gs`."""
    model = normalise_propagation_model(propagation_model)
    distance = (
        focal_length_m
        if propagation_distance_m is None
        else propagation_distance_m
    )
    if model == "fraunhofer":
        return _centered_ifft2(output_field)
    input_phase, lens_phase, output_phase, _ = _fresnel_phase_factors(
        output_field.shape,
        dx_input_m,
        wavelength_m,
        distance,
        focal_length_m,
    )
    weighted_input = _centered_ifft2(
        output_field * np.conj(output_phase)
    )
    return weighted_input * np.conj(lens_phase * input_phase)


def propagate_phase_plate(
    plate_field: np.ndarray,
    dx_plate_m: float,
    wavelength_m: float,
    focal_length_m: float,
    propagation_model: str = "fraunhofer",
    propagation_distance_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a plate field and return intensity, grids, and complex field."""
    field = np.asarray(plate_field)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(
            f"plate_field must be square and 2-D, got {field.shape}"
        )
    model = normalise_propagation_model(propagation_model)
    distance = (
        focal_length_m
        if propagation_distance_m is None
        else propagation_distance_m
    )
    spectrum = forward_for_gs(
        field,
        dx_plate_m,
        wavelength_m,
        focal_length_m,
        model,
        distance,
    )
    output_axis = propagation_axis(
        field.shape[0], dx_plate_m, wavelength_m, distance
    )
    if model == "fraunhofer":
        output_field = (
            spectrum
            * dx_plate_m**2
            / (1j * wavelength_m * focal_length_m)
        )
    else:
        # forward_for_gs already contains the output quadratic phase.
        wave_number = 2 * np.pi / wavelength_m
        output_field = (
            np.exp(1j * wave_number * distance)
            * spectrum
            * dx_plate_m**2
            / (1j * wavelength_m * distance)
        )
    X_m, Y_m = np.meshgrid(output_axis, output_axis, indexing="xy")
    return np.abs(output_field) ** 2, X_m, Y_m, output_field


def _has_fft_friendly_size(samples: int) -> bool:
    remaining = int(samples)
    for factor in (2, 3, 5):
        while remaining > 1 and remaining % factor == 0:
            remaining //= factor
    return remaining == 1


def next_fft_friendly_size(minimum_samples: int, parity: int = 0) -> int:
    samples = max(1, int(np.ceil(minimum_samples)))
    parity %= 2
    if samples % 2 != parity:
        samples += 1
    while not _has_fft_friendly_size(samples):
        samples += 2
    return samples


def resolve_propagation_samples(
    plate_samples: int,
    plate_size_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    *,
    pad_factor: float = 1.0,
    output_pixel_size_m: float | None = None,
    maximum_samples: int = 4096,
) -> dict[str, float | int]:
    """Choose a symmetric FFT grid and report the achieved focal pitch."""
    dx_plate_m = plate_size_m / plate_samples
    minimum = max(plate_samples, int(round(pad_factor * plate_samples)))
    if output_pixel_size_m is not None:
        minimum = max(
            minimum,
            int(
                np.ceil(
                    wavelength_m
                    * propagation_distance_m
                    / (dx_plate_m * output_pixel_size_m)
                )
            ),
        )
    samples = next_fft_friendly_size(
        minimum, parity=plate_samples % 2
    )
    if samples > maximum_samples:
        raise ValueError(
            f"Requested sampling needs {samples} propagation samples, "
            f"above maximum {maximum_samples}"
        )
    padding = (samples - plate_samples) // 2
    achieved_pitch = (
        wavelength_m
        * propagation_distance_m
        / (samples * dx_plate_m)
    )
    return {
        "plate_samples": plate_samples,
        "propagation_samples": samples,
        "padding_per_side": padding,
        "plate_pixel_size_m": dx_plate_m,
        "output_pixel_size_m": achieved_pitch,
        "output_extent_m": samples * achieved_pitch,
    }


def normalise_intensity(intensity: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(intensity, dtype=float), 0, None)
    peak = result.max(initial=0.0)
    return result / peak if peak > 0 else result


def focal_metrics(
    test: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    test_n = normalise_intensity(test)
    reference_n = normalise_intensity(reference)
    if test_n.shape != reference_n.shape:
        raise ValueError(
            f"Shape mismatch: {test_n.shape} versus {reference_n.shape}"
        )
    x = test_n.ravel() - test_n.mean()
    y = reference_n.ravel() - reference_n.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    pcc = float(np.dot(x, y) / denominator) if denominator else float("nan")
    reference_norm = np.linalg.norm(reference_n)
    nrmse = (
        float(np.linalg.norm(test_n - reference_n) / reference_norm)
        if reference_norm
        else float("inf")
    )
    return {"pcc": pcc, "relative_nrmse": nrmse}
