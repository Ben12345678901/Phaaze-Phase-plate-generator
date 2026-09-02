# `optics.py`

[View source](../../src/phase_plate_generator/optics.py)

## Purpose

This module contains the shared spatial grids, interpolation, FFT propagation,
sampling selection, intensity normalisation and focal-shape metrics. It is the
numerical optics layer used by both retrieval and final verification.

## Function map

| Function | Responsibility |
|---|---|
| `normalise_propagation_model()` | Canonicalise Fraunhofer/Fresnel aliases |
| `centered_axis()`, `make_pupil()` | Physical sample coordinates and aperture |
| `pad_center()` | Symmetric crop or zero-padding |
| `bilinear_resample()` | Shape-based bilinear interpolation |
| `resample_phase_to_shape()` | Seam-aware wrapped-phase resize |
| `resample_intensity_to_grid()` | Physical-pitch intensity interpolation |
| `_centered_fft2()`, `_centered_ifft2()` | Centred discrete transform pair |
| `propagation_axis()` | Map FFT frequency to metres |
| `_fresnel_phase_factors()` | Input, lens and output quadratic phases |
| `forward_for_gs()`, `inverse_for_gs()` | Discrete retrieval operator pair |
| `propagate_phase_plate()` | Physically scaled final propagation |
| `_has_fft_friendly_size()`, `next_fft_friendly_size()` | Select 2/3/5-factor FFT sizes |
| `resolve_propagation_samples()` | Resolve padding and achieved focal pitch |
| `normalise_intensity()`, `focal_metrics()` | Shape normalization and comparison |

## Grids, pupils and resampling

`centered_axis(N, dx)` returns `(j - floor(N/2))*dx`.
`make_pupil()` evaluates either `x^2+y^2 <= (D/2)^2` or the corresponding
square bounds. `pad_center()` symmetrically crops or zero-pads a 2-D array.

`bilinear_resample()` evaluates the four surrounding pixels with separable
linear weights. `resample_intensity_to_grid()` performs the same interpolation
using physical source and destination pitches and sets samples outside the
source support to zero.

`resample_phase_to_shape()` avoids averaging across the `0/2*pi` seam. A phase
with at most 16 detected values uses nearest-neighbour sampling to preserve
steps. A more continuous phase is converted to `exp(i phi)`, the real and
imaginary parts are interpolated, and the output argument is taken.

## Centred discrete Fourier transform

The internal pair is:

```text
F = fftshift(fft2(ifftshift(U)))
U = fftshift(ifft2(ifftshift(F))).
```

The shifts place the spatial origin and zero spatial frequency at the central
array sample. The transform conventions and inverse scaling follow
[NumPy's DFT definition](https://numpy.org/doc/stable/reference/routines.fft.html).

For input pitch `dx` and propagation distance `z`, spatial frequency maps to
physical output coordinate

```text
x_out = lambda z f_x,
dx_out = lambda z / (N dx).
```

This reciprocal relationship is why zero-padding decreases focal-plane sample
pitch without adding physical aperture information.

## Fraunhofer branch

`forward_for_gs()` returns only the centred FFT because constant complex
prefactors do not affect a phase-only alternating-projection update.
`propagate_phase_plate()` restores physical quadrature scaling:

```text
U_f = dx^2 F / (i lambda f)
I_f = |U_f|^2.
```

The omitted global carrier phase has no effect on intensity.

## Fresnel branch with a thin lens

For `k = 2*pi/lambda`, input radius `r_1`, output radius `r_2`, distance `z`
and focal length `f`, the code multiplies by:

```text
input quadratic: exp(+i k r_1^2 / 2z)
lens:            exp(-i k r_1^2 / 2f)
output quadratic:exp(+i k r_2^2 / 2z).
```

The weighted field is Fourier transformed. The final propagator adds
`exp(i k z) dx^2/(i lambda z)`. At `z=f`, the input propagation and lens
quadratics cancel, recovering the familiar focal-plane Fourier transform.
`inverse_for_gs()` conjugates both quadratic phase factors around the inverse
FFT, making it the exact discrete inverse of the implemented forward operator.

This is the scalar, paraxial Fresnel approximation. It is not an angular
spectrum or vector diffraction calculation.

## Sampling selection

`resolve_propagation_samples()` starts with the plate grid and requested pad
factor. If an output pitch is requested, it enforces

```text
N >= lambda z / (dx_plate dx_output_requested).
```

It then chooses the next same-parity integer factorable only by 2, 3 and 5 for
efficient FFT execution. A maximum size prevents accidental excessive memory
use. The returned dictionary records the achieved pitch; it should be used
instead of assuming the requested pitch was obtained exactly.

## Metrics

`normalise_intensity()` clips negative numerical values and divides by the
peak. `focal_metrics()` compares peak-normalised arrays using:

```text
PCC = dot(T-mean(T), R-mean(R)) /
      (norm(T-mean(T)) norm(R-mean(R)))

relative_NRMSE = norm(T-R) / norm(R).
```

PCC measures linear shape correlation after mean removal. Relative NRMSE
measures residual Euclidean error. Because both arrays are peak-normalised,
neither metric measures absolute energy throughput.

## Assumptions and failure modes

- Arrays must be square for Fresnel propagation and final propagation.
- Coordinates are paraxial and wavelength is the vacuum wavelength.
- Uniform square sampling and a centred optical axis are assumed.
- FFT wrap-around can contaminate results when the computational window is too
  small; increase padding and check convergence with grid refinement.
- Bilinear interpolation does not preserve integrated optical power unless the
  caller separately accounts for pixel area.
- Fraunhofer validity is a physical approximation that the code does not
  automatically test.

## References

- S. N. Dixit et al., "Kinoform phase plates for focal plane irradiance
  profile control," *Optics Letters* **19**, 417-419 (1994),
  [doi:10.1364/OL.19.000417](https://doi.org/10.1364/OL.19.000417).
- [NumPy discrete Fourier transform conventions](https://numpy.org/doc/stable/reference/routines.fft.html)
- [NumPy `fftshift` reference](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html)
