# `modes/surface.py`

[View source](../../../src/phase_plate_generator/modes/surface.py)

## Purpose

This module contains the reusable numerical operations for a real, unwrapped
continuous-relief phase surface: Gaussian correlation filtering, phase
unwrapping, branch lifting, RMS control, thickness conversion and correlation
measurement.

## Gaussian transfer and correlation length

`gaussian_transfer()` constructs the Fourier transform of a Gaussian kernel:

```text
H(fx,fy) = exp[-2*pi^2*sigma_px^2(fx^2 + fy^2)].
```

Filtering white noise with kernel standard deviation `sigma` gives an
autocorrelation with standard deviation `sqrt(2)*sigma`. Its full width at half
maximum is consequently

```text
L_FWHM = 4 sqrt(ln 2) sigma.
```

`correlation_fwhm_to_sigma_px()` rearranges this expression and divides by
physical pixel size.

`filter_surface()` performs convolution by FFT. It convolves the mask by the
same transfer function and divides by those weights, avoiding artificial
depression near a clear-aperture boundary. The pupil minimum is then removed.

## Branch lifting

`lift_near_surface()` selects the branch of a wrapped candidate closest to a
reference surface:

```text
S_lift = S_ref + arg(exp[i(phi_wrapped - S_ref)]).
```

This minimizes the pointwise phase update modulo `2*pi`; it does not by itself
enforce spatial slope.

## Two-dimensional unwrapping

`unwrap_surface_2d()`:

1. calculates wrapped forward differences on valid horizontal and vertical
   pupil edges;
2. forms their discrete divergence;
3. solves the Poisson equation in Fourier space using the discrete Laplacian

   ```text
   L(kx,ky) = 2 cos(kx) + 2 cos(ky) - 4;
   ```

4. sets the undefined zero-frequency piston to zero; and
5. shifts the pupil minimum to zero.

This is an unweighted least-squares-style Fourier Poisson unwrap. Fast-transform
least-squares phase unwrapping is treated by
[Ghiglia and Romero (1994)](https://doi.org/10.1364/JOSAA.11.000107), but the
present masked periodic FFT implementation is a simplified project method,
not their complete robust weighted algorithm.

## RMS normalization

`normalize_surface_rms()` subtracts the pupil mean, multiplies by
`required_rms/current_rms`, masks the exterior, then subtracts the new pupil
minimum. Adding this piston leaves RMS and optical intensity unchanged.

## Phase to thickness

For a transmissive relief of refractive index `n` surrounded by air at normal
incidence, excess optical path is `(n-1)t`, so:

```text
phi = 2*pi*(n-1)t/lambda
t = phi lambda / [2*pi*(n-1)].
```

`phase_to_thickness_m()` applies this thin-element relation to the supplied
wrapped or unwrapped phase. The index is assumed real and constant.

## Correlation measurement

`estimate_correlation_fwhm_m()` subtracts the pupil mean, computes circular
autocovariance with the Wiener-Khinchin FFT product, and divides by the
autocorrelation of the mask to account for valid sample pairs. It averages the
positive and negative horizontal/vertical profiles, locates the first
half-maximum crossing by linear interpolation, doubles that half-width and
converts pixels to metres.

This is an axial mean FWHM, not a full two-dimensional anisotropy or power
spectral density analysis.

## Limitations

- FFT convolution and Poisson inversion imply periodic computational
  boundaries.
- Mask normalisation reduces edge bias but does not impose a physical boundary
  condition for the unwrap.
- Unwrapping can fail around residues or under-sampled gradients.
- Thickness ignores material dispersion, incidence angle, reflection,
  absorption and substrate/coating structure.
- Correlation estimates are returned as `NaN` if no half-height crossing lies
  in the inspected quarter-grid lag.

## References

- D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
  unweighted phase unwrapping that uses fast transforms and iterative
  methods," *JOSA A* **11**, 107-117 (1994),
  [doi:10.1364/JOSAA.11.000107](https://doi.org/10.1364/JOSAA.11.000107).
- [NumPy FFT reference](https://numpy.org/doc/stable/reference/routines.fft.html)

