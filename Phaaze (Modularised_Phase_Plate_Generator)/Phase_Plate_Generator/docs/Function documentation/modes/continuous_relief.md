# `modes/continuous_relief.py`

[View source](../../../src/phase_plate_generator/modes/continuous_relief.py)

## Purpose

`ContinuousReliefMode` is the modular replacement for the legacy
`true_continuous` option. It stores one real, unwrapped phase sheet throughout
retrieval and marks the target constraint as adiabatic.

## Initial surface

A white Gaussian random array is convolved with a Gaussian kernel whose
autocorrelation FWHM is the requested correlation length. Aperture-edge bias is
removed by normalised masked convolution. The resulting surface is rescaled to
`initial_rms_rad`, shifted so its pupil minimum is zero, and converted to a
wrapped phasor for propagation.

An imported wrapped phase is first unwrapped with the Fourier Poisson solver,
then filtered and RMS-normalised in the same way.

## Per-iteration projection

The inverse-propagated candidate supplies only a wrapped phase. For each
sample, `lift_near_surface()` chooses the `2*pi` branch nearest the previous
real surface:

```text
delta = arg(exp[i(phi_candidate - S_previous)])
S_new = S_previous + delta.
```

The pupil minimum is subtracted as an optically irrelevant piston, and
`arg(exp(i S_new))` is returned to GS. Unlike distributed mode, the surface is
not correlation-filtered or RMS-normalised after each iteration.

`finalize()` returns both `S mod 2*pi` and the real sheet `S`.

## Relation to the literature

[Dixit et al. (1996)](https://doi.org/10.1364/OL.21.001715) report an
iterative method that maintains a continuous phase screen while producing a
high-order super-Gaussian focal profile. This implementation follows that
design objective through a continuous sheet state and gradual target
constraint.

The Gaussian-correlated random initializer, simplified branch lifting and
exact linear target ramp are project-specific numerical choices. This module
must not be described as a line-for-line reproduction of the paper without a
full equation- and figure-level benchmark.

## Limitations

- Discrete branch proximity does not impose a maximum slope or curvature.
- Continuity between samples and manufacturability are not guaranteed.
- A phase singularity or poor sampling can still lead to steep branch
  structure.
- The configured correlation length describes the initial surface only; it is
  not re-imposed during continuous-relief iterations.

## Reference

- S. N. Dixit, M. D. Feit, M. D. Perry and H. T. Powell, "Designing fully
  continuous phase screens for tailoring focal-plane irradiance profiles,"
  *Optics Letters* **21**, 1715-1717 (1996),
  [doi:10.1364/OL.21.001715](https://doi.org/10.1364/OL.21.001715).

