# `modes/continuous_wrapped.py`

[View source](../../../src/phase_plate_generator/modes/continuous_wrapped.py)

## Purpose

`ContinuousWrappedMode` is the unconstrained phase-only kinoform. It inherits
the base initializer, projector and finalizer unchanged and supplies the
canonical name `continuous_wrapped`.

## Numerical recipe

At every GS inverse step, the candidate angle is accepted at each pupil pixel:

```text
phi_(n+1)(x,y) = arg(U'_n(x,y)).
```

The final array is reduced modulo `2*pi`. There is no quantisation, spatial
filter, slope constraint, correlation constraint or unwrapped surface state.

## Physics and naming

This is the closest mode to the phase-only kinoform retrieval reported by
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417). "Continuous"
means the allowed phase value is not discretised. It does **not** mean the
corresponding material relief is spatially continuous: neighbouring values
near zero and `2*pi` can represent a full-wave height step after a naive
phase-to-thickness conversion.

Use `continuous_relief` when a single unwrapped surface sheet is required.

## Reference

- S. N. Dixit et al., "Kinoform phase plates for focal plane irradiance
  profile control," *Optics Letters* **19**, 417-419 (1994),
  [doi:10.1364/OL.19.000417](https://doi.org/10.1364/OL.19.000417).

