# `config.py`

[View source](../../src/phase_plate_generator/config.py)

## Purpose

This module defines the typed schema for an experiment. Each input-deck
section has a dataclass, and `ExperimentDeck` combines them:

| Section | Controls |
|---|---|
| `FacilityConfig` | Descriptive run label only |
| `GridConfig` | Plate extent, plate samples and propagation padding |
| `OpticsConfig` | Wavelength, focal length, propagation model and index |
| `BeamConfig` | Incident-amplitude model |
| `TargetConfig` | Desired focal intensity and constraint window |
| `AlgorithmConfig` | GS iteration and progress settings |
| `PhasePlateConfig` | Phase representation and its parameters |
| `ManufacturingConfig` | Element geometry, pitch, iterative grid constraint and binary rounding |
| `OutputConfig` | Saving, display and focal plot window |

The facility name does not change any physics. It labels plots and metadata.

## Methods

`as_dict()` recursively converts dataclasses and paths into a JSON-compatible
mapping. `observation_distance_m` uses the focal length when no separate
propagation distance is declared. `validate()` rejects unknown names,
non-positive physical quantities, invalid fractions, incompatible sample
limits and missing imported files before allocating the main arrays.

## Physical conventions

Lengths are metres, phase is radians, and target arrays are intensity unless a
field name says otherwise. The refractive index must exceed one because
surface thickness is obtained from the excess optical path relative to air:

```text
phi = 2*pi*(n - 1)*t/lambda.
```

The field `propagation_distance_m` matters for Fresnel propagation. In the
Fraunhofer branch, the focal length is used as the transform distance so the
output coordinates describe the lens focal plane.

`ManufacturingConfig.iterate_on_element_grid` is an opt-in multilevel-only
constraint. It requires square or hexagonal elements. When enabled, every GS
near-field update is element-averaged before multilevel projection, so the
iteration PCC/NRMSE history measures the physical grid rather than the fine
calculation raster.

For a binary multilevel kinoform (`mode="multilevel", levels=2`),
`PhasePlateConfig.binarization_method` accepts `threshold`, `median`, or
`amplitude_weighted`. Threshold preserves nearest-phase quantisation; median
balances physical area; amplitude-weighted balances incident field amplitude.
The latter two values are rejected for nonbinary or non-multilevel modes.

## Validation is not experimental qualification

The checks establish internal numerical consistency only. They do not confirm
that a chosen wavelength, material index, element pitch, aperture, focal
length or fluence is suitable for a real facility. Dispersion, laser damage,
minimum feature size and fabrication tolerances must be supplied from the
actual optic specification.

## References

- [Python `dataclasses` documentation](https://docs.python.org/3/library/dataclasses.html)
- S. N. Dixit et al., "Kinoform phase plates for focal plane irradiance
  profile control," *Optics Letters* **19**, 417-419 (1994),
  [doi:10.1364/OL.19.000417](https://doi.org/10.1364/OL.19.000417).
