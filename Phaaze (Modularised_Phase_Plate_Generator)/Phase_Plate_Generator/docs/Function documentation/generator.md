# `generator.py`

[View source](../../src/phase_plate_generator/generator.py)

## Purpose

This is the high-level numerical coordinator. It turns one validated
`ExperimentDeck` into a `GenerationResult`, but delegates the individual
recipes to the beam, target, optics, algorithm, mode and manufacturing
modules.

`GenerationResult` keeps both the ideal fine-grid kinoform and the exported
physical-element realization:

- input amplitude and pupil;
- wrapped, unwrapped and thickness arrays before and after manufacturing;
- ideal and manufactured focal intensities;
- target and focal-plane coordinates;
- element labels, convergence history and metadata.

## Function map

- `_load_initial_phase()` validates and maps a saved phase onto the calculation
  grid.
- `_crop_physical_plate()` extracts the central unpadded plate.
- `_mode_levels()` tells manufacturing whether the mode has two or `N`
  discrete levels.
- `generate()` coordinates the complete calculation.

## End-to-end recipe

`generate()` performs these operations:

1. Validate the deck and canonicalise the propagation model.
2. Choose an FFT size and output pitch using
   `resolve_propagation_samples()`.
3. Build the incident amplitude on the physical plate and centre-pad it onto
   the propagation grid.
4. Build the target intensity and optional finite constraint mask on the
   physically scaled focal grid.
5. Set the default phase-feature scale to
   `lambda * distance / target_diameter`.
6. Build the selected phase mode and optional imported phase seed.
7. Optionally build a reusable element grid and run the common
   Gerchberg-Saxton engine with each multilevel update constrained to it.
8. Crop the propagation grid back to the physical plate.
9. Average or quantise the design onto requested square/hexagonal elements.
10. Re-propagate that manufactured array rather than reusing the ideal focal
    field.
11. Calculate ideal and manufactured PCC/NRMSE, convert phase to thickness,
    measure continuous-surface statistics, and optionally save.

Steps 9-10 are important: the final metrics describe the actual array written
to disk as the gridded plate, so manufacturing loss is visible.

When `manufacturing.iterate_on_element_grid` is true, the convergence history
already describes the selected square or hexagonal element grid. The cropped
phase is circular-mean averaged per element before every multilevel update;
labels are precomputed once so the geometry is not rebuilt on each cycle.
For binary median or amplitude-weighted projection, those labels also ensure
that balancing selects complete physical elements. The selected method is
recorded in the manufacturing metadata.

## Imported phase handling

`_load_initial_phase()` accepts any finite 2-D array. A propagation-grid array
is used directly. A physical-grid array is centre-padded, while any other size
is first phase-aware resampled to the physical plate size. The phase is only an
initial condition; it does not override the chosen mode's projection rule.

## Physics represented

The workflow assumes a phase-only, normally illuminated thin element:

```text
U_after(x,y) = A(x,y) exp[i phi(x,y)].
```

The target is an irradiance constraint in a Fourier-related focal plane, the
kinoform design approach reported by
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417). The reciprocal
feature estimate follows the Fourier scaling `x_f = lambda f f_x`; it is a
useful default element/correlation scale, not a fabrication tolerance.

## Important limitations

- The scalar propagators do not model vector or high-NA effects.
- Phase-to-thickness uses one constant refractive index at one wavelength.
- Manufacturing projection represents piecewise-constant cell values but not
  edge slope, etch bias, roughness, coating or alignment.
- Peak-normalised focal metrics test shape, not absolute transmitted energy.
- The generator is deterministic only when the same numerical environment,
  deck and random seed are used.

## References

- S. N. Dixit et al., "Kinoform phase plates for focal plane irradiance
  profile control," *Optics Letters* **19**, 417-419 (1994),
  [doi:10.1364/OL.19.000417](https://doi.org/10.1364/OL.19.000417).
- Y. Lin, T. J. Kessler and G. N. Lawrence, "Distributed phase plates for
  super-Gaussian focal-plane irradiance profiles," *Optics Letters* **20**,
  764-766 (1995),
  [doi:10.1364/OL.20.000764](https://doi.org/10.1364/OL.20.000764).
