# `manufacturing.py`

[View source](../../src/phase_plate_generator/manufacturing.py)

## Purpose

This module replaces the ideal pixel-scale phase with a piecewise-constant
plate on square or hexagonal elements. `ManufacturedPhase` stores the raster,
one value per element, the integer label image, element count and achieved
pitch.

## Function map

- `_square_labels()` partitions the raster into rectangular index blocks.
- `_round_hex()` rounds fractional axial coordinates through constrained cube
  coordinates.
- `_hex_labels()` assigns one integer to every in-pupil hexagonal cell.
- `_circular_cell_means()` reduces wrapped pixel phases to cell phasor means.
- `build_element_grid()` precomputes reusable labels for post-processing or
  iterative projection.
- `project_to_elements()` performs the selected geometry and cell reduction.
- `round_binary_boundaries()` smooths a binary topology while preserving its
  `pi` area fraction.

## Element labelling

For a square grid, rounded, equally spaced pixel edges divide each axis into
`elements_across`. The achieved pitch is exactly
`plate_size_m / elements_across`.

For a hexagonal grid, pixel coordinates are converted to fractional
pointy-top axial coordinates:

```text
q = 2 x / (sqrt(3) p)
r = y/p - x/(sqrt(3) p),
```

where `p` is the centre-to-centre pitch in pixels. Axial coordinates are
converted to cube coordinates satisfying `x_c + y_c + z_c = 0`, rounded, then
the component with the largest rounding error is corrected to restore that
constraint. Unique axial pairs inside the pupil become integer element labels.
The exact cube-rounding construction is documented by
[Patel's hexagonal-grid reference](https://www.redblobgames.com/grids/hexagons/#rounding).

## Reducing fine pixels to one cell value

Wrapped phase must be averaged on the unit circle because `0` and `2*pi`
represent the same phasor. For a general cell:

```text
phi_cell = atan2(sum sin(phi_j), sum cos(phi_j)) mod 2*pi.
```

For a two-level plate, `threshold` uses the sign of `sum cos(phi_j)` to select
zero or `pi`. `median` instead balances illuminated element area, and
`amplitude_weighted` balances the summed incident field amplitude. The latter
two rules rank whole cells by the magnitude of their circular-mean phase.
Multilevel values are snapped to the requested phase alphabet.

`ElementGrid.project_wrapped()` can also apply those same labels during every
GS iteration. Iterative use first takes the unquantized circular cell mean;
the multilevel mode then applies its configured quantization ramp. This order
keeps every update cell-constant without bypassing the ramp schedule.

An unwrapped relief is different: arithmetic means are taken in each cell,
then a global offset makes the smallest surface value zero. The wrapped export
is the resulting relief modulo `2*pi`.

`geometry="none"` returns a copy of the fine design and an all-`-1` label map.

## Binary boundary rounding

`round_binary_boundaries()`:

1. classifies each pixel by whether its wrapped phase is closer to zero or
   `pi`;
2. records the `pi` area fraction;
3. applies a Gaussian low-pass filter by multiplication in Fourier space;
4. chooses a quantile threshold that preserves that area fraction for
   `threshold`, or reapplies the configured area/amplitude balance; and
5. returns a strict zero/`pi` raster.

The Gaussian transfer function is

```text
H(fx,fy) = exp[-2*pi^2*sigma_px^2(fx^2 + fy^2)].
```

This smooths narrow pixel-scale boundary excursions but is not a geometrical
model of a particular etch or polishing process.

## Physics and interpretation

Hexagonal zero/`pi` elements are historically relevant to large random phase
plates; Nova phase plates are described by
[Dixit et al. (1993)](https://doi.org/10.1364/AO.32.002543). The code also
supports continuous and multilevel cell values. Discretising a kinoform alters
its complex transmission and can lower target fidelity and diffraction
efficiency, hence the mandatory re-propagation in `generator.py`. The
fabrication role of multiple phase levels is discussed experimentally by
[Hasman, Davidson and Friesem (1991)](https://doi.org/10.1364/OL.16.000423).

## Limitations

- `requested_pitch_m` is rasterised; square pitch is adjusted to tile the
  plate and hex pitch cannot be below one fine-grid pixel.
- Edge elements are clipped by the pupil and need not be full hexagons.
- Cell averaging is area-weighted only through the number of pixels, so a
  coarse computational grid biases partial elements.
- Amplitude-weighted binary balancing assumes the supplied beam array is field
  amplitude; supplying intensity would apply the wrong physical weights.
- The model has vertical, piecewise-constant phase boundaries and no
  fabrication-error distribution.

## References

- S. N. Dixit et al., "Random phase plates for beam smoothing on the Nova
  laser," *Applied Optics* **32**, 2543-2554 (1993),
  [doi:10.1364/AO.32.002543](https://doi.org/10.1364/AO.32.002543).
- E. Hasman, N. Davidson and A. A. Friesem, "Efficient multilevel phase
  holograms for CO2 lasers," *Optics Letters* **16**, 423-425 (1991),
  [doi:10.1364/OL.16.000423](https://doi.org/10.1364/OL.16.000423).
