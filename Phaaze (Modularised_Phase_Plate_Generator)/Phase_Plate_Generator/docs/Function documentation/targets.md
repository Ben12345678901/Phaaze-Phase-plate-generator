# `targets.py`

[View source](../../src/phase_plate_generator/targets.py)

## Purpose

This module builds the desired focal-plane **intensity** and the region in
which that target is enforced. `normalise_target_name()` converts aliases to
one of four canonical profiles.

`_stripe_coordinate()` selects the `x` or `y` modulation coordinate,
`build_target_intensity()` implements all analytic/imported profiles, and
`build_constraint_mask()` defines the region used by GS.

## Target recipes

For target radius `a = diameter_m/2`, radial coordinate
`r = sqrt(x^2+y^2)` and order `m`, the super-Gaussian target is:

```text
I_target(x,y) = exp[-(r/a)^(2m)].
```

This uses `a` as the `1/e` intensity radius. Increasing `m` makes the central
region flatter and the edge steeper. Super-Gaussian focal envelopes are a
central use case in the DPP literature, including
[Lin, Kessler and Lawrence (1995)](https://doi.org/10.1364/OL.20.000764).

`striped_supergaussian` multiplies that envelope by:

```text
1 + depth * cos(2*pi*s/period),
period = diameter/stripes,
```

where `s` is `x` or `y`. The multiplicative intensity stays non-negative
because the validated depth lies in `[0,1]`.

`striped_tophat` divides the diameter into `2*stripes - 1` equal bands, giving
exactly `stripes` bright bands separated by dark bands. The levels are
`1+depth` and `1-depth`, clipped by the circular radius. This striped target is
a project feature rather than a claim from the cited kinoform papers.

`array` loads a finite 2-D intensity map. Without a declared source pixel
size, it is resized to the target shape. With a source pitch, it is
interpolated onto the physical focal grid and zero-filled outside the source
extent. Every target is clipped non-negative and peak-normalised.

## Constraint mask

`build_constraint_mask()` returns all samples when
`dark_region_factor=None`. Otherwise it defines a centred square with
half-width:

```text
dark_region_factor * target_diameter / 2.
```

During GS, target amplitude is imposed only inside this square; the calculated
amplitude is retained outside it. It is therefore a freedom window, not a
physical aperture.

## Physics and cautions

The GS engine uses `sqrt(I_target)` because Fourier propagation acts on the
complex field amplitude. Supplying amplitude as an imported target would
therefore square-root it again and give the wrong constraint.

Peak normalisation specifies shape only. The target's integral is rescaled
inside a partial constraint mask during each iteration, but no requested
absolute focal energy or laser power is represented.

## References

- Y. Lin, T. J. Kessler and G. N. Lawrence, "Distributed phase plates for
  super-Gaussian focal-plane irradiance profiles," *Optics Letters* **20**,
  764-766 (1995),
  [doi:10.1364/OL.20.000764](https://doi.org/10.1364/OL.20.000764).
- S. N. Dixit et al., "Designing fully continuous phase screens for tailoring
  focal-plane irradiance profiles," *Optics Letters* **21**, 1715-1717 (1996),
  [doi:10.1364/OL.21.001715](https://doi.org/10.1364/OL.21.001715).
