# `modes/random_binary.py`

[View source](../../../src/phase_plate_generator/modes/random_binary.py)

## Purpose

`RandomBinaryMode` generates a stochastic zero/`pi` random phase plate and
deliberately skips target-driven GS retrieval.

## Recipe

For each pupil sample, a uniform random number is compared with
`pi_fraction`:

```text
phi(x,y) = pi  if u < pi_fraction
           0   otherwise.
```

The seeded `numpy.random.Generator` makes the sampled raster reproducible.
Samples outside the pupil are zero. Physical square or hexagonal cells are
created later by `manufacturing.py`; for a binary plate, each cell uses the
majority phasor sign.

## Physics

A zero/`pi` plate changes phase without deliberately modulating amplitude. Its
random cells divide a coherent beam into many interfering contributions,
producing a broadened envelope with fine speckle. Random phase plates have
been used for target-plane smoothing in high-power laser experiments; design
and fabrication on Nova are reported by
[Dixit et al. (1993)](https://doi.org/10.1364/AO.32.002543).

This code is a statistical raster model. It does not reproduce a particular
Nova layout, sol-gel process or measured plate map. A static coherent
propagation also retains speckle; temporal smoothing and plasma thermal
smoothing are outside the model.

## Distinction from binary kinoform

`random_binary` never uses the selected target to choose phase. Use
`multilevel` with `levels=2` for a GS-optimised binary plate. The two may have
the same phase alphabet but are physically and numerically different designs.

## Reference

- S. N. Dixit et al., "Random phase plates for beam smoothing on the Nova
  laser," *Applied Optics* **32**, 2543-2554 (1993),
  [doi:10.1364/AO.32.002543](https://doi.org/10.1364/AO.32.002543).

