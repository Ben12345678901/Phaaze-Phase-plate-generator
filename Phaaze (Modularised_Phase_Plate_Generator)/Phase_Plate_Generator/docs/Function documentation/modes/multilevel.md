# `modes/multilevel.py`

[View source](../../../src/phase_plate_generator/modes/multilevel.py)

## Purpose

This module implements target-optimised stepped phase plates. `levels=2`
produces a binary kinoform; it is not the stochastic `random_binary` mode.
`quantize_phase()` supplies the phase alphabet and `MultilevelMode` applies it
during and after retrieval.

## Quantisation

`quantize_phase()` uses `N` equally spaced values over one optical cycle:

```text
Delta = 2*pi/N
phi_q = round((phi mod 2*pi)/Delta) Delta,
```

with a rounded `2*pi` mapped back to zero. Thus the admissible values are
`0, Delta, ..., (N-1)Delta`.

For `levels=2`, `phase_plate.binarization_method` selects one of three rules:

- `threshold` is the historical nearest-phase rule. Samples whose principal
  phase magnitude is at least `pi/2` become `pi`; the remainder become zero.
- `median` ranks samples by principal phase magnitude and assigns `pi` to the
  highest-ranked samples until their illuminated area is as close as possible
  to half the pupil area.
- `amplitude_weighted` uses the same ranking but balances incident electric-
  field amplitude instead of area. It therefore minimizes the coherent
  on-axis sum for a nonuniform incident beam. The weights are field amplitude,
  not intensity.

When a physical element grid is active, the ranking and balancing operate on
whole square or hexagonal elements. Clipped edge-element area is included, and
amplitude-weighted mode sums all incident-amplitude samples in each element.
No balancing rule can split one physical element between zero and `pi`.

## Gradual projection

Quantisation is introduced over the final
`round(total_iterations*ramp_fraction)` iterations. At ramp fraction `alpha`,
the continuous and quantised phasors are blended on the unit circle:

```text
phi_next = arg[(1-alpha) exp(i phi) + alpha exp(i phi_q)].
```

Complex interpolation avoids an artificial jump across the `0/2*pi` seam.
The finalizer always performs exact quantisation. Therefore
`ramp_fraction=0` means "quantise only at finalisation", not "disable
quantisation".

## Physics

Multiple discrete relief depths approximate a continuous kinoform. More phase
levels can reduce quantisation error, but the actual target fidelity also
depends on element pitch, etch accuracy and the designed phase distribution.
High efficiency from manufactured multilevel phase holograms and sensitivity
to step errors are discussed by
[Hasman, Davidson and Friesem (1991)](https://doi.org/10.1364/OL.16.000423).

The gradual projector is a numerical continuation method selected by this
project; it is not claimed as the fabrication method in that paper.

## Limitations

- Phase levels are uniformly spaced even if a fabrication process would
  benefit from calibrated nonuniform depths.
- Quantisation acts on phase at one wavelength and one refractive index.
- The mode does not enforce physical element size; that occurs later in
  `manufacturing.py`.
- Median and amplitude-weighted balancing are binary-only. They can slightly
  trade target-shape fidelity for reduced coherent zero-order leakage.
- Amplitude balancing uses the configured scalar incident amplitude and does
  not include fabrication-dependent transmission variations.
- Phasor interpolation can be ill-conditioned if its two terms nearly cancel.

## Reference

- E. Hasman, N. Davidson and A. A. Friesem, "Efficient multilevel phase
  holograms for CO2 lasers," *Optics Letters* **16**, 423-425 (1991),
  [doi:10.1364/OL.16.000423](https://doi.org/10.1364/OL.16.000423).
