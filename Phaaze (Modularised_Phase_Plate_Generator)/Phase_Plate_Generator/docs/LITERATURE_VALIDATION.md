# Literature validation

This note distinguishes direct validation from project-specific modeling
choices. Access was limited to publisher abstracts/metadata where full text was
not openly available.

## Gerchberg-Saxton / kinoform retrieval

The common engine alternates between the near and far fields, replacing the
far-field magnitude with the square root of target intensity and restoring the
known incident amplitude in the plate plane. This matches the defining
Gerchberg-Saxton alternating-constraint structure and the kinoform phase-plate
description by Dixit et al.

- R. W. Gerchberg and W. O. Saxton, “A practical algorithm for the
  determination of phase from image and diffraction plane pictures,” *Optik*
  **35**, 237–246 (1972).
- S. N. Dixit et al., “Kinoform phase plates for focal plane irradiance profile
  control,” *Optics Letters* **19**, 417–419 (1994),
  https://doi.org/10.1364/OL.19.000417.

The configurable finite target window and hard central warm-up are later
engineering constraints inherited from the source scripts, not claims about
the original GS paper.

## Continuous-relief mode

The legacy `true continuous` option is renamed `continuous_relief` because it
tracks a real, unwrapped surface and introduces the target constraint
adiabatically. This is conceptually aligned with the paper’s requirement that
the phase remain continuous throughout the iterative process.

- S. N. Dixit et al., “Designing fully continuous phase screens for tailoring
  focal-plane irradiance profiles,” *Optics Letters* **21**, 1715–1717 (1996),
  https://doi.org/10.1364/OL.21.001715.

The Gaussian correlation filter, branch-lifting implementation, and exact
linear target ramp are numerical choices from this project. They should not be
described as an exact reproduction of the paper without comparison against its
full method and figures.

## Distributed-relief mode

The DPP paper starts from a strictly continuous distributed phase plate and
uses a small number of phase-retrieval cycles to improve a super-Gaussian far
field while reducing discontinuity-driven scattering. The modular
`distributed_relief` mode satisfies the strict-continuity requirement.

- Y. Lin, T. J. Kessler, and G. N. Lawrence, “Distributed phase plates for
  super-Gaussian focal-plane irradiance profiles,” *Optics Letters* **20**,
  764–766 (1995), https://doi.org/10.1364/OL.20.000764.

The code additionally re-imposes Gaussian autocorrelation and phase RMS after
every iteration. That is a legacy/project-specific constrained projector, not
established by the accessible abstract. It is intentionally isolated in
`modes/distributed_relief.py` so it can be replaced after a full paper-level
benchmark.

## Random binary phase plates

`random_binary` is deliberately separate from an optimized two-level kinoform.
It creates a stochastic 0/π plate and skips target-driven GS retrieval. This
matches the role of an RPP as a robust beam-homogenizing optic, while the
current implementation does not claim to reproduce a particular fabrication
layout from the paper.

- C. L. S. Lewis et al., “Use of a random phase plate as a KrF laser beam
  homogenizer for thin film deposition applications,” *Review of Scientific
  Instruments* **70**, 2116–2121 (1999).

## OMEGA starter deck

LLE identifies OMEGA as a 351 nm, 60-beam system using DPPs. Public OMEGA-EP
capability information describes DPP-generated smooth super-Gaussian spots and
an eighth-order exponent. Those facts support the wavelength, DPP mode, and
target family in the starter deck. The remaining optical geometry comes from
the legacy control panel and must be checked for a real shot.

- LLE OMEGA Laser System:
  https://www.lle.rochester.edu/omega-laser-facility/omega-laser-system/
- LaserNetUS OMEGA EP:
  https://new.lasernetus.org/facilities/omega-ep

