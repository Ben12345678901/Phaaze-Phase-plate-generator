# `modes/base.py`

[View source](../../../src/phase_plate_generator/modes/base.py)

## Purpose

`PhaseMode` is the stateful projection interface between the generic GS loop
and a particular phase representation. `ModeResult` contains wrapped phase and
an optional unwrapped relief.

## Default recipe

The base initializer chooses independent values from
`{-pi, -pi/2, 0, pi/2, pi}` and multiplies by the pupil. This supplies a
reproducible random phase seed through the generator passed into the
constructor.

`initialize_from()` reduces an imported array to its principal phase using
`angle(exp(i phi))`. `project()` accepts the candidate wrapped phase inside the
pupil without further constraints. `finalize()` maps it to `[0,2*pi)` and
keeps the exterior at zero.

## Extension flags

- `canonical_name`: stable metadata and artifact label.
- `requires_adiabatic_target`: asks the GS engine to ramp the focal constraint
  and disables the hard-zero warm-up.
- `skip_phase_retrieval`: finalises the initial phase without GS iterations.

## Physics

The projection method represents the admissible set in an alternating-
projection calculation. The base class admits any phase-only phasor, which is
the wrapped-kinoform case described by
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417). Derived modes
restrict that set further.

## Contract for new modes

Arrays must match the pupil, contain phase in radians, and remain zero outside
the pupil. A mode that represents a real surface should keep that surface as
internal state and return it as `unwrapped_phase_rad`; otherwise the
phase-to-thickness conversion can only describe a single wrapped relief order.

