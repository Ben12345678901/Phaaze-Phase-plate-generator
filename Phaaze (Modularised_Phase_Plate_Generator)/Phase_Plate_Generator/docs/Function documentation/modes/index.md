# `modes/__init__.py`

[View source](../../../src/phase_plate_generator/modes/__init__.py)

## Purpose

This is the registry and factory for phase-plate representations. It translates
legacy/user aliases into canonical scientific names and constructs the correct
`PhaseMode` for the common GS engine.

| Canonical mode | Physical/numerical meaning |
|---|---|
| `continuous_wrapped` | Unquantised phase-only kinoform; wrapping may hide discontinuous relief |
| `continuous_relief` | One tracked real, unwrapped surface sheet |
| `distributed_relief` | Continuous random relief with projected RMS and correlation scale |
| `multilevel` | Target-optimised kinoform on `N` equally spaced phase levels |
| `random_binary` | Stochastic zero/`pi` plate with no target-driven retrieval |

`normalise_mode_name()` performs the alias lookup and rejects unknown values.
`build_mode()` is the factory used by the generator.

The aliases `true_continuous`, `distributed`, `dpp`, `binary`, `random` and
others are retained for old decks, but metadata always records a canonical
name.

## Factory recipe

`build_mode()` passes the pupil, physical pixel size and seeded NumPy random
generator to every mode. It then supplies only the parameters relevant to the
selected representation. If no correlation length is given for a relief, the
generator's reciprocal feature estimate is used.

The alias `binary` maps to `multilevel`, where `levels` must be two to obtain a
binary **optimised** optic. A genuinely random binary phase plate must use
`random_binary`.

## Physics

The separation is deliberate. Kinoforms are iterative phase-only designs
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417); strictly
continuous screens and distributed phase plates impose different surface
constraints
([Dixit et al. 1996](https://doi.org/10.1364/OL.21.001715);
[Lin et al. 1995](https://doi.org/10.1364/OL.20.000764)); random phase plates
are stochastic beam-smoothing optics
[Dixit et al. (1993)](https://doi.org/10.1364/AO.32.002543).
Putting them behind one interface does not make them physically equivalent.

## Maintenance rule

A new mode needs its own module, canonical name, aliases, factory branch and
tests. Its `project()` method must return a wrapped near-field phase on the
same grid and keep samples outside the pupil at zero.
