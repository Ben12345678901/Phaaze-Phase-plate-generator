# Function documentation

This directory is the file-by-file technical reference for
`src/phase_plate_generator`. It explains what each module does, the numerical
recipe it implements, the associated optics, the array and unit conventions,
and the important limitations.

## Calculation flow

```text
Python input deck
    |
    v
decks.py -> config.py validation
    |
    v
generator.py
    |-- beams.py: incident amplitude and aperture
    |-- optics.py: sampling grids and propagation operators
    |-- targets.py: required focal-plane intensity
    |-- modes/: phase representation and projection rule
    `-- algorithms/gerchberg_saxton.py: alternating constraints
    |
    v
manufacturing.py: square/hexagonal physical elements
    |
    v
optics.py: re-propagate the exported optic
    |
    v
outputs.py + plotting.py
```

The fine-grid result is called the **ideal kinoform** in saved artifacts. When
manufacturing geometry is enabled, a second result is produced by averaging
the ideal design onto square or hexagonal elements. Metrics for both are
retained so the numerical design and the physical-element approximation are
not confused.

## Documentation map

| Source file | Documentation |
|---|---|
| `phase_plate_generator/__init__.py` | [Package API](package_init.md) |
| `phase_plate_generator/__main__.py` | [Module entry point](package_main.md) |
| `phase_plate_generator/algorithms/__init__.py` | [Algorithms package](algorithms/index.md) |
| `phase_plate_generator/algorithms/gerchberg_saxton.py` | [Gerchberg-Saxton engine](algorithms/gerchberg_saxton.md) |
| `phase_plate_generator/artifacts.py` | [Artifact naming](artifacts.md) |
| `phase_plate_generator/beams.py` | [Incident beams](beams.md) |
| `phase_plate_generator/cli.py` | [Legacy command-line interface](cli.md) |
| `phase_plate_generator/config.py` | [Typed configuration](config.md) |
| `phase_plate_generator/decks.py` | [Deck construction](decks.md) |
| `phase_plate_generator/generator.py` | [Generation workflow](generator.md) |
| `phase_plate_generator/manufacturing.py` | [Manufacturing projection](manufacturing.md) |
| `phase_plate_generator/modes/__init__.py` | [Mode registry](modes/index.md) |
| `phase_plate_generator/modes/base.py` | [Mode interface](modes/base.md) |
| `phase_plate_generator/modes/continuous_relief.py` | [Continuous relief](modes/continuous_relief.md) |
| `phase_plate_generator/modes/continuous_wrapped.py` | [Wrapped kinoform](modes/continuous_wrapped.md) |
| `phase_plate_generator/modes/distributed_relief.py` | [Distributed relief](modes/distributed_relief.md) |
| `phase_plate_generator/modes/multilevel.py` | [Multilevel mode](modes/multilevel.md) |
| `phase_plate_generator/modes/random_binary.py` | [Random binary mode](modes/random_binary.md) |
| `phase_plate_generator/modes/surface.py` | [Continuous-surface numerics](modes/surface.md) |
| `phase_plate_generator/optics.py` | [Grids and diffraction](optics.md) |
| `phase_plate_generator/output_layout.py` | [Output directory layout](output_layout.md) |
| `phase_plate_generator/outputs.py` | [Saved arrays and metadata](outputs.md) |
| `phase_plate_generator/plotting.py` | [Diagnostic plots](plotting.md) |
| `phase_plate_generator/progress.py` | [Progress reporting](progress.md) |
| `phase_plate_generator/runner.py` | [Timestamped runs](runner.md) |
| `phase_plate_generator/targets.py` | [Focal-plane targets](targets.md) |

## Global conventions

- All physical lengths are in metres inside the generator.
- All optical phase arrays are in radians.
- Beam arrays contain field amplitude. Target and propagated focal arrays
  contain intensity.
- Array coordinates use `indexing="xy"` and the zero-frequency sample is
  shifted to the array centre.
- `wrapped_phase_rad` is in the interval `[0, 2*pi)`. A non-wrapped physical
  surface is present only when `unwrapped_phase_rad` is not `None`.
- Intensity comparisons use peak-normalised arrays.
- The implementation is scalar, monochromatic and paraxial. It does not model
  polarisation, dispersion, vector diffraction, coating loss, damage,
  fabrication tolerances or temporal smoothing.

## Principal literature

The design architecture is based on the alternating Fourier-plane constraints
of Gerchberg and Saxton and their use for kinoform phase plates by
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417). The two
continuous-surface modes are compared separately with
[Dixit et al. (1996)](https://doi.org/10.1364/OL.21.001715) and
[Lin, Kessler and Lawrence (1995)](https://doi.org/10.1364/OL.20.000764).
The detailed pages identify where the implementation follows these sources and
where it makes an independent numerical choice. See also the project's
[literature validation note](../LITERATURE_VALIDATION.md).

