# `artifacts.py`

[View source](../../src/phase_plate_generator/artifacts.py)

## Purpose

This module is the single naming policy for generated files. Keeping naming
out of `outputs.py` and `plotting.py` ensures that saved arrays and comparison
figures use the same mode and geometry labels.

| Function | Result |
|---|---|
| `safe_name_component(value)` | Replaces non-alphanumeric runs by `_` |
| `ideal_kinoform_stem(mode)` | `Ideal_<mode>_Kinoform` |
| `grid_phase_stem(mode, geometry)` | `Hex_grid_<mode>`, `Square_grid_<mode>`, or `None` |
| `comparison_plot_name(mode, geometry)` | Joins the ideal and gridded stems into a comparison filename |

## Numerical and physical content

There is none. "Ideal" means the fine computational grid before manufacturing
projection; it does not mean an experimentally perfect or fabrication-ready
optic. "Kinoform" denotes the phase-only fine-grid design, consistent with the
phase-plate usage in [Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417).

## Maintenance rule

Change canonical output names here, then update the output-name tests. Do not
duplicate filename strings in plotting or saving code.

