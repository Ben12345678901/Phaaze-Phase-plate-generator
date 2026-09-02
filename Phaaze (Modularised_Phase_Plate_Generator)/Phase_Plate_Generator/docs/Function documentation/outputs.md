# `outputs.py`

[View source](../../src/phase_plate_generator/outputs.py)

## Purpose

`save_result()` writes arrays and metadata with stable, self-describing names.
`_json_safe()` recursively converts NumPy scalars and paths and replaces
non-finite floats by JSON `null`.

## Saved arrays

The `Arrays` folder contains:

| Artifact | Meaning |
|---|---|
| `Input_beam.npy` | Physical-plate field amplitude |
| `Ideal_<mode>_Kinoform.npy` | Fine-grid wrapped phase before element projection |
| `Ideal_<mode>_Kinoform_unwrapped_phase_rad.npy` | Continuous ideal relief, when present |
| `Ideal_<mode>_Kinoform_thickness_m.npy` | Ideal phase converted to material height |
| `Ideal_<mode>_Kinoform_focal_spot.npy` | Peak-normalised ideal focal intensity |
| `<Hex/Square>_grid_<mode>.npy` | Manufactured wrapped phase |
| `<grid>_unwrapped_phase_rad.npy` | Manufactured continuous relief, when present |
| `<grid>_thickness_m.npy` | Manufactured material height |
| `<grid>_focal_spot.npy` | Re-propagated manufactured focal intensity |
| `<grid>_element_labels.npy` | Integer cell identity; `-1` lies outside the pupil |
| `Target_focal_spot_structure.npy` | Requested peak-normalised target intensity |
| `Input_beam_pupil_mask.npy` | Clear-aperture Boolean mask |
| `Focal_spot_coordinates_um.npz` | Full `X_um` and `Y_um` coordinate arrays |
| `Focal_spot_extent_um.npy` | `[xmin, xmax, ymin, ymax]` in micrometres |

Grid-named arrays are omitted when manufacturing geometry is `none`.

## Saved JSON

- `metadata.json`: sampling, mode, propagation, metrics, manufacturing and
  surface summary.
- `convergence.json`: one PCC, relative NRMSE and RMS phase-step record per GS
  iteration. It is empty for random binary mode.
- `resolved_input_deck.json`: the complete, validated deck with absolute paths.

## Scientific interpretation

NumPy files preserve array shape and dtype without lossy image scaling.
Coordinates are saved separately so a focal array is not mistaken for an
unscaled pixel image. Intensities are normalised, so absolute SI irradiance
cannot be reconstructed from these artifacts alone.

The names "ideal" and "grid" distinguish numerical design from the
piecewise-constant physical approximation. They do not certify manufacturing
feasibility.

## References

- [NumPy binary I/O (`save`, `savez`, `load`)](https://numpy.org/doc/stable/reference/routines.io.html)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)

