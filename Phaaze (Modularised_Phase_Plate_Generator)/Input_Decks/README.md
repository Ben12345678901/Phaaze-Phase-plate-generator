# Input decks

An input deck is one runnable Python file. Its `INPUTS` dictionary contains
the complete optical system, target, algorithm, phase representation,
manufacturing projection, and output configuration. Edit that dictionary and
run the same file from the IDE:

```python
INPUTS = {
    "facility": {"name": "OMEGA"},
    "target": {
        "profile": "supergaussian",
        "diameter_m": 800e-6,
        "supergaussian_order": 8.0,
    },
    "phase_plate": {"mode": "distributed_relief"},
    # The real decks include every remaining section and value here.
}

def main():
    deck = deck_from_mapping(INPUTS, source_path=DECK_PATH)
    return run_configured_deck(deck, DECK_PATH)


if __name__ == "__main__":
    main()
```

`Input_Decks` sits beside the modular `Phase_Plate_Generator` package, matching
the proven sibling-file launch pattern used by `Source_Files`. Relative paths
inside `INPUTS` are resolved from the Python deck's directory.
Facility files are starter profiles, not shot-authoritative specifications.
The OMEGA wavelength is supported by the LLE facility documentation, while
focal length, clear aperture, material index, input beam, and target
definition must be checked for the actual beamline and campaign. The GSI
starter values are migrated from the legacy generator and likewise need
experiment-owner review.

Every deck contains its own imports, package-path bootstrap, `main()` function,
and run block. To run the OMEGA facility deck, edit and run
`facilities/omega.py`. Likewise, an experiment is configured and run entirely
from its file in `examples/`.

The shared runner appends the input-deck name and local date/time to the
configured output root, for example:

```text
outputs/omega_distributed_800um_2026-07-25_18-42-10/
```

The complete runnable Python deck is copied into the run directory for
traceability. Generated JSON output records capture the resolved inputs,
metrics, and convergence history.

Every timestamped run uses the same grouped layout:

```text
<deck_name>_<date>_<time>/
  plots/             PNG figures
  json_files/        metadata, convergence, and resolved inputs
  Arrays/            NPY and NPZ numerical artifacts
  Input deck used/   exact Python deck source used for the run
```

Progress and plot controls are also part of each deck:

```python
"algorithm": {
    # ...
    "show_progress": True,
    "progress_updates": 20,
},
"output": {
    # ...
    "save_plots": True,
    "show_plots": True,
    "plot_dpi": 160,
    "focal_plot_window_factor": 2.0,
},
```

The live bar reports iteration percentage, PCC, NRMSE, and elapsed time.
After the calculation, the convergence and legacy-style result figures are
saved to the timestamped output directory and displayed. The main overview
shows the pre-manufacturing GS solution (fine-grid normally, or grid-
constrained when iterative projection is enabled). Square and hexagonal decks
receive a separate clean physical-element comparison without contrasting edge
highlighting.

For a multilevel plate, the GS loop can optionally use the physical square or
hexagonal grid during every iteration:

```python
"manufacturing": {
    "element_geometry": "hexagonal",
    "element_pitch_m": 1.0e-3,
    "iterate_on_element_grid": True,
    "element_grid_start_fraction": 0.0,
    "boundary_rounding_sigma_px": 0.0,
},
```

With the start fraction left at `0.0`, each near-field update is reduced to one
circular-mean phase per element before the multilevel projection, preserving
the original behaviour. Set `element_grid_start_fraction` between zero and one
to keep that fraction of GS on the fine phase grid before switching on the
physical geometry. For example, `0.5` activates the element grid for the
second half of the iterations. To align the switch with the start of the
quantization ramp, use `1 - quantization_ramp_fraction`. The fine numerical
raster, propagation sampling, saved array shapes, and final manufacturing pass
remain unchanged. With iterative projection disabled, the element grid is
applied only after retrieval; the final manufactured PCC/NRMSE is still
recorded in `metadata.json`.

Binary multilevel decks can select the final zero/`pi` assignment rule:

```python
"phase_plate": {
    "mode": "multilevel",
    "levels": 2,
    # "threshold", "median", or "amplitude_weighted"
    "binarization_method": "amplitude_weighted",
},
```

`threshold` preserves the previous nearest-phase result. `median` balances
illuminated plate area. `amplitude_weighted` balances incident electric-field
amplitude and should be used only when `beam.array_path` contains amplitude,
not measured intensity. On a manufacturing grid, complete elements are
balanced so a square or hexagonal cell is never split.

Saved arrays use the canonical phase mode and physical geometry in their
filenames. For a `continuous_relief` plate on a hexagonal grid, the principal
artifacts are:

```text
Ideal_continuous_relief_Kinoform.npy
Ideal_continuous_relief_Kinoform_focal_spot.npy
Hex_grid_continuous_relief.npy
Hex_grid_continuous_relief_focal_spot.npy
Input_beam.npy
Target_focal_spot_structure.npy
Focal_spot_coordinates_um.npz
Focal_spot_extent_um.npy
Ideal_continuous_relief_Kinoform_vs_Hex_grid_continuous_relief_comparison.png
```

Canonical phase modes are:

- `continuous_wrapped`: unquantized kinoform, not guaranteed to be a smooth
  physical surface.
- `continuous_relief`: strictly continuous unwrapped surface relief.
- `distributed_relief`: continuous random relief with retained correlation
  and phase-RMS constraints.
- `multilevel`: optimized N-level plate (`levels: 2` is binary).
- `random_binary`: stochastic 0/π random phase plate without GS optimization.

Legacy names such as `continuous`, `true_continuous`, `distributed`,
`quantized`, `binary`, `random`, and `dpp` remain accepted aliases.
