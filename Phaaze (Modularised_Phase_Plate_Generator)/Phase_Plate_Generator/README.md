# Modular phase-plate generator

This package combines the shared functionality of the two legacy generators
under `../Source_Files` while keeping phase representations, focal targets,
propagation, physical element geometry, and facility cases independently
editable.

## Run an input deck directly

Each input deck is one self-contained Python file. All editable inputs live in
its `INPUTS` dictionary, so there is no separate JSON configuration to keep in
sync. Open the deck you want and press **Run** in the IDE:

```text
../Input_Decks/
  facilities/
    omega.py                         edit and run this file
    gsi.py                           edit and run this file
  examples/
    omega_distributed_800um.py       edit and run this file
    gsi_striped_tophat_binary.py     edit and run this file
    quick_demo.py                    small validation case
```

The deck locates the package and starts the generator itself. No package
installation, selector edit, JSON file, or command-line argument is required.

Every run creates a separate directory:

```text
outputs/
  gsi_striped_tophat_binary_2026-07-25_18-42-10/
    plots/
      convergence.png
      results_overview.png
      Ideal_multilevel_Kinoform_vs_Hex_grid_multilevel_comparison.png
    json_files/
      resolved_input_deck.json
      metadata.json
      convergence.json
    Arrays/
      Input_beam.npy
      Target_focal_spot_structure.npy
      Ideal_multilevel_Kinoform.npy
      Ideal_multilevel_Kinoform_focal_spot.npy
      Hex_grid_multilevel.npy
      Hex_grid_multilevel_focal_spot.npy
      Focal_spot_extent_um.npy
      ...
    Input deck used/
      gsi_striped_tophat_binary.py
```

If two runs start in the same second, the later folder receives `_02`, `_03`,
and so on. A failed calculation retains the selected deck and writes
`json_files/run_failed.json` with the exception information. The JSON files
inside an output folder's `json_files` directory are generated records of the
resolved configuration, metrics, and convergence history; they are not
hand-maintained input files.

## Layout

```text
src/phase_plate_generator/
  algorithms/          common Gerchberg-Saxton loop
  modes/               one file per phase-plate representation
  beams.py             incident beam construction
  targets.py           focal intensity targets
  optics.py            grids and Fraunhofer/Fresnel propagation
  manufacturing.py     square/hexagonal physical element projection
  generator.py         input-deck orchestration
  runner.py            timestamped Python-deck execution
  outputs.py           stable saved artifact names
  output_layout.py     grouped folders inside each timestamped run
  plotting.py          convergence, overview, and element-grid figures
../Input_Decks/
  facilities/          reusable OMEGA and GSI starter profiles
  examples/            experiment-specific complete decks
docs/
  LITERATURE_VALIDATION.md
  MIGRATION.md
tests/
```

The numerical workflow saves the selected Python deck, resolved inputs,
wrapped phase, optional unwrapped surface, thickness, manufactured focal
intensity, target intensity, physical focal coordinates, convergence history,
metadata, and the fully resolved deck. Metrics are calculated after
manufacturing projection and re-propagation. Multilevel runs can also set
`manufacturing.iterate_on_element_grid=True` so convergence can use square-
or hex-grid-constrained phase updates. By default the grid is active from the
first iteration. Set `manufacturing.element_grid_start_fraction` to delay its
activation while retaining the same fine numerical raster; for example, `0.5`
keeps the first half of GS on the fine phase grid and activates the physical
element geometry for the second half.

Each runnable deck enables a live GS progress bar and post-run plots by
default. `results_overview.png` follows the legacy 2-by-3 result layout.
That main overview contains the pre-manufacturing GS solution: the ideal
fine-grid design normally, or the element-grid-constrained design when
iterative grid projection is enabled.
For square or hexagonal manufacturing, the mode-specific comparison PNG shows
the fine-grid design beside the clean physical element realization. Its
filename includes both the canonical mode and grid geometry.

Facility decks are starting points rather than facility-approved shot
specifications. Check all optical and material values with the relevant
campaign owner before manufacturing or experimental use.
