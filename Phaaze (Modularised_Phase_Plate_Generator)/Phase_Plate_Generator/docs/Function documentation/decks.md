# `decks.py`

[View source](../../src/phase_plate_generator/decks.py)

## Purpose

This module converts an input mapping into the typed dataclasses in
`config.py`. It also retains support for inherited legacy JSON decks.

## Data recipe

`deck_from_mapping()`:

1. chooses the Python deck file as `source_path`;
2. resolves declared array, initial-phase and output paths relative to that
   file, not relative to VS Code's current working directory;
3. rejects unknown top-level sections and unknown keys inside each dataclass;
4. constructs `ExperimentDeck`; and
5. calls `validate()`.

This path rule is what lets a deck be opened and run directly from VS Code
without relying on a particular terminal directory.

For old JSON files, `_read_with_inheritance()` follows an `extends` key,
detects cycles, resolves paths at the file that declared them, and recursively
merges child dictionaries over their parents. `load_deck()` then feeds the
resolved mapping through the same typed construction path.

## Function map

- `_deep_merge()` recursively overlays nested dictionaries.
- `_resolve_declared_paths()` resolves only fields explicitly declared by the
  current mapping.
- `_read_with_inheritance()` loads the legacy parent chain with cycle
  detection.
- `_construct()` rejects unexpected dataclass keys before construction.
- `_resolve_optional_path()` converts an optional relative string to a path.
- `deck_from_mapping()` is the Python `INPUTS` entry point.
- `load_deck()` is the legacy JSON entry point.

## Numerical and physical content

There is no optical calculation here. Its scientific importance is
reproducibility: explicit typing, early validation and source-relative file
resolution prevent a run from silently using the wrong beam, target or phase
array.

## Limitations

- Python decks do not use the JSON inheritance mechanism.
- A Python deck is executable code and should only be run when trusted.
- Deep merge applies only to dictionaries; a child scalar or list replaces
  the parent value.

## References

- [Python `pathlib` documentation](https://docs.python.org/3/library/pathlib.html)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
