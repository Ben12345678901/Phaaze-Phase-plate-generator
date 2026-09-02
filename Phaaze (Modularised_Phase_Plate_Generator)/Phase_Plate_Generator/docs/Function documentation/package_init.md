# `phase_plate_generator/__init__.py`

[View source](../../src/phase_plate_generator/__init__.py)

## Purpose

This is the package's small public facade. It re-exports the objects needed by
an input deck:

| Export | Role |
|---|---|
| `ExperimentDeck` | Typed top-level configuration |
| `GenerationResult` | In-memory result arrays and metadata |
| `deck_from_mapping` | Convert an `INPUTS` dictionary into a validated deck |
| `generate` | Run the numerical workflow |
| `load_deck` | Load a legacy JSON deck |
| `run_configured_deck` | Run a Python deck into a timestamped folder |
| `run_deck_file` | Run a legacy JSON deck |

It also declares package version `0.1.0`. There is no numerical recipe or
optical model in this file; it deliberately prevents input decks from needing
to know the internal module layout.

## Design note

`__all__` is an API declaration, not access control. Internal modules remain
importable, but code outside the package should prefer these names so that
future refactoring does not require every deck to change.

## Reference

- [Python tutorial: packages](https://docs.python.org/3/tutorial/modules.html#packages)

