# `runner.py`

[View source](../../src/phase_plate_generator/runner.py)

## Purpose

This module gives every execution a unique, auditable output folder and
preserves the input source used for the run.

`_safe_deck_name()` removes characters unsuitable for a directory name.
`create_timestamped_run_directory()` reserves the unique folder,
`_run_loaded_deck()` carries out the shared run, and the two public entry
points select Python or legacy JSON input.

## Run recipe

`create_timestamped_run_directory()` sanitises the deck filename and creates:

```text
<deck-name>_YYYY-MM-DD_HH-MM-SS
```

under the configured outputs root. If that name already exists, `_02`, `_03`
and so on are tried using atomic `mkdir(exist_ok=False)`.

`_run_loaded_deck()` then:

1. creates the standard artifact subfolders;
2. updates the deck's output directory to the reserved run directory;
3. copies each unique deck/launcher source into `Input deck used`;
4. calls `generate(save=False)`;
5. adds run provenance to metadata;
6. saves arrays and JSON;
7. creates plots when requested; or
8. writes `json_files/run_failed.json` before re-raising any exception.

`run_configured_deck()` is the public path for a self-contained `.py` input
deck. `run_deck_file()` is the compatibility path for JSON.

## Reproducibility boundary

The preserved deck, resolved deck and random seed make a run traceable, but
bit-for-bit reproducibility also depends on the package source, package
version, Python/NumPy versions and FFT implementation. The current runner
records the package version but does not snapshot the complete source tree or
environment lock file.

## Safety and failure behaviour

The run directory is reserved before the calculation, so a failed run does not
overwrite a previous success. A failure record contains exception type and
message but not a full traceback. Source copies are ordinary files, not
cryptographic integrity records.

## References

- [Python `datetime` documentation](https://docs.python.org/3/library/datetime.html)
- [Python `shutil.copy2` documentation](https://docs.python.org/3/library/shutil.html#shutil.copy2)
