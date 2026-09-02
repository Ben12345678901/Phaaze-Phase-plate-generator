# `output_layout.py`

[View source](../../src/phase_plate_generator/output_layout.py)

## Purpose

`RunFolders` is an immutable record of the four artifact groups for one run.
`ensure_run_folders()` resolves the run root and creates:

```text
<run>/
|-- plots/
|-- json_files/
|-- Arrays/
`-- Input deck used/
```

Repeated calls are safe because the root uses `parents=True, exist_ok=True`
and child folders use `exist_ok=True`. The timestamped run directory itself is
reserved separately by `runner.py`.

## Numerical and physical content

None. This module exists so plotting, array saving and runner source capture
cannot drift onto different directory conventions.

## Reference

- [Python `pathlib` documentation](https://docs.python.org/3/library/pathlib.html)

