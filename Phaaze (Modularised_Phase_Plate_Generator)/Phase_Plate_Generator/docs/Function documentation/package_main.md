# `phase_plate_generator/__main__.py`

[View source](../../src/phase_plate_generator/__main__.py)

## Purpose and execution

This four-line adapter makes the installed package runnable as:

```text
python -m phase_plate_generator ...
```

It imports `cli.main()`, uses the returned integer as the process exit status,
and raises `SystemExit`. It contains no optics or numerical calculation. The
preferred project workflow is the self-contained Python input deck, so this
entry point mainly preserves command-line compatibility for legacy JSON decks.

## Reference

- [Python documentation: `__main__`](https://docs.python.org/3/library/__main__.html)

