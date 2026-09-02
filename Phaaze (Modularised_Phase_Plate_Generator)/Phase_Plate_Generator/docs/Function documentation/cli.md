# `cli.py`

[View source](../../src/phase_plate_generator/cli.py)

## Purpose

This is the legacy JSON command-line interface. It is not used when a
self-contained Python input deck is opened and run in VS Code.

`build_parser()` defines:

- an optional JSON deck path;
- `--validate-only`, which resolves and prints the deck without generating;
- `--no-save`, which returns an in-memory result without artifacts; and
- `--list-modes`, which prints canonical mode names.

`main()` parses arguments, loads the deck, selects the requested action, and
prints final PCC and relative NRMSE. A normal run delegates timestamped output
creation to `runner.run_deck_file`.

## Numerical and physical content

None is implemented locally. The metrics displayed here are computed in
`optics.focal_metrics`, and all propagation occurs through `generator.py`.

## Reference

- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)

