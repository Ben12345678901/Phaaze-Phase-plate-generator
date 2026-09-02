# `progress.py`

[View source](../../src/phase_plate_generator/progress.py)

## Purpose

`TerminalProgress` renders an in-place progress bar for a GS run without an
extra dependency.

`start()` records a monotonic timer and displays zero progress. `update()`
prints approximately `updates` evenly spaced reports, always including the
last iteration. Each report includes percentage, iteration count, PCC,
relative NRMSE and elapsed seconds. `skipped()` explains that no iterative
retrieval is required, as in random binary mode.

## Numerical and physical content

The class does not calculate metrics; it displays the values supplied by the
GS engine. `time.perf_counter()` measures elapsed wall-clock duration but is
not a reproducible benchmark because machine load, FFT library and array size
all affect it.

## Reference

- [Python `time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)

