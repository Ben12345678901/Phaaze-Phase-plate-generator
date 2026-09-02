# `algorithms/__init__.py`

[View source](../../../src/phase_plate_generator/algorithms/__init__.py)

## Purpose

This package initializer exposes only `GSResult` and `gerchberg_saxton` from
the common retrieval engine. It provides a stable import boundary so
`generator.py` does not depend on the engine's physical filename.

There is no numerical calculation in this file. New retrieval algorithms
should live beside `gerchberg_saxton.py` and be explicitly exported here only
after they have a clear result contract.

## Reference

- R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the
  determination of phase from image and diffraction plane pictures,"
  *Optik* **35**, 237-246 (1972),
  [bibliographic record](https://cir.nii.ac.jp/crid/1571980075954436992).

