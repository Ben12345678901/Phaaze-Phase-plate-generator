# `plotting.py`

[View source](../../src/phase_plate_generator/plotting.py)

## Purpose

This module turns a completed `GenerationResult` into diagnostic figures. It
does not recalculate the optical fields.

## Function map

- `_plate_extent_mm()` and `_focal_extent_um()` derive image extents.
- `_set_focal_window()` applies the configured display crop.
- `_save_figure()` centralises DPI and tight bounding-box saving.
- `_convergence_figure()`, `_overview_figure()` and
  `_manufacturing_comparison_figure()` create the three diagnostics.
- `plot_generation_result()` selects, saves, displays and closes them.

## Figures

`_convergence_figure()` plots PCC, relative NRMSE and

```text
100 * max(PCC, 0) * max(1 - NRMSE, 0).
```

The last curve is a project-specific visual score, not a standard statistical
accuracy measure and not the GS objective function.

`_overview_figure()` shows the ideal calculation only:

1. input beam amplitude;
2. target focal intensity;
3. log10 ideal focal intensity, clipped below `10^-8`;
4. linear ideal focal intensity and PCC;
5. ideal fine-grid wrapped phase; and
6. ideal thickness.

`_manufacturing_comparison_figure()` is produced only for square or hexagonal
elements. It compares the fine phase, grid phase, fine focal intensity and
re-propagated grid focal intensity. The grid is visible through the
piecewise-constant cell values; cell boundary outlines are intentionally not
overplotted.

`plot_generation_result()` chooses a non-interactive Matplotlib backend when
figures are only saved, creates the standard plot folder, writes the enabled
figures and optionally displays them.

## Coordinate and colour conventions

Plate axes are millimetres and focal axes are micrometres. Focal panels can be
cropped to `focal_plot_window_factor * target_diameter`; this changes only the
view, not the underlying result or metric domain. Wrapped phase is displayed
from zero to `2*pi`.

## Limitations

- Image interpolation performed by the display backend is visual only.
- The log panel clips low intensity and must not be used to measure dynamic
  range below `10^-8`.
- Plot labels report normalised intensity, not W/m2.
- The combined accuracy curve should not be compared with a published optical
  efficiency.

## References

- [Matplotlib `imshow` documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- [Matplotlib `savefig` documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
