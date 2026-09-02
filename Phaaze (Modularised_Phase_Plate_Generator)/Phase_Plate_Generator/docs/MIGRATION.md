# Migration from the two source generators

| Legacy feature | New location |
|---|---|
| GS loop in both scripts | `algorithms/gerchberg_saxton.py` |
| Fraunhofer/Fresnel helpers | `optics.py` |
| Circular/square incident beams | `beams.py` |
| Super-Gaussian and striped targets | `targets.py` |
| Old `continuous` | `modes/continuous_wrapped.py` |
| Old `true continuous` | `modes/continuous_relief.py` |
| Old `distributed` / DPP | `modes/distributed_relief.py` |
| Old `quantized` | `modes/multilevel.py` |
| Explicit random RPP | `modes/random_binary.py` |
| Square/hexagonal element projection | `manufacturing.py` |
| Editable control panels | Self-contained Python files in `../Input_Decks/` |
| Direct IDE launchers | Each deck's own `.py` file |
| Timestamped run folders | `runner.py` |
| Array/metadata export | `outputs.py` |
| Per-run artifact subfolders | `output_layout.py` |
| GS percentage/progress output | `progress.py` |
| Legacy result and convergence figures | `plotting.py` |
| Square/hex physical comparison | `plotting.py` using manufacturing labels |

The original files in `../Source_Files` remain unchanged as numerical
references. The new package is the maintained path. Plot-heavy exploratory
helpers from the legacy scripts were not copied into the computational core;
their underlying arrays and metrics are exported under stable names so
plotting can be a separate analysis layer.
