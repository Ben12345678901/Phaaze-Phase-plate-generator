# `algorithms/gerchberg_saxton.py`

[View source](../../../src/phase_plate_generator/algorithms/gerchberg_saxton.py)

## Purpose

This is the common alternating-projection engine for every phase
representation. It owns propagation and plane constraints; a `PhaseMode` owns
the permitted phase/surface representation after each inverse propagation.

`GSResult` contains the mode's final phase, the final complex focal field,
focal intensity and per-iteration history.

## Function map

- `_warmup_mask()` constructs the normalized circular Fourier-grid mask.
- `_prepare_initial_phase()` validates, centre-pads and wraps an imported seed.
- `_apply_phase_projector()` validates an optional spatial phase constraint.
- `gerchberg_saxton()` performs the full retrieval described below.

## Standard alternating-constraint recipe

Let `A_0` be the known input amplitude, `phi_n` the current plate phase and
`I_t` the peak-normalised target intensity. One normal iteration is:

```text
U_n       = A_0 exp(i phi_n)
V_n       = P_forward[U_n]
A_t       = sqrt(I_t)
V'_n      = A_next exp(i arg(V_n))
U'_n      = P_inverse[V'_n]
phi_(n+1) = mode.project(arg(U'_n)).
```

An optional spatial phase projector is applied to the initialized phase and
to `arg(U'_n)` before `mode.project()`. The generator uses this hook for
multilevel element-grid iterations: phase is averaged per square or hexagonal
cell, then the multilevel quantization rule is applied.

Restoring `A_0` before every forward propagation and replacing focal
amplitude while retaining focal phase is the Gerchberg-Saxton
alternating-constraint structure
[Gerchberg and Saxton (1972)](https://cir.nii.ac.jp/crid/1571980075954436992).
Its use for phase-only focal irradiance control follows
[Dixit et al. (1994)](https://doi.org/10.1364/OL.19.000417).

The square root is essential: the requested input is intensity, while the
propagator acts on complex amplitude.

## Finite constraint region and power scaling

When the constraint mask covers only part of the focal plane, the target
amplitude inside it is scaled by

```text
s = sqrt(sum_mask |V_n|^2 / sum_mask A_t^2).
```

The code replaces amplitude inside the mask by `s A_t` and leaves the current
amplitude outside. This avoids forcing the unconstrained outer field to zero.
When the mask covers the complete grid in an ordinary wrapped/stepped mode,
the unscaled `A_t` is imposed.

## Continuous-mode target ramp

For modes marked `requires_adiabatic_target`, and when `target_ramp=True`, the
constrained amplitude at iteration `n` is:

```text
A_next = (1-alpha)|V_n| + alpha s A_t,
alpha = (n+1)/N.
```

This gradually introduces the focal constraint so the continuous sheet is not
asked to jump immediately to a difficult target. It implements the
continuity-preserving design principle associated with
[Dixit et al. (1996)](https://doi.org/10.1364/OL.21.001715), but the exact
linear schedule is a project-specific choice.

## Hard central warm-up

For an initial fraction of iterations in wrapped and stepped modes, the
constrained far field is multiplied by a circular low-frequency mask whose
normalised array radius is at most `warmup_cutoff_fraction`. It is disabled for
continuous relief because hard zeros can create phase singularities. This
warm-up is inherited engineering logic, not part of the original GS
publication.

## Initial phase and non-iterative modes

An imported phase is centre-padded if necessary and reduced to its principal
phasor angle. The mode then converts it to its own internal state. Otherwise
the mode supplies its initializer.

If `mode.skip_phase_retrieval` is true, the initialized plate is finalised and
propagated once. This is how a genuinely random binary phase plate remains
separate from a target-optimised two-level kinoform.

## Convergence records

After each forward propagation, the current peak-normalised intensity is
compared with the target inside the constraint mask. The record stores:

- PCC;
- relative NRMSE; and
- RMS of the wrapped phase change inside the pupil,
  `sqrt(mean(angle(exp(i(phi_new-phi_old)))^2))`.

If the element-grid projector is active, that forward field is produced by
the grid-constrained phase, so PCC and relative NRMSE are the element-grid
iteration metrics.

The metric is evaluated from the forward field at the start of that iteration,
before its newly constrained inverse update. A final result is propagated
again after `mode.finalize()`, so final exported metrics are calculated later
from the actual final array.

## Assumptions and limitations

- GS is an alternating projection, not a global optimiser; results depend on
  initialization, target feasibility, sampling and mode constraints.
- No monotonic decrease of NRMSE is guaranteed.
- The implementation fixes near-field amplitude, so it cannot optimize an
  amplitude-modulating optic.
- Peak-normalised errors do not measure absolute diffraction efficiency.
- A finite constraint mask can move unwanted energy into the unconstrained
  region.

## References

- R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the
  determination of phase from image and diffraction plane pictures,"
  *Optik* **35**, 237-246 (1972),
  [bibliographic record](https://cir.nii.ac.jp/crid/1571980075954436992).
- S. N. Dixit et al., "Kinoform phase plates for focal plane irradiance
  profile control," *Optics Letters* **19**, 417-419 (1994),
  [doi:10.1364/OL.19.000417](https://doi.org/10.1364/OL.19.000417).
- S. N. Dixit et al., "Designing fully continuous phase screens for tailoring
  focal-plane irradiance profiles," *Optics Letters* **21**, 1715-1717 (1996),
  [doi:10.1364/OL.21.001715](https://doi.org/10.1364/OL.21.001715).
