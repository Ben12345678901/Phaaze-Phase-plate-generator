# `modes/distributed_relief.py`

[View source](../../../src/phase_plate_generator/modes/distributed_relief.py)

## Purpose

`DistributedReliefMode` extends continuous relief with a statistical
projection. It keeps the phase RMS and Gaussian autocorrelation scale near
configured values during every GS cycle.

## Projection recipe

After choosing the branch nearest the previous surface:

1. Gaussian-filter the lifted surface inside the pupil.
2. Rescale the filtered surface to `phase_rms_rad`.
3. Blend lifted and correlated surfaces:

   ```text
   S_blend = (1-beta) S_lift + beta S_correlated,
   ```

   where `beta = distributed_projection_strength`.

4. Rescale the blend once more to the requested RMS.
5. Return its wrapped phasor angle to GS.

At `beta=1`, the update is fully projected onto the correlated surface. At
`beta=0`, correlation filtering does not affect the shape, but the final RMS
normalisation still applies. The target constraint is introduced
adiabatically because the mode inherits `requires_adiabatic_target=True`.

## Relation to distributed phase plates

[Lin, Kessler and Lawrence (1995)](https://doi.org/10.1364/OL.20.000764)
describe starting from a strictly continuous DPP and applying a small number
of phase-retrieval cycles to improve a fourth-order super-Gaussian envelope
while reducing scattering associated with phase discontinuities. The present
mode shares the continuous starting surface and constrained retrieval goal.

Re-imposing a Gaussian correlation transfer function and exact RMS on every
cycle is a project-specific extension inherited from the source generator.
The accessible paper metadata does not establish this exact projector. Its
parameters should therefore be calibrated against measured or published DPP
statistics before treating it as a facility model.

## Physical interpretation

The correlation length controls the lateral feature scale of the continuous
random relief. RMS phase controls modulation strength. Both influence the
angular/focal spread, but there is no one-to-one analytic guarantee that the
requested pair produces a particular target diameter once GS and a finite
pupil are included.

## Limitations

- Gaussian autocorrelation is an imposed model, not measured surface
  metrology.
- Filtering is performed on a periodic FFT grid with aperture
  renormalisation.
- No surface slope, tool radius or power spectral density tolerance is
  enforced.
- Strong projection can compete with focal-target fidelity; compare the
  recorded PCC, NRMSE and measured correlation FWHM.

## Reference

- Y. Lin, T. J. Kessler and G. N. Lawrence, "Distributed phase plates for
  super-Gaussian focal-plane irradiance profiles," *Optics Letters* **20**,
  764-766 (1995),
  [doi:10.1364/OL.20.000764](https://doi.org/10.1364/OL.20.000764).

