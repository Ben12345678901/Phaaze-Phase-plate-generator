import unittest

import numpy as np

from phase_plate_generator.config import PhasePlateConfig
from phase_plate_generator.modes import build_mode, normalise_mode_name
from phase_plate_generator.modes.multilevel import quantize_phase


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.pupil = np.ones((32, 32), dtype=bool)
        self.rng = np.random.default_rng(3)

    def test_legacy_names_have_unambiguous_canonical_names(self):
        self.assertEqual(
            normalise_mode_name("continuous"), "continuous_wrapped"
        )
        self.assertEqual(
            normalise_mode_name("true continuous"), "continuous_relief"
        )
        self.assertEqual(
            normalise_mode_name("distributed"), "distributed_relief"
        )
        self.assertEqual(normalise_mode_name("quantized"), "multilevel")
        self.assertEqual(normalise_mode_name("random"), "random_binary")

    def test_multilevel_finalize_has_requested_levels(self):
        mode = build_mode(
            PhasePlateConfig(mode="multilevel", levels=4),
            pupil=self.pupil,
            pixel_size_m=1e-4,
            rng=self.rng,
            default_correlation_length_m=1e-3,
        )
        candidate = self.rng.uniform(-np.pi, np.pi, self.pupil.shape)
        result = mode.finalize(candidate)
        self.assertLessEqual(
            len(np.unique(np.round(result.wrapped_phase_rad, 12))), 4
        )

    def test_median_binarization_balances_valid_area(self):
        phase = np.linspace(-np.pi, np.pi, 20).reshape(4, 5)
        pupil = np.ones(phase.shape, dtype=bool)
        pupil[0, 0] = False
        pupil[-1, -1] = False

        binary = quantize_phase(
            phase,
            2,
            binarization_method="median",
            valid_mask=pupil,
        )

        pi_samples = np.isclose(binary[pupil], np.pi)
        self.assertEqual(int(np.count_nonzero(pi_samples)), pupil.sum() // 2)
        self.assertTrue(np.all(binary[~pupil] == 0))

    def test_amplitude_weighted_binarization_balances_field_weight(self):
        phase = np.array([[0.1, 3.0, 2.0, 1.0]])
        amplitude = np.array([[8.0, 4.0, 3.0, 1.0]])

        binary = quantize_phase(
            phase,
            2,
            binarization_method="amplitude_weighted",
            amplitude_weights=amplitude,
        )

        pi_weight = float(amplitude[np.isclose(binary, np.pi)].sum())
        self.assertEqual(pi_weight, float(amplitude.sum() / 2))
        self.assertEqual(int(np.count_nonzero(np.isclose(binary, np.pi))), 3)

    def test_distributed_projection_preserves_required_rms(self):
        required_rms = 1.7
        mode = build_mode(
            PhasePlateConfig(
                mode="distributed_relief",
                correlation_length_m=8e-4,
                phase_rms_rad=required_rms,
                distributed_projection_strength=1,
            ),
            pupil=self.pupil,
            pixel_size_m=1e-4,
            rng=self.rng,
            default_correlation_length_m=1e-3,
        )
        initial = mode.initialize()
        projected = mode.project(initial + 0.1, 0, 2)
        result = mode.finalize(projected)
        self.assertIsNotNone(result.unwrapped_phase_rad)
        self.assertAlmostEqual(
            float(np.std(result.unwrapped_phase_rad[self.pupil])),
            required_rms,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
