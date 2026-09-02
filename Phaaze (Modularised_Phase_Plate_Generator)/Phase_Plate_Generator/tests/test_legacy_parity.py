"""Regression checks against the two untouched source generators."""

from pathlib import Path
import sys
import unittest

import numpy as np

from phase_plate_generator.modes.multilevel import quantize_phase
from phase_plate_generator.modes.surface import gaussian_transfer
from phase_plate_generator.optics import forward_for_gs, inverse_for_gs
from phase_plate_generator.targets import build_target_intensity


SOURCE_FILES = Path(__file__).resolve().parents[2] / "Source_Files"
sys.path.insert(0, str(SOURCE_FILES))

from Phase_plate_generator_MK2_hexagonal_tophat_7stripes import (  # noqa: E402
    generate_focal_spot_modulated_tophat_x,
)
from Phase_plate_generator_with_distributed_mode import (  # noqa: E402
    backpropagate_for_gs as legacy_inverse,
    build_isotropic_manufacturing_filter as legacy_filter,
    propagate_for_gs as legacy_forward,
    quantize_phase as legacy_quantize,
)


class LegacyParityTests(unittest.TestCase):
    def test_shared_numerical_primitives_match_exactly(self):
        rng = np.random.default_rng(17)
        field = rng.normal(size=(32, 32)) + 1j * rng.normal(
            size=(32, 32)
        )
        for model in ("fraunhofer", "fresnel"):
            with self.subTest(model=model):
                old = legacy_forward(
                    field, 1e-4, 351e-9, 1.8, model, 1.8
                )
                new = forward_for_gs(
                    field, 1e-4, 351e-9, 1.8, model, 1.8
                )
                np.testing.assert_array_equal(new, old)
                np.testing.assert_array_equal(
                    inverse_for_gs(
                        new, 1e-4, 351e-9, 1.8, model, 1.8
                    ),
                    legacy_inverse(
                        old, 1e-4, 351e-9, 1.8, model, 1.8
                    ),
                )

        np.testing.assert_array_equal(
            gaussian_transfer((32, 32), 2.3),
            legacy_filter((32, 32), 2.3),
        )
        phase = rng.uniform(-3 * np.pi, 3 * np.pi, size=(32, 32))
        np.testing.assert_array_equal(
            quantize_phase(phase, 5), legacy_quantize(phase, 5)
        )

    def test_seven_stripe_target_matches_exactly(self):
        legacy, x, y = generate_focal_spot_modulated_tophat_x(
            pixels=401,
            extent_m=2e-3,
            radius_m=0.5e-3,
            mod_amp=1,
            mod_period_m=1e-3 / 7,
            I_peak=1,
            num_stripes=7,
        )
        X, Y = np.meshgrid(x, y, indexing="xy")
        modular = build_target_intensity(
            X,
            Y,
            profile="striped_tophat",
            diameter_m=1e-3,
            supergaussian_order=5.2,
            stripes=7,
            modulation_depth=1,
            stripe_axis="x",
        )
        np.testing.assert_array_equal(modular, legacy)


if __name__ == "__main__":
    unittest.main()

