import unittest

import numpy as np

from phase_plate_generator.optics import (
    forward_for_gs,
    inverse_for_gs,
    normalise_intensity,
    propagate_phase_plate,
)


class OpticsTests(unittest.TestCase):
    def test_forward_inverse_round_trip(self):
        rng = np.random.default_rng(4)
        field = rng.normal(size=(32, 32)) + 1j * rng.normal(
            size=(32, 32)
        )
        for model in ("fraunhofer", "fresnel"):
            with self.subTest(model=model):
                propagated = forward_for_gs(
                    field,
                    1e-4,
                    532e-9,
                    0.5,
                    model,
                    0.5,
                )
                recovered = inverse_for_gs(
                    propagated,
                    1e-4,
                    532e-9,
                    0.5,
                    model,
                    0.5,
                )
                np.testing.assert_allclose(
                    recovered, field, rtol=1e-11, atol=1e-11
                )

    def test_fresnel_at_focus_matches_fraunhofer_intensity(self):
        rng = np.random.default_rng(5)
        field = np.exp(
            1j * rng.uniform(-np.pi, np.pi, size=(32, 32))
        )
        fraunhofer, *_ = propagate_phase_plate(
            field, 1e-4, 351e-9, 1.8, "fraunhofer", 1.8
        )
        fresnel, *_ = propagate_phase_plate(
            field, 1e-4, 351e-9, 1.8, "fresnel", 1.8
        )
        np.testing.assert_allclose(
            normalise_intensity(fresnel),
            normalise_intensity(fraunhofer),
            rtol=1e-11,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()

