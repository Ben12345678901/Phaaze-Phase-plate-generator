import unittest

import numpy as np

from phase_plate_generator.targets import build_target_intensity


class TargetTests(unittest.TestCase):
    def test_striped_tophat_has_exact_bright_band_count(self):
        axis = np.linspace(-1e-3, 1e-3, 1001)
        X, Y = np.meshgrid(axis, axis, indexing="xy")
        target = build_target_intensity(
            X,
            Y,
            profile="striped_tophat",
            diameter_m=1e-3,
            supergaussian_order=5,
            stripes=7,
            modulation_depth=1,
            stripe_axis="x",
            array_pixel_size_m=None,
        )
        row = target[target.shape[0] // 2] > 0.5
        starts = np.flatnonzero(row & ~np.roll(row, 1))
        self.assertEqual(len(starts), 7)
        self.assertEqual(target.max(), 1)
        self.assertEqual(target[0, 0], 0)

    def test_supergaussian_is_radially_symmetric(self):
        axis = np.linspace(-1e-3, 1e-3, 129)
        X, Y = np.meshgrid(axis, axis, indexing="xy")
        target = build_target_intensity(
            X,
            Y,
            profile="supergaussian",
            diameter_m=1e-3,
            supergaussian_order=8,
            stripes=7,
            modulation_depth=1,
            stripe_axis="x",
            array_pixel_size_m=None,
        )
        np.testing.assert_allclose(target, target.T)


if __name__ == "__main__":
    unittest.main()
