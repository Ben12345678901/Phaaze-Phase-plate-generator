from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np

from phase_plate_generator.config import ExperimentDeck
from phase_plate_generator.decks import deck_from_mapping
from phase_plate_generator.generator import generate
from phase_plate_generator.manufacturing import ElementGrid


PROJECT = Path(__file__).resolve().parents[1]
INPUT_DECKS = PROJECT.parent / "Input_Decks"


class GeneratorTests(unittest.TestCase):
    def test_quick_deck_runs_and_exports_binary_plate_in_memory(self):
        path = INPUT_DECKS / "examples" / "quick_demo.py"
        namespace = runpy.run_path(
            str(path), run_name="input_deck_test"
        )
        deck = deck_from_mapping(namespace["INPUTS"], source_path=path)
        deck.algorithm.show_progress = False
        result = generate(deck, save=False)
        self.assertEqual(result.wrapped_phase_rad.shape, (64, 64))
        self.assertEqual(result.focal_intensity.shape, (64, 64))
        self.assertEqual(len(result.history), 8)
        unique = np.unique(np.round(result.wrapped_phase_rad, 12))
        self.assertLessEqual(len(unique), 2)
        self.assertTrue(np.isfinite(result.focal_intensity).all())
        self.assertIn("pcc", result.metadata["metrics"])
        self.assertEqual(
            result.pre_manufacturing_phase_rad.shape, (64, 64)
        )
        self.assertEqual(
            result.pre_manufacturing_thickness_m.shape, (64, 64)
        )
        self.assertTrue(
            np.any(result.manufacturing_element_labels >= 0)
        )

    def test_random_mode_skips_gs(self):
        deck = ExperimentDeck()
        deck.grid.plate_pixels = 32
        deck.grid.maximum_propagation_samples = 64
        deck.algorithm.iterations = 5
        deck.phase_plate.mode = "random_binary"
        deck.manufacturing.element_geometry = "none"
        deck.output.save_arrays = False
        deck.output.save_metadata = False
        result = generate(deck, save=False)
        self.assertEqual(result.history, [])
        self.assertLessEqual(
            len(np.unique(np.round(result.wrapped_phase_rad, 12))), 2
        )

    def test_continuous_relief_exports_unwrapped_surface(self):
        deck = ExperimentDeck()
        deck.grid.plate_size_m = 0.01
        deck.grid.plate_pixels = 32
        deck.grid.maximum_propagation_samples = 64
        deck.algorithm.iterations = 3
        deck.phase_plate.mode = "continuous_relief"
        deck.phase_plate.correlation_length_m = 0.002
        deck.manufacturing.element_geometry = "none"
        deck.output.save_arrays = False
        deck.output.save_metadata = False
        result = generate(deck, save=False)
        self.assertIsNotNone(result.unwrapped_phase_rad)
        self.assertTrue(np.isfinite(result.thickness_m).all())
        self.assertGreater(
            np.ptp(result.unwrapped_phase_rad[result.pupil]), 0
        )

    def test_multilevel_can_iterate_on_hexagonal_element_grid(self):
        deck = ExperimentDeck()
        deck.grid.plate_size_m = 0.01
        deck.grid.plate_pixels = 32
        deck.grid.maximum_propagation_samples = 64
        deck.algorithm.iterations = 4
        deck.phase_plate.mode = "multilevel"
        deck.phase_plate.levels = 4
        deck.manufacturing.element_geometry = "hexagonal"
        deck.manufacturing.element_pitch_m = 0.001
        deck.manufacturing.iterate_on_element_grid = True
        deck.output.save_arrays = False
        deck.output.save_metadata = False

        result = generate(deck, save=False)

        self.assertEqual(len(result.history), 4)
        self.assertTrue(
            all(
                np.isfinite(item[metric])
                for item in result.history
                for metric in ("pcc", "relative_nrmse")
            )
        )
        self.assertTrue(
            result.metadata["manufacturing"][
                "iterate_on_element_grid"
            ]
        )
        labels = result.manufacturing_element_labels
        for label in np.unique(labels[labels >= 0]):
            cell_phase = result.pre_manufacturing_phase_rad[
                labels == label
            ]
            self.assertEqual(
                len(np.unique(np.round(cell_phase, 12))), 1
            )

    def test_element_grid_can_start_partway_through_iterations(self):
        deck = ExperimentDeck()
        deck.grid.plate_size_m = 0.01
        deck.grid.plate_pixels = 32
        deck.grid.maximum_propagation_samples = 64
        deck.algorithm.iterations = 4
        deck.phase_plate.mode = "multilevel"
        deck.phase_plate.levels = 4
        deck.manufacturing.element_geometry = "hexagonal"
        deck.manufacturing.element_pitch_m = 0.001
        deck.manufacturing.iterate_on_element_grid = True
        deck.manufacturing.element_grid_start_fraction = 0.5
        deck.output.save_arrays = False
        deck.output.save_metadata = False

        calls = 0
        original = ElementGrid.project_wrapped

        def record_projection(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return original(self, *args, **kwargs)

        with patch.object(
            ElementGrid, "project_wrapped", record_projection
        ):
            result = generate(deck, save=False)

        # Two late GS projections plus the unchanged final manufacturing pass.
        self.assertEqual(calls, 3)
        self.assertEqual(
            result.pre_manufacturing_phase_rad.shape, (32, 32)
        )
        self.assertEqual(result.wrapped_phase_rad.shape, (32, 32))
        self.assertEqual(
            result.metadata["manufacturing"][
                "element_grid_start_fraction"
            ],
            0.5,
        )
        self.assertEqual(
            result.metadata["manufacturing"][
                "element_grid_start_iteration"
            ],
            3,
        )

    def test_element_grid_iteration_requires_multilevel_grid(self):
        deck = ExperimentDeck()
        deck.phase_plate.mode = "multilevel"
        deck.manufacturing.iterate_on_element_grid = True
        with self.assertRaisesRegex(ValueError, "element geometry"):
            deck.validate()

        deck.manufacturing.element_geometry = "hexagonal"
        deck.phase_plate.mode = "continuous_relief"
        with self.assertRaisesRegex(ValueError, "multilevel"):
            deck.validate()

        deck.phase_plate.mode = "multilevel"
        deck.manufacturing.element_grid_start_fraction = 1.1
        with self.assertRaisesRegex(ValueError, "start_fraction"):
            deck.validate()

    def test_amplitude_weighted_binary_grid_balances_whole_elements(self):
        deck = ExperimentDeck()
        deck.grid.plate_size_m = 0.01
        deck.grid.plate_pixels = 32
        deck.grid.maximum_propagation_samples = 64
        deck.algorithm.iterations = 4
        deck.phase_plate.mode = "multilevel"
        deck.phase_plate.levels = 2
        deck.phase_plate.binarization_method = "amplitude_weighted"
        deck.manufacturing.element_geometry = "hexagonal"
        deck.manufacturing.element_pitch_m = 0.001
        deck.manufacturing.iterate_on_element_grid = True
        deck.output.save_arrays = False
        deck.output.save_metadata = False

        result = generate(deck, save=False)

        labels = result.manufacturing_element_labels
        phase = result.wrapped_phase_rad
        amplitude = result.input_amplitude
        valid = labels >= 0
        for label in np.unique(labels[valid]):
            self.assertEqual(
                len(np.unique(np.round(phase[labels == label], 12))), 1
            )
        pi_mask = valid & np.isclose(phase, np.pi)
        pi_weight = float(amplitude[pi_mask].sum())
        total_weight = float(amplitude[valid].sum())
        cell_weights = np.bincount(
            labels[valid], weights=amplitude[valid]
        )
        self.assertLessEqual(
            abs(pi_weight - total_weight / 2),
            float(cell_weights.max(initial=0) / 2) + 1e-12,
        )
        self.assertEqual(
            result.metadata["manufacturing"]["binarization_method"],
            "amplitude_weighted",
        )

    def test_balanced_binarization_requires_binary_multilevel_mode(self):
        deck = ExperimentDeck()
        deck.phase_plate.mode = "multilevel"
        deck.phase_plate.levels = 4
        deck.phase_plate.binarization_method = "median"
        with self.assertRaisesRegex(ValueError, "levels=2"):
            deck.validate()

        deck.phase_plate.levels = 2
        deck.phase_plate.mode = "continuous_wrapped"
        with self.assertRaisesRegex(ValueError, "multilevel"):
            deck.validate()


if __name__ == "__main__":
    unittest.main()
