from pathlib import Path
import runpy
import unittest

from phase_plate_generator.decks import deck_from_mapping


PROJECT = Path(__file__).resolve().parents[1]
INPUT_DECKS = PROJECT.parent / "Input_Decks"


def load_python_deck(path: Path):
    namespace = runpy.run_path(str(path), run_name="input_deck_test")
    return deck_from_mapping(namespace["INPUTS"], source_path=path)


class DeckTests(unittest.TestCase):
    def test_omega_example_contains_complete_configuration(self):
        deck = load_python_deck(
            INPUT_DECKS
            / "examples"
            / "omega_distributed_800um.py"
        )
        self.assertEqual(deck.facility.name, "OMEGA")
        self.assertEqual(deck.optics.wavelength_m, 351e-9)
        self.assertEqual(deck.phase_plate.mode, "distributed_relief")
        self.assertEqual(deck.target.diameter_m, 800e-6)
        self.assertTrue(deck.output.directory.is_absolute())

    def test_gsi_migrates_striped_target(self):
        deck = load_python_deck(
            INPUT_DECKS / "facilities" / "gsi.py"
        )
        self.assertEqual(deck.grid.aperture_shape, "square")
        self.assertEqual(deck.beam.profile, "square_supergaussian")
        self.assertEqual(deck.target.profile, "striped_tophat")
        self.assertEqual(deck.target.stripes, 7)
        self.assertEqual(deck.phase_plate.levels, 2)
        self.assertTrue(deck.manufacturing.iterate_on_element_grid)
        self.assertEqual(
            deck.manufacturing.element_grid_start_fraction, 0.0
        )


if __name__ == "__main__":
    unittest.main()
