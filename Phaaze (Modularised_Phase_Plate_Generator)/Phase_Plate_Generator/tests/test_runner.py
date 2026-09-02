from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from phase_plate_generator.runner import (
    create_timestamped_run_directory,
)


PROJECT = Path(__file__).resolve().parents[1]
INPUT_DECKS = PROJECT.parent / "Input_Decks"
QUICK_SCRIPT = INPUT_DECKS / "examples" / "quick_demo.py"
QUICK_DECK_NAMESPACE = runpy.run_path(
    str(QUICK_SCRIPT), run_name="input_deck_test"
)
run_quick_deck = QUICK_DECK_NAMESPACE["main"]

RUN_TIME = datetime(
    2026,
    7,
    25,
    18,
    42,
    10,
    tzinfo=timezone(timedelta(hours=1)),
)


class RunnerTests(unittest.TestCase):
    def test_timestamped_directory_handles_same_second_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = create_timestamped_run_directory(
                QUICK_SCRIPT, temporary, started_at=RUN_TIME
            )
            second = create_timestamped_run_directory(
                QUICK_SCRIPT, temporary, started_at=RUN_TIME
            )
            self.assertEqual(
                first.name, "quick_demo_2026-07-25_18-42-10"
            )
            self.assertEqual(
                second.name, "quick_demo_2026-07-25_18-42-10_02"
            )

    def test_deck_run_saves_traceable_timestamped_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_quick_deck(
                outputs_root=temporary,
                started_at=RUN_TIME,
                show_plots=False,
                show_progress=False,
            )
            output = result.output_directory
            self.assertIsNotNone(output)
            self.assertEqual(
                output.name, "quick_demo_2026-07-25_18-42-10"
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"plots", "json_files", "Arrays", "Input deck used"},
            )
            json_directory = output / "json_files"
            arrays_directory = output / "Arrays"
            plots_directory = output / "plots"
            deck_directory = output / "Input deck used"
            expected_json_files = {
                "resolved_input_deck.json",
                "metadata.json",
                "convergence.json",
            }
            expected_array_files = {
                "Input_beam.npy",
                "Input_beam_pupil_mask.npy",
                "Target_focal_spot_structure.npy",
                "Focal_spot_coordinates_um.npz",
                "Focal_spot_extent_um.npy",
                "Ideal_multilevel_Kinoform.npy",
                "Ideal_multilevel_Kinoform_thickness_m.npy",
                "Ideal_multilevel_Kinoform_focal_spot.npy",
                "Square_grid_multilevel.npy",
                "Square_grid_multilevel_thickness_m.npy",
                "Square_grid_multilevel_focal_spot.npy",
                "Square_grid_multilevel_element_labels.npy",
            }
            expected_plot_files = {
                "convergence.png",
                "results_overview.png",
                (
                    "Ideal_multilevel_Kinoform_vs_"
                    "Square_grid_multilevel_comparison.png"
                ),
            }
            self.assertTrue(
                expected_json_files.issubset(
                    {path.name for path in json_directory.iterdir()}
                )
            )
            self.assertTrue(
                expected_array_files.issubset(
                    {path.name for path in arrays_directory.iterdir()}
                )
            )
            self.assertTrue(
                expected_plot_files.issubset(
                    {path.name for path in plots_directory.iterdir()}
                )
            )
            self.assertEqual(
                {path.name for path in deck_directory.iterdir()},
                {"quick_demo.py"},
            )
            metadata = json.loads(
                (json_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["run"]["input_deck_name"], "quick_demo"
            )
            self.assertEqual(
                metadata["run"]["started_at"],
                "2026-07-25T18:42:10+01:00",
            )
            self.assertEqual(
                Path(metadata["run"]["launcher_path"]), QUICK_SCRIPT
            )
            resolved = json.loads(
                (json_directory / "resolved_input_deck.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                Path(resolved["output"]["directory"]), output
            )


if __name__ == "__main__":
    unittest.main()
