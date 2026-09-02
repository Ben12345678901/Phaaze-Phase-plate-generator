from contextlib import redirect_stdout
import io
import unittest

from phase_plate_generator.progress import TerminalProgress


class ProgressTests(unittest.TestCase):
    def test_terminal_progress_reaches_one_hundred_percent(self):
        output = io.StringIO()
        progress = TerminalProgress(total=4, updates=2, width=10)
        with redirect_stdout(output):
            progress.start()
            for completed in range(1, 5):
                progress.update(
                    completed,
                    4,
                    {"pcc": 0.75, "relative_nrmse": 0.25},
                )
        rendered = output.getvalue()
        self.assertIn("100.0%", rendered)
        self.assertIn("(4/4)", rendered)
        self.assertIn("PCC 0.7500", rendered)
