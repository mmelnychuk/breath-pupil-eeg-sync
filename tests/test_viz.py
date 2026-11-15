"""The live view shares one Unix-time axis. No display required."""

import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sniff_sync import trigger
from sniff_sync.simulate import Session


class ViewTests(unittest.TestCase):
    def test_window_marks_one_trigger_on_unix_time(self):
        from PySide6 import QtWidgets

        from sniff_sync.viz import Monitor

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tmp = tempfile.TemporaryDirectory()
        session = Session(tmp.name)
        session.start()
        try:
            trigger("stimulus_onset")
            self.assertTrue(session.wait_for_label("stimulus_onset"))
            deadline = time.time() + 2
            while time.time() < deadline and not session.feed.snapshot()["pupil"]:
                time.sleep(0.05)
            window = Monitor(session.feed)
            app.processEvents()
            window.refresh()
            pupil_x = list(window.eye0.getData()[0])
            breath_x = list(window.breath_curve.getData()[0])
            self.assertTrue(pupil_x)
            self.assertTrue(breath_x)
            self.assertGreater(min(pupil_x), 1_000_000_000)
            self.assertGreater(min(breath_x), 1_000_000_000)
            self.assertEqual(len(window._seen), 1)
            self.assertIn("Pupil diameter", window.pupil_plot.titleLabel.text)
            self.assertNotIn("gaze", window.pupil_plot.titleLabel.text.lower())
            window.close()
        finally:
            session.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
