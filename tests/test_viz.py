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
            while time.time() < deadline:
                snap = session.feed.snapshot()
                if snap["pupil"] and len(snap["breath"]) >= 8 and snap["eeg"]:
                    break
                time.sleep(0.05)
            window = Monitor(session.feed)
            app.processEvents()
            window.refresh()
            pupil_x = list(window.eye0.getData()[0])
            self.assertTrue(pupil_x)
            self.assertGreater(min(pupil_x), 1_000_000_000)
            point_x = list(window.breath_point.getData()[0])
            self.assertEqual(len(point_x), 1)
            self.assertEqual(len(window._seen), 1)
            self.assertIn("Pupil diameter", window.pupil_plot.getPlotItem().titleLabel.text)
            self.assertNotIn("gaze", window.pupil_plot.getPlotItem().titleLabel.text.lower())
            self.assertIn("phase", window.polar_plot.getPlotItem().titleLabel.text.lower())
            self.assertTrue(window.channel_boxes["AF3"].isChecked())
            window.channel_boxes["AF3"].setChecked(False)
            window.refresh()
            hidden = window.eeg_curves["AF3"].getData()[0]
            self.assertTrue(hidden is None or len(list(hidden)) == 0)
            self.assertIn("breath", window._panels)
            self.assertIn("pupil", window._panels)
            self.assertIn("eeg", window._panels)
            window.close()
        finally:
            session.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
