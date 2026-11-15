"""Fake triggers, no headset and no glasses."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from sniff_sync import trigger
from sniff_sync.check import compare_session
from sniff_sync.diameter import sample_from_pupil_message
from sniff_sync.eeg import channels_from_packet, emokit_status
from sniff_sync.simulate import Session
from sniff_sync.timefmt import TIMESTAMP_FORMAT, unix_seconds


class Packet:
    def __init__(self, sensors):
        self.sensors = sensors
        self.counter = 3


class SyncTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("SNIFF_SYNC_PORT")
        self._tmp = tempfile.TemporaryDirectory()
        self.session = Session(self._tmp.name)
        self.session.start()

    def tearDown(self):
        self.session.stop()
        self._tmp.cleanup()
        if self._env is None:
            os.environ.pop("SNIFF_SYNC_PORT", None)
        else:
            os.environ["SNIFF_SYNC_PORT"] = self._env

    def test_one_trigger_matches_in_all_three_recordings(self):
        before = time.time()
        trigger("stimulus_onset")
        self.assertTrue(self.session.wait_for_label("stimulus_onset", timeout=3))
        result = compare_session(self.session.directory)
        self.assertTrue(result["ok"], result["problems"])
        self.assertEqual(result["compared"], 1)
        text = (self.session.directory / "breathing.csv").read_text()
        self.assertIn("stimulus_onset", text)
        self.assertIn("Unix Time", text)
        self.assertIn("Nasal Pressure", text)
        stamp = _trigger_stamp(self.session.directory / "breathing.csv")
        self.assertGreater(stamp, before - 1)
        self.assertLess(stamp, time.time() + 1)
        self.assertGreater(stamp, 1_000_000_000)

    def test_two_labels_stay_distinct(self):
        trigger("task_start")
        trigger("keypress")
        self.assertTrue(self.session.wait_for_label("keypress"))
        result = compare_session(self.session.directory)
        self.assertTrue(result["ok"], result["problems"])
        self.assertEqual(result["compared"], 2)

    def test_breathing_samples_are_not_triggers(self):
        time.sleep(0.3)
        result = compare_session(self.session.directory)
        self.assertFalse(result["ok"])
        self.assertEqual(result["compared"], 0)
        text = (self.session.directory / "breathing.csv").read_text()
        self.assertIn("Nasal Pressure", text)
        self.assertNotIn("stimulus_onset", text)

    def test_pupil_diameter_reaches_the_view_in_unix_time(self):
        deadline = time.time() + 2
        snap = {"pupil": []}
        while time.time() < deadline:
            snap = self.session.feed.snapshot()
            eyes = {eye for _t, eye, _v in snap["pupil"]}
            if eyes == {0, 1}:
                break
            time.sleep(0.05)
        self.assertEqual({eye for _t, eye, _v in snap["pupil"]}, {0, 1})
        for unix, _eye, diameter in snap["pupil"]:
            self.assertGreater(unix, 1_000_000_000)
            self.assertGreater(diameter, 2.0)
            self.assertLess(diameter, 8.0)

    def test_missing_hub_does_not_raise(self):
        os.environ["SNIFF_SYNC_PORT"] = "53921"
        trigger("nobody_home")


class ParserTests(unittest.TestCase):
    def test_diameter_prefers_millimetres_and_ignores_gaze(self):
        gaze = sample_from_pupil_message("gaze.3d.01.", {"topic": "gaze.3d.01.", "norm_pos": [0.2, 0.4], "timestamp": 1, "confidence": 1})
        self.assertIsNone(gaze)
        both = sample_from_pupil_message(
            "pupil.1.3d",
            {"id": 1, "topic": "pupil.1.3d", "diameter": 80.0, "diameter_3d": 4.1, "timestamp": 10, "confidence": 0.9},
        )
        self.assertEqual(both["unit"], "mm")
        self.assertEqual(both["diameter"], 4.1)
        self.assertEqual(both["eye"], 1)
        weak = sample_from_pupil_message("pupil.0.2d", {"id": 0, "diameter": 40, "timestamp": 1, "confidence": 0.1})
        self.assertIsNone(weak)

    def test_emokit_packet_shape(self):
        packet = Packet({"AF3": {"value": 110, "quality": 8}, "F7": {"value": 63, "quality": 1}, "X": {"value": 1}})
        channels = channels_from_packet(packet)
        self.assertEqual(channels["AF3"], 110)
        self.assertEqual(channels["F7"], 63)
        self.assertNotIn("X", channels)

    def test_emokit_import_is_reported(self):
        status = emokit_status()
        self.assertIn("EPOC", status["supported_models"][0])
        self.assertTrue(status["unsupported_models"])
        # pycrypto is not installed in this environment. The report must say so
        # rather than claiming a headset could be opened.
        self.assertIn("Crypto", status["import_error"])

    def test_timestamp_format_matches_the_plugin(self):
        import sniff_trigger_capture

        self.assertEqual(sniff_trigger_capture.TIMESTAMP_FORMAT, TIMESTAMP_FORMAT)
        self.assertEqual(sniff_trigger_capture.format_timestamp(1.25), format(1.25, TIMESTAMP_FORMAT))

    def test_polar_display_milliseconds_become_seconds(self):
        seconds = 1_700_000_000.5
        self.assertEqual(unix_seconds(seconds), seconds)
        self.assertEqual(unix_seconds(seconds * 1000.0), seconds)


def _trigger_stamp(path: Path) -> float:
    import csv

    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("Trigger") or "").strip():
                return float(row["Unix Time"])
    raise AssertionError("no trigger row")


if __name__ == "__main__":
    unittest.main()
