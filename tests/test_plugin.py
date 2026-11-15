"""The Pupil plugin keeps the Unix time it was given."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pupil_plugin"))

import sniff_trigger_capture


class Pool:
    def get_timestamp(self):
        return 0.0


class PluginTests(unittest.TestCase):
    def test_unix_time_is_stored_and_placement_is_not_replaced(self):
        directory = tempfile.mkdtemp()
        plugin = sniff_trigger_capture.Sniff_Trigger_Capture(Pool())
        plugin._recording = True
        plugin._rec_dir = directory
        note = {
            "subject": "sniff_sync.trigger",
            "label": "stimulus_onset",
            "trigger_id": "trg_abc",
            "unix_time": 1_700_000_000.125,
            "timestamp": 5000.5,
            "duration": 0.0,
        }
        plugin.on_notify(note)
        events = {"annotation": [{"topic": "annotation", "label": "kept", "timestamp": 1}]}
        plugin.recent_events(events)
        stored = events["annotation"][-1]
        self.assertEqual(stored["unix_time"], 1_700_000_000.125)
        self.assertEqual(stored["timestamp"], 5000.5)
        self.assertEqual(events["annotation"][0]["label"], "kept")
        text = open(os.path.join(directory, "sniff_sync_triggers.csv")).read()
        self.assertIn("1700000000.125000", text)
        self.assertIn("stimulus_onset", text)
        plugin.cleanup()

    def test_idle_recording_does_not_store(self):
        plugin = sniff_trigger_capture.Sniff_Trigger_Capture(Pool())
        plugin.on_notify(
            {
                "subject": "sniff_sync.trigger",
                "label": "early",
                "trigger_id": "trg_early",
                "unix_time": 10.0,
                "timestamp": 1.0,
            }
        )
        events = {}
        plugin.recent_events(events)
        self.assertNotIn("annotation", events)
        self.assertEqual(plugin.missed_while_idle, 1)


if __name__ == "__main__":
    unittest.main()
