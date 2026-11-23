"""The live coupling worker: start, stop, a result about once a second, and a known direction."""

import csv
import os
import tempfile
import time
import unittest

import numpy as np

from sniff_sync.coupling import (
    APPROXIMATE_NAME,
    SIGNAL_NAMES,
    LiveCoupling,
    _WorkerMemory,
    score_triplet,
)
from sniff_sync.rolling import RollingHistory


class WorkerTests(unittest.TestCase):
    def test_the_worker_starts_and_stops_with_the_hub(self):
        from sniff_sync.hub import Hub

        temporary = tempfile.TemporaryDirectory()
        hub = Hub(temporary.name, pupil_port=9, port=0, enable_coupling=True)
        hub.start()
        try:
            self.assertIsNotNone(hub.coupling._process)
            self.assertTrue(hub.coupling._process.is_alive())
            latest = hub.latest_coupling()
            self.assertEqual(latest["label"], "approximate")
            self.assertIn(latest["status"], ("warming up", "skipped", "ready"))
        finally:
            hub.close()
            temporary.cleanup()
        self.assertTrue(hub.coupling._process is None or not hub.coupling._process.is_alive())

    def test_results_arrive_about_once_a_second_in_simulate_mode(self):
        from sniff_sync.simulate import Session

        previous = os.environ.get("SNIFF_SYNC_PORT")
        temporary = tempfile.TemporaryDirectory()
        session = Session(temporary.name, enable_coupling=True)
        session.start()
        try:
            time.sleep(8.0)
            path = session.directory / APPROXIMATE_NAME
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
        finally:
            session.stop()
            temporary.cleanup()
            if previous is None:
                os.environ.pop("SNIFF_SYNC_PORT", None)
            else:
                os.environ["SNIFF_SYNC_PORT"] = previous
        self.assertGreaterEqual(len(rows), 5)
        self.assertLessEqual(len(rows), 12)
        stamps = [float(row["logged_unix"]) for row in rows]
        gaps = np.diff(stamps)
        self.assertGreater(float(np.median(gaps)), 0.6)
        self.assertLess(float(np.median(gaps)), 1.6)
        self.assertTrue(all(row["label"] == "approximate" for row in rows))

    def test_a_known_direction_is_recovered(self):
        embedding = {name: {"delay": 2, "dimension": 3} for name in SIGNAL_NAMES}
        series = _driven_series(seed=2)
        interdependence, forecast, _theiler = score_triplet(series, embedding)
        forward = interdependence["breathing → electroencephalography"]
        backward = interdependence["electroencephalography → breathing"]
        self.assertGreater(forward, backward)
        self.assertGreater(forward, 0.0)
        self.assertGreater(
            forecast["breathing → electroencephalography"],
            forecast["electroencephalography → breathing"],
        )
        # The worker reports the same comparison from a window payload.
        memory = _WorkerMemory()
        mask = np.ones(len(series["breathing"]), dtype=bool)
        job = {
            "values": {
                "breathing": series["breathing"],
                "pupil": series["pupil diameter"],
                "eeg": series["electroencephalography"],
            },
            "valid": mask,
            "window_end_unix": 100.0,
            "window_seconds": 60.0,
            "eeg_signal": "delta",
            "eeg_channels": ["AF3"],
            "filter_profile": "live",
            "neighbor_count": 8,
            "pool_size": 40,
            "horizon_samples": 4,
            "surrogates": False,
            "hand_embedding": embedding,
            "delays_seconds": {},
        }
        result = memory.compute(job)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["label"], "approximate")
        self.assertEqual(result["embedding_source"], "hand")
        summary = result["directions"]["breathing and electroencephalography"]
        self.assertEqual(summary["interdependence_stronger"], "breathing → electroencephalography")


def _driven_series(seed):
    """Breathing drives the electroencephalography trace. Pupil diameter is separate."""
    generator = np.random.default_rng(seed)
    count = 800
    rate = 10.0
    breathing = np.zeros(count)
    electroencephalography = np.zeros(count)
    pupil = np.zeros(count)
    for index in range(1, count):
        breathing[index] = np.sin(2.0 * np.pi * 0.25 * index / rate) + 0.05 * generator.normal()
        drive = max(breathing[index - 5], 0.0)
        electroencephalography[index] = 0.4 * electroencephalography[index - 1] + 1.4 * drive + 0.05 * generator.normal()
        pupil[index] = 0.85 * pupil[index - 1] + 0.2 * generator.normal()
    return {
        "breathing": breathing,
        "pupil diameter": pupil,
        "electroencephalography": electroencephalography,
    }


if __name__ == "__main__":
    unittest.main()
