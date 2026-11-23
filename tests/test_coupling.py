"""The live coupling worker: start, stop, a result about once a second, and a known direction."""

import csv
import os
import tempfile
import time
import unittest

import numpy as np

from sniff_sync.coupling import (
    APPROXIMATE_NAME,
    LOG_COLUMNS,
    SIGNAL_NAMES,
    LiveCoupling,
    _WorkerMemory,
    _log_row,
    score_triplet,
)
from sniff_sync.nlid.measures import recency_weights, _weighted_mean
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
        self.assertFalse(result["recency_weighting"])

    def test_recency_weights_favour_the_newest_reference_point(self):
        weights = recency_weights(5, half_life_seconds=1.0, sample_rate_hz=1.0)
        self.assertAlmostEqual(weights[-1], 1.0)
        self.assertAlmostEqual(weights[-2], 0.5)
        self.assertAlmostEqual(weights[0], 0.0625)
        values = np.array([0.0, 0.0, 0.0, 0.0, 8.0])
        weighted = _weighted_mean(values, weights)
        self.assertGreater(weighted, float(np.mean(values)))
        self.assertIsNone(recency_weights(5, None, 1.0))

    def test_recency_weighting_changes_the_window_score_and_is_logged_unsmoothed(self):
        embedding = {name: {"delay": 2, "dimension": 2} for name in SIGNAL_NAMES}
        series = _recent_drive()
        plain, plain_forecast, _theiler = score_triplet(series, embedding)
        weighted, weighted_forecast, _theiler = score_triplet(series, embedding, recency_half_life_seconds=4.0)
        arrow = "breathing → electroencephalography"
        self.assertGreater(weighted[arrow], plain[arrow])
        self.assertNotAlmostEqual(weighted_forecast[arrow], plain_forecast[arrow], places=4)
        memory = _WorkerMemory()
        mask = np.ones(len(series["breathing"]), dtype=bool)
        job = {
            "values": {
                "breathing": series["breathing"],
                "pupil": series["pupil diameter"],
                "eeg": series["electroencephalography"],
            },
            "valid": mask,
            "window_end_unix": 80.0,
            "window_seconds": 50.0,
            "eeg_signal": "delta",
            "eeg_channels": ["AF3"],
            "filter_profile": "live",
            "neighbor_count": 8,
            "pool_size": 40,
            "horizon_samples": 4,
            "surrogates": False,
            "recency_weighting": True,
            "recency_half_life_seconds": 4.0,
            "hand_embedding": embedding,
            "delays_seconds": {},
        }
        result = memory.compute(job)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["recency_weighting"])
        self.assertEqual(result["recency_half_life_seconds"], 4.0)
        self.assertAlmostEqual(result["interdependence"][arrow], weighted[arrow])
        row = _log_row(result)
        self.assertEqual(row["recency_weighting"], "yes")
        self.assertEqual(row["recency_half_life_seconds"], "4.000000")
        self.assertIn("recency_weighting", LOG_COLUMNS)
        self.assertEqual(
            row["interdependence__breathing_to_electroencephalography"],
            "%.6f" % result["interdependence"][arrow],
        )

    def test_the_reference_uses_the_same_recency_setting(self):
        from unittest.mock import patch

        import sniff_sync.reference as reference

        temporary = tempfile.TemporaryDirectory()
        root = temporary.name
        _write_approximate(root, recency="yes", half_life="4.500000")
        captured = {}

        def fake_score(*_args, **kwargs):
            captured["half_life"] = kwargs.get("recency_half_life_seconds")
            interdependence = {"%s → %s" % (source, target): 0.2 for source in SIGNAL_NAMES for target in SIGNAL_NAMES if source != target}
            forecast = {"%s → %s" % (source, target): 0.1 for source in SIGNAL_NAMES for target in SIGNAL_NAMES if source != target}
            return interdependence, forecast, 2

        grid = np.arange(0.0, 40.0, 0.1)
        signals = {
            "time": grid,
            "breathing": np.sin(grid),
            "pupil diameter": np.cos(grid),
            "delta": np.sin(grid),
            "alpha": np.ones_like(grid),
            "delta-to-alpha": np.sin(grid),
        }
        try:
            with patch.object(reference, "load_breathing", return_value=(grid, np.sin(grid))), patch.object(
                reference, "load_eeg", return_value=(grid, {"AF3": np.sin(grid)})
            ), patch.object(
                reference, "load_pupil", return_value=[{"eye": 0, "time": grid, "diameter": np.ones_like(grid), "confidence": np.ones_like(grid)}]
            ), patch.object(reference, "reference_signals", return_value=signals), patch.object(
                reference, "score_triplet", side_effect=fake_score
            ):
                destination = reference.write_reference(root, pupil_path=os.path.join(root, "pupil_diameter.csv"))
            weighted_half_life = captured["half_life"]
            with open(destination, newline="") as handle:
                rows = list(csv.DictReader(handle))
            _write_approximate(root, recency="no", half_life="10.000000")
            with patch.object(reference, "load_breathing", return_value=(grid, np.sin(grid))), patch.object(
                reference, "load_eeg", return_value=(grid, {"AF3": np.sin(grid)})
            ), patch.object(
                reference, "load_pupil", return_value=[{"eye": 0, "time": grid, "diameter": np.ones_like(grid), "confidence": np.ones_like(grid)}]
            ), patch.object(reference, "reference_signals", return_value=signals), patch.object(
                reference, "score_triplet", side_effect=fake_score
            ):
                reference.write_reference(root, pupil_path=os.path.join(root, "pupil_diameter.csv"))
        finally:
            temporary.cleanup()
        self.assertEqual(weighted_half_life, 4.5)
        self.assertEqual(rows[0]["label"], "reference")
        self.assertEqual(rows[0]["status"], "ready")
        self.assertEqual(rows[0]["recency_weighting"], "yes")
        self.assertEqual(rows[0]["recency_half_life_seconds"], "4.500000")
        self.assertEqual(rows[0]["interdependence__breathing_to_pupil_diameter"], "0.200000")
        self.assertIsNone(captured["half_life"])


def _recent_drive():
    """Coupling from breathing into electroencephalography only in the last few seconds."""
    generator = np.random.default_rng(7)
    count = 500
    rate = 10.0
    times = np.arange(count) / rate
    breathing = np.sin(2.0 * np.pi * 0.25 * times)
    pupil = np.sin(2.0 * np.pi * 0.13 * times + 0.7) + 0.05 * generator.normal(size=count)
    electroencephalography = 0.15 * generator.normal(size=count)
    start = count - 100
    electroencephalography[start:] = breathing[start - 5 : count - 5]
    return {
        "breathing": breathing,
        "pupil diameter": pupil,
        "electroencephalography": electroencephalography,
    }


def _write_approximate(directory, recency, half_life):
    path = os.path.join(directory, APPROXIMATE_NAME)
    row = {column: "" for column in LOG_COLUMNS}
    row.update(
        {
            "label": "approximate",
            "status": "ready",
            "window_end_unix": "30.000000",
            "window_seconds": "30.000000",
            "eeg_signal": "delta-to-alpha",
            "eeg_channels": "AF3 AF4 F3 F4",
            "filter_profile": "live",
            "embedding_source": "hand",
            "neighbor_count": "8",
            "pool_size": "40",
            "horizon_samples": "4",
            "surrogates": "no",
            "recency_weighting": recency,
            "recency_half_life_seconds": half_life,
            "valid_fraction": "1.000000",
        }
    )
    for name in SIGNAL_NAMES:
        key = name.replace(" ", "_")
        row["embedding_delay_%s" % key] = "2"
        row["embedding_dimension_%s" % key] = "2"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        writer.writerow(row)


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
