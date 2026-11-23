"""The live-versus-reference comparison: agreement numbers, and writing the reference when it is missing."""

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from sniff_sync.compare import compare_session
from sniff_sync.coupling import APPROXIMATE_NAME, LOG_COLUMNS, SIGNAL_NAMES, _arrow_column


class CompareTests(unittest.TestCase):
    def test_correlation_difference_and_direction_match(self):
        temporary = tempfile.TemporaryDirectory()
        try:
            _write_pair(temporary.name)
            summary = compare_session(temporary.name)
            self.assertEqual(summary["windows"], 4)
            self.assertTrue(os.path.isfile(summary["table"]))
            self.assertTrue(os.path.isfile(summary["figure"]))
            driven = _link(summary, "interdependence", "breathing", "electroencephalography")
            self.assertGreater(driven["correlation"], 0.99)
            self.assertAlmostEqual(driven["median_absolute_difference"], 0.01, places=6)
            pair = _pair(summary, "interdependence", "breathing", "electroencephalography")
            self.assertEqual(pair["direction_matches"], 4)
            self.assertEqual(pair["direction_match_fraction"], 1.0)
            reversed_pair = _pair(summary, "forecast_gain", "breathing", "pupil diameter")
            self.assertEqual(reversed_pair["direction_matches"], 0)
            self.assertEqual(reversed_pair["direction_match_fraction"], 0.0)
        finally:
            temporary.cleanup()

    def test_a_missing_reference_file_is_written_first(self):
        temporary = tempfile.TemporaryDirectory()
        try:
            _write_pair(temporary.name, include_reference=False)
            called = {}

            def fake_reference(session, **_kwargs):
                called["session"] = str(session)
                _write_reference_only(session)
                return os.path.join(str(session), "coupling_reference.csv")

            with patch("sniff_sync.reference.write_reference", side_effect=fake_reference):
                summary = compare_session(temporary.name)
            self.assertEqual(called["session"], temporary.name)
            self.assertEqual(summary["windows"], 4)
        finally:
            temporary.cleanup()


def _link(summary, measure, source, target):
    for item in summary["links"]:
        if item["measure"] == measure and item["source"] == source and item["target"] == target:
            return item
    raise AssertionError("missing link")


def _pair(summary, measure, left, right):
    for item in summary["pairs"]:
        if item["measure"] == measure and item["left"] == left and item["right"] == right:
            return item
    raise AssertionError("missing pair")


def _write_pair(directory, include_reference=True):
    live_rows = []
    reference_rows = []
    for index, end in enumerate((10.0, 11.0, 12.0, 13.0)):
        live = _blank(end, "approximate", "live")
        reference = _blank(end, "reference", "reference")
        # Breathing drives electroencephalography. The live score sits 0.01 above the reference.
        _set(live, "interdependence", "breathing", "electroencephalography", 0.80 + 0.01 * index)
        _set(reference, "interdependence", "breathing", "electroencephalography", 0.79 + 0.01 * index)
        _set(live, "interdependence", "electroencephalography", "breathing", 0.20)
        _set(reference, "interdependence", "electroencephalography", "breathing", 0.19)
        # This pair disagrees: live says breathing, reference says pupil diameter.
        _set(live, "forecast_gain", "breathing", "pupil diameter", 0.50)
        _set(live, "forecast_gain", "pupil diameter", "breathing", 0.10)
        _set(reference, "forecast_gain", "breathing", "pupil diameter", 0.10)
        _set(reference, "forecast_gain", "pupil diameter", "breathing", 0.50)
        for source in SIGNAL_NAMES:
            for target in SIGNAL_NAMES:
                if source == target:
                    continue
                if _get(live, "interdependence", source, target) == "":
                    _set(live, "interdependence", source, target, 0.05)
                    _set(reference, "interdependence", source, target, 0.05)
                if _get(live, "forecast_gain", source, target) == "":
                    _set(live, "forecast_gain", source, target, 0.05)
                    _set(reference, "forecast_gain", source, target, 0.05)
        live_rows.append(live)
        reference_rows.append(reference)
    _dump(os.path.join(directory, APPROXIMATE_NAME), live_rows)
    if include_reference:
        _dump(os.path.join(directory, "coupling_reference.csv"), reference_rows)


def _write_reference_only(directory):
    with open(os.path.join(directory, APPROXIMATE_NAME), newline="") as handle:
        live_rows = list(csv.DictReader(handle))
    reference_rows = []
    for live in live_rows:
        reference = dict(live)
        reference["label"] = "reference"
        reference["filter_profile"] = "reference"
        reference["interdependence__breathing_to_electroencephalography"] = "%.6f" % (
            float(live["interdependence__breathing_to_electroencephalography"]) - 0.01
        )
        reference["interdependence__electroencephalography_to_breathing"] = "0.190000"
        reference["forecast_gain__breathing_to_pupil_diameter"] = "0.100000"
        reference["forecast_gain__pupil_diameter_to_breathing"] = "0.500000"
        reference_rows.append(reference)
    _dump(os.path.join(str(directory), "coupling_reference.csv"), reference_rows)


def _blank(end, label, profile):
    row = {column: "" for column in LOG_COLUMNS}
    row["label"] = label
    row["status"] = "ready"
    row["window_end_unix"] = "%.6f" % end
    row["window_seconds"] = "30.000000"
    row["filter_profile"] = profile
    return row


def _set(row, measure, source, target, value):
    row[_arrow_column(measure, source, target)] = "%.6f" % value


def _get(row, measure, source, target):
    return row[_arrow_column(measure, source, target)]


def _dump(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
