"""Loaders, band-power ratios, and the nearest-neighbour indices. No headset."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.nlid.embed import choose_delay, choose_dimension
from sniff_sync.nlid.load import load_breathing, load_eeg, load_pupil
from sniff_sync.nlid.measures import bivariate_interdependence
from sniff_sync.nlid.preprocess import attentional_signals, clean_pupil
from sniff_sync.nlid.surrogates import benjamini_hochberg, bonferroni, iaaft, p_value_greater
from sniff_sync.records import BreathingLog, EegLog
from sniff_sync.timefmt import format_timestamp


class LoaderTests(unittest.TestCase):
    def test_breathing_skips_triggers_and_accepts_milliseconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "breathing.csv"
            log = BreathingLog(path)
            stamp = 1_700_000_000.5
            log.write_sample(stamp, 1.25)
            log.write_trigger(type("Event", (), {"unix_time": stamp + 1, "label": "onset", "trigger_id": "trg_x"})())
            log.close()
            # A second file in milliseconds, the Sniff_polar_display.py convention.
            milli = Path(tmp) / "milli.csv"
            with open(milli, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Unix Time", "Nasal Pressure", "Trigger"])
                writer.writerow([format_timestamp((stamp + 2) * 1000.0), "0.500000", ""])
            times, values = load_breathing(path)
            self.assertEqual(len(times), 1)
            self.assertAlmostEqual(times[0], stamp, places=5)
            self.assertAlmostEqual(values[0], 1.25, places=5)
            times, values = load_breathing(milli)
            self.assertAlmostEqual(times[0], stamp + 2, places=4)
            self.assertAlmostEqual(values[0], 0.5, places=5)

    def test_pupil_blink_is_interpolated_and_offset_is_applied(self):
        times = np.linspace(0, 2, 81)
        diameter = 4.0 + 0.1 * times
        confidence = np.ones_like(times)
        # A 0.2 second blink in the middle.
        blink = (times > 0.9) & (times < 1.1)
        confidence[blink] = 0.0
        diameter = diameter.copy()
        diameter[blink] = 0.0
        cleaned = clean_pupil(times, diameter, confidence, max_gap_s=0.5, pad_s=0.0)
        self.assertTrue(np.all(np.isfinite(cleaned)))
        self.assertLess(np.max(np.abs(cleaned - (4.0 + 0.1 * times))), 0.05)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pupil_positions.csv"
            offset = 12.5
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pupil_timestamp", "eye_id", "confidence", "diameter", "diameter_3d"])
                for index, stamp in enumerate(times):
                    eye = index % 2
                    writer.writerow([stamp + offset, eye, 0.99, 80.0, 4.2 + 0.01 * eye])
            series = load_pupil(path, offset=offset)
            self.assertEqual({item["eye"] for item in series}, {0, 1})
            for item in series:
                self.assertGreaterEqual(item["time"][0], -1e-6)
                self.assertLess(item["time"][0], 0.05)
                self.assertGreater(item["time"][-1], 1.9)
                self.assertLess(item["time"].max(), 3.0)
                # Millimetres from diameter_3d, not the pixel diameter.
                self.assertGreater(item["diameter"].mean(), 4.0)
                self.assertLess(item["diameter"].mean(), 4.3)

    def test_eeg_trigger_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eeg.csv"
            log = EegLog(path)
            log.write_sample(1_700_000_000.0, {name: float(index) for index, name in enumerate(EPOC_CHANNELS)})
            log.write_trigger(type("Event", (), {"unix_time": 1_700_000_000.1, "label": "onset", "trigger_id": "trg_y"})())
            log.write_sample(1_700_000_000.2, {name: 1.0 for name in EPOC_CHANNELS})
            log.close()
            times, channels = load_eeg(path)
            self.assertEqual(len(times), 2)
            self.assertAlmostEqual(channels["AF3"][0], 0.0, places=3)
            self.assertIn("AF4", channels)


class PreprocessTests(unittest.TestCase):
    def test_delta_alpha_ratio_tracks_the_modulator(self):
        sample_rate = 128.0
        duration = 30.0
        times = np.arange(0, duration, 1.0 / sample_rate)
        modulator = np.sin(2 * np.pi * 0.2 * times)
        alpha = (1.2 + 0.8 * modulator) * np.sin(2 * np.pi * 10.0 * times)
        delta = (1.2 - 0.8 * modulator) * np.sin(2 * np.pi * 2.0 * times)
        channels = {name: np.zeros_like(times) for name in ("AF3", "AF4", "F3", "F4")}
        for name in channels:
            channels[name] = alpha + delta
        _grid, signals, warnings = attentional_signals(times, channels)
        self.assertIn("delta/alpha power ratio", signals)
        ratio = signals["delta/alpha power ratio"]
        # The modulator at the power's time base.
        grid = _grid
        finite = np.isfinite(ratio)
        reference = np.interp(grid[finite], times, -modulator)
        correlation = np.corrcoef(ratio[finite], reference)[0, 1]
        self.assertGreater(correlation, 0.7, msg="correlation %.3f warnings %s" % (correlation, warnings))


class MeasureTests(unittest.TestCase):
    def test_delay_and_dimension_for_a_noisy_sine(self):
        generator = np.random.default_rng(0)
        samples = np.arange(800)
        series = np.sin(2 * np.pi * samples / 40.0) + 0.05 * generator.normal(size=800)
        delay, _scores = choose_delay(series, max_lag=30)
        self.assertGreaterEqual(delay, 6)
        self.assertLessEqual(delay, 14)
        dimension, fractions = choose_dimension(series, delay, max_dimension=4)
        self.assertIn(dimension, (2, 3, 4))
        self.assertLess(fractions[dimension - 1], 0.15)

    def test_identical_series_scores_higher_than_noise(self):
        generator = np.random.default_rng(1)
        samples = np.arange(700)
        series = np.sin(2 * np.pi * 0.05 * samples) + 0.05 * generator.normal(size=700)
        same = bivariate_interdependence(series, series, (3, 3), (5, 5), 8, 15)
        other = bivariate_interdependence(series, generator.normal(size=700), (3, 3), (5, 5), 8, 15)
        self.assertGreater(same["andrzejak_M"], 0.8)
        self.assertGreater(same["similarity_S"], 0.5)
        self.assertLess(abs(other["andrzejak_M"]), 0.15)

    def test_coupled_autoregression_is_significant_and_an_independent_one_is_not(self):
        from sniff_sync.nlid.analyze import analyse_triplet, is_detected

        generator = np.random.default_rng(2)
        n_samples = 800
        driver = np.zeros(n_samples)
        response = np.zeros(n_samples)
        other = generator.normal(size=n_samples)
        for time_index in range(1, n_samples):
            driver[time_index] = 0.6 * np.sin(driver[time_index - 1]) + 0.5 * generator.normal()
            response[time_index] = 0.5 * response[time_index - 1] + 0.8 * driver[time_index - 1] + 0.3 * generator.normal()
        embedding = {name: {"dimension": 3, "delay": 1} for name in ("driver", "response", "other")}
        rows, _resolved = analyse_triplet(
            {"driver": driver, "response": response, "other": other},
            sample_rate_hz=1.0,
            attentional_signal="response",
            embedding=embedding,
            neighbor_count=8,
            pool_size=30,
            theiler_window=5,
            n_surrogates=19,
            seed=2,
            include_prediction=False,
            include_joint=False,
            include_partial_cross_map=False,
            minimum_shift_fraction=0.1,
            prediction_horizons_s=(1,),
            max_cross_map_lag_s=2,
        )
        coupled = next(row for row in rows if row["measure"] == "Andrzejak M" and row["source"] == "driver" and row["target"] == "response" and not row["conditioned_on"])
        independent = next(row for row in rows if row["measure"] == "Andrzejak M" and row["source"] == "other" and row["target"] == "response" and not row["conditioned_on"])
        self.assertTrue(is_detected(coupled["value"], coupled["p_value"], 0.05), msg=str(coupled))
        self.assertFalse(is_detected(independent["value"], independent["p_value"], 0.05), msg=str(independent))

    def test_iaaft_keeps_the_spectrum_and_corrections_rank_p_values(self):
        generator = np.random.default_rng(3)
        series = np.cumsum(generator.normal(size=512))
        surrogate = iaaft(series, generator, iterations=20)
        spectrum = np.abs(np.fft.rfft(series))
        surrogate_spectrum = np.abs(np.fft.rfft(surrogate))
        correlation = np.corrcoef(spectrum[1:], surrogate_spectrum[1:])[0, 1]
        self.assertGreater(correlation, 0.99)
        raw = np.array([0.001, 0.02, 0.03, 0.5])
        _bonferroni_p, bonferroni_flag = bonferroni(raw, alpha=0.05)
        _fdr_p, fdr_flag = benjamini_hochberg(raw, alpha=0.05)
        self.assertTrue(bonferroni_flag[0])
        self.assertFalse(bonferroni_flag[1])
        self.assertTrue(fdr_flag[0] and fdr_flag[1] and fdr_flag[2])
        self.assertFalse(fdr_flag[3])
        self.assertLessEqual(p_value_greater(1.0, [0.0, 0.2, 0.4]), 0.25)
        self.assertGreater(p_value_greater(0.0, [0.2, 0.4, 0.6]), 0.5)


class CommandTests(unittest.TestCase):
    def test_simulated_session_round_trip(self):
        """The hub's --simulate files, plus a pupil export on the same Unix clock, run end to end."""
        import os
        import time

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from sniff_sync.nlid.__main__ import main
        from sniff_sync.simulate import Session

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.start()
            try:
                time.sleep(8)
            finally:
                session.stop()
            times, _pressure = load_breathing(root / "breathing.csv")
            pupil = ["unix_time,eye_id,confidence,diameter_3d"]
            for index, stamp in enumerate(np.linspace(times[0], times[-1], 200)):
                pupil.append("%.6f,%d,0.99,%.4f" % (stamp, index % 2, 4.2 + 0.05 * np.sin(index / 8.0)))
            (root / "pupil_positions.csv").write_text("\n".join(pupil) + "\n")
            code = main([str(root), "--surrogates", "9", "--signals", "alpha power", "--sample-rate", "16", "--seed", "1"])
            self.assertEqual(code, 0)
            text = (root / "nlid" / "interdependence.csv").read_text()
            self.assertIn("breathing", text)
            self.assertIn("pupil diameter", text)
            self.assertIn("alpha power", text)
            self.assertTrue((root / "nlid" / "summary.png").is_file())

    def test_command_line_on_hub_shaped_files(self):
        from sniff_sync.nlid.__main__ import main

        generator = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            breath = BreathingLog(root / "breathing.csv")
            eeg = EegLog(root / "eeg.csv")
            start = 1_700_000_100.0
            pupil_rows = ["unix_time,eye_id,confidence,diameter_3d"]
            for index, stamp in enumerate(np.arange(start, start + 25.0, 1.0 / 20.0)):
                breath.write_sample(stamp, np.sin(2 * np.pi * 0.25 * (stamp - start)))
            breath.write_trigger(type("Event", (), {"unix_time": start + 3, "label": "onset", "trigger_id": "trg_z"})())
            breath.close()
            for step, stamp in enumerate(np.arange(start, start + 25.0, 1.0 / 128.0)):
                elapsed = stamp - start
                modulator = np.sin(2 * np.pi * 0.25 * elapsed)
                alpha = (1.0 + 0.6 * modulator) * np.sin(2 * np.pi * 10.0 * elapsed)
                delta = (1.0 - 0.6 * modulator) * np.sin(2 * np.pi * 2.0 * elapsed)
                values = {name: alpha + delta + 0.05 * generator.normal() for name in EPOC_CHANNELS}
                eeg.write_sample(stamp, values, counter=step)
            eeg.close()
            for index, stamp in enumerate(np.arange(start, start + 25.0, 1.0 / 40.0)):
                confidence = 0.0 if 4.0 < stamp - start < 4.2 else 0.99
                diameter = 0.0 if confidence == 0.0 else 4.1 + 0.2 * np.sin(2 * np.pi * 0.2 * (stamp - start))
                pupil_rows.append("%.6f,%d,%.2f,%.4f" % (stamp, index % 2, confidence, diameter))
            (root / "pupil_positions.csv").write_text("\n".join(pupil_rows) + "\n")
            code = main(
                [
                    str(root),
                    "--surrogates",
                    "9",
                    "--signals",
                    "alpha power,delta/alpha power ratio",
                    "--sample-rate",
                    "16",
                    "--seed",
                    "4",
                ]
            )
            self.assertEqual(code, 0)
            table = (root / "nlid" / "interdependence.csv").read_text()
            self.assertIn("Andrzejak M", table)
            self.assertIn("conditional Andrzejak M", table)
            self.assertIn("direct-link index", table)
            self.assertIn("delta/alpha power ratio", table)
            self.assertTrue((root / "nlid" / "summary.png").is_file())
            self.assertTrue((root / "nlid" / "summary.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
