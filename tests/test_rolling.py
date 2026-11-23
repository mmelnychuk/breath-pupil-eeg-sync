"""The rolling history puts breathing, pupil diameter, and electroencephalography on one grid."""

import os
import tempfile
import time
import unittest

import numpy as np

from sniff_sync.feed import Feed
from sniff_sync.rolling import (
    DEFAULT_EEG_CHANNELS,
    OUTPUT_RATE_HZ,
    PupilStage,
    RollingHistory,
)


START = 1_700_000_000.0


def _bump(times, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((times - center) / sigma) ** 2)


def _add_breath(history, start, duration, rate=20.0, pressure=None):
    times = start + np.arange(0.0, duration, 1.0 / rate)
    if pressure is None:
        pressure = 0.2 * np.sin(2.0 * np.pi * 0.25 * (times - start))
    for stamp, value in zip(times, pressure):
        history.add_breath(float(stamp), float(value))
    return times


def _add_pupil(history, start, duration, rate=40.0, eyes=(0, 1), diameter=None):
    times = start + np.arange(0.0, duration, 1.0 / rate)
    if diameter is None:
        diameter = np.full(times.shape, 4.5)
    for stamp, value in zip(times, diameter):
        for eye in eyes:
            history.add_pupil(float(stamp), eye, float(value))
    return times


def _add_eeg(history, start, duration, rate, values, channels=DEFAULT_EEG_CHANNELS, skip=None):
    times = start + np.arange(0.0, duration, 1.0 / rate)
    if values is None:
        values = np.zeros(times.shape)
    for stamp, value in zip(times, values):
        if skip is not None and skip[0] <= stamp < skip[1]:
            continue
        sample = {name: float(value) for name in channels}
        history.add_eeg(float(stamp), sample)
    return times


def _steady(history, duration, eeg_rate=32.0, start=START):
    """Enough of each stream for a window, with a slow shape so the filters have an input."""
    times = start + np.arange(0.0, duration, 1.0 / eeg_rate)
    relative = times - start
    wave = (
        (20.0 + 8.0 * np.sin(2.0 * np.pi * 0.05 * relative)) * np.sin(2.0 * np.pi * 2.5 * relative)
        + (24.0 + 6.0 * np.sin(2.0 * np.pi * 0.04 * relative)) * np.sin(2.0 * np.pi * 10.0 * relative)
    )
    _add_breath(history, start, duration)
    _add_pupil(history, start, duration)
    _add_eeg(history, start, duration, eeg_rate, wave)
    return start


def _peak_time(time_seconds, values, center, half_width=2.5):
    """Time of a bump, matched to a smooth template so one ripply sample cannot move it."""
    select = (time_seconds >= center - half_width) & (time_seconds <= center + half_width) & np.isfinite(values)
    if int(np.count_nonzero(select)) < 5:
        raise AssertionError("no finite samples near %.3f" % center)
    times = time_seconds[select]
    chunk = values[select]
    chunk = chunk - np.mean(chunk)
    best_shift = 0.0
    best_score = -np.inf
    for shift in np.arange(-0.4, 0.401, 0.1):
        template = np.exp(-0.5 * ((times - (center + shift)) / 0.6) ** 2)
        template = template - np.mean(template)
        score = float(np.dot(chunk, template))
        if score > best_score:
            best_score = score
            best_shift = float(shift)
    return float(center + best_shift)


class SharedEventTests(unittest.TestCase):
    def test_one_event_lands_on_the_same_grid_time(self):
        for eeg_rate in (128.0, 32.0):
            with self.subTest(eeg_rate=eeg_rate):
                self._event_at_rate(eeg_rate)

    def _event_at_rate(self, eeg_rate):
        duration = 45.0
        event = START + 30.0
        history = RollingHistory()
        breath_times = START + np.arange(0.0, duration, 1.0 / 20.0)
        breath = 0.15 * np.sin(2.0 * np.pi * 0.25 * (breath_times - START)) + _bump(breath_times, event, 0.35, 2.0)
        _add_breath(history, START, duration, pressure=breath)
        pupil_times = START + np.arange(0.0, duration, 1.0 / 40.0)
        pupil = 4.2 + 0.05 * np.sin(2.0 * np.pi * 0.2 * (pupil_times - START)) + _bump(pupil_times, event, 0.35, 1.5)
        for stamp, value in zip(pupil_times, pupil):
            history.add_pupil(float(stamp), 0, float(value))
            history.add_pupil(float(stamp), 1, float(value) + 0.1)
        eeg_times = START + np.arange(0.0, duration, 1.0 / eeg_rate)
        # The delta bump is much larger than the alpha bump, so the ratio
        # peaks with the delta envelope rather than sitting on a flat top.
        delta_amplitude = 25.0 + _bump(eeg_times, event, 0.8, 40.0)
        alpha_amplitude = 30.0 + _bump(eeg_times, event, 0.8, 2.0)
        wave = delta_amplitude * np.sin(2.0 * np.pi * 2.5 * (eeg_times - START))
        wave = wave + alpha_amplitude * np.sin(2.0 * np.pi * 10.0 * (eeg_times - START))
        _add_eeg(history, START, duration, eeg_rate, wave)

        window = history.resample()
        self.assertGreater(int(np.count_nonzero(window.valid)), 100)
        peaks = {
            "breathing": _peak_time(window.time_seconds, window.breathing, event),
            "pupil diameter": _peak_time(window.time_seconds, window.pupil_diameter, event),
            "delta envelope": _peak_time(window.time_seconds, window.delta_envelope, event),
            "alpha envelope": _peak_time(window.time_seconds, window.alpha_envelope, event),
            "delta-to-alpha": _peak_time(window.time_seconds, window.delta_to_alpha, event),
        }
        for name, peak in peaks.items():
            self.assertLessEqual(
                abs(peak - event),
                1.0 / OUTPUT_RATE_HZ,
                "%s peak at %.4f, event at %.4f, delays %s" % (name, peak, event, window.delays_seconds),
            )


class EnvelopeTests(unittest.TestCase):
    def test_envelopes_follow_known_amplitude_changes(self):
        for eeg_rate in (128.0, 32.0):
            with self.subTest(eeg_rate=eeg_rate):
                self._follow(eeg_rate)

    def _follow(self, eeg_rate):
        duration = 55.0
        history = RollingHistory()
        _add_breath(history, START, duration)
        _add_pupil(history, START, duration)
        times = START + np.arange(0.0, duration, 1.0 / eeg_rate)
        relative = times - START
        delta_amplitude = 20.0 + 12.0 * np.sin(2.0 * np.pi * 0.05 * relative)
        alpha_amplitude = 28.0 + 10.0 * np.sin(2.0 * np.pi * 0.033 * relative + 0.7)
        wave = delta_amplitude * np.sin(2.0 * np.pi * 2.5 * relative)
        wave = wave + alpha_amplitude * np.sin(2.0 * np.pi * 10.0 * relative)
        _add_eeg(history, START, duration, eeg_rate, wave)

        window = history.resample()
        known_delta = 20.0 + 12.0 * np.sin(2.0 * np.pi * 0.05 * (window.time_seconds - START))
        known_alpha = 28.0 + 10.0 * np.sin(2.0 * np.pi * 0.033 * (window.time_seconds - START) + 0.7)
        usable = window.valid.copy()
        # Drop the ends of the valid stretch so a filter restart cannot dominate the score.
        if np.any(usable):
            indices = np.flatnonzero(usable)
            usable[: indices[0] + int(2 * OUTPUT_RATE_HZ)] = False
            usable[indices[-1] - int(2 * OUTPUT_RATE_HZ) :] = False
        self.assertGreater(int(np.count_nonzero(usable)), 200)
        self.assertGreater(_correlation(window.delta_envelope[usable], known_delta[usable]), 0.9)
        self.assertGreater(_correlation(window.alpha_envelope[usable], known_alpha[usable]), 0.9)
        self.assertGreater(
            _correlation(window.delta_to_alpha[usable], (known_delta / known_alpha)[usable]),
            0.85,
        )
        self.assertLessEqual(abs(_best_lag(known_delta[usable], window.delta_envelope[usable])), 1)
        self.assertLessEqual(abs(_best_lag(known_alpha[usable], window.alpha_envelope[usable])), 1)
        high = known_delta[usable] >= np.quantile(known_delta[usable], 0.75)
        low = known_delta[usable] <= np.quantile(known_delta[usable], 0.25)
        self.assertGreater(float(np.mean(window.delta_envelope[usable][high])), float(np.mean(window.delta_envelope[usable][low])))


class GapTests(unittest.TestCase):
    def test_long_gaps_are_missing_and_short_gaps_are_bridged(self):
        duration = 50.0
        history = RollingHistory()
        _add_breath(history, START, duration)
        short = (START + 15.0, START + 15.15)
        long = (START + 22.0, START + 24.5)
        one_eye = (START + 30.0, START + 34.0)
        pupil_times = START + np.arange(0.0, duration, 1.0 / 40.0)
        for stamp in pupil_times:
            stamp = float(stamp)
            if long[0] <= stamp < long[1] or short[0] <= stamp < short[1]:
                continue
            history.add_pupil(stamp, 0, 4.0)
            if not (one_eye[0] <= stamp < one_eye[1]):
                history.add_pupil(stamp, 1, 6.0)
        eeg_gap = (START + 36.0, START + 39.0)
        eeg_times = START + np.arange(0.0, duration, 1.0 / 128.0)
        relative = eeg_times - START
        wave = 30.0 * np.sin(2.0 * np.pi * 2.5 * relative) + 20.0 * np.sin(2.0 * np.pi * 10.0 * relative)
        _add_eeg(history, START, duration, 128.0, wave, skip=eeg_gap)

        window = history.resample()
        self._region_is_valid(window, short[0], short[1])
        self._region_is_missing(window, long[0] + 0.4, long[1] - 0.4)
        self._region_is_missing(window, eeg_gap[0] + 0.4, eeg_gap[1] - 0.4)
        both_eyes = (window.time_seconds >= START + 10.0) & (window.time_seconds < START + 14.0) & window.valid
        only_left = (window.time_seconds >= START + 31.2) & (window.time_seconds < START + 33.0) & np.isfinite(window.pupil_diameter)
        self.assertGreater(int(np.count_nonzero(both_eyes)), 10)
        self.assertGreater(int(np.count_nonzero(only_left)), 5)
        self.assertAlmostEqual(float(np.median(window.pupil_diameter[both_eyes])), 5.0, delta=0.25)
        self.assertAlmostEqual(float(np.median(window.pupil_diameter[only_left])), 4.0, delta=0.25)
        # Breathing did not have a hole, so its values stay present through the pupil gap.
        pupil_hole = (window.time_seconds >= long[0] + 0.4) & (window.time_seconds < long[1] - 0.4)
        self.assertTrue(np.all(np.isfinite(window.breathing[pupil_hole])))
        self.assertTrue(np.all(~np.isfinite(window.pupil_diameter[pupil_hole])))

    def _region_is_valid(self, window, begin, end):
        select = (window.time_seconds >= begin) & (window.time_seconds < end)
        self.assertGreater(int(np.count_nonzero(select)), 0)
        self.assertTrue(np.all(window.valid[select]), "expected valid samples from %.3f to %.3f" % (begin, end))

    def _region_is_missing(self, window, begin, end):
        select = (window.time_seconds >= begin) & (window.time_seconds < end)
        self.assertGreater(int(np.count_nonzero(select)), 0)
        self.assertTrue(np.all(~window.valid[select]))


class WindowTests(unittest.TestCase):
    def test_window_length_setting(self):
        history = RollingHistory()
        self.assertEqual(history.window_seconds, 60.0)
        _steady(history, 150.0, eeg_rate=32.0)
        full = history.resample()
        self.assertAlmostEqual(full.time_seconds[-1] - full.time_seconds[0], 60.0, places=6)
        self.assertEqual(len(full.time_seconds), 601)
        self.assertAlmostEqual(float(np.median(np.diff(full.time_seconds))), 0.1, places=6)

        history.set_window_seconds(30)
        short = history.resample()
        self.assertAlmostEqual(short.time_seconds[-1] - short.time_seconds[0], 30.0, places=6)
        self.assertEqual(len(short.time_seconds), 301)
        np.testing.assert_allclose(short.time_seconds, full.time_seconds[-len(short.time_seconds) :])
        np.testing.assert_allclose(short.breathing, full.breathing[-len(short.breathing) :], equal_nan=True)
        np.testing.assert_allclose(short.delta_envelope, full.delta_envelope[-len(short.delta_envelope) :], equal_nan=True)

        history.set_window_seconds(120)
        long = history.resample()
        self.assertAlmostEqual(long.time_seconds[-1] - long.time_seconds[0], 120.0, places=6)
        self.assertEqual(len(long.time_seconds), 1201)
        self.assertGreater(float(np.mean(long.valid)), 0.9)

        with self.assertRaises(ValueError):
            history.set_window_seconds(20)
        with self.assertRaises(ValueError):
            history.set_window_seconds(121)
        self.assertEqual(history.window_seconds, 120.0)
        with self.assertRaises(ValueError):
            RollingHistory(window_seconds=10)

    def test_history_is_longer_than_the_plot_and_does_not_grow_without_limit(self):
        self.assertEqual(Feed().horizon_s, 20.0)
        history = RollingHistory()
        _add_breath(history, START, 200.0, rate=20.0)
        span = history._breath[-1][0] - history._breath[0][0]
        self.assertGreater(span, 120.0)
        self.assertLess(span, 120.0 + 8.0 + 1.0)

    def test_default_channels_are_the_frontal_set_and_fz_is_rejected(self):
        history = RollingHistory()
        self.assertEqual(history.eeg_channels, ("AF3", "AF4", "F3", "F4"))
        history.eeg_channels = ("F3", "F4")
        self.assertEqual(history.eeg_channels, ("F3", "F4"))
        with self.assertRaises(ValueError):
            history.eeg_channels = ("Fz",)
        with self.assertRaises(ValueError):
            history.eeg_channels = ()


class PupilStageTests(unittest.TestCase):
    def test_a_declared_stage_delay_is_removed(self):
        duration = 45.0
        event = START + 30.0
        history = RollingHistory()
        breath_times = START + np.arange(0.0, duration, 1.0 / 20.0)
        _add_breath(history, START, duration, pressure=0.1 * np.sin(2 * np.pi * 0.25 * (breath_times - START)) + _bump(breath_times, event, 0.35, 2.0))
        pupil_times = START + np.arange(0.0, duration, 1.0 / 40.0)
        pupil = 4.0 + _bump(pupil_times, event, 0.35, 1.5)
        for stamp, value in zip(pupil_times, pupil):
            history.add_pupil(float(stamp), 0, float(value))
            history.add_pupil(float(stamp), 1, float(value))
        eeg_times = START + np.arange(0.0, duration, 1.0 / 128.0)
        wave = 25.0 * np.sin(2 * np.pi * 2.5 * (eeg_times - START)) + 20.0 * np.sin(2 * np.pi * 10.0 * (eeg_times - START))
        wave = wave + _bump(eeg_times, event, 0.8, 30.0) * np.sin(2 * np.pi * 2.5 * (eeg_times - START))
        _add_eeg(history, START, duration, 128.0, wave)
        # Where a blink correction step will go: after the short-gap bridge, before the low-pass.
        history.pupil_stages.insert(1, _HalfSecondDelay())
        window = history.resample()
        names = [name for name, _delay in window.pupil_stage_delays_seconds]
        self.assertEqual(names, ["short-gap bridge", "half-second delay", "linear-phase low-pass"])
        self.assertAlmostEqual(window.delays_seconds["pupil diameter"], sum(delay for _name, delay in window.pupil_stage_delays_seconds), places=6)
        self.assertAlmostEqual(dict(window.pupil_stage_delays_seconds)["half-second delay"], 0.5, places=6)
        self.assertAlmostEqual(dict(window.pupil_stage_delays_seconds)["short-gap bridge"], 0.0, places=6)
        peak = _peak_time(window.time_seconds, window.pupil_diameter, event)
        self.assertLessEqual(abs(peak - event), 1.0 / OUTPUT_RATE_HZ, window.delays_seconds)


class SimulateFeedTests(unittest.TestCase):
    def test_a_simulated_session_fills_the_history(self):
        from sniff_sync import trigger
        from sniff_sync.simulate import Session

        previous = os.environ.get("SNIFF_SYNC_PORT")
        temporary = tempfile.TemporaryDirectory()
        session = Session(temporary.name)
        session.start()
        try:
            time.sleep(1.2)
            trigger("marker")
            self.assertTrue(session.wait_for_label("marker", timeout=3))
            history = session.hub.history
            self.assertGreater(len(history._breath), 5)
            self.assertGreater(len(history._pupil), 5)
            self.assertGreater(len(history._eeg), 5)
            self.assertTrue(any(label == "marker" for _stamp, label in history.triggers()))
            window = history.resample()
            self.assertEqual(len(window.time_seconds), 601)
            self.assertEqual(window.breathing.shape, window.time_seconds.shape)
            self.assertEqual(window.pupil_diameter.shape, window.time_seconds.shape)
            self.assertEqual(window.delta_envelope.shape, window.time_seconds.shape)
            self.assertEqual(window.alpha_envelope.shape, window.time_seconds.shape)
            self.assertEqual(window.delta_to_alpha.shape, window.time_seconds.shape)
            self.assertEqual(window.valid.shape, window.time_seconds.shape)
            self.assertIn("delta envelope", window.delays_seconds)
        finally:
            session.stop()
            temporary.cleanup()
            if previous is None:
                os.environ.pop("SNIFF_SYNC_PORT", None)
            else:
                os.environ["SNIFF_SYNC_PORT"] = previous


class _HalfSecondDelay(PupilStage):
    """Move samples 0.5 seconds later and say so.

    The value left at each time is the value from half a second earlier.
    ``delay_s`` tells the resampler to put those times back.
    """

    name = "half-second delay"
    delay_s = 0.5

    def apply(self, times, values):
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        if times.size < 3:
            return times, values
        period = float(np.median(np.diff(times)))
        if period <= 0:
            return times, values
        shift = int(round(0.5 / period))
        shifted = np.full(values.shape, np.nan)
        if 0 < shift < values.size:
            shifted[shift:] = values[:-shift]
        return times, shifted


def _correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _best_lag(reference, measured):
    reference = np.asarray(reference, dtype=float)
    measured = np.asarray(measured, dtype=float)
    reference = reference - np.mean(reference)
    measured = measured - np.mean(measured)
    scores = np.correlate(measured, reference, mode="full")
    return int(np.argmax(scores) - (reference.size - 1))


if __name__ == "__main__":
    unittest.main()
