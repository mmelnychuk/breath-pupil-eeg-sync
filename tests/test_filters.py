"""Linear-phase filters: exact symmetry, integer delay, and a real frequency response."""

import unittest

import numpy as np
from scipy.signal import freqz, lfilter

from sniff_sync.filters import (
    GROUP_DELAY_MINIMUM_GAIN,
    GROUP_DELAY_TOLERANCE_SAMPLES,
    PHASE_IMAGINARY_TOLERANCE,
    PROFILES,
    compensated_response,
    design,
    gain_db,
    group_delay_samples,
)


class SymmetryTests(unittest.TestCase):
    def test_every_filter_is_symmetric_with_an_integer_delay(self):
        for profile, specs in PROFILES.items():
            for sample_rate in (20.0, 32.0, 40.0, 120.0, 128.0):
                for spec in specs.values():
                    with self.subTest(profile=profile, name=spec.name, sample_rate=sample_rate):
                        try:
                            fir = design(spec, sample_rate)
                        except ValueError:
                            # A 12 hertz band does not fit under a 20 hertz sample rate.
                            continue
                        coefficients = fir.coefficients
                        self.assertEqual(coefficients.size % 2, 1)
                        self.assertTrue(np.array_equal(coefficients, coefficients[::-1]))
                        self.assertEqual(fir.delay_samples, (coefficients.size - 1) // 2)
                        self.assertIsInstance(fir.delay_samples, int)

    def test_compensated_response_is_real_and_the_group_delay_is_flat(self):
        frequencies = np.linspace(0.05, 60.0, 2048)
        for profile, specs in PROFILES.items():
            for spec in specs.values():
                with self.subTest(profile=profile, name=spec.name):
                    fir = design(spec, 128.0)
                    usable = frequencies < 128.0 / 2.0
                    compensated = compensated_response(fir, frequencies[usable])
                    scale = 1.0 + np.abs(np.real(compensated))
                    imaginary = np.max(np.abs(np.imag(compensated)) / scale)
                    self.assertLessEqual(imaginary, PHASE_IMAGINARY_TOLERANCE)
                    # Where the response is not a deep zero, the phase is 0 or pi
                    # and the group delay matches the integer tap delay.
                    magnitude = np.abs(compensated)
                    kept = magnitude >= GROUP_DELAY_MINIMUM_GAIN
                    phase = np.angle(compensated[kept])
                    distance = np.minimum(np.abs(phase), np.abs(np.abs(phase) - np.pi))
                    self.assertLessEqual(float(np.max(distance)), 1e-6)
                    delays = group_delay_samples(fir, frequencies[usable][kept])
                    finite = np.isfinite(delays)
                    self.assertGreater(int(np.count_nonzero(finite)), 10)
                    self.assertLessEqual(
                        float(np.max(np.abs(delays[finite] - fir.delay_samples))),
                        GROUP_DELAY_TOLERANCE_SAMPLES,
                    )


class CleanAccuracyTests(unittest.TestCase):
    def test_clean_filters_meet_the_stated_stopband(self):
        checks = {
            "anti_alias": ((0.25, 1.0), (5.0,)),
            "delta": ((1.0, 2.5, 4.0), (0.2, 6.0)),
            "alpha": ((8.0, 10.0, 12.0), (6.0, 16.0)),
            "envelope": ((0.1, 0.5), (4.0,)),
        }
        specs = PROFILES["clean"]
        for key, (passband, stop) in checks.items():
            fir = design(specs[key], 128.0)
            passed = gain_db(fir, passband)
            stopped = gain_db(fir, stop)
            self.assertTrue(np.all(passed >= -1.0), "%s passband %s" % (key, passed))
            self.assertTrue(np.all(passed <= 1.0), "%s passband %s" % (key, passed))
            self.assertTrue(np.all(stopped <= -40.0), "%s stopband %s" % (key, stopped))


class EnvelopeAndResampleTests(unittest.TestCase):
    def test_envelope_delay_matches_the_sum_of_the_two_filters(self):
        from sniff_sync.filters import apply_fir

        sample_rate = 128.0
        bandpass = design(PROFILES["live"]["delta"], sample_rate)
        smoother = design(PROFILES["live"]["envelope"], sample_rate)
        expected = (bandpass.delay_samples + smoother.delay_samples) / sample_rate
        duration = 30.0
        times = np.arange(0.0, duration, 1.0 / sample_rate)
        for rate in (0.05, 0.10, 0.20):
            amplitude = 1.0 + 0.5 * np.sin(2.0 * np.pi * rate * times)
            wave = amplitude * np.sin(2.0 * np.pi * 2.5 * times)
            band, _band_delay = apply_fir(wave, bandpass)
            rectified = np.abs(band)
            rectified[~np.isfinite(band)] = np.nan
            # The smoother does not accept internal missing samples, so fill the blank head with zeros
            # only for this delay measurement: the blank is the filter startup, before the tone is used.
            rectified = np.where(np.isfinite(rectified), rectified, 0.0)
            envelope, _smooth_delay = apply_fir(rectified, smoother)
            shift = bandpass.delay_samples + smoother.delay_samples
            start = int(8 * sample_rate)
            stop = int(-2 * sample_rate)
            reference = amplitude[start:stop]
            measured = envelope[start + shift : stop + shift]
            count = min(len(reference), len(measured))
            reference = reference[:count]
            measured = measured[:count]
            self.assertGreater(np.corrcoef(reference, measured)[0, 1], 0.98)
            reference = reference - np.mean(reference)
            measured = measured - np.mean(measured)
            scores = np.correlate(measured, reference, mode="full")
            lag = int(np.argmax(scores) - (len(reference) - 1))
            self.assertLessEqual(abs(lag) / sample_rate, 1.0 / sample_rate)

    def test_resampling_and_the_gap_bridge_do_not_shift_a_slow_wave(self):
        from sniff_sync.rolling import OUTPUT_RATE_HZ, ShortGapBridge, linear_phase_lowpass, onto_grid

        sample_rate = 20.0
        duration = 40.0
        times = np.arange(0.0, duration, 1.0 / sample_rate)
        # Uneven timestamps: each sample is moved by up to a fifth of the period.
        jitter = 0.2 / sample_rate * np.sin(2.0 * np.pi * 0.07 * times)
        stamps = times + jitter
        order = np.argsort(stamps)
        stamps = stamps[order]
        wave = np.sin(2.0 * np.pi * 0.25 * stamps)
        filtered_times, filtered, delay = linear_phase_lowpass(stamps, wave, profile="live")
        grid = np.arange(5.0, 35.0, 1.0 / OUTPUT_RATE_HZ)
        sampled = onto_grid(grid, filtered_times, filtered, delay)
        expected = np.sin(2.0 * np.pi * 0.25 * grid)
        finite = np.isfinite(sampled)
        self.assertGreater(int(np.count_nonzero(finite)), 100)
        sampled = sampled[finite]
        expected = expected[finite]
        sampled = sampled - np.mean(sampled)
        expected = expected - np.mean(expected)
        scores = np.correlate(sampled, expected, mode="full")
        lag = int(np.argmax(scores) - (len(expected) - 1))
        # One 10 hertz sample is the tolerance. The measured lag should be much smaller.
        self.assertLessEqual(abs(lag) / OUTPUT_RATE_HZ, 0.05)

        # A 0.15 second hole, bridged, then filtered. The slow wave should not move.
        gap = (stamps >= 20.0) & (stamps < 20.15)
        stamps_g = stamps[~gap]
        wave_g = np.sin(2.0 * np.pi * 0.25 * stamps_g)
        uniform_times, uniform_values, _rate = _uniform(stamps_g, wave_g)
        bridged_times, bridged = ShortGapBridge(max_gap_seconds=0.20).apply(uniform_times, uniform_values)
        filtered_times, filtered, delay = linear_phase_lowpass(bridged_times, bridged, profile="live")
        sampled = onto_grid(grid, filtered_times, filtered, delay)
        finite = np.isfinite(sampled)
        sampled = sampled[finite] - np.mean(sampled[finite])
        expected = np.sin(2.0 * np.pi * 0.25 * grid[finite])
        expected = expected - np.mean(expected)
        scores = np.correlate(sampled, expected, mode="full")
        lag = int(np.argmax(scores) - (len(expected) - 1))
        self.assertLessEqual(abs(lag) / OUTPUT_RATE_HZ, 0.05)


def _uniform(times, values):
    from sniff_sync.rolling import _as_uniform

    prepared = _as_uniform(times, values)
    return prepared


if __name__ == "__main__":
    unittest.main()
