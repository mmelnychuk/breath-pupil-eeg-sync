"""Linear-phase filters for the live view, and zero-phase filters for the reference.

A symmetric finite impulse response filter with an odd number of taps
delays every frequency by the same whole number of samples,
(taps − 1) / 2. Subtracting that delay leaves a real frequency response:
the phase is 0 where the filter is positive and pi where the sign flips.

The live path uses the shorter set, called "live". It is for the display
shown while someone is being recorded, and every live result is approximate.
The longer set, called "clean", is the shortest symmetric filter that still
meets the accuracy tolerances below. The post-session reference does not use
either of these. It uses forward-backward filtering on the whole recording,
so the reference has no delay at all except at the very start and end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, firwin, freqz, group_delay, kaiserord, lfilter, sosfiltfilt

# After the delay is removed, the imaginary part of the frequency response
# must stay under this fraction of (1 + |real part|). That is about 1e-8
# radians of phase, in the passband and in the stopband.
PHASE_IMAGINARY_TOLERANCE = 1e-8

# Where the gain is above -60 dB, the group delay must stay within this
# many samples of (taps - 1) / 2. Deep zeros make the group delay undefined,
# so they are left out of the check.
GROUP_DELAY_TOLERANCE_SAMPLES = 0.05
GROUP_DELAY_MINIMUM_GAIN = 10 ** (-60.0 / 20.0)

# Clean accuracy. The stated passband stays within 1 dB of unity gain.
# The named stop frequencies are at least 40 dB down.
CLEAN_PASSBAND_RIPPLE_DB = 1.0
CLEAN_STOPBAND_DB = 40.0

# Live accuracy that the half-second design is still required to meet.
# A steady tone's envelope, peak to peak, stays within these fractions of
# its own median. A pure theta tone (6 hertz) produces an envelope no larger
# than this fraction of an equal delta tone.
LIVE_ENVELOPE_RIPPLE_AT_1_HZ = 0.15
LIVE_ENVELOPE_RIPPLE_AT_2_HZ = 0.10
LIVE_THETA_LEAKAGE = 0.10


@dataclass(frozen=True)
class FirSpec:
    """One Kaiser design. ``cuts_hz`` is one frequency for a low-pass, or two for a band-pass."""

    name: str
    kind: str
    cuts_hz: tuple
    transition_hz: float
    attenuation_db: float
    # Frequencies used only when checking the clean accuracy tolerance.
    passband_hz: tuple
    stop_hz: tuple


@dataclass(frozen=True)
class FirFilter:
    """A designed filter. ``delay_samples`` is the exact integer (taps - 1) / 2."""

    spec: FirSpec
    sample_rate_hz: float
    coefficients: np.ndarray
    delay_samples: int

    @property
    def taps(self) -> int:
        return int(self.coefficients.size)

    @property
    def delay_seconds(self) -> float:
        return self.delay_samples / self.sample_rate_hz

    @property
    def transition_bands_hz(self) -> tuple:
        """The two transition bands, each ``transition_hz`` wide and centred on a cut."""
        width = self.spec.transition_hz
        bands = []
        for cut in self.spec.cuts_hz:
            bands.append((cut - width / 2.0, cut + width / 2.0))
        return tuple(bands)


# The live set is the default. Its delta path is the shortest cascade whose
# envelope ripple and theta leakage stay inside the live tolerances above.
# At 128 hertz the delta path (band-pass plus envelope smoother) is 0.547 seconds.
LIVE_ANTI_ALIAS = FirSpec(
    "breathing and pupil low-pass",
    "lowpass",
    (2.5,),
    2.5,
    30.0,
    (0.0, 1.0),
    (5.0,),
)
LIVE_DELTA = FirSpec(
    "delta band-pass",
    "bandpass",
    (0.5, 4.5),
    2.2,
    25.0,
    (2.0, 3.5),
    (6.0,),
)
LIVE_ALPHA = FirSpec(
    "alpha band-pass",
    "bandpass",
    (8.0, 12.0),
    2.2,
    25.0,
    (9.0, 11.0),
    (6.0, 16.0),
)
LIVE_ENVELOPE = FirSpec(
    "envelope low-pass",
    "lowpass",
    (1.0,),
    2.2,
    25.0,
    (0.0, 0.4),
    (4.0,),
)

# Shortest designs that still meet the clean ripple and stopband tolerances.
CLEAN_ANTI_ALIAS = FirSpec(
    "breathing and pupil low-pass",
    "lowpass",
    (3.0,),
    2.0,
    40.0,
    (0.0, 2.0),
    (5.0,),
)
CLEAN_DELTA = FirSpec(
    "delta band-pass",
    "bandpass",
    (0.65, 4.35),
    0.70,
    40.0,
    (1.0, 4.0),
    (0.2, 6.0),
)
CLEAN_ALPHA = FirSpec(
    "alpha band-pass",
    "bandpass",
    (7.5, 12.5),
    1.0,
    40.0,
    (8.0, 12.0),
    (6.0, 16.0),
)
CLEAN_ENVELOPE = FirSpec(
    "envelope low-pass",
    "lowpass",
    (1.0,),
    1.0,
    40.0,
    (0.0, 0.5),
    (4.0,),
)

PROFILES = {
    "live": {
        "anti_alias": LIVE_ANTI_ALIAS,
        "delta": LIVE_DELTA,
        "alpha": LIVE_ALPHA,
        "envelope": LIVE_ENVELOPE,
    },
    "clean": {
        "anti_alias": CLEAN_ANTI_ALIAS,
        "delta": CLEAN_DELTA,
        "alpha": CLEAN_ALPHA,
        "envelope": CLEAN_ENVELOPE,
    },
}

_CACHE = {}


def _symmetrize(coefficients):
    """Make the two sides of the filter the same float, not merely close."""
    coefficients = np.array(coefficients, dtype=float, copy=True)
    half = coefficients.size // 2
    for index in range(half):
        value = 0.5 * (coefficients[index] + coefficients[-1 - index])
        coefficients[index] = value
        coefficients[-1 - index] = value
    return coefficients


def design(spec: FirSpec, sample_rate_hz: float) -> FirFilter:
    """Design one odd-length symmetric filter at this sample rate.

    The same specification is reused. A rate that cannot support the cuts
    raises ``ValueError``.
    """
    sample_rate_hz = float(sample_rate_hz)
    key = (
        spec.name,
        spec.kind,
        tuple(round(cut, 6) for cut in spec.cuts_hz),
        round(spec.transition_hz, 6),
        round(spec.attenuation_db, 6),
        round(sample_rate_hz, 6),
    )
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    nyquist = sample_rate_hz / 2.0
    if any(cut <= 0.0 or cut >= nyquist * 0.98 for cut in spec.cuts_hz):
        raise ValueError(
            "The %s cuts %s hertz do not fit under a %.1f hertz sample rate."
            % (spec.name, spec.cuts_hz, sample_rate_hz)
        )
    width = spec.transition_hz / nyquist
    if width <= 0.0 or width >= 1.0:
        raise ValueError(
            "The %s transition of %.2f hertz does not fit a %.1f hertz sample rate."
            % (spec.name, spec.transition_hz, sample_rate_hz)
        )
    taps, beta = kaiserord(spec.attenuation_db, width)
    if taps % 2 == 0:
        taps += 1
    window = ("kaiser", beta)
    if spec.kind == "lowpass":
        coefficients = firwin(taps, spec.cuts_hz[0], fs=sample_rate_hz, window=window)
    elif spec.kind == "bandpass":
        coefficients = firwin(
            taps, list(spec.cuts_hz), pass_zero=False, fs=sample_rate_hz, window=window
        )
    else:
        raise ValueError("Unknown filter kind %r." % spec.kind)
    coefficients = _symmetrize(coefficients)
    if coefficients.size % 2 == 0:
        raise RuntimeError("The %s filter was not given an odd length." % spec.name)
    designed = FirFilter(
        spec=spec,
        sample_rate_hz=sample_rate_hz,
        coefficients=coefficients,
        delay_samples=(coefficients.size - 1) // 2,
    )
    _CACHE[key] = designed
    return designed


def apply_fir(values, fir: FirFilter):
    """Run one filter forward. The first ``taps - 1`` samples of each stretch are missing.

    Returns the filtered values, still lined up with the input index, and the
    integer delay in samples. The value at index n describes the input at
    index n minus that delay.
    """
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    taps = fir.taps
    finite = np.isfinite(values)
    if not np.any(finite):
        return output, fir.delay_samples
    padded = np.concatenate([[False], finite, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        if int(stop - start) <= taps:
            continue
        filtered = lfilter(fir.coefficients, [1.0], values[int(start) : int(stop)])
        filtered = np.array(filtered, dtype=float, copy=True)
        filtered[: taps - 1] = np.nan
        output[int(start) : int(stop)] = filtered
    return output, fir.delay_samples


def compensated_response(fir: FirFilter, frequencies_hz):
    """Frequency response after removing the exact integer delay. It is real for a symmetric filter."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    _frequencies, response = freqz(fir.coefficients, worN=frequencies, fs=fir.sample_rate_hz)
    correction = np.exp(1j * 2.0 * np.pi * frequencies * fir.delay_samples / fir.sample_rate_hz)
    return response * correction


def group_delay_samples(fir: FirFilter, frequencies_hz):
    """Group delay in samples at the requested frequencies."""
    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    _grid, delays = group_delay((fir.coefficients, [1.0]), w=frequencies, fs=fir.sample_rate_hz)
    return np.asarray(delays, dtype=float)


def gain_db(fir: FirFilter, frequencies_hz):
    frequencies = np.asarray(frequencies_hz, dtype=float)
    _grid, response = freqz(fir.coefficients, worN=frequencies, fs=fir.sample_rate_hz)
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))


def profile_specs(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError("The filter profile must be 'live' or 'clean', not %r." % profile)
    return PROFILES[profile]


def path_delay_seconds(profile: str, sample_rate_hz: float) -> dict:
    """Delay of each live or clean path, in seconds, at one sample rate.

    The envelope path delay is the band-pass delay plus the smoother delay,
    because rectification itself has no memory.
    """
    specs = profile_specs(profile)
    anti = design(specs["anti_alias"], sample_rate_hz)
    delta = design(specs["delta"], sample_rate_hz)
    alpha = design(specs["alpha"], sample_rate_hz)
    envelope = design(specs["envelope"], sample_rate_hz)
    return {
        "anti-alias": anti.delay_seconds,
        "delta band-pass": delta.delay_seconds,
        "alpha band-pass": alpha.delay_seconds,
        "envelope low-pass": envelope.delay_seconds,
        "delta envelope": delta.delay_seconds + envelope.delay_seconds,
        "alpha envelope": alpha.delay_seconds + envelope.delay_seconds,
        "aligned grid": max(
            anti.delay_seconds,
            delta.delay_seconds + envelope.delay_seconds,
            alpha.delay_seconds + envelope.delay_seconds,
        ),
    }


def describe(profile: str, sample_rate_hz: float = 128.0) -> list:
    """One row per filter: taps, transition, design attenuation, and delay."""
    rows = []
    for spec in profile_specs(profile).values():
        fir = design(spec, sample_rate_hz)
        rows.append(
            {
                "profile": profile,
                "name": spec.name,
                "sample_rate_hz": float(sample_rate_hz),
                "taps": fir.taps,
                "transition_hz": spec.transition_hz,
                "attenuation_db": spec.attenuation_db,
                "cuts_hz": spec.cuts_hz,
                "delay_samples": fir.delay_samples,
                "delay_seconds": fir.delay_seconds,
            }
        )
    return rows


def zero_phase_lowpass(values, sample_rate_hz, cutoff_hz):
    """Forward-backward low-pass. There is no delay. The ends of the record are the unreliable part."""
    sections = butter(4, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sections, np.asarray(values, dtype=float))


def zero_phase_bandpass(values, sample_rate_hz, low_hz, high_hz):
    """Forward-backward band-pass. There is no delay."""
    sections = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sections, np.asarray(values, dtype=float))


def zero_phase_envelope(values, sample_rate_hz, low_hz, high_hz, smoother_hz=1.0):
    """Forward-backward band-pass, absolute value, then a forward-backward smoother.

    Rectification has no delay of its own. Both linear filters are applied
    in both directions, so the envelope is lined up with the input.
    A unit sine comes back near amplitude 1.
    """
    band = zero_phase_bandpass(values, sample_rate_hz, low_hz, high_hz)
    smoother = butter(4, smoother_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(smoother, np.abs(band)) * (np.pi / 2.0)
