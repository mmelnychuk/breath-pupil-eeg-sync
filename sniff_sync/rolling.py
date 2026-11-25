"""A rolling history and a causal resampler for live coupling.

The live plot keeps about 20 seconds. Coupling needs a longer window, so
this buffer is separate and does not change that plot. Call ``resample``
about once a second. It returns breathing, pupil diameter, and the
electroencephalography (EEG) envelopes on one grid at 10 hertz. Times
are Unix seconds.

The live filters are symmetric finite impulse response filters with an
odd number of taps. Every frequency is delayed by the same whole number
of samples, (taps − 1) / 2. That delay is subtracted from the timestamps,
so a change that happened at one moment in the raw streams shows up at
that same moment on the grid. The result is for the live display and is
approximate. The post-session reference uses forward-backward filtering
instead, and has no delay.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.filters import apply_fir, design, profile_specs

# One sample every 0.1 seconds.
OUTPUT_RATE_HZ = 10.0

DEFAULT_WINDOW_SECONDS = 60.0
MINIMUM_WINDOW_SECONDS = 30.0
MAXIMUM_WINDOW_SECONDS = 120.0

# Raw samples kept before the window, so the filters have already settled
# when the window starts. The slowest step used here settles in about two
# seconds; eight seconds leaves a margin.
WARMUP_SECONDS = 8.0

# The live profile is the display default. "clean" is the longer
# linear-phase set that meets the tighter stopband tolerance.
DEFAULT_FILTER_PROFILE = "live"

DELTA_BAND_HZ = (1.0, 4.0)
ALPHA_BAND_HZ = (8.0, 12.0)

# The original Emotiv EPOC has no electrode at Fz. AF3 and F3 are the
# frontal electrodes on the left, AF4 and F4 the frontal electrodes on
# the right. F7 and F8 sit further to the side and pick up more muscle.
# FC5 and FC6 sit lower, toward the motor strip. This is the same frontal
# set the offline analysis averages.
DEFAULT_EEG_CHANNELS = ("AF3", "AF4", "F3", "F4")

# On the merged pupil trace, a hole this long is drawn as a straight line.
# Anything longer is left missing. Blink correction is not this step; it
# can be inserted later and will have its own delay.
PUPIL_SHORT_GAP_SECONDS = 0.20

# A near-zero alpha envelope must not send the ratio to infinity.
# The floor is this fraction of the median alpha in the window.
RATIO_FLOOR_FRACTION = 0.001

_HISTORY_SECONDS = MAXIMUM_WINDOW_SECONDS + WARMUP_SECONDS


def _check_eeg_channels(names) -> tuple:
    chosen = tuple(names)
    if not chosen:
        raise ValueError("Choose at least one electroencephalography channel.")
    unknown = [name for name in chosen if name not in EPOC_CHANNELS]
    if unknown:
        raise ValueError(
            "Unknown electroencephalography channel(s): %s. "
            "The original EPOC channels are: %s."
            % (", ".join(unknown), ", ".join(EPOC_CHANNELS))
        )
    return chosen


def _check_window_seconds(seconds: float) -> float:
    seconds = float(seconds)
    if seconds < MINIMUM_WINDOW_SECONDS or seconds > MAXIMUM_WINDOW_SECONDS:
        raise ValueError(
            "The window length must be between %g and %g seconds. %g is outside that range."
            % (MINIMUM_WINDOW_SECONDS, MAXIMUM_WINDOW_SECONDS, seconds)
        )
    return seconds


def _last_unique(times, values):
    """Sort by time and keep the last value when a time is repeated."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size == 0:
        return times, values
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    if times.size == 1:
        return times, values
    change = np.flatnonzero(np.diff(times) > 0)
    keep = np.concatenate([change, [times.size - 1]])
    return times[keep], values[keep]


def _finite_runs(values):
    """Index ranges ``[start, stop)`` where the values are finite."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    padded = np.concatenate([[False], finite, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        yield int(start), int(stop)


def fill_short_gaps(times, values, max_gap_seconds):
    """Fill internal holes with a straight line. Leave long holes missing.

    A hole that reaches the first or the last sample is left missing,
    because the value on the far side has not been seen. ``max_gap_seconds``
    is the time from the last present sample before the hole to the first
    present sample after it.
    """
    times = np.asarray(times, dtype=float)
    filled = np.array(values, dtype=float, copy=True)
    finite = np.isfinite(filled)
    if finite.all() or not np.any(finite):
        return filled
    missing = np.flatnonzero(~finite)
    splits = np.flatnonzero(np.diff(missing) > 1) + 1
    for gap in np.split(missing, splits):
        begin = int(gap[0])
        finish = int(gap[-1])
        if begin == 0 or finish == filled.size - 1:
            continue
        duration = float(times[finish + 1] - times[begin - 1])
        # A tenth of a millisecond of slack, so a gap that lands on the
        # limit is still filled when the sample times are not exact.
        if duration <= max_gap_seconds + 1e-4:
            filled[begin : finish + 1] = np.interp(
                times[begin : finish + 1],
                [times[begin - 1], times[finish + 1]],
                [filled[begin - 1], filled[finish + 1]],
            )
    return filled


def _few_samples_seconds(period_seconds: float) -> float:
    """A hole of one or two samples, with a little slack for uneven spacing."""
    return 3.3 * float(period_seconds)


def interpolate_with_gaps(times, values, grid, max_gap_seconds):
    """Straight-line samples on ``grid``. Holes longer than ``max_gap_seconds`` stay missing."""
    times, values = _last_unique(times, values)
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    out = np.full(grid.shape, np.nan, dtype=float)
    if times.size == 0:
        return out
    if times.size == 1:
        close = np.abs(grid - times[0]) <= max_gap_seconds
        out[close] = values[0]
        return out
    right = np.searchsorted(times, grid, side="right")
    left = right - 1
    has_left = left >= 0
    has_right = right < times.size
    dist_left = np.full(grid.shape, np.inf, dtype=float)
    dist_right = np.full(grid.shape, np.inf, dtype=float)
    dist_left[has_left] = grid[has_left] - times[left[has_left]]
    dist_right[has_right] = times[right[has_right]] - grid[has_right]
    inside = has_left & has_right
    gap = np.full(grid.shape, np.inf, dtype=float)
    gap[inside] = dist_left[inside] + dist_right[inside]
    usable = inside & (gap <= max_gap_seconds) & (gap > 0)
    if np.any(usable):
        span = gap[usable]
        weight = (grid[usable] - times[left[usable]]) / span
        out[usable] = values[left[usable]] * (1.0 - weight) + values[right[usable]] * weight
    # A grid point just after the last sample before a hole has no near
    # neighbour on the far side. Keep it when a real sample is close, so
    # the edge of the hole is not thrown away.
    nearest = np.minimum(dist_left, dist_right)
    period = float(np.median(np.diff(times)))
    edge = ~np.isfinite(out) & (nearest <= max(period, 0.0))
    take_left = edge & (dist_left <= dist_right) & has_left
    take_right = edge & (dist_right < dist_left) & has_right
    if np.any(take_left):
        out[take_left] = values[left[take_left]]
    if np.any(take_right):
        out[take_right] = values[right[take_right]]
    return out


def _uniform_from_samples(times, values):
    """Copy irregular samples onto an even grid at their own median rate.

    One or two missed samples are filled. Longer holes stay missing.
    Returns ``(grid, values, sample_rate_hz)`` or ``None`` when the series
    is too short.
    """
    finite = np.isfinite(times) & np.isfinite(values)
    times = np.asarray(times, dtype=float)[finite]
    values = np.asarray(values, dtype=float)[finite]
    if times.size < 4:
        return None
    times, values = _last_unique(times, values)
    if times.size < 4:
        return None
    period = float(np.median(np.diff(times)))
    if not math.isfinite(period) or period <= 0:
        return None
    sample_rate = 1.0 / period
    start = float(times[0])
    stop = float(times[-1])
    count = int(math.floor((stop - start) / period)) + 1
    if count < 4:
        return None
    grid = start + np.arange(count) * period
    filled = interpolate_with_gaps(times, values, grid, _few_samples_seconds(period))
    return grid, filled, sample_rate


def _as_uniform(times, values):
    """Even grid. A series that is already even keeps its missing samples."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 4:
        return None
    period = float(np.median(np.diff(times)))
    if not math.isfinite(period) or period <= 0:
        return None
    deltas = np.diff(times)
    already_even = np.max(np.abs(deltas - period)) <= 0.05 * period
    if already_even:
        filled = fill_short_gaps(times, values, _few_samples_seconds(period))
        return times, filled, 1.0 / period
    return _uniform_from_samples(times, values)


def linear_phase_lowpass(times, values, profile=DEFAULT_FILTER_PROFILE):
    """Low-pass onto an even grid with a symmetric finite impulse response filter.

    Returns ``(times, values, delay_seconds)``. The times are not shifted.
    ``delay_seconds`` is exactly (taps − 1) / 2 divided by the sample rate.
    The first taps − 1 samples of each unbroken stretch are missing, because
    the filter has not seen a full window yet.
    """
    prepared = _as_uniform(times, values)
    if prepared is None:
        return np.asarray(times, dtype=float), np.full(len(times), np.nan), float("nan")
    grid, samples, sample_rate = prepared
    try:
        fir = design(profile_specs(profile)["anti_alias"], sample_rate)
    except ValueError:
        return grid, np.full(grid.shape, np.nan), float("nan")
    filtered, delay_samples = apply_fir(samples, fir)
    return grid, filtered, delay_samples / sample_rate


def _band_envelope(samples, sample_rate_hz, band_name, profile=DEFAULT_FILTER_PROFILE):
    """Amplitude envelope of one band, and its delay in seconds.

    The series is band-pass filtered, rectified, and smoothed. Both filters
    are symmetric and odd-length, so each delays every frequency by the same
    whole number of samples. Rectification has no memory, so the envelope
    delay is the sum of the two filter delays. A unit sine comes back near
    amplitude 1, because the mean of the absolute value of a unit sine is
    2 / pi and that factor is put back.
    """
    specs = profile_specs(profile)
    try:
        bandpass = design(specs[band_name], sample_rate_hz)
        smoother = design(specs["envelope"], sample_rate_hz)
    except ValueError:
        return None
    band, band_delay = apply_fir(samples, bandpass)
    rectified = np.abs(band)
    rectified[~np.isfinite(band)] = np.nan
    envelope, smooth_delay = apply_fir(rectified, smoother)
    envelope = envelope * (np.pi / 2.0)
    return envelope, (band_delay + smooth_delay) / sample_rate_hz


def merge_eyes(samples):
    """One pupil series from the two eyes.

    When both eyes have a sample at the same grid time, the value is their
    mean. When only one eye does, that eye is used. A gap of one or two
    samples in a single eye is bridged; a longer gap in that eye is missing
    and the other eye can still fill it.
    """
    by_eye = {0: [], 1: []}
    for stamp, eye, value in samples:
        if eye not in by_eye:
            continue
        if not math.isfinite(stamp) or not math.isfinite(value):
            continue
        by_eye[eye].append((float(stamp), float(value)))
    present = {eye: pairs for eye, pairs in by_eye.items() if len(pairs) >= 2}
    if not present:
        return None
    pieces = []
    for pairs in present.values():
        eye_times = np.array([pair[0] for pair in pairs], dtype=float)
        eye_values = np.array([pair[1] for pair in pairs], dtype=float)
        eye_times, eye_values = _last_unique(eye_times, eye_values)
        pieces.append((eye_times, eye_values))
    all_times = np.unique(np.concatenate([eye_times for eye_times, _values in pieces]))
    if all_times.size < 4:
        return None
    period = float(np.median(np.diff(all_times)))
    if not math.isfinite(period) or period <= 0:
        return None
    start = float(all_times[0])
    stop = float(all_times[-1])
    count = int(math.floor((stop - start) / period)) + 1
    if count < 4:
        return None
    grid = start + np.arange(count) * period
    columns = []
    for eye_times, eye_values in pieces:
        eye_period = float(np.median(np.diff(eye_times))) if eye_times.size > 1 else period
        if not math.isfinite(eye_period) or eye_period <= 0:
            eye_period = period
        max_gap = max(_few_samples_seconds(eye_period), period)
        columns.append(interpolate_with_gaps(eye_times, eye_values, grid, max_gap))
    stack = np.vstack(columns)
    count_present = np.sum(np.isfinite(stack), axis=0)
    averaged = np.full(grid.shape, np.nan, dtype=float)
    usable = count_present > 0
    if np.any(usable):
        averaged[usable] = np.nanmean(stack[:, usable], axis=0)
    return grid, averaged


def onto_grid(grid, times, values, delay_seconds):
    """Read a series on ``grid`` after shifting its times earlier by the delay.

    Interpolation does not cross a missing stretch, so a long gap stays missing.
    """
    grid = np.asarray(grid, dtype=float)
    out = np.full(grid.shape, np.nan, dtype=float)
    if times is None or delay_seconds is None or not math.isfinite(delay_seconds):
        return out
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size == 0 or values.size == 0 or times.size != values.size:
        return out
    compensated = times - float(delay_seconds)
    for start, stop in _finite_runs(values):
        if stop - start < 2:
            continue
        piece_t = compensated[start:stop]
        piece_v = values[start:stop]
        if np.any(np.diff(piece_t) <= 0):
            piece_t, piece_v = _last_unique(piece_t, piece_v)
            if piece_t.size < 2:
                continue
        select = (grid >= piece_t[0]) & (grid <= piece_t[-1] + 1e-9)
        if np.any(select):
            out[select] = np.interp(grid[select], piece_t, piece_v)
    return out


def _output_grid(end_seconds, window_seconds):
    """Inclusive 10 hertz grid of about ``window_seconds``, ending at ``end_seconds``."""
    steps = int(round(window_seconds * OUTPUT_RATE_HZ))
    span = steps / OUTPUT_RATE_HZ
    grid = end_seconds - span + np.arange(steps + 1) / OUTPUT_RATE_HZ
    grid[-1] = end_seconds
    return grid


def _ratio(delta, alpha):
    ratio = np.full(delta.shape, np.nan, dtype=float)
    both = np.isfinite(delta) & np.isfinite(alpha)
    if not np.any(both):
        return ratio
    positive = alpha[both & (alpha > 0)]
    if positive.size == 0:
        return ratio
    floor = RATIO_FLOOR_FRACTION * float(np.median(positive))
    if not math.isfinite(floor) or floor <= 0:
        return ratio
    ratio[both] = delta[both] / np.maximum(alpha[both], floor)
    return ratio


class PupilStage:
    """One step on the pupil path.

    A blink-correction step can be inserted into
    ``RollingHistory.pupil_stages``. It should leave the returned values
    lined up with the times it was given, and set ``delay_s`` to the
    number of seconds it shifts the diameter later. The resampler adds
    that to the other pupil delays and shifts the timestamps back.
    """

    name = "pupil stage"
    delay_s = 0.0

    def apply(self, times, values):
        raise NotImplementedError


class ShortGapBridge(PupilStage):
    """Fill a short pupil hole with a straight line. Leave a long hole missing."""

    name = "short-gap bridge"

    def __init__(self, max_gap_seconds=PUPIL_SHORT_GAP_SECONDS):
        self.max_gap_seconds = float(max_gap_seconds)
        self.delay_s = 0.0

    def apply(self, times, values):
        return np.asarray(times, dtype=float), fill_short_gaps(times, values, self.max_gap_seconds)


class LinearPhaseLowpass(PupilStage):
    """Symmetric low-pass, so the 10 hertz grid does not alias.

    ``delay_s`` is filled in when the step runs. It is (taps − 1) / 2
    divided by the sample rate of the series it is given.
    """

    name = "linear-phase low-pass"

    def __init__(self, profile=DEFAULT_FILTER_PROFILE):
        self.profile = profile
        self.delay_s = 0.0

    def apply(self, times, values):
        filtered_times, filtered_values, delay = linear_phase_lowpass(times, values, profile=self.profile)
        self.delay_s = 0.0 if not math.isfinite(delay) else float(delay)
        return filtered_times, filtered_values


def default_pupil_stages(profile=DEFAULT_FILTER_PROFILE):
    """The pupil path with no blink correction.

    Put a blink-correction step after the short-gap bridge and before
    the low-pass. Give it a ``delay_s``.
    """
    return [ShortGapBridge(), LinearPhaseLowpass(profile=profile)]


@dataclass
class ResampledWindow:
    """The three signals on one Unix-time grid.

    ``valid`` is true only where breathing, pupil diameter, both envelopes,
    and the ratio all have a number. A missing sample is not-a-number in
    its own array as well. ``delays_seconds`` is how far each path was
    shifted earlier, in seconds. ``pupil_stage_delays_seconds`` adds up
    to the pupil delay.
    """

    time_seconds: np.ndarray
    breathing: np.ndarray
    pupil_diameter: np.ndarray
    delta_envelope: np.ndarray
    alpha_envelope: np.ndarray
    delta_to_alpha: np.ndarray
    valid: np.ndarray
    delays_seconds: dict
    pupil_stage_delays_seconds: tuple
    filter_profile: str = DEFAULT_FILTER_PROFILE


class RollingHistory:
    """Raw breathing, pupil diameter, and electroencephalography for the last two minutes.

    The plot's feed is not this object. Append here from the same places
    that append to the feed. ``resample`` does the filtering, and it is
    the call to make about once a second.
    """

    def __init__(
        self,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        eeg_channels=DEFAULT_EEG_CHANNELS,
        filter_profile=DEFAULT_FILTER_PROFILE,
    ):
        self._lock = threading.Lock()
        self._breath = deque()
        self._pupil = deque()
        self._eeg = deque()
        self._triggers = deque()
        self._window_seconds = DEFAULT_WINDOW_SECONDS
        self._eeg_channels = DEFAULT_EEG_CHANNELS
        self._filter_profile = DEFAULT_FILTER_PROFILE
        self.pupil_stages = default_pupil_stages(filter_profile)
        self.set_window_seconds(window_seconds)
        self.eeg_channels = eeg_channels
        self.filter_profile = filter_profile

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def set_window_seconds(self, seconds: float) -> None:
        """How many seconds ``resample`` should return. Allowed range is 30 to 120."""
        self._window_seconds = _check_window_seconds(seconds)

    @property
    def eeg_channels(self) -> tuple:
        return self._eeg_channels

    @eeg_channels.setter
    def eeg_channels(self, names) -> None:
        self._eeg_channels = _check_eeg_channels(names)

    @property
    def filter_profile(self) -> str:
        return self._filter_profile

    @filter_profile.setter
    def filter_profile(self, profile: str) -> None:
        profile_specs(profile)
        self._filter_profile = profile
        for stage in self.pupil_stages:
            if isinstance(stage, LinearPhaseLowpass):
                stage.profile = profile

    def add_breath(self, unix_time: float, pressure: float) -> None:
        parsed = _pair(unix_time, pressure)
        if parsed is None:
            return
        self._append(self._breath, parsed)

    def add_pupil(self, unix_time: float, eye: int, diameter_mm: float) -> None:
        try:
            eye_id = int(eye)
        except (TypeError, ValueError):
            return
        if eye_id not in (0, 1):
            return
        parsed = _pair(unix_time, diameter_mm)
        if parsed is None:
            return
        self._append(self._pupil, (parsed[0], eye_id, parsed[1]))

    def add_eeg(self, unix_time: float, channels: dict) -> None:
        try:
            stamp = float(unix_time)
        except (TypeError, ValueError):
            return
        if not math.isfinite(stamp):
            return
        clean = {}
        for name, value in dict(channels).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[str(name)] = number
        if not clean:
            return
        self._append(self._eeg, (stamp, clean))

    def add_trigger(self, unix_time: float, label: str) -> None:
        try:
            stamp = float(unix_time)
        except (TypeError, ValueError):
            return
        if not math.isfinite(stamp):
            return
        text = str(label).strip()
        if not text:
            return
        self._append(self._triggers, (stamp, text))

    def triggers(self):
        """``(unix_time, label)`` pairs still inside the history."""
        with self._lock:
            return list(self._triggers)

    def _append(self, rows, item) -> None:
        stamp = item[0]
        with self._lock:
            newest = stamp
            if rows:
                newest = max(stamp, rows[-1][0])
            cutoff = newest - _HISTORY_SECONDS
            if stamp < cutoff:
                return
            rows.append(item)
            while rows and rows[0][0] < cutoff:
                rows.popleft()

    def resample(self) -> ResampledWindow:
        """The last window, on one 10 hertz grid, with each path's delay removed.

        The grid ends at the latest time where breathing, pupil diameter,
        and both envelopes are all present. That time sits behind the newest
        packet by the longest filter delay. Samples inside a long gap are
        not-a-number, and ``valid`` is false there. The result is approximate:
        it is the live display path, not the post-session reference.
        """
        with self._lock:
            breath = list(self._breath)
            pupil = list(self._pupil)
            eeg = list(self._eeg)
            window_seconds = self._window_seconds
            channels = self._eeg_channels
            stages = list(self.pupil_stages)
            profile = self._filter_profile

        if not breath or not pupil or not eeg:
            return _empty_window(stages, profile=profile)

        # The buffer keeps the longest window. Filtering only needs this
        # window plus the warm-up, so a 30 second window does less work
        # than a 120 second one. The older samples stay in the buffer.
        keep = window_seconds + WARMUP_SECONDS
        breath = _recent(breath, keep)
        pupil = _recent(pupil, keep)
        eeg = _recent(eeg, keep)

        breath_times, breath_values, breath_delay = _breath_path(breath, profile)
        pupil_times, pupil_values, pupil_delay, stage_delays = _pupil_path(pupil, stages)
        eeg_times, delta, alpha, delta_delay, alpha_delay = _eeg_path(eeg, channels, profile)

        paths = (
            (breath_times, breath_values, breath_delay),
            (pupil_times, pupil_values, pupil_delay),
            (eeg_times, delta, delta_delay),
            (eeg_times, alpha, alpha_delay),
        )
        grid = _window_grid(
            paths,
            window_seconds,
            breath,
            pupil,
            eeg,
            breath_delay,
            pupil_delay,
            delta_delay,
            alpha_delay,
        )
        if grid is None:
            return _empty_window(stages, stage_delays, breath_delay, pupil_delay, delta_delay, alpha_delay, profile)

        breathing = onto_grid(grid, breath_times, breath_values, breath_delay)
        pupil_diameter = onto_grid(grid, pupil_times, pupil_values, pupil_delay)
        delta_envelope = onto_grid(grid, eeg_times, delta, delta_delay)
        alpha_envelope = onto_grid(grid, eeg_times, alpha, alpha_delay)
        delta_to_alpha = _ratio(delta_envelope, alpha_envelope)
        valid = (
            np.isfinite(breathing)
            & np.isfinite(pupil_diameter)
            & np.isfinite(delta_envelope)
            & np.isfinite(alpha_envelope)
            & np.isfinite(delta_to_alpha)
        )
        delays = {
            "breathing": float(breath_delay),
            "pupil diameter": float(pupil_delay),
            "delta envelope": float(delta_delay),
            "alpha envelope": float(alpha_delay),
        }
        return ResampledWindow(
            time_seconds=grid,
            breathing=breathing,
            pupil_diameter=pupil_diameter,
            delta_envelope=delta_envelope,
            alpha_envelope=alpha_envelope,
            delta_to_alpha=delta_to_alpha,
            valid=valid,
            delays_seconds=delays,
            pupil_stage_delays_seconds=tuple(stage_delays),
            filter_profile=profile,
        )


def _recent(samples, seconds):
    """Samples whose time is within ``seconds`` of the newest one."""
    if len(samples) < 2:
        return samples
    cutoff = samples[-1][0] - seconds
    low = 0
    high = len(samples)
    while low < high:
        mid = (low + high) // 2
        if samples[mid][0] < cutoff:
            low = mid + 1
        else:
            high = mid
    return samples[low:]


def _pair(unix_time, value):
    try:
        stamp = float(unix_time)
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp) or not math.isfinite(number):
        return None
    return stamp, number


def _empty_window(
    stages,
    stage_delays=None,
    breath_delay=float("nan"),
    pupil_delay=float("nan"),
    delta_delay=float("nan"),
    alpha_delay=float("nan"),
    profile=DEFAULT_FILTER_PROFILE,
):
    if stage_delays is None:
        stage_delays = tuple((stage.name, float(getattr(stage, "delay_s", 0.0))) for stage in stages)
    empty = np.array([], dtype=float)
    return ResampledWindow(
        time_seconds=empty,
        breathing=empty.copy(),
        pupil_diameter=empty.copy(),
        delta_envelope=empty.copy(),
        alpha_envelope=empty.copy(),
        delta_to_alpha=empty.copy(),
        valid=np.array([], dtype=bool),
        delays_seconds={
            "breathing": float(breath_delay) if breath_delay is not None else float("nan"),
            "pupil diameter": float(pupil_delay) if pupil_delay is not None else float("nan"),
            "delta envelope": float(delta_delay) if delta_delay is not None else float("nan"),
            "alpha envelope": float(alpha_delay) if alpha_delay is not None else float("nan"),
        },
        pupil_stage_delays_seconds=tuple(stage_delays),
        filter_profile=profile,
    )


def align_arrays(breath_samples, pupil_samples, eeg_samples, profile=DEFAULT_FILTER_PROFILE, eeg_channels=DEFAULT_EEG_CHANNELS):
    """The whole overlap of the three signals on a 10 hertz grid.

    This is the same filtering as ``resample``, without cutting to the last
    N seconds. It is what the agreement study uses to compare filter profiles.
    """
    breath_times, breath_values, breath_delay = _breath_path(breath_samples, profile)
    pupil_times, pupil_values, pupil_delay, _stage_delays = _pupil_path(
        pupil_samples, default_pupil_stages(profile)
    )
    eeg_times, delta, alpha, delta_delay, alpha_delay = _eeg_path(eeg_samples, eeg_channels, profile)
    paths = (
        (breath_times, breath_values, breath_delay),
        (pupil_times, pupil_values, pupil_delay),
        (eeg_times, delta, delta_delay),
        (eeg_times, alpha, alpha_delay),
    )
    if any(times is None or not math.isfinite(delay) for times, _values, delay in paths):
        raise ValueError("One of the signals could not be filtered.")
    start = max(float(np.min(times[np.isfinite(values)] - delay)) for times, values, delay in paths if np.any(np.isfinite(values)))
    end = min(float(np.max(times[np.isfinite(values)] - delay)) for times, values, delay in paths if np.any(np.isfinite(values)))
    grid = _output_grid(end, end - start)
    breathing = onto_grid(grid, breath_times, breath_values, breath_delay)
    pupil = onto_grid(grid, pupil_times, pupil_values, pupil_delay)
    delta_on = onto_grid(grid, eeg_times, delta, delta_delay)
    alpha_on = onto_grid(grid, eeg_times, alpha, alpha_delay)
    ratio = _ratio(delta_on, alpha_on)
    return {
        "time": grid,
        "breathing": breathing,
        "pupil diameter": pupil,
        "delta": delta_on,
        "alpha": alpha_on,
        "delta-to-alpha": ratio,
        "delays_seconds": {
            "breathing": float(breath_delay),
            "pupil diameter": float(pupil_delay),
            "delta envelope": float(delta_delay),
            "alpha envelope": float(alpha_delay),
        },
        "filter_profile": profile,
    }


def _breath_path(samples, profile):
    times = np.array([sample[0] for sample in samples], dtype=float)
    values = np.array([sample[1] for sample in samples], dtype=float)
    filtered_times, filtered, delay = linear_phase_lowpass(times, values, profile=profile)
    if filtered_times.size == 0:
        return None, None, float("nan")
    return filtered_times, filtered, delay


def _pupil_path(samples, stages):
    merged = merge_eyes(samples)
    stage_delays = []
    if merged is None:
        return None, None, float("nan"), tuple(stage_delays)
    times, values = merged
    total = 0.0
    for stage in stages:
        times, values = stage.apply(times, values)
        delay = float(stage.delay_s)
        stage_delays.append((str(stage.name), delay))
        total += delay
    return times, values, total, tuple(stage_delays)


def _eeg_path(samples, channel_names, profile):
    times = np.array([sample[0] for sample in samples], dtype=float)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    ordered = [samples[int(index)][1] for index in order]
    if times.size < 8:
        return None, None, None, float("nan"), float("nan")
    period = float(np.median(np.diff(times)))
    if not math.isfinite(period) or period <= 0:
        return None, None, None, float("nan"), float("nan")
    sample_rate = 1.0 / period
    start = float(times[0])
    stop = float(times[-1])
    count = int(math.floor((stop - start) / period)) + 1
    if count < 8:
        return None, None, None, float("nan"), float("nan")
    grid = start + np.arange(count) * period
    max_gap = _few_samples_seconds(period)
    delta_columns = []
    alpha_columns = []
    delta_delay = float("nan")
    alpha_delay = float("nan")
    for name in channel_names:
        raw = np.array([row.get(name, np.nan) for row in ordered], dtype=float)
        uniform = interpolate_with_gaps(times, raw, grid, max_gap)
        if not np.any(np.isfinite(uniform)):
            continue
        delta = _band_envelope(uniform, sample_rate, "delta", profile)
        alpha = _band_envelope(uniform, sample_rate, "alpha", profile)
        if delta is not None:
            delta_columns.append(delta[0])
            delta_delay = delta[1]
        if alpha is not None:
            alpha_columns.append(alpha[0])
            alpha_delay = alpha[1]
    if not delta_columns or not alpha_columns:
        nan_values = np.full(grid.shape, np.nan)
        return grid, nan_values, nan_values.copy(), delta_delay, alpha_delay
    return grid, _mean_present(delta_columns), _mean_present(alpha_columns), delta_delay, alpha_delay


def _mean_present(columns):
    """Average the rows that have a number. An all-missing column stays missing."""
    stack = np.vstack(columns)
    count = np.sum(np.isfinite(stack), axis=0)
    averaged = np.full(stack.shape[1], np.nan, dtype=float)
    usable = count > 0
    if np.any(usable):
        averaged[usable] = np.mean(stack[:, usable], axis=0, where=np.isfinite(stack[:, usable]))
    return averaged


def _window_grid(paths, window_seconds, breath, pupil, eeg, breath_delay, pupil_delay, delta_delay, alpha_delay):
    """10 hertz grid ending at the latest time every path has a sample.

    When the filters have not settled yet, the grid still has the requested
    length and the samples come back missing.
    """
    delays = [delay for _times, _values, delay in paths]
    if any(not math.isfinite(delay) for delay in delays):
        return _fallback_grid(
            window_seconds, breath, pupil, eeg, breath_delay, pupil_delay, delta_delay, alpha_delay
        )
    diagnostic = _diagnostic_grid(paths)
    if diagnostic is not None:
        joint = np.ones(diagnostic.shape, dtype=bool)
        for times, values, delay in paths:
            joint &= np.isfinite(onto_grid(diagnostic, times, values, delay))
        if np.any(joint):
            return _output_grid(float(diagnostic[np.flatnonzero(joint)[-1]]), window_seconds)
    return _fallback_grid(
        window_seconds, breath, pupil, eeg, breath_delay, pupil_delay, delta_delay, alpha_delay
    )


def _fallback_grid(window_seconds, breath, pupil, eeg, breath_delay, pupil_delay, delta_delay, alpha_delay):
    stamps = [breath[-1][0], pupil[-1][0], eeg[-1][0]]
    delays = [delay for delay in (breath_delay, pupil_delay, delta_delay, alpha_delay) if math.isfinite(delay)]
    end = max(stamps) - (max(delays) if delays else 0.0)
    return _output_grid(float(end), window_seconds)


def _diagnostic_grid(paths):
    starts = []
    ends = []
    for times, values, delay in paths:
        finite = np.isfinite(values)
        if not np.any(finite):
            return None
        compensated = np.asarray(times, dtype=float) - float(delay)
        finite_times = compensated[finite]
        starts.append(float(np.min(finite_times)))
        ends.append(float(np.max(finite_times)))
    start = min(starts)
    end = max(ends)
    if end <= start:
        return None
    count = int(math.floor((end - start) * OUTPUT_RATE_HZ)) + 1
    if count < 2:
        return None
    return start + np.arange(count) / OUTPUT_RATE_HZ
