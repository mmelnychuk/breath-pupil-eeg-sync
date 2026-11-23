"""Put breathing, pupil diameter, and electroencephalography on one time grid.

Electroencephalography is not analysed raw. Each channel is band-passed into
delta, theta, alpha, beta, and low gamma. The Hilbert amplitude is squared to
give instantaneous power, smoothed, and brought down to the common grid
(10–20 Hz is the useful range; the default is 16 Hz). Every ordered ratio of
those powers is kept. Delta/alpha is one of them, not a privileged one.

Frontal channels AF3, AF4, F3 and F4 are the default average. The original
EPOC headset has no Fz.

Blinks and other dropouts in pupil diameter are removed from the confidence
flag (the same 0.6 threshold the live view uses) and from non-positive
diameters, then a short margin on either side of each gap is removed as well.
Gaps up to half a second are filled by linear interpolation. Longer gaps are
left missing, and the analysis keeps the longest stretch where every signal
is present.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.nlid.load import find_pupil_file, load_breathing, load_eeg, load_pupil

BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low gamma", 30.0, 45.0),
)

CHANNEL_GROUPS = {
    "frontal": ("AF3", "AF4", "F3", "F4"),
    "frontal left": ("AF3", "F7", "F3", "FC5"),
    "frontal right": ("AF4", "F8", "F4", "FC6"),
    "temporal": ("T7", "T8"),
    "posterior": ("P7", "O1", "O2", "P8"),
    "all": tuple(EPOC_CHANNELS),
}

PUPIL_CONFIDENCE_MINIMUM = 0.6


def longest_finite_run(mask):
    """Start and stop (stop excluded) of the longest stretch of True values."""
    if not np.any(mask):
        return 0, 0
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(stops[best])


def interpolate_short_gaps(times, values, max_gap_s):
    """Linearly fill internal gaps whose duration is at most ``max_gap_s``."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.all() or not finite.any():
        return values
    missing = np.flatnonzero(~finite)
    gaps = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for gap in gaps:
        begin = int(gap[0])
        finish = int(gap[-1])
        if begin == 0 or finish == len(values) - 1:
            continue
        duration = times[finish + 1] - times[begin - 1]
        if duration <= max_gap_s:
            values[begin : finish + 1] = np.interp(
                times[begin : finish + 1],
                [times[begin - 1], times[finish + 1]],
                [values[begin - 1], values[finish + 1]],
            )
    return values


def _dilate_missing(times, missing, pad_s):
    if pad_s <= 0 or not np.any(missing):
        return missing
    out = missing.copy()
    bad = np.flatnonzero(missing)
    left = 0
    right = 0
    n_samples = len(times)
    for index in bad:
        while left < n_samples and times[index] - times[left] > pad_s:
            left += 1
        if right < left:
            right = left
        while right < n_samples and times[right] - times[index] <= pad_s:
            right += 1
        out[left:right] = True
    return out


def clean_pupil(times, diameter, confidence=None, max_gap_s=0.5, pad_s=0.05, minimum_confidence=PUPIL_CONFIDENCE_MINIMUM):
    """Mark blinks as missing and interpolate the short ones.

    A sample is a blink when its confidence is below ``minimum_confidence``,
    when the diameter is missing or not positive, or when it sits within
    ``pad_s`` of such a sample. Confidence is ignored when the whole column is
    missing.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(diameter, dtype=float).copy()
    missing = ~np.isfinite(values) | (values <= 0)
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        if np.any(np.isfinite(confidence)):
            missing = missing | (np.isfinite(confidence) & (confidence < minimum_confidence))
    missing = _dilate_missing(times, missing, pad_s)
    values[missing] = np.nan
    return interpolate_short_gaps(times, values, max_gap_s)


def _uniform(times, values, sample_rate):
    start = float(times[0])
    stop = float(times[-1])
    n_samples = int(np.floor((stop - start) * sample_rate)) + 1
    if n_samples < 4:
        raise ValueError("The time span is too short to resample at %.3f Hz." % sample_rate)
    grid = start + np.arange(n_samples) / sample_rate
    resampled = np.interp(grid, times, values)
    return grid, resampled


def _bandpass(values, sample_rate, low_hz, high_hz, order=4):
    nyquist = sample_rate / 2.0
    if low_hz <= 0 or high_hz >= nyquist * 0.95 or low_hz >= high_hz:
        return None
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return None


def _lowpass(values, sample_rate, cutoff_hz, order=2):
    nyquist = sample_rate / 2.0
    cutoff_hz = min(float(cutoff_hz), nyquist * 0.8)
    if cutoff_hz <= 0:
        return values
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return values


def band_powers(times, channels, envelope_lowpass_hz=2.0):
    """Smoothed instantaneous power of each band, averaged across channels.

    Returns ``(native_time, {band name: power})`` and a list of warning strings
    for bands the sample rate or the record length cannot support.
    """
    warnings = []
    arrays = [np.asarray(values, dtype=float) for values in channels.values()]
    # Interpolate each channel through its own missing samples before filtering,
    # using the shared time stamps. A channel that is entirely missing is dropped.
    usable = []
    for values in arrays:
        if np.all(~np.isfinite(values)):
            continue
        filled = values.copy()
        finite = np.isfinite(filled)
        if not finite.all():
            filled[~finite] = np.interp(times[~finite], times[finite], filled[finite])
        usable.append(filled)
    if not usable:
        raise ValueError("Every electroencephalography channel is empty.")
    native_rate = 1.0 / float(np.median(np.diff(times)))
    grid, _example = _uniform(times, usable[0], native_rate)
    resampled = []
    for values in usable:
        _grid, uniform_values = _uniform(times, values, native_rate)
        resampled.append(uniform_values[: len(grid)])
    powers = {}
    for name, low_hz, high_hz in BANDS:
        channel_powers = []
        for values in resampled:
            filtered = _bandpass(values, native_rate, low_hz, high_hz)
            if filtered is None:
                continue
            power = np.square(np.abs(hilbert(filtered)))
            channel_powers.append(_lowpass(power, native_rate, envelope_lowpass_hz))
        if not channel_powers:
            warnings.append(
                "Skipped the %s band (%.1f–%.1f Hz): the electroencephalography sample rate is %.1f Hz "
                "or the record is too short for that filter." % (name, low_hz, high_hz, native_rate)
            )
            continue
        stacked = np.vstack(channel_powers)
        averaged = np.mean(stacked, axis=0)
        # Drop the filter warm-up at each end so it is not treated as brain activity.
        trim = int(round(0.5 * native_rate))
        if trim > 0 and len(averaged) > 4 * trim:
            averaged = averaged.copy()
            averaged[:trim] = np.nan
            averaged[-trim:] = np.nan
        powers[name] = averaged
    return grid, powers, warnings


def power_ratios(powers):
    """Every ordered ratio of band powers. The denominator is floored at 0.1% of its median."""
    names = list(powers)
    ratios = {}
    for numerator in names:
        for denominator in names:
            if numerator == denominator:
                continue
            bottom = powers[denominator]
            scale = np.nanmedian(bottom)
            floor = 1e-8 if not np.isfinite(scale) or scale <= 0 else 0.001 * scale
            ratios["%s/%s power ratio" % (numerator, denominator)] = powers[numerator] / np.maximum(bottom, floor)
    return ratios


def attentional_signals(times, channels, envelope_lowpass_hz=2.0):
    """Band amplitudes are not returned; the candidates are smoothed power and the ratios.

    Each band's smoothed power is itself a candidate, named e.g. "alpha power",
    and so is every ordered ratio.
    """
    grid, powers, warnings = band_powers(times, channels, envelope_lowpass_hz=envelope_lowpass_hz)
    signals = {"%s power" % name: values for name, values in powers.items()}
    signals.update(power_ratios(powers))
    return grid, signals, warnings


def _average_eyes(series, max_gap_s, pad_s):
    cleaned = []
    for item in series:
        values = clean_pupil(item["time"], item["diameter"], item["confidence"], max_gap_s=max_gap_s, pad_s=pad_s)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        cleaned.append((item["time"], values))
    if not cleaned:
        raise ValueError("Pupil diameter is missing on every sample after blink removal.")
    start = min(times[np.isfinite(values)][0] for times, values in cleaned)
    stop = max(times[np.isfinite(values)][-1] for times, values in cleaned)
    rate = np.median([1.0 / np.median(np.diff(times)) for times, _values in cleaned])
    grid = np.arange(start, stop, 1.0 / rate)
    stack = []
    for times, values in cleaned:
        finite = np.isfinite(values)
        interpolated = np.interp(grid, times[finite], values[finite], left=np.nan, right=np.nan)
        stack.append(interpolated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        averaged = np.nanmean(np.vstack(stack), axis=0)
    return grid, averaged


def on_grid(times, values, grid):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("Fewer than two finite samples, so the signal cannot be resampled.")
    resampled = np.full(grid.shape, np.nan)
    inside = (grid >= times[finite][0]) & (grid <= times[finite][-1])
    resampled[inside] = np.interp(grid[inside], times[finite], values[finite])
    return resampled


def assemble_record(
    breath_time,
    breath,
    pupil_series,
    eeg_time,
    eeg_channels,
    sample_rate_hz=16.0,
    channels=None,
    max_blink_s=0.5,
    blink_pad_s=0.05,
    envelope_lowpass_hz=2.0,
    attentional_names=None,
):
    """Resample everything onto one grid and build the attentional candidates.

    ``pupil_series`` is the list returned by ``load_pupil``. ``channels`` selects
    which electroencephalography channels are averaged. Returns the grid, the
    breathing and pupil arrays on that grid, the attentional signals, and warnings.
    """
    if channels:
        eeg_channels = {name: values for name, values in eeg_channels.items() if name in channels}
        missing = [name for name in channels if name not in eeg_channels]
        if missing:
            raise ValueError(
                "These electroencephalography channels are not in the file: %s. Present: %s."
                % (", ".join(missing), ", ".join(eeg_channels) or "none")
            )
    pupil_time, pupil = _average_eyes(pupil_series, max_blink_s, blink_pad_s)
    eeg_grid, attention, warnings = attentional_signals(
        eeg_time, eeg_channels, envelope_lowpass_hz=envelope_lowpass_hz
    )
    if attentional_names:
        unknown = [name for name in attentional_names if name not in attention]
        if unknown:
            raise ValueError(
                "Unknown attentional signal(s): %s. Available: %s."
                % (", ".join(unknown), ", ".join(sorted(attention)))
            )
        attention = {name: attention[name] for name in attentional_names}
    start = max(breath_time[0], pupil_time[0], eeg_grid[0])
    stop = min(breath_time[-1], pupil_time[-1], eeg_grid[-1])
    if stop - start <= 1.0 / sample_rate_hz:
        raise ValueError("Breathing, pupil diameter, and electroencephalography do not overlap in time.")
    n_samples = int(np.floor((stop - start) * sample_rate_hz))
    grid = start + np.arange(n_samples) / sample_rate_hz
    breath_on_grid = on_grid(breath_time, breath, grid)
    pupil_on_grid = on_grid(pupil_time, pupil, grid)
    attention_on_grid = {name: on_grid(eeg_grid, values, grid) for name, values in attention.items()}
    present = np.isfinite(breath_on_grid) & np.isfinite(pupil_on_grid)
    for values in attention_on_grid.values():
        present &= np.isfinite(values)
    run_start, run_stop = longest_finite_run(present)
    if run_stop - run_start < 80:
        raise ValueError(
            "The longest stretch with breathing, pupil diameter, and electroencephalography all present "
            "is %d samples. That is too short for a neighbour search." % (run_stop - run_start)
        )
    if run_stop - run_start < n_samples:
        warnings.append(
            "Using the longest complete stretch, %.1f of %.1f seconds. The rest has a gap longer than "
            "the blink interpolation allows."
            % ((run_stop - run_start) / sample_rate_hz, n_samples / sample_rate_hz)
        )
    grid = grid[run_start:run_stop]
    breath_on_grid = breath_on_grid[run_start:run_stop]
    pupil_on_grid = pupil_on_grid[run_start:run_stop]
    attention_on_grid = {name: values[run_start:run_stop] for name, values in attention_on_grid.items()}
    return {
        "time_s": grid,
        "breathing": breath_on_grid,
        "pupil diameter": pupil_on_grid,
        "attentional_signals": attention_on_grid,
        "warnings": warnings,
        "sample_rate_hz": float(sample_rate_hz),
    }


def load_session(
    session,
    pupil_path=None,
    pupil_offset=None,
    sample_rate_hz=16.0,
    channels=None,
    group="frontal",
    max_blink_s=0.5,
    blink_pad_s=0.05,
    envelope_lowpass_hz=2.0,
    attentional_names=None,
):
    """Load a hub session directory (or an explicit pupil export) into analysis arrays."""
    root = Path(session)
    breath_path = root / "breathing.csv"
    eeg_path = root / "eeg.csv"
    if not breath_path.is_file() or not eeg_path.is_file():
        raise ValueError("Expected breathing.csv and eeg.csv in %s." % root)
    if channels is None:
        if group not in CHANNEL_GROUPS:
            known = ", ".join(sorted(CHANNEL_GROUPS))
            raise ValueError("Unknown channel group %r. Known groups: %s." % (group, known))
        channels = list(CHANNEL_GROUPS[group])
    pupil_file = Path(pupil_path) if pupil_path else find_pupil_file(root)
    if pupil_file is None:
        raise ValueError(
            "No pupil-diameter file in %s. The live hub plots pupil diameter but does not save it. "
            "Copy a Pupil Core pupil_positions.csv into the session, or pass --pupil. "
            "If the timestamps are Pupil's own clock, also pass --pupil-offset "
            "(pupil time minus Unix time, the offset the hub prints at startup)." % root
        )
    breath_time, breath = load_breathing(breath_path)
    eeg_time, eeg_channels = load_eeg(eeg_path, channels=channels)
    pupil_series = load_pupil(pupil_file, offset=pupil_offset)
    assembled = assemble_record(
        breath_time,
        breath,
        pupil_series,
        eeg_time,
        eeg_channels,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        max_blink_s=max_blink_s,
        blink_pad_s=blink_pad_s,
        envelope_lowpass_hz=envelope_lowpass_hz,
        attentional_names=attentional_names,
    )
    assembled["pupil_file"] = str(pupil_file)
    return assembled
