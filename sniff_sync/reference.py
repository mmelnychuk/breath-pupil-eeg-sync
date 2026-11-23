"""Post-session reference coupling, with no filter delay.

This reads the raw session, filters it forward and backward over the whole
recording, and scores the same windows the live worker used. The live file
is approximate. This file is the reference. The display itself is separate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from sniff_sync.coupling import (
    APPROXIMATE_NAME,
    BASELINE_SECONDS,
    EEG_SIGNALS,
    MINIMUM_VALID_FRACTION,
    SIGNAL_NAMES,
    _log_row,
    _longest_valid_seconds,
    _series_map,
    score_triplet,
)
from sniff_sync.filters import zero_phase_envelope, zero_phase_lowpass
from sniff_sync.nlid.load import find_pupil_file, load_breathing, load_eeg, load_pupil
from sniff_sync.rolling import (
    DEFAULT_EEG_CHANNELS,
    RATIO_FLOOR_FRACTION,
    ShortGapBridge,
    _as_uniform,
    _output_grid,
    merge_eyes,
    onto_grid,
)

REFERENCE_NAME = "coupling_reference.csv"


def _ratio(delta, alpha):
    ratio = np.full(delta.shape, np.nan, dtype=float)
    both = np.isfinite(delta) & np.isfinite(alpha)
    positive = alpha[both & (alpha > 0)] if np.any(both) else np.array([])
    if positive.size == 0:
        return ratio
    floor = RATIO_FLOOR_FRACTION * float(np.median(positive))
    ratio[both] = delta[both] / np.maximum(alpha[both], floor)
    return ratio


def _filter_runs(values, function):
    """Apply a zero-phase filter on each unbroken stretch. Short stretches stay missing."""
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    padded = np.concatenate([[False], finite, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        if int(stop - start) < 80:
            continue
        try:
            output[int(start) : int(stop)] = function(values[int(start) : int(stop)])
        except ValueError:
            continue
    return output


def _pupil_rows(series):
    rows = []
    for item in series:
        eye = 0 if item["eye"] is None else int(item["eye"])
        for stamp, value in zip(item["time"], item["diameter"]):
            rows.append((float(stamp), eye, float(value)))
    return rows


def reference_signals(breath_time, breath, pupil_series, eeg_time, eeg_channels):
    """Breathing, pupil diameter, and both envelopes on one 10 hertz grid, with no delay.

    The filters run forward and backward on the whole recording. Only the
    very start and the very end of the recording carry the filter's edge.
    """
    breath_uniform = _as_uniform(breath_time, breath)
    if breath_uniform is None:
        raise ValueError("Breathing is too short to filter.")
    breath_grid, breath_values, breath_rate = breath_uniform
    breath_filtered = _filter_runs(
        breath_values, lambda values: zero_phase_lowpass(values, breath_rate, 4.0)
    )

    merged = merge_eyes(_pupil_rows(pupil_series))
    if merged is None:
        raise ValueError("Pupil diameter is too short to filter.")
    pupil_grid, pupil_values = ShortGapBridge().apply(*merged)
    pupil_rate = 1.0 / float(np.median(np.diff(pupil_grid)))
    pupil_filtered = _filter_runs(
        pupil_values, lambda values: zero_phase_lowpass(values, pupil_rate, 4.0)
    )

    eeg_uniform = _as_uniform(eeg_time, next(iter(eeg_channels.values())))
    if eeg_uniform is None:
        raise ValueError("Electroencephalography is too short to filter.")
    eeg_grid, _example, eeg_rate = eeg_uniform
    delta_columns = []
    alpha_columns = []
    for values in eeg_channels.values():
        prepared = _as_uniform(eeg_time, values)
        if prepared is None:
            continue
        _grid, uniform, rate = prepared
        count = min(len(uniform), len(eeg_grid))
        delta_columns.append(
            _filter_runs(uniform[:count], lambda samples, rate=rate: zero_phase_envelope(samples, rate, 1.0, 4.0))
        )
        alpha_columns.append(
            _filter_runs(uniform[:count], lambda samples, rate=rate: zero_phase_envelope(samples, rate, 8.0, 12.0))
        )
    if not delta_columns or not alpha_columns:
        raise ValueError("The electroencephalography channels could not be filtered.")
    with np.errstate(all="ignore"):
        delta = np.nanmean(np.vstack([column[: len(eeg_grid)] for column in delta_columns]), axis=0)
        alpha = np.nanmean(np.vstack([column[: len(eeg_grid)] for column in alpha_columns]), axis=0)
    ratio = _ratio(delta, alpha)

    start = max(breath_grid[0], pupil_grid[0], eeg_grid[0])
    stop = min(breath_grid[-1], pupil_grid[-1], eeg_grid[-1])
    count = int(np.floor((stop - start) * 10.0)) + 1
    grid = start + np.arange(count) / 10.0
    return {
        "time": grid,
        "breathing": onto_grid(grid, breath_grid, breath_filtered, 0.0),
        "pupil diameter": onto_grid(grid, pupil_grid, pupil_filtered, 0.0),
        "delta": onto_grid(grid, eeg_grid, delta, 0.0),
        "alpha": onto_grid(grid, eeg_grid, alpha, 0.0),
        "delta-to-alpha": onto_grid(grid, eeg_grid, ratio, 0.0),
    }


def _load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _embedding_from_row(row):
    embedding = {}
    for name in SIGNAL_NAMES:
        key = name.replace(" ", "_")
        delay = row.get("embedding_delay_%s" % key) or ""
        dimension = row.get("embedding_dimension_%s" % key) or ""
        if delay == "" or dimension == "":
            return None
        embedding[name] = {"delay": int(float(delay)), "dimension": int(float(dimension))}
    return embedding


def _window_at(signals, end, window_seconds, eeg_signal):
    grid = _output_grid(float(end), float(window_seconds))
    # onto_grid with delay 0 reads the reference at the same Unix times.
    field = EEG_SIGNALS[eeg_signal]
    # reference_signals uses "delta-to-alpha" with a hyphen; the live name matches.
    eeg_key = "delta-to-alpha" if eeg_signal == "delta-to-alpha" else eeg_signal
    values = {
        "breathing": onto_grid(grid, signals["time"], signals["breathing"], 0.0),
        "pupil": onto_grid(grid, signals["time"], signals["pupil diameter"], 0.0),
        "eeg": onto_grid(grid, signals["time"], signals[eeg_key], 0.0),
    }
    valid = np.isfinite(values["breathing"]) & np.isfinite(values["pupil"]) & np.isfinite(values["eeg"])
    return values, valid


def write_reference(session, pupil_path=None, pupil_offset=None, channels=None):
    """Score the live worker's window times with zero-phase filtering.

    Returns the path of ``coupling_reference.csv``.
    """
    root = Path(session)
    approximate = root / APPROXIMATE_NAME
    if not approximate.is_file():
        raise ValueError(
            "No approximate coupling file in %s. Run the live hub first so the window times exist." % root
        )
    rows = _load_rows(approximate)
    ready = [row for row in rows if (row.get("status") or "") == "ready" and (row.get("window_end_unix") or "")]
    if not ready:
        raise ValueError("The approximate file has no ready windows to recompute.")
    breath_time, breath = load_breathing(root / "breathing.csv")
    channel_names = channels
    if channel_names is None:
        text = (ready[0].get("eeg_channels") or "").split()
        channel_names = text or list(DEFAULT_EEG_CHANNELS)
    eeg_time, eeg_channels = load_eeg(root / "eeg.csv", channels=channel_names)
    pupil_file = Path(pupil_path) if pupil_path else find_pupil_file(root)
    if pupil_file is None:
        raise ValueError(
            "No pupil-diameter file in %s. The hub writes pupil_diameter.csv, or a Pupil Capture "
            "pupil_positions.csv can be placed in the session." % root
        )
    pupil_series = load_pupil(pupil_file, offset=pupil_offset)
    signals = reference_signals(breath_time, breath, pupil_series, eeg_time, eeg_channels)
    destination = root / REFERENCE_NAME
    with open(destination, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_log_columns(), extrasaction="ignore")
        writer.writeheader()
        for row in ready:
            embedding = _embedding_from_row(row)
            end = float(row["window_end_unix"])
            window_seconds = float(row["window_seconds"])
            eeg_signal = row.get("eeg_signal") or "delta-to-alpha"
            result = {
                "label": "reference",
                "status": "skipped",
                "reason": "",
                "logged_unix": "",
                "window_end_unix": end,
                "window_seconds": window_seconds,
                "eeg_signal": eeg_signal,
                "eeg_channels": channel_names,
                "filter_profile": "reference",
                "embedding": embedding,
                "embedding_source": row.get("embedding_source") or "",
                "neighbor_count": int(float(row["neighbor_count"])) if row.get("neighbor_count") else 8,
                "pool_size": int(float(row["pool_size"])) if row.get("pool_size") else 40,
                "horizon_samples": int(float(row["horizon_samples"])) if row.get("horizon_samples") else 4,
                "surrogates": (row.get("surrogates") or "") == "yes",
                "interdependence": {},
                "forecast_gain": {},
                "directions": {},
                "valid_fraction": "",
                "compute_seconds": "",
                "delays_seconds": {},
                "theiler_window_samples": "",
            }
            if embedding is None:
                result["reason"] = "This window has no embedding in the approximate file, so it was not recomputed."
                writer.writerow(_log_row(result))
                continue
            values, valid = _window_at(signals, end, window_seconds, eeg_signal)
            fraction = float(np.mean(valid)) if valid.size else 0.0
            result["valid_fraction"] = fraction
            if fraction < MINIMUM_VALID_FRACTION or _longest_valid_seconds(valid) < BASELINE_SECONDS:
                result["reason"] = (
                    "The reference window ending at %.3f does not have enough present samples." % end
                )
                writer.writerow(_log_row(result))
                continue
            series = _series_map(values, valid)
            if series is None:
                result["reason"] = "The reference window could not be cut into one complete stretch."
                writer.writerow(_log_row(result))
                continue
            try:
                interdependence, forecast, theiler = score_triplet(
                    series,
                    embedding,
                    neighbor_count=result["neighbor_count"],
                    pool_size=result["pool_size"],
                    horizon_samples=result["horizon_samples"],
                )
            except ValueError as exc:
                result["reason"] = "The reference score failed: %s" % exc
                writer.writerow(_log_row(result))
                continue
            from sniff_sync.coupling import direction_summary

            result["status"] = "ready"
            result["interdependence"] = interdependence
            result["forecast_gain"] = forecast
            result["directions"] = direction_summary(interdependence, forecast)
            result["theiler_window_samples"] = theiler
            writer.writerow(_log_row(result))
    return destination


def _log_columns():
    from sniff_sync.coupling import LOG_COLUMNS

    return LOG_COLUMNS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute live coupling windows with zero-phase filters and write the reference file."
    )
    parser.add_argument("session")
    parser.add_argument("--pupil", default="", help="Pupil diameter file. Unix time, or Pupil time with --pupil-offset.")
    parser.add_argument("--pupil-offset", type=float, default=None, help="Pupil time minus Unix time.")
    parser.add_argument("--channels", nargs="*", default=None)
    args = parser.parse_args(argv)
    try:
        path = write_reference(
            args.session,
            pupil_path=args.pupil or None,
            pupil_offset=args.pupil_offset,
            channels=args.channels,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
