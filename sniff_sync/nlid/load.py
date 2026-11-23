"""Read breathing, pupil diameter, and electroencephalography onto arrays.

Breathing times follow the hub: Unix seconds, or Unix milliseconds when the
numbers are large enough to be ``time.time()`` multiplied by 1000. Trigger
rows, which carry a label and an empty pressure, are skipped.

Pupil Capture exports are not Unix time unless a column already says so. A
``pupil_timestamp`` column is converted with the same offset the hub prints
(pupil time minus Unix time): Unix time = pupil time − offset. Diameter in
millimetres (``diameter_3d``) is preferred over the pixel diameter.

Electroencephalography rows with kind ``trigger``, or with no channel values,
are skipped. Channel names are the original Emotiv EPOC set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.timefmt import unix_seconds

BREATH_TIME_COLUMNS = ("unix time", "unix_time", "time", "timestamp")
BREATH_VALUE_COLUMNS = ("nasal pressure", "nasal_pressure", "pressure")
PUPIL_TIME_COLUMNS = ("unix time", "unix_time", "timestamp", "world_timestamp", "pupil_timestamp")
PUPIL_DIAMETER_COLUMNS = ("diameter_3d", "diameter_mm", "pupil_diameter", "diameter")
PUPIL_CONFIDENCE_COLUMNS = ("confidence",)
PUPIL_EYE_COLUMNS = ("eye_id", "eye", "eye id")


def _normalise(name):
    return " ".join(name.strip().lower().replace("_", " ").split())


def _find_column(fieldnames, options):
    if not fieldnames:
        return None
    lookup = {_normalise(name): name for name in fieldnames}
    for option in options:
        found = lookup.get(_normalise(option))
        if found:
            return found
    return None


def _unique_sorted(times, values):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) == 0:
        raise ValueError("The file has no samples.")
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    unique, starts = np.unique(times, return_index=True)
    if len(unique) == len(times):
        return unique, values
    averaged = np.empty(len(unique))
    edges = list(starts) + [len(times)]
    for index, (begin, finish) in enumerate(zip(edges[:-1], edges[1:])):
        averaged[index] = np.nanmean(values[begin:finish])
    return unique, averaged


def load_breathing(path):
    """Return Unix times and nasal pressure, without trigger rows."""
    path = Path(path)
    times = []
    values = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        time_column = _find_column(reader.fieldnames, BREATH_TIME_COLUMNS)
        value_column = _find_column(reader.fieldnames, BREATH_VALUE_COLUMNS)
        trigger_column = _find_column(reader.fieldnames, ("trigger",))
        if time_column is None or value_column is None:
            raise ValueError(
                "%s needs a Unix time column and a nasal-pressure column. Found: %s"
                % (path, ", ".join(reader.fieldnames or []))
            )
        for row in reader:
            if trigger_column and (row.get(trigger_column) or "").strip():
                continue
            raw_time = (row.get(time_column) or "").strip()
            raw_value = (row.get(value_column) or "").strip()
            if not raw_time or not raw_value:
                continue
            times.append(unix_seconds(raw_time))
            values.append(float(raw_value))
    if not times:
        raise ValueError("%s has time stamps but no nasal-pressure samples." % path)
    return _unique_sorted(times, values)


def load_eeg(path, channels=None):
    """Return Unix times and a dict of channel arrays.

    ``channels`` limits the set. The default is every original-EPOC channel
    present in the file.
    """
    path = Path(path)
    wanted = list(channels) if channels else list(EPOC_CHANNELS)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        time_column = _find_column(reader.fieldnames, ("unix time", "unix_time", "time", "timestamp"))
        kind_column = _find_column(reader.fieldnames, ("kind",))
        if time_column is None:
            raise ValueError("%s has no Unix time column." % path)
        available = {_normalise(name): name for name in (reader.fieldnames or [])}
        selected = []
        for name in wanted:
            found = available.get(_normalise(name))
            if found:
                selected.append((name, found))
        if not selected:
            raise ValueError(
                "%s has none of the requested electroencephalography channels (%s)."
                % (path, ", ".join(wanted))
            )
        buckets = {name: [] for name, _column in selected}
        times = []
        for row in reader:
            kind = (row.get(kind_column) or "").strip().lower() if kind_column else ""
            if kind and kind != "sample":
                continue
            raw_time = (row.get(time_column) or "").strip()
            if not raw_time:
                continue
            parsed = {}
            any_value = False
            for name, column in selected:
                raw = (row.get(column) or "").strip()
                if not raw:
                    parsed[name] = np.nan
                    continue
                parsed[name] = float(raw)
                any_value = True
            if not any_value:
                continue
            times.append(unix_seconds(raw_time))
            for name, _column in selected:
                buckets[name].append(parsed[name])
    if not times:
        raise ValueError("%s has no electroencephalography samples." % path)
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    channels_out = {name: np.asarray(values, dtype=float)[order] for name, values in buckets.items()}
    return times, channels_out


def _as_unix(raw_time, offset):
    number = float(raw_time)
    # Already Unix seconds, or Unix milliseconds.
    if number > 1e8:
        return unix_seconds(number)
    if offset is None:
        raise ValueError(
            "Pupil timestamps look like Pupil Capture's own clock (values near %.3f), not Unix time. "
            "Pass --pupil-offset using the hub's offset (pupil time minus Unix time)." % number
        )
    return number - float(offset)


def load_pupil(path, offset=None):
    """Return a list of per-eye series: ``{"eye", "time", "diameter", "confidence"}``.

    One series, with eye ``None``, is returned when the file has no eye column.
    """
    path = Path(path)
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        time_column = _find_column(reader.fieldnames, PUPIL_TIME_COLUMNS)
        diameter_columns = [
            _find_column(reader.fieldnames, (option,))
            for option in PUPIL_DIAMETER_COLUMNS
        ]
        diameter_columns = [column for column in diameter_columns if column]
        confidence_column = _find_column(reader.fieldnames, PUPIL_CONFIDENCE_COLUMNS)
        eye_column = _find_column(reader.fieldnames, PUPIL_EYE_COLUMNS)
        if time_column is None or not diameter_columns:
            raise ValueError(
                "%s needs a time column and a pupil-diameter column. Found: %s"
                % (path, ", ".join(reader.fieldnames or []))
            )
        # Prefer an explicit Unix column over pupil_timestamp when both exist.
        unix_column = _find_column(reader.fieldnames, ("unix time", "unix_time"))
        if unix_column:
            time_column = unix_column
        for row in reader:
            raw_time = (row.get(time_column) or "").strip()
            raw_diameter = ""
            for column in diameter_columns:
                raw_diameter = (row.get(column) or "").strip()
                if raw_diameter:
                    break
            if not raw_time or not raw_diameter:
                continue
            confidence = np.nan
            if confidence_column and (row.get(confidence_column) or "").strip():
                confidence = float(row[confidence_column])
            eye = None
            if eye_column and (row.get(eye_column) or "").strip():
                eye = int(float(row[eye_column]))
            rows.append((_as_unix(raw_time, None if unix_column else offset), float(raw_diameter), confidence, eye))
    if not rows:
        raise ValueError("%s has no pupil-diameter samples." % path)
    eyes = sorted({item[3] for item in rows})
    series = []
    for eye in eyes:
        chosen = [item for item in rows if item[3] == eye]
        times, diameters = _unique_sorted([item[0] for item in chosen], [item[1] for item in chosen])
        _times_c, confidence = _unique_sorted([item[0] for item in chosen], [item[2] for item in chosen])
        series.append({"eye": eye, "time": times, "diameter": diameters, "confidence": confidence})
    return series


def find_pupil_file(session):
    """Look for a pupil export inside a session directory. Return None if there is not one."""
    root = Path(session)
    direct = (
        "pupil.csv",
        "pupil_diameter.csv",
        "pupil_positions.csv",
        "pupil_recording/pupil_positions.csv",
    )
    for name in direct:
        path = root / name
        if path.is_file():
            return path
    matches = []
    for pattern in ("pupil_recording/**/pupil_positions.csv", "exports/**/pupil_positions.csv", "**/pupil_positions.csv"):
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    unique = []
    seen = set()
    for path in matches:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique[0] if unique else None
