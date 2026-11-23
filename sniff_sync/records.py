"""Session files. Every time column is Unix time."""

from __future__ import annotations

import csv
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.timefmt import format_timestamp

TRIGGER_COLUMNS = ["trigger_id", "unix_time", "label"]


def new_trigger_id() -> str:
    return "trg_" + uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Trigger:
    trigger_id: str
    label: str
    unix_time: float


class _Csv:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._columns = columns
        self._lock = threading.Lock()
        self._fh = open(path, "a+", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=columns, extrasaction="ignore")
        self._fh.seek(0)
        if self._fh.read(1) == "":
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


class BreathingLog(_Csv):
    """getSniffs.py columns, plus a trigger label when a trigger arrives.

    getSniffs.py writes ``Unix Time`` in seconds and ``Nasal Pressure``
    in pascal. A pressure sample leaves Trigger empty. A trigger row
    carries the label and the same Unix seconds, and leaves pressure empty.
    """

    columns = ["Unix Time", "Nasal Pressure", "Trigger", "trigger_id"]

    def __init__(self, path):
        super().__init__(Path(path), self.columns)

    def write_sample(self, unix_time: float, nasal_pressure: float) -> None:
        self.write(
            {
                "Unix Time": format_timestamp(unix_time),
                "Nasal Pressure": format(float(nasal_pressure), ".6f"),
                "Trigger": "",
                "trigger_id": "",
            }
        )

    def write_trigger(self, event: Trigger) -> None:
        self.write(
            {
                "Unix Time": format_timestamp(event.unix_time),
                "Nasal Pressure": "",
                "Trigger": event.label,
                "trigger_id": event.trigger_id,
            }
        )


class PupilDiameterLog(_Csv):
    """Continuous pupil diameter, in millimetres, on Unix time.

    The live view already receives these samples. This file keeps them so
    the post-session reference can read the same diameters. Eye is 0 or 1.
    """

    columns = ["unix_time", "eye", "diameter_mm"]

    def __init__(self, path):
        super().__init__(Path(path), self.columns)

    def write_sample(self, unix_time: float, eye: int, diameter_mm: float) -> None:
        self.write(
            {
                "unix_time": format_timestamp(unix_time),
                "eye": str(int(eye)),
                "diameter_mm": format(float(diameter_mm), ".6f"),
            }
        )


class EegLog(_Csv):
    """EEG samples and trigger markers in one file, on Unix time.

    Channel values are whatever emokit decoded (raw sensor units, about
    0.51 microvolts per step on the original EPOC). Trigger rows leave
    the channel cells empty.
    """

    columns = ["unix_time", "kind", "label", "trigger_id", "counter"] + list(EPOC_CHANNELS)

    def __init__(self, path):
        super().__init__(Path(path), self.columns)

    def write_sample(self, unix_time: float, channels: dict, counter="") -> None:
        row = {
            "unix_time": format_timestamp(unix_time),
            "kind": "sample",
            "label": "",
            "trigger_id": "",
            "counter": "" if counter is None else counter,
        }
        for name in EPOC_CHANNELS:
            value = channels.get(name, "")
            row[name] = "" if value == "" or value is None else format(float(value), ".3f")
        self.write(row)

    def write_trigger(self, event: Trigger) -> None:
        row = {name: "" for name in self.columns}
        row.update(
            {
                "unix_time": format_timestamp(event.unix_time),
                "kind": "trigger",
                "label": event.label,
                "trigger_id": event.trigger_id,
                "counter": "",
            }
        )
        self.write(row)
