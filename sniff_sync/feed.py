"""Samples and triggers for the live view. Times are Unix time."""

from __future__ import annotations

import threading


class Feed:
    def __init__(self, horizon_s: float = 20.0):
        self.horizon_s = horizon_s
        self._lock = threading.Lock()
        self.breath = []
        self.pupil = []  # (unix, eye, mm)
        self.eeg = []  # (unix, {channel: value})
        self.triggers = []  # (unix, label, trigger_id)

    def add_breath(self, unix_time: float, pressure: float) -> None:
        with self._lock:
            self.breath.append((float(unix_time), float(pressure)))
            self._trim(self.breath)

    def add_pupil(self, unix_time: float, eye: int, diameter_mm: float) -> None:
        with self._lock:
            self.pupil.append((float(unix_time), int(eye), float(diameter_mm)))
            self._trim(self.pupil)

    def add_eeg(self, unix_time: float, channels: dict) -> None:
        with self._lock:
            self.eeg.append((float(unix_time), dict(channels)))
            self._trim(self.eeg)

    def add_trigger(self, unix_time: float, label: str, trigger_id: str) -> None:
        with self._lock:
            self.triggers.append((float(unix_time), str(label), str(trigger_id)))
            self._trim(self.triggers)

    def _trim(self, rows) -> None:
        if not rows:
            return
        newest = rows[-1][0]
        cutoff = newest - self.horizon_s
        while rows and rows[0][0] < cutoff:
            rows.pop(0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "breath": list(self.breath),
                "pupil": list(self.pupil),
                "eeg": list(self.eeg),
                "triggers": list(self.triggers),
            }
