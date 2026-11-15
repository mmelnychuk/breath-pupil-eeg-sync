"""Fake breathing, pupil diameter, and EEG on one machine. No hardware.

Triggers are not invented from the breathing trace. They arrive only
through trigger(label).
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

from sniff_sync.eeg_channels import EPOC_CHANNELS
from sniff_sync.feed import Feed
from sniff_sync.hub import Hub
from sniff_sync.mock_pupil import MockPupil

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "pupil_plugin"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


class Session:
    def __init__(self, directory, port: int = 0):
        self.directory = Path(directory)
        self._port = port
        self.directory.mkdir(parents=True, exist_ok=True)
        self.feed = Feed()
        self.mock = MockPupil()
        self.hub = None
        self._stop = threading.Event()
        self._threads = []
        self._plugin = None

    def start(self) -> Hub:
        self.mock.start()
        self.hub = Hub(self.directory, pupil_port=self.mock.port, port=self._port, feed=self.feed)
        self.hub.start()
        os.environ["SNIFF_SYNC_PORT"] = str(self.hub.port)
        self._plugin = _Bridge(self.mock, self.directory / "pupil_recording", self._stop)
        self._threads.append(threading.Thread(target=self._plugin.run, name="pupil-plugin", daemon=True))
        self._threads.append(threading.Thread(target=self._streams, name="fake-streams", daemon=True))
        for thread in self._threads:
            thread.start()
        return self.hub

    def _streams(self) -> None:
        start = time.perf_counter()
        next_breath = 0.0
        next_pupil = 0.0
        next_eeg = 0.0
        counter = 0
        while not self._stop.is_set():
            now = time.perf_counter() - start
            if now >= next_breath:
                pressure = 0.25 * math.sin(2 * math.pi * 0.25 * now)
                unix = time.time()
                self.hub.breathing.write_sample(unix, pressure)
                self.feed.add_breath(unix, pressure)
                next_breath += 0.05
            if now >= next_pupil:
                for eye in (0, 1):
                    millimetres = 4.2 - 0.15 * eye + 0.08 * math.sin(2 * math.pi * 0.2 * now + eye)
                    self.mock.publish(
                        "pupil.%d.3d" % eye,
                        {
                            "id": eye,
                            "topic": "pupil.%d.3d" % eye,
                            "method": "3d c++",
                            "norm_pos": [0.5, 0.5],
                            "diameter": millimetres * 20.0,
                            "diameter_3d": millimetres,
                            "timestamp": self.mock.pupil_time(),
                            "confidence": 0.99,
                        },
                    )
                next_pupil += 1.0 / 40.0
            if now >= next_eeg:
                unix = time.time()
                channels = {}
                for index, name in enumerate(EPOC_CHANNELS):
                    channels[name] = 400.0 + 30.0 * math.sin(2 * math.pi * (8 + index) * now)
                self.hub.eeg.write_sample(unix, channels, counter=counter)
                self.feed.add_eeg(unix, channels)
                counter += 1
                next_eeg += 1.0 / 32.0
            self._stop.wait(0.01)

    def wait_for_label(self, label: str, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        path = self.directory / "pupil_recording" / "sniff_sync_triggers.csv"
        while time.time() < deadline:
            if path.exists() and label in path.read_text():
                return True
            time.sleep(0.02)
        return False

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        if self.hub is not None:
            self.hub.close()
        self.mock.stop()
        if self._plugin is not None:
            self._plugin.close()


class _Pool:
    def get_timestamp(self):
        return 0.0


class _Bridge:
    """Runs the real plugin against the mock's notification queue."""

    def __init__(self, mock, rec_dir, stop):
        from sniff_trigger_capture import Sniff_Trigger_Capture

        self.mock = mock
        self.stop = stop
        self.plugin = Sniff_Trigger_Capture(_Pool())
        self.plugin._recording = True
        self.plugin._rec_dir = str(rec_dir)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                note = self.mock.notes.get(timeout=0.05)
            except Exception:
                continue
            self.plugin.on_notify(note)
            self.plugin.recent_events({})

    def close(self) -> None:
        self.plugin.cleanup()
