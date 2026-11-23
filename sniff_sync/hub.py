"""Receive a label, stamp Unix time, write it to all three recordings."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import msgpack
import zmq

from sniff_sync.diameter import sample_from_pupil_message
from sniff_sync.trigger import DEFAULT_PORT
from sniff_sync.feed import Feed
from sniff_sync.pupil_link import PupilLink
from sniff_sync.records import BreathingLog, EegLog, Trigger, new_trigger_id
from sniff_sync.rolling import RollingHistory

log = logging.getLogger("sniff_sync")


class Hub:
    def __init__(
        self,
        session,
        pupil_port: int = 50020,
        port: int = DEFAULT_PORT,
        feed: Feed | None = None,
        history: RollingHistory | None = None,
    ):
        self.session = Path(session)
        self.session.mkdir(parents=True, exist_ok=True)
        self.breathing = BreathingLog(self.session / "breathing.csv")
        self.eeg = EegLog(self.session / "eeg.csv")
        self.pupil = PupilLink(port=pupil_port)
        self.feed = feed if feed is not None else Feed()
        # Longer than the plot. The plot keeps 20 seconds; this keeps the
        # coupling window. resample() is not called from the live loop.
        self.history = history if history is not None else RollingHistory()
        self.port = port
        self._ctx = None
        self._rep = None
        self._sub = None
        self._stop = threading.Event()
        self._threads = []

    def start(self) -> None:
        self.pupil.connect()
        self._ctx = zmq.Context.instance()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind("tcp://127.0.0.1:%s" % (self.port or "*"))
        endpoint = self._rep.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(endpoint, bytes):
            endpoint = endpoint.decode("utf-8")
        self.port = int(endpoint.rsplit(":", 1)[1])
        self._threads.append(threading.Thread(target=self._serve, name="trigger-input", daemon=True))
        if self.pupil.connected and self.pupil.sub_port:
            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.connect("tcp://127.0.0.1:%s" % self.pupil.sub_port)
            self._sub.setsockopt(zmq.SUBSCRIBE, b"pupil.")
            self._sub.setsockopt(zmq.RCVTIMEO, 100)
            self._threads.append(threading.Thread(target=self._diameters, name="pupil-diameter", daemon=True))
        for thread in self._threads:
            thread.start()

    def accept(self, label: str) -> Trigger:
        """Stamp one label on arrival and fan it out. This is the hub side of trigger(label)."""
        text = str(label).strip() or "trigger"
        event = Trigger(new_trigger_id(), text.splitlines()[0][:200], unix_time=time.time())
        self.breathing.write_trigger(event)
        self.eeg.write_trigger(event)
        self.pupil.send_trigger(event)
        self.feed.add_trigger(event.unix_time, event.label, event.trigger_id)
        self.history.add_trigger(event.unix_time, event.label)
        log.info("trigger %s %s at %.6f", event.trigger_id, event.label, event.unix_time)
        return event

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        while not self._stop.is_set():
            if self._rep not in dict(poller.poll(100)):
                continue
            label = self._rep.recv_string()
            event = self.accept(label)
            from sniff_sync.timefmt import format_timestamp

            self._rep.send_string(format_timestamp(event.unix_time))

    def _diameters(self) -> None:
        while not self._stop.is_set():
            try:
                topic, payload = self._sub.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            datum = msgpack.unpackb(payload, raw=False)
            text = topic.decode("utf-8", "replace") if isinstance(topic, bytes) else str(topic)
            sample = sample_from_pupil_message(text, datum)
            if sample is None or sample["unit"] != "mm":
                continue
            unix = self.pupil.pupil_to_unix(sample["timestamp"])
            if unix is None:
                continue
            self.feed.add_pupil(unix, sample["eye"], sample["diameter"])
            self.history.add_pupil(unix, sample["eye"], sample["diameter"])

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        if self._rep is not None:
            self._rep.close(linger=0)
        if self._sub is not None:
            self._sub.close(linger=0)
        self.pupil.close()
        self.breathing.close()
        self.eeg.close()
