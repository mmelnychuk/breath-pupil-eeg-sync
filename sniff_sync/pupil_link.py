"""Talk to Pupil Capture on this computer.

Pupil Capture keeps its own clock ("Pupil time"). We measure the
difference between that clock and Unix time once, at startup, with the
Pupil Remote "t" command. The round trip is timed from the local clock
before the request and after the reply; Pupil's manual uses the midpoint.

We do not send the "T" command. That command resets Pupil's clock for
the cameras that are already running, and the manual warns that the
command itself arrives late.

Annotations still need a Pupil-time placement so they sit on the right
frame. That placement is unix_time plus the one offset. The value we
store, and the value the other recordings share, is the Unix time.
"""

from __future__ import annotations

import logging
import time

import msgpack
import zmq

log = logging.getLogger("sniff_sync")


class PupilLink:
    def __init__(self, port: int = 50020):
        self.port = port
        self.connected = False
        self.offset = None  # pupil_time = unix_time + offset
        self.sub_port = None
        self._ctx = None
        self._req = None

    def connect(self) -> bool:
        self._ctx = zmq.Context.instance()
        self._req = self._ctx.socket(zmq.REQ)
        self._req.setsockopt(zmq.RCVTIMEO, 400)
        self._req.setsockopt(zmq.SNDTIMEO, 400)
        self._req.setsockopt(zmq.LINGER, 0)
        try:
            self._req.setsockopt(zmq.REQ_RELAXED, 1)
            self._req.setsockopt(zmq.REQ_CORRELATE, 1)
        except (AttributeError, zmq.ZMQError):
            pass
        self._req.connect("tcp://127.0.0.1:%d" % self.port)
        try:
            self._req.send_string("v")
            self._req.recv_string()
            offsets = [self._one_offset() for _ in range(5)]
            self.offset = _trimmed_mean(offsets)
            self._req.send_string("SUB_PORT")
            self.sub_port = self._req.recv_string().strip()
        except Exception as exc:
            log.warning("Pupil Capture is not on localhost:%s (%s)", self.port, exc)
            self.close()
            return False
        self.connected = True
        return True

    def _one_offset(self) -> float:
        before = time.time()
        self._req.send_string("t")
        pupil_time = float(self._req.recv_string())
        after = time.time()
        return pupil_time - (before + after) / 2.0

    def pupil_to_unix(self, pupil_time: float):
        if self.offset is None:
            return None
        return float(pupil_time) - self.offset

    def send_trigger(self, event) -> bool:
        """Hand the hub's trigger to the Sniff Trigger Capture plugin.

        The plugin stores unix_time. timestamp is only the placement on
        Pupil's clock.
        """
        if not self.connected or self._req is None or self.offset is None:
            return False
        note = {
            "subject": "sniff_sync.trigger",
            "label": event.label,
            "unix_time": event.unix_time,
            "timestamp": event.unix_time + self.offset,
            "duration": 0.0,
            "trigger_id": event.trigger_id,
        }
        try:
            self._req.send_string("notify." + note["subject"], flags=zmq.SNDMORE)
            self._req.send(msgpack.packb(note, use_bin_type=True))
            self._req.recv_string()
        except Exception as exc:
            log.warning("Pupil did not take trigger %s (%s)", event.trigger_id, exc)
            self.connected = False
            return False
        return True

    def close(self) -> None:
        self.connected = False
        if self._req is not None:
            self._req.close(linger=0)
            self._req = None


def _trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) >= 4:
        ordered = ordered[1:-1]
    return sum(ordered) / float(len(ordered))
