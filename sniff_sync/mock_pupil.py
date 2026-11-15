"""A Pupil Remote stand-in for tests. No headset required.

Its clock is not Unix time. It starts at an arbitrary epoch so a test
can see that we converted it.
"""

from __future__ import annotations

import queue
import threading
import time

import msgpack
import zmq


class MockPupil:
    def __init__(self):
        self._epoch = 5000.0
        self._perf0 = time.perf_counter()
        self.notes = queue.Queue()
        self._ctx = zmq.Context()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind("tcp://127.0.0.1:*")
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind("tcp://127.0.0.1:*")
        self.port = _port(self._rep)
        self.sub_port = _port(self._pub)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def pupil_time(self) -> float:
        return self._epoch + (time.perf_counter() - self._perf0)

    def start(self) -> None:
        self._thread.start()

    def publish(self, topic: str, datum: dict) -> None:
        self._pub.send_multipart([topic.encode("utf-8"), msgpack.packb(datum, use_bin_type=True)])

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        while not self._stop.is_set():
            if self._rep not in dict(poller.poll(100)):
                continue
            parts = self._rep.recv_multipart()
            if len(parts) == 1:
                self._rep.send_string(self._command(parts[0].decode("utf-8")))
            else:
                note = msgpack.unpackb(parts[1], raw=False)
                self.notes.put(note)
                self._rep.send_string("Notification received.")

    def _command(self, text: str) -> str:
        if text == "t":
            return repr(self.pupil_time())
        if text == "v":
            return "mock-pupil 1"
        if text == "SUB_PORT":
            return str(self.sub_port)
        if text == "PUB_PORT":
            return str(self.sub_port)
        if text == "R" or text.startswith("R "):
            return "OK"
        if text == "r":
            return "OK"
        return "Unknown command."

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._rep.close(linger=0)
        self._pub.close(linger=0)
        self._ctx.term()


def _port(socket) -> int:
    endpoint = socket.getsockopt(zmq.LAST_ENDPOINT)
    if isinstance(endpoint, bytes):
        endpoint = endpoint.decode("utf-8")
    return int(endpoint.rsplit(":", 1)[1])
