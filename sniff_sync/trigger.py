"""The one way to fire a trigger: trigger(label).

The function sends the label to the hub on this computer. The hub stamps
Unix time when the label arrives. Calling this from a task script, a
button handler, or a hardware alert is the same call.
"""

from __future__ import annotations

import logging
import os

import zmq

log = logging.getLogger("sniff_sync")

DEFAULT_PORT = 8765


def trigger(label: str) -> None:
    """Send one trigger label to the local hub.

    Does nothing noisy to the caller when the hub is not running: the
    experiment script keeps going, and a warning is logged.
    """
    port = int(os.environ.get("SNIFF_SYNC_PORT", str(DEFAULT_PORT)))
    text = str(label).strip()
    if not text:
        text = "trigger"
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 400)
    socket.setsockopt(zmq.SNDTIMEO, 400)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.setsockopt(zmq.REQ_RELAXED, 1)
        socket.setsockopt(zmq.REQ_CORRELATE, 1)
    except (AttributeError, zmq.ZMQError):
        pass
    try:
        socket.connect("tcp://127.0.0.1:%d" % port)
        socket.send_string(text.splitlines()[0][:200])
        socket.recv_string()
    except Exception:
        log.warning("trigger %r did not reach the local hub", text)
    finally:
        socket.close()
