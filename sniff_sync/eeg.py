"""Read EEG from emokit, or from a stand-in when no headset is attached.

A real headset is opened only when asked. Importing this module does
not import emokit, because emokit imports an old crypto package at
import time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from sniff_sync.eeg_channels import EPOC_CHANNELS

_EMOKIT_PYTHON = Path(__file__).resolve().parents[1] / "third_party" / "emokit" / "python"


def emokit_status() -> dict:
    """Say whether the vendored library can be imported, and which headsets it knows."""
    status = {
        "vendored": str(_EMOKIT_PYTHON.parent),
        "commit": "7f25321a1c3a240f5b64a1572e5f106807b1beea",
        "supported_models": ["original EPOC"],
        "unsupported_models": [
            "EPOC+",
            "EPOC X",
            "Insight",
            "Flex",
            "MN8",
        ],
        "import_error": "",
    }
    if str(_EMOKIT_PYTHON) not in sys.path:
        sys.path.insert(0, str(_EMOKIT_PYTHON))
    try:
        import emokit.emotiv  # noqa: F401
    except Exception as exc:
        status["import_error"] = "%s: %s" % (type(exc).__name__, exc)
    return status


def channels_from_packet(packet) -> dict:
    """Pull the fourteen EEG channels off an emokit packet.

    emokit stores each sensor as packet.sensors[name]['value'].
    """
    sensors = getattr(packet, "sensors", None) or {}
    values = {}
    for name in EPOC_CHANNELS:
        sensor = sensors.get(name)
        if not isinstance(sensor, dict) or sensor.get("value") in (None, "?"):
            continue
        try:
            values[name] = float(sensor["value"])
        except (TypeError, ValueError):
            continue
    return values


class EmokitHeadset:
    """Live reader. Stamps each dequeued packet with Unix time.

    emokit's own stamp is datetime.now(). That value is ignored.
    """

    def __init__(self):
        self._headset = None
        self.error = ""

    def open(self) -> bool:
        status = emokit_status()
        if status["import_error"]:
            self.error = status["import_error"]
            return False
        from emokit.emotiv import Emotiv

        try:
            # Original consumer EPOC: the old packet layout and the original
            # AES key (crypto_key, not the EPOC+ key). is_research stays false.
            self._headset = Emotiv(
                display_output=False,
                write=False,
                verbose=False,
                is_research=False,
                force_old_crypto=True,
                force_epoc_mode=False,
            )
        except Exception as exc:
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self._headset = None
            return False
        return True

    def read(self):
        """Return {unix_time, counter, channels} or None if no packet is waiting."""
        if self._headset is None:
            return None
        packet = self._headset.dequeue()
        if packet is None:
            return None
        return {
            "unix_time": time.time(),
            "counter": getattr(packet, "counter", ""),
            "channels": channels_from_packet(packet),
        }

    def close(self) -> None:
        if self._headset is not None:
            try:
                self._headset.stop()
            except Exception:
                pass
            self._headset = None
