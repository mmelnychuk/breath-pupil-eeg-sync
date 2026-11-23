"""Adapt the current ``hid`` package to the function names emokit already calls.

``pip install hidapi`` provides ``import hid``. This module is used only when
``import hidapi`` itself is not available. The headset is still read as a
human-interface device. Nothing here changes packet bytes.
"""

from __future__ import absolute_import


def _library():
    try:
        import hid
    except ImportError as exc:
        raise ImportError(
            "Opening the headset needs the hidapi package. Install it with: pip install hidapi"
        ) from exc
    return hid


class _DeviceInfo(object):
    """One enumerated device, with the attribute names the reader already uses."""

    def __init__(self, info):
        self.vendor_id = info.get("vendor_id")
        self.product_id = info.get("product_id")
        self.serial_number = _text(info.get("serial_number"))
        self.path = info.get("path")
        self.product_string = _text(info.get("product_string"))
        self.manufacturer_string = _text(info.get("manufacturer_string"))


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def hid_init():
    return None


def hid_exit():
    return None


def hid_enumerate():
    devices = []
    for info in _library().enumerate():
        devices.append(_DeviceInfo(info))
    return devices


def hid_open_path(path):
    device = _library().device()
    device.open_path(path)
    return device


def hid_read_timeout(device, length, timeout_ms):
    data = device.read(int(length), int(timeout_ms))
    if not data:
        return []
    return list(data)


def hid_close(device):
    if device is not None:
        device.close()
