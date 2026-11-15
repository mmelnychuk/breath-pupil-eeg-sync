"""Pupil diameter from a Pupil Core pupil message.

Gaze (where the eye is pointed) is not used. Diameter is the size of
the pupil. The 3d detector's diameter_3d is in millimetres. The 2d
detector's diameter is in pixels. Millimetres are preferred.
"""

from __future__ import annotations

MIN_CONFIDENCE = 0.6


def sample_from_pupil_message(topic: str, datum: dict):
    """Return {eye, diameter, unit, timestamp} or None.

    timestamp here is still Pupil's own clock. The caller converts it
    to Unix time with the one offset measured at startup.
    """
    if not isinstance(datum, dict):
        return None
    text = topic.decode("utf-8", "replace") if isinstance(topic, bytes) else str(topic)
    if text.startswith("gaze"):
        return None
    if not text.startswith("pupil"):
        inner = str(datum.get("topic") or "")
        if not inner.startswith("pupil"):
            return None
        text = inner
    try:
        confidence = float(datum.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if confidence < MIN_CONFIDENCE:
        return None
    eye = datum.get("id")
    if eye not in (0, 1):
        parts = text.split(".")
        if len(parts) >= 2 and parts[1] in ("0", "1"):
            eye = int(parts[1])
        else:
            return None
    value = None
    unit = None
    if datum.get("diameter_3d") not in (None, ""):
        try:
            value = float(datum["diameter_3d"])
            unit = "mm"
        except (TypeError, ValueError):
            value = None
    if value is None and datum.get("diameter") not in (None, ""):
        try:
            value = float(datum["diameter"])
            unit = "px"
        except (TypeError, ValueError):
            return None
    if value is None or value <= 0:
        return None
    try:
        timestamp = float(datum["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"eye": int(eye), "diameter": value, "unit": unit, "timestamp": timestamp}
