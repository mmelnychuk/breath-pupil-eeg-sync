"""Common triggers for one computer.

The sniff sensor, Pupil Capture, the Emotiv headset, and the live view
all sit on this machine. The clock is Unix time: time.time(), seconds
since 1970-01-01 UTC (Coordinated Universal Time).

The public call is trigger(label).
"""

from sniff_sync.trigger import trigger

__all__ = ["trigger"]
__version__ = "0.1.0"
