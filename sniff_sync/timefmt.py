"""How Unix timestamps are written.

The Pupil plugin copies this format string. Both sides format the same
float, so the files can be compared as text.
"""

TIMESTAMP_FORMAT = ".6f"


def format_timestamp(value: float) -> str:
    return format(float(value), TIMESTAMP_FORMAT)


def unix_seconds(value: float) -> float:
    """Return Unix time in seconds.

    getSniffs.py already writes seconds. Sniff_polar_display.py writes
    time.time() multiplied by 1000, which is the same clock in milliseconds.
    A value that large is divided by 1000. Seconds are left alone.
    """
    number = float(value)
    if number > 1e11:
        return number / 1000.0
    return number
