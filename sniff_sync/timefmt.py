"""How Unix timestamps are written.

The Pupil plugin copies this format string. Both sides format the same
float, so the files can be compared as text.
"""

TIMESTAMP_FORMAT = ".6f"


def format_timestamp(value: float) -> str:
    return format(float(value), TIMESTAMP_FORMAT)
