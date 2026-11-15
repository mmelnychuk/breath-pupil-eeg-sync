"""The only call a task script needs.

The hub must already be running on this computer
(python -m sniff_sync). trigger() sends the label. The hub stamps
Unix time when it arrives and writes that one value into the breathing
log, the Pupil recording, and the EEG file.

In KORNHUBER_Exp.py, next to the existing time.time() write inside on_key:

    from sniff_sync import trigger
    trigger("keypress")

When the task window opens, before mainloop:

    trigger("task_start")

A button handler or a hardware alert uses the same call:

    trigger("mark")

Nothing here reads the sniff sensor. Breathing is a separate stream.
"""

from sniff_sync import trigger


def on_key(event):
    trigger("keypress")


def on_task_start():
    trigger("task_start")


if __name__ == "__main__":
    trigger("example")
