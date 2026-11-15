"""Pupil Capture user plugin: store common triggers as annotations.

Copy this file to ~/pupil_capture_settings/plugins/ and enable
"Sniff Trigger Capture" in the Plugin Manager. It runs in the world
process. Triggers arrive as notifications with subject
sniff_sync.trigger, sent over the local ZeroMQ (ZMQ) network API.

The annotation's unix_time field is the shared clock (seconds since
1970-01-01 UTC). The timestamp field is the placement on Pupil's own
clock, already filled in by the hub. This plugin does not replace it.

Pupil Capture's bundled Python is older than the hub's. This file stays
on that older language: no dataclasses, no list[str] syntax.
"""

from __future__ import annotations

import csv
import json
import os

try:
    from plugin import Plugin
except ImportError:  # outside Pupil Capture, tests construct the class directly

    class Plugin(object):
        def __init__(self, g_pool):
            self.g_pool = g_pool

        def add_menu(self):
            self.menu = []

        def remove_menu(self):
            self.menu = None

        def notify_all(self, notification):
            pass


# Keep this identical to sniff_sync.timefmt.TIMESTAMP_FORMAT.
TIMESTAMP_FORMAT = ".6f"
COLUMNS = ["trigger_id", "unix_time", "label"]


def format_timestamp(value):
    return format(float(value), TIMESTAMP_FORMAT)


def annotation_from_trigger(notification, fallback_timestamp=None):
    """Build one annotation dict, or return None if this is not our trigger."""
    if not isinstance(notification, dict):
        return None
    if notification.get("subject") not in (None, "sniff_sync.trigger"):
        return None
    trigger_id = str(notification.get("trigger_id") or "")
    if not trigger_id:
        return None
    if notification.get("unix_time") is None:
        return None
    if notification.get("timestamp") is None:
        if fallback_timestamp is None:
            return None
        placement = float(fallback_timestamp)
    else:
        placement = float(notification["timestamp"])
    return {
        "topic": "annotation",
        "label": str(notification.get("label") or "trigger"),
        "timestamp": placement,
        "duration": float(notification.get("duration") or 0.0),
        "trigger_id": trigger_id,
        "unix_time": float(notification["unix_time"]),
        "source": "sniff_sync",
    }


class Sniff_Trigger_Capture(Plugin):
    uniqueness = "by_class"
    order = 0.95
    icon_chr = "S"
    icon_font = "roboto"

    def __init__(self, g_pool, **kwargs):
        super(Sniff_Trigger_Capture, self).__init__(g_pool)
        self._pending = []
        self._seen = set()
        self._recording = False
        self._rec_dir = ""
        self._sidecar = None
        self.trigger_count = 0
        self.missed_while_idle = 0
        self.status_text = "No triggers yet"
        self.eye0_text = "Eye 0: no diameter yet"
        self.eye1_text = "Eye 1: no diameter yet"

    def init_ui(self):
        from pyglui import ui

        self.add_menu()
        self.menu.label = "Sniff sync"
        self.menu.append(
            ui.Info_Text(
                "Stores triggers as annotations. The unix_time field is "
                "Unix time from this computer. Start the recording before "
                "the first trigger."
            )
        )
        self.menu.append(ui.Text_Input("status_text", self, label="Status", setter=lambda _v: None))
        self.menu.append(ui.Text_Input("eye0_text", self, label="Eye 0", setter=lambda _v: None))
        self.menu.append(ui.Text_Input("eye1_text", self, label="Eye 1", setter=lambda _v: None))

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {}

    def on_notify(self, notification):
        subject = notification.get("subject", "")
        if subject == "recording.started":
            self._recording = True
            for key in ("rec_path", "record_path", "rec_dir"):
                if notification.get(key):
                    self._rec_dir = notification[key]
                    break
            self.status_text = "Recording"
            return
        if subject == "recording.stopped":
            self._recording = False
            self.status_text = "Recording stopped. %d triggers stored" % self.trigger_count
            return
        if subject != "sniff_sync.trigger":
            return
        fallback = None
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "get_timestamp"):
            fallback = self.g_pool.get_timestamp()
        annotation = annotation_from_trigger(notification, fallback_timestamp=fallback)
        if annotation is None:
            return
        if annotation["trigger_id"] in self._seen:
            return
        self._seen.add(annotation["trigger_id"])
        if not self._recording:
            self.missed_while_idle += 1
            self.status_text = "Trigger while not recording (%d)" % self.missed_while_idle
            return
        self._pending.append(annotation)
        self.trigger_count += 1
        self.status_text = "%d triggers. Last: %s" % (self.trigger_count, annotation["label"])

    def recent_events(self, events):
        self._note_diameter(events)
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        existing = list(events.get("annotation") or [])
        existing.extend(batch)
        events["annotation"] = existing
        self._append_sidecar(batch)

    def _note_diameter(self, events):
        for datum in events.get("pupil") or []:
            eye = datum.get("id")
            if eye not in (0, 1):
                continue
            value = datum.get("diameter_3d")
            unit = "mm"
            if value in (None, "", 0, 0.0):
                value = datum.get("diameter")
                unit = "px"
            if value in (None, "", 0, 0.0):
                continue
            text = "Eye %s: %.2f %s" % (eye, float(value), unit)
            if eye == 0:
                self.eye0_text = text
            else:
                self.eye1_text = text

    def _append_sidecar(self, annotations):
        if not self._rec_dir:
            return
        path = os.path.join(self._rec_dir, "sniff_sync_triggers.csv")
        if self._sidecar is None:
            directory = os.path.dirname(path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            self._sidecar = open(path, "a", newline="")
            self._writer = csv.DictWriter(self._sidecar, fieldnames=COLUMNS)
            if new_file:
                self._writer.writeheader()
        for annotation in annotations:
            self._writer.writerow(
                {
                    "trigger_id": annotation["trigger_id"],
                    "unix_time": format_timestamp(annotation["unix_time"]),
                    "label": annotation["label"],
                }
            )
        self._sidecar.flush()

    def cleanup(self):
        if self._sidecar is not None:
            self._sidecar.close()
            self._sidecar = None
