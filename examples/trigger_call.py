"""The only call: trigger("label").

The hub must already be running on this computer
(python -m sniff_sync). It stamps Unix time, in seconds, when the
label arrives, and writes that value into the breathing log, the Pupil
recording, and the EEG file.
"""

from sniff_sync import trigger

trigger("label")
