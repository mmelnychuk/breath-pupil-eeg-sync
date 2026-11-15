# Sniff sync

This runs on **one computer**. The sniffBasic nasal sensor, Pupil Capture, the Emotiv headset reader, and the live view all use that machine. Nothing here synchronises clocks between computers. Pupil Capture is reached on localhost.

The reference clock is **Unix time**: `time.time()`, seconds since 1970-01-01 00:00:00 UTC (Coordinated Universal Time). Every trigger, the breathing log, the electroencephalography (EEG) markers, the `unix_time` field of each Pupil annotation, and the live view's time axis use that clock. Pupil Capture has its own clock. At startup the hub asks Pupil for the time once (the network API `t` command), measures the offset, and converts Pupil sample times into Unix time. The hub does not send the `T` command, which would reset Pupil's clock under the cameras that are already running.

Breathing is a recorded stream. It is not a trigger. A trigger is a label sent with `trigger("label")`. The hub stamps Unix time when that label arrives, then writes the same value into the breathing log, the Pupil recording, and the EEG file.

## Clocks in the existing sniff scripts

Those scripts already call `time.time()`. The notes are in [docs/unix-time-audit.md](docs/unix-time-audit.md). `getSniffs.py` writes seconds. `Sniff_polar_display.py` writes the same Unix clock in milliseconds (`time.time() * 1000`). Point the live view at a `getSniffs.py` file, not the polar-display file.

## Install

```bash
pip install -e .
cp pupil_plugin/sniff_trigger_capture.py ~/pupil_capture_settings/plugins/
```

Restart Pupil Capture and enable **Sniff Trigger Capture** in the Plugin Manager (world window). Also enable the Network API plugin. Leave it on `127.0.0.1` port `50020`.

## Run a session

1. Start Pupil Capture and start the recording before the first trigger.
2. Start the hub in the experiment folder, and point it at the pressure file `getSniffs.py` is writing:

```bash
python -m sniff_sync --session recordings/P01 --sniff-csv subject_data/P01/KH_sniff.csv
```

3. From the task script, send a label. This is the whole public interface:

```python
from sniff_sync import trigger

trigger("task_start")   # when the task window opens
trigger("keypress")     # inside the existing key handler, next to time.time()
```

A button handler or a hardware alert uses that same call. `examples/trigger_call.py` shows the two lines. The hub must be running first. If it is not, `trigger()` logs a warning and does not raise, so the task continues.

Any other program on this machine can send one ZeroMQ (ZMQ) request string to `tcp://127.0.0.1:8765` and read the reply. The reply is the Unix timestamp the hub assigned. The Python function above is the call to use from the experiment scripts.

4. After the session, confirm the three copies match:

```bash
python -m sniff_sync.check recordings/P01
```

## What gets written

| File | Contents |
| --- | --- |
| `breathing.csv` | Trigger rows (`kind` = `trigger`) on Unix time. In a simulated session it also holds nasal-pressure samples, in pascal, the same quantity `getSniffs.py` stores. |
| `eeg.csv` | EEG samples and a trigger row for each label, on Unix time. |
| `pupil_recording/sniff_sync_triggers.csv` | The same triggers, written by the plugin into the Pupil recording directory. |

`getSniffs.py` keeps its own pressure file (`Unix Time`, `Nasal Pressure`). The hub does not rewrite that file. `--sniff-csv` only feeds the live view.

Inside Pupil, each annotation has `unix_time` (the shared clock) and `timestamp` (where to draw it on Pupil's own timeline: Unix time plus the one measured offset). Compare `unix_time`, not `timestamp`, when you line the recordings up.

## EEG (emokit)

The reader is the vendored copy of [openyou/emokit](https://github.com/openyou/emokit) at commit `7f25321`, under `third_party/emokit`. Limits and the two small fixes are in [third_party/emokit/VENDOR.md](third_party/emokit/VENDOR.md).

- Headsets it can read: EPOC, and EPOC+ from before 2016.
- It cannot read EPOC+ from 2016 onward, EPOC X, Insight, Flex, or MN8. Those need Emotiv's Cortex service, which this project does not speak.
- The library decrypts packets with `pycrypto==2.6.1` and reads USB (Universal Serial Bus) through `hidapi`. `pycrypto` 2.6.1 does not build on Python 3.12. On this environment the import fails, and the hub still records trigger markers in `eeg.csv` with a clear error. A real headset has to be tried on the rig's own Python.
- emokit stamps packets with `datetime.now()`. This project ignores that and stores `time.time()` when the packet is dequeued.
- Channel numbers are emokit's raw sensor values. Their FAQ says one step is about 0.51 microvolts on the original EPOC. Treat that as unchecked until you confirm it on the headset.

The fourteen channels saved are AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

## Live view

The window is [pyqtgraph](https://www.pyqtgraph.org/). It updates the lines in place, which keeps up with an EEG stream on the same machine without a browser in the middle. Matplotlib redraws too slowly for that, and a web page would be a second process this rig does not need.

Three plots share one Unix-time axis (drawn as clock time):

- Pupil diameter for eye 0 and eye 1, in millimetres when the 3d detector is on. Gaze position is not plotted.
- Nasal pressure, in pascal.
- The EEG channels, stacked so they can be read together. The file stores the raw values; the stack is only for display.

Each trigger is a vertical line through all three plots at the Unix time the hub stamped.

## Simulated session

No sensor, no glasses, no headset:

```bash
python -m sniff_sync --simulate --session recordings/sim
```

Add `--no-window` to skip the plot, and `--seconds 5` to stop on its own. Then:

```bash
python -c 'from sniff_sync import trigger; trigger("stimulus_onset")'
python -m sniff_sync.check recordings/sim
```

Tests, also with no hardware:

```bash
python -m unittest discover -s tests
```

## On the real rig

- The sniff serial port in `getSniffs.py` is hard-coded. If the device name changed, `ls /dev/tty*` and update that path. The hub does not open the serial port.
- Feed the live view a `getSniffs.py` CSV (seconds). The polar-display CSV is milliseconds and will not sit on this axis.
- Pupil's Network API is on localhost port 50020, and Sniff Trigger Capture is enabled. Start the recording before `trigger()`.
- After the hub connects, a fresh Pupil `t` query should sit within a few milliseconds of `time.time()` plus the printed offset. The link is local, so a gap of tens of milliseconds means the offset was measured badly and should be redone by restarting the hub before the recording.
- Pupil diameter in millimetres appears only when the 3d detector is running. Otherwise the plugin menu shows the 2d diameter in pixels, and the live view waits for millimetres.
- Confirm the Emotiv model against the list above before expecting samples. Install `hidapi` and a crypto package that matches the rig's Python, and install emokit's udev rule (`third_party/emokit/linux/epoc.rules`) so the USB device is readable.
- Research-edition headsets may need emokit's research flag. That is a headset setting, not something the simulated tests cover.
- Check one real trigger: the Unix time in `breathing.csv`, `eeg.csv`, and `sniff_sync_triggers.csv` should be the same number, and the annotation should sit on the right moment in Pupil Player.
- The 0.51 microvolt step is from emokit's FAQ. Confirm it before reporting EEG in microvolts.
