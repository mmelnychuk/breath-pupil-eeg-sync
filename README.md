# Sniff sync

This runs on **one computer**. The sniffBasic nasal sensor, Pupil Capture, the Emotiv headset reader, and the live view all use that machine. Nothing here synchronises clocks between computers. Pupil Capture is reached on localhost.

The reference clock is **Unix time**: `time.time()`, seconds since 1970-01-01 00:00:00 UTC (Coordinated Universal Time). Every trigger, the breathing log, the electroencephalography (EEG) markers, the `unix_time` field of each Pupil annotation, and the live view's time axis use that clock. Pupil Capture has its own clock. At startup the hub asks Pupil for the time once (the network API `t` command), measures the offset, and converts Pupil sample times into Unix time. The hub does not send the `T` command, which would reset Pupil's clock under the cameras that are already running.

Breathing is a recorded stream. It is not a trigger. A trigger is a label sent with `trigger("label")`. The hub stamps Unix time when that label arrives, then writes the same value into the breathing log, the Pupil recording, and the EEG file.

## Clocks in the existing sniff scripts

The breathing capture that matters here is `getSniffs.py`. It writes `Unix Time` in seconds and `Nasal Pressure`. `Sniff_polar_display.py` writes that same Unix clock in milliseconds (`time.time() * 1000`). This project stores and plots seconds. If a file's time values are in milliseconds, they are divided by 1000 when read. `checkSniff.py` only checks that the sensor is connected. It does not timestamp anything. Notes on the other scripts are in [docs/unix-time-audit.md](docs/unix-time-audit.md); they are not part of this tool.

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

3. Send a label. This is the whole public interface:

```python
from sniff_sync import trigger

trigger("label")
```

`examples/trigger_call.py` is that call. The hub must be running first. If it is not, `trigger()` logs a warning and does not raise. Any other program on this machine can send one ZeroMQ (ZMQ) request string to `tcp://127.0.0.1:8765` and read the reply. The reply is the Unix timestamp, in seconds, that the hub assigned on arrival.

4. After the session, confirm the three copies match:

```bash
python -m sniff_sync.check recordings/P01
```

## What gets written

| File | Contents |
| --- | --- |
| `breathing.csv` | The `getSniffs.py` columns `Unix Time` (seconds) and `Nasal Pressure` (pascal). A trigger row uses the same columns and puts the label in `Trigger`. |
| `eeg.csv` | EEG samples and a trigger row for each label, on Unix time. |
| `pupil_recording/sniff_sync_triggers.csv` | The same triggers, written by the plugin into the Pupil recording directory. |

`getSniffs.py` keeps its own pressure file (`Unix Time`, `Nasal Pressure`). The hub does not rewrite that file. `--sniff-csv` only feeds the live view.

Inside Pupil, each annotation has `unix_time` (the shared clock) and `timestamp` (where to draw it on Pupil's own timeline: Unix time plus the one measured offset). Compare `unix_time`, not `timestamp`, when you line the recordings up.

## EEG (emokit)

The reader is the vendored copy of [openyou/emokit](https://github.com/openyou/emokit) at commit `7f25321`, under `third_party/emokit`. Limits and the two small fixes are in [third_party/emokit/VENDOR.md](third_party/emokit/VENDOR.md).

- The headset this project opens is the original Emotiv EPOC, not an EPOC+. The hub asks emokit for the original decryption key (`crypto_key`, consumer model, not the research or EPOC+ key). The key schedule itself was not edited. Two crash fixes in the vendored copy are described in [third_party/emokit/VENDOR.md](third_party/emokit/VENDOR.md): a Python 2 exception attribute, and a missing serial number.
- EPOC+, EPOC X, Insight, Flex, and MN8 are not read. Newer headsets need Emotiv's Cortex service, which this project does not speak.
- The library decrypts packets with `pycrypto==2.6.1` and reads USB (Universal Serial Bus) through `hidapi`. `pycrypto` 2.6.1 does not build on Python 3.12. On this environment the import fails, and the hub still records trigger markers in `eeg.csv` with a clear error. A real headset has to be tried on the rig's own Python.
- emokit stamps packets with `datetime.now()`. This project ignores that and stores `time.time()` when the packet is dequeued.
- Channel numbers are emokit's raw sensor values. Their FAQ says one step is about 0.51 microvolts on the original EPOC. Treat that as unchecked until you confirm it on the headset.

The fourteen channels saved are AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

## Live view

The window is [pyqtgraph](https://www.pyqtgraph.org/). It updates the lines in place, which keeps up with an EEG stream on the same machine without a browser in the middle. Matplotlib redraws too slowly for that, and a web page would be a second process this rig does not need.

The window is split left and right.

- Left: a polar breath-phase display in the style of `Sniff_polar_display.py`. The angle is the phase of the recent nasal-pressure signal. The distance from the centre is the size of the latest sample. Red is an inhale (positive pressure), blue an exhale.
- Right, top: pupil diameter for eye 0 and eye 1, in millimetres when the 3d detector is on. Gaze position is not plotted.
- Right, bottom: the EEG channels. Each channel has a checkbox. A switch chooses raw sensor values or power in the 8–12 Hz band. The axis is Unix time in seconds.

Triggers are marked on the polar plot and as vertical lines on the pupil and EEG panels. A later dynamical-system panel is another `add_panel()` call on the same window.

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

## Nonlinear interdependence

Performance of pairwise M, conditional M and partial cross mapping on simulated data from the 2018 coupled-oscillator model is in [docs/nlid-performance-report.md](docs/nlid-performance-report.md).

## On the real rig

- The sniff serial port in `getSniffs.py` is hard-coded. If the device name changed, `ls /dev/tty*` and update that path. The hub does not open the serial port.
- Feed the live view a `getSniffs.py` CSV (`Unix Time` in seconds, `Nasal Pressure`). A `Sniff_polar_display.py` file is milliseconds and is divided by 1000 on read. Our own files stay in seconds.
- Pupil's Network API is on localhost port 50020, and Sniff Trigger Capture is enabled. Start the recording before `trigger()`.
- After the hub connects, a fresh Pupil `t` query should sit within a few milliseconds of `time.time()` plus the printed offset. The link is local, so a gap of tens of milliseconds means the offset was measured badly and should be redone by restarting the hub before the recording.
- Pupil diameter in millimetres appears only when the 3d detector is running. Otherwise the plugin menu shows the 2d diameter in pixels, and the live view waits for millimetres.
- The headset must be an original consumer EPOC. EPOC+ will not use the right key. Install `hidapi` and a crypto package that matches the rig's Python (`pycrypto` 2.6.1 does not build on Python 3.12), and install emokit's udev rule (`third_party/emokit/linux/epoc.rules`) so the USB device is readable.
- Check one real trigger: the Unix time in `breathing.csv`, `eeg.csv`, and `sniff_sync_triggers.csv` should be the same number, and the annotation should sit on the right moment in Pupil Player.
- The 0.51 microvolt step is from emokit's FAQ. Confirm it before reporting EEG in microvolts.
