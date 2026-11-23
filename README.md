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

## Aligned window

The live plot keeps about 20 seconds. A coupling window is longer than that, so a second buffer keeps the raw samples. It is filled from the same three places as the plot: the nasal-pressure file, the pupil-diameter messages, and the electroencephalography (EEG) packets. The plot still keeps 20 seconds. This buffer is not drawn yet.

Ask for the window when you want it, about once a second. On a running hub that is:

```python
window = hub.history.resample()
```

`hub` is the `Hub` the program already builds. `window.time_seconds` is Unix time, one sample every 0.1 seconds (10 hertz). The other fields are on that same grid.

| Field | What it is |
| --- | --- |
| `breathing` | Nasal pressure, in the units of the file. |
| `pupil_diameter` | Pupil diameter in millimetres. Both eyes when both are present, otherwise the eye that is present. |
| `delta_envelope` | Amplitude of the 1 to 4 hertz band. |
| `alpha_envelope` | Amplitude of the 8 to 12 hertz band. |
| `delta_to_alpha` | The delta envelope divided by the alpha envelope. |
| `valid` | True only where all five of those have a number. |
| `delays_seconds` | How far each path was shifted earlier, in seconds. |

A missing sample is not-a-number in its own array, and `valid` is false there. The length defaults to 60 seconds. `hub.history.set_window_seconds(45)` accepts any length from 30 to 120 seconds. The buffer keeps 120 seconds plus 8 seconds of filter warm-up whatever length you have selected, so raising the length does not start from an empty buffer. NumPy and SciPy do the filtering. They are installed with the package.

Breathing goes through a forward low-pass filter at 4 hertz, then onto the grid. Forward means each output sample uses the current input and inputs that have already arrived, not inputs that are still in the future. The low-pass is what stops higher frequencies from folding into the 10 hertz grid.

The pupil path is a short list of steps, `hub.history.pupil_stages`. Each step has a name and a `delay_s`, and the names and delays come back on `window.pupil_stage_delays_seconds`. The first step fills a hole of up to 0.20 seconds with a straight line and adds no delay. A longer hole is left missing. The second step is the same kind of low-pass used for breathing. Blink correction is not one of these steps. To add it later, insert a step between those two. The step should leave its values lined up with the times it was given, and set `delay_s` to the number of seconds it shifts the diameter later. The resampler adds that to the pupil delay and shifts the timestamps back.

Electroencephalography is not used as the raw trace. The default channels are AF3, AF4, F3 and F4. The original EPOC headset has no electrode at Fz. These four are the frontal ones, AF3 and F3 on the left and AF4 and F4 on the right. F7 and F8 sit further to the side and pick up more muscle activity. FC5 and FC6 sit lower, toward the motor strip. `hub.history.eeg_channels = ("F3", "F4")` selects a different subset. Each selected channel is band-pass filtered, the amplitude is taken with a forward smoother at 1 hertz, and the channels are averaged. A sine wave of amplitude A comes back near A. The alpha value in the ratio is not allowed to fall below one thousandth of the median alpha in the window, so a near-zero alpha does not send the ratio off to infinity.

A forward filter delays what comes out of it. Breathing and pupil diameter are shifted by the group delay of their low-pass at 0.25 hertz, which is about the rate of a breath. Each envelope is shifted by the average group delay of its band-pass across that band, plus the group delay of the 1 hertz smoother at 0.25 hertz. For a 128 hertz electroencephalography stream, a 20 hertz nasal-pressure stream, and a 120 hertz pupil stream, those delays are 48.9 milliseconds for breathing, 56.3 milliseconds for pupil diameter, 406.5 milliseconds for the delta envelope, and 363.0 milliseconds for the alpha envelope. A 32 hertz simulated electroencephalography stream is within a few milliseconds of the same envelope delays. The numbers that were actually used are always in `window.delays_seconds`, because they move a little when the sample rate moves. The grid stops at the latest time for which breathing, pupil diameter, and both envelopes are all present. That is a few hundred milliseconds behind the newest packet, set by the slower envelope. The same lag on every signal is acceptable for a display. A different lag on one signal would bias the direction of coupling.

The group delay inside the delta band is not flat. It is shorter near 4 hertz than near 1 hertz, by a few tenths of a second. The single delta delay above is the average across 1 to 4 hertz. A burst that sits at one edge of that band can still be early or late by more than one sample of this grid. The alpha band is flatter.

One or two missed samples in breathing or in the electroencephalography are filled with a straight line. A longer hole stays missing. After a hole the filters start again, and the stretch while they settle stays missing too. That settling is about a third of a second for the low-pass and about two seconds for the envelopes.

The call is meant to be made about once a second. With a 128 hertz electroencephalography stream, a 120 hertz pupil stream and a 20 hertz nasal-pressure stream, one call took 9 milliseconds for a 30 second window, 15 milliseconds for 60 seconds, and 27 milliseconds for 120 seconds. Using all fourteen channels for a 60 second window took 32 milliseconds.

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
pip install -e '.[analysis]'
python -m unittest discover -s tests
```

## Offline coupling analysis

After a session, nearest-neighbour interdependence among breathing, pupil diameter, and the electroencephalography can be estimated without starting the hub:

```bash
python -m sniff_sync.nlid recordings/P01
```

The live hub plots pupil diameter but does not save it. Put a Pupil Core `pupil_positions.csv` in the session (or pass `--pupil`). If that file is still on Pupil's clock, pass `--pupil-offset` as the offset the hub printed (pupil time minus Unix time).

Electroencephalography is not used raw. Each channel is split into delta, theta, alpha, beta, and low gamma; the squared Hilbert amplitude is smoothed; and every ratio of those powers is kept as a candidate attentional signal. The default channels are the frontal set AF3, AF4, F3, F4 (this headset has no Fz). Delta/alpha is not treated as special.

The table and the figure land in `recordings/P01/nlid/`. The measures, the conditional test that separates a direct breathing–attention link from one that only passes through pupil diameter, and the partial cross mapping comparison are written out in [docs/nonlinear_interdependence.md](docs/nonlinear_interdependence.md).

## On the real rig

- The sniff serial port in `getSniffs.py` is hard-coded. If the device name changed, `ls /dev/tty*` and update that path. The hub does not open the serial port.
- Feed the live view a `getSniffs.py` CSV (`Unix Time` in seconds, `Nasal Pressure`). A `Sniff_polar_display.py` file is milliseconds and is divided by 1000 on read. Our own files stay in seconds.
- Pupil's Network API is on localhost port 50020, and Sniff Trigger Capture is enabled. Start the recording before `trigger()`.
- After the hub connects, a fresh Pupil `t` query should sit within a few milliseconds of `time.time()` plus the printed offset. The link is local, so a gap of tens of milliseconds means the offset was measured badly and should be redone by restarting the hub before the recording.
- Pupil diameter in millimetres appears only when the 3d detector is running. Otherwise the plugin menu shows the 2d diameter in pixels, and the live view waits for millimetres.
- The headset must be an original consumer EPOC. EPOC+ will not use the right key. Install `hidapi` and a crypto package that matches the rig's Python (`pycrypto` 2.6.1 does not build on Python 3.12), and install emokit's udev rule (`third_party/emokit/linux/epoc.rules`) so the USB device is readable.
- Check one real trigger: the Unix time in `breathing.csv`, `eeg.csv`, and `sniff_sync_triggers.csv` should be the same number, and the annotation should sit on the right moment in Pupil Player.
- The 0.51 microvolt step is from emokit's FAQ. Confirm it before reporting EEG in microvolts.
