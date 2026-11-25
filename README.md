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
| `pupil_diameter.csv` | Pupil diameter in millimetres for each eye, on Unix time, so a later reference pass can read it. |
| `pupil_recording/sniff_sync_triggers.csv` | The same triggers, written by the plugin into the Pupil recording directory. |
| `coupling_approximate.csv` | Live coupling scores. Every row is labelled approximate. |
| `coupling_reference.csv` | The same windows recomputed after the session with zero-phase filters. |

`getSniffs.py` keeps its own pressure file (`Unix Time`, `Nasal Pressure`). The hub does not rewrite that file. `--sniff-csv` only feeds the live view.

Inside Pupil, each annotation has `unix_time` (the shared clock) and `timestamp` (where to draw it on Pupil's own timeline: Unix time plus the one measured offset). Compare `unix_time`, not `timestamp`, when you line the recordings up.

## EEG (emokit)

The reader is the vendored copy of [openyou/emokit](https://github.com/openyou/emokit) at commit `7f25321`, under `third_party/emokit`. Limits and the two small fixes are in [third_party/emokit/VENDOR.md](third_party/emokit/VENDOR.md).

- The headset this project opens is the original Emotiv EPOC, not an EPOC+. The hub asks emokit for the original decryption key (`crypto_key`, consumer model, not the research or EPOC+ key). The key schedule itself was not edited. The library changes are described in [third_party/emokit/VENDOR.md](third_party/emokit/VENDOR.md).
- EPOC+, EPOC X, Insight, Flex, and MN8 are not read. Newer headsets need Emotiv's Cortex service, which this project does not speak.
- Packets are decrypted with `pycryptodome` (`Crypto.Cipher.AES`), which installs on Python 3.9 through 3.12. The older `pycrypto` 2.6.1 package does not build on Python 3.12 and is not used. The headset is still read as a Universal Serial Bus human-interface device through the `hidapi` package. Check the import before a session with `emokit_status()`, which reports an empty `import_error` when decryption can load.
- emokit stamps packets with `datetime.now()`. This project ignores that and stores `time.time()` when the packet is dequeued.
- Channel numbers are the values emokit produces from the 14-bit packet. The decoder multiplies each 14-bit level by 0.5151515151. The library's own notes call one step about 0.51 microvolts on the original EPOC. Treat that scale as unchecked until you confirm it on the headset in front of you.

The fourteen channels saved are AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

## Live view

The window is [pyqtgraph](https://www.pyqtgraph.org/). It updates the lines in place, which keeps up with an EEG stream on the same machine without a browser in the middle. Matplotlib redraws too slowly for that, and a web page would be a second process this rig does not need.

The window is split left and right.

- Left: a polar breath-phase display in the style of `Sniff_polar_display.py`. The angle is the phase of the recent nasal-pressure signal. The distance from the centre is the size of the latest sample. Red is an inhale (positive pressure), blue an exhale.
- Right, top: pupil diameter for eye 0 and eye 1, in millimetres when the 3d detector is on. Gaze position is not plotted.
- Right, bottom: the EEG channels. Each channel has a checkbox. A switch chooses raw sensor values or power in the 8–12 Hz band. The axis is Unix time in seconds.

Triggers are marked on the polar plot and as vertical lines on the pupil and electroencephalography panels. The directed coupling panel sits under those plots. It is another `add_panel()` call on the same window, and it does not redraw the polar, pupil, or electroencephalography traces.

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

There are two filter paths.

The live path is only for the display. It uses symmetric finite impulse response filters with an odd number of taps, so every frequency in a filter is delayed by the same whole number of samples, (taps − 1) / 2. That delay is subtracted from the timestamps. After the subtraction the frequency response is real: the phase is 0 where the filter is positive and pi where the sign flips. The stated check is that the imaginary part stays below one hundred-millionth of (1 + the absolute real part), and that the group delay stays within 0.05 samples of the exact delay wherever the gain is above −60 decibels. Measured residuals are far inside that: the phase error is about a hundred-trillionth of a radian, and the group-delay error is about a trillionth of a sample.

The live default is the shortest set that still keeps the envelope usable, which lands at about half a second of lag. A shorter cascade, about 0.48 seconds, let the envelope of a 1 hertz tone ripple by 54 percent of its height, so it was rejected. The numbers below are for a 128 hertz electroencephalography stream unless another rate is named.

| Filter | Taps | Transition width | Design attenuation | Delay |
| --- | --- | --- | --- | --- |
| Breathing low-pass at 20 hertz | 15 | 2.5 hertz | 30 decibels | 0.350 seconds |
| Pupil low-pass at 120 hertz | 75 | 2.5 hertz | 30 decibels | 0.308 seconds |
| Delta band-pass | 71 | 2.2 hertz | 25 decibels | 0.273 seconds |
| Alpha band-pass | 71 | 2.2 hertz | 25 decibels | 0.273 seconds |
| Envelope low-pass | 71 | 2.2 hertz | 25 decibels | 0.273 seconds |

The envelope is the absolute value of the band-pass, then the envelope low-pass. Rectification has no memory, so the envelope delay is the sum, 0.547 seconds for both delta and alpha. That is also the lag of the aligned grid, because the breathing and pupil delays are shorter. The grid therefore ends 0.547 seconds behind the newest electroencephalography packet. `window.delays_seconds` has the delays actually used, which move by a few milliseconds when the sample rate is not 128 hertz. A 32 hertz simulated stream gives an envelope delay of 0.5625 seconds.

What this half-second set gives up, compared with the clean linear-phase set below: the transition is 2.2 hertz wide instead of 0.70 hertz, and the Kaiser design asks for 25 decibels of stopband attenuation instead of 40. A tone at 1 hertz or at 4 hertz, the edges of the delta band, is about 4 decibels down instead of within 0.1 decibel. A 0.2 hertz drift is only about 6.5 decibels down, instead of about 42. A pure 6 hertz theta tone still comes through about 39 decibels down, and its envelope is about 1 percent of an equal-sized 2.5 hertz tone. The envelope of a steady 2 hertz tone ripples by about 3.5 percent of its height, and a 2.5 hertz tone by about 0.4 percent. Slow amplitude changes at 0.05, 0.10 and 0.20 hertz are delayed by the sum of the two filters with no extra lag, to the nearest sample at 128 hertz.

The clean linear-phase set is the shortest odd-tap filter that keeps the stated passband within 1 decibel and the named stop frequencies at least 40 decibels down (0.2 hertz and 6 hertz for delta, 6 hertz and 16 hertz for alpha, 5 hertz for the anti-alias low-pass, and 4 hertz for the envelope smoother). It cannot get near half a second. The delta band-pass alone is 1.602 seconds.

| Clean filter, 128 hertz | Taps | Transition width | Design attenuation | Delay |
| --- | --- | --- | --- | --- |
| Breathing and pupil low-pass | 145 | 2.0 hertz | 40 decibels | 0.562 seconds |
| Delta band-pass | 411 | 0.70 hertz | 40 decibels | 1.602 seconds |
| Alpha band-pass | 287 | 1.0 hertz | 40 decibels | 1.117 seconds |
| Envelope low-pass | 287 | 1.0 hertz | 40 decibels | 1.117 seconds |

The clean delta path is 1.602 + 1.117 = 2.719 seconds, and that is the lag of an aligned grid built from the clean set. The live display does not use this set. It is here so the cost of the half-second choice is visible.

Placing the filtered samples on the 10 hertz grid, including uneven timestamps and the 0.20 second pupil gap bridge, did not move a 0.25 hertz wave. The measured lag was 0 samples. The tolerance used for that check is 0.05 seconds.

The pupil path is a short list of steps, `hub.history.pupil_stages`. Each step has a name and a `delay_s`. The first step fills a hole of up to 0.20 seconds with a straight line and adds no delay. A longer hole is left missing. The second step is the linear-phase low-pass. Blink correction is not one of these steps. To add it later, insert a step between those two, leave its values lined up with the times it was given, and set `delay_s` to the lag it adds.

Electroencephalography uses AF3, AF4, F3 and F4. The original EPOC headset has no electrode at Fz. These four are the frontal electrodes, two on each side. `hub.history.eeg_channels = ("F3", "F4")` selects others. Each channel is filtered, the envelopes are averaged, and a sine of amplitude A comes back near A. The alpha value in the ratio is not allowed to fall below one thousandth of the median alpha in the window.

One or two missed samples are filled with a straight line. A longer hole stays missing, and the filter starts again after it, so the first taps of that new stretch stay missing too.

## Live coupling

The hub starts a separate process that scores the aligned window about once a second. Neighbour searches run in that process, so they do not hold the global interpreter lock against the readers or the plot. If a score is not finished when the next second arrives, that window is skipped rather than queued. The latest result is `hub.latest_coupling()`. Every live result is labelled approximate.

The scores are conditional Andrzejak interdependence and the forward forecast gain, for all six directions among breathing, pupil diameter, and the electroencephalography trace. Each direction is conditioned on the third signal. The electroencephalography trace defaults to the delta-to-alpha ratio. `hub.coupling.eeg_signal` may be set to `"delta"` or `"alpha"`. The forecast horizon is 0.4 seconds (4 samples). The search uses 8 neighbours from a pool of 40. Surrogate tests are off. `hub.coupling.surrogates` is kept so they can be switched on later. This checkpoint records the flag and does not run surrogates.

The embedding delay and dimension are chosen once, from the first window that has at least 20 seconds with at least 80 percent of its samples present. The delay is the first minimum of the delayed mutual information, and the dimension is the false-nearest-neighbour choice, the same two methods as the offline analysis. They are then held fixed and reported on the result. `hub.coupling.set_embedding(...)` sets them by hand instead. Until that baseline exists, the status is "warming up". A later window with too many missing samples is "skipped", with the reason written out.

In a simulated session the baseline chose, for a 60 second window, a breathing delay of 7 samples and dimension 2, a pupil delay of 6 samples and dimension 2, and an electroencephalography delay of 5 samples and dimension 3. Those numbers move a little from one run to the next.

One update took 0.036 seconds for a 30 second window, 0.074 seconds for 60 seconds, and 0.189 seconds for 120 seconds, measured inside the worker while simulate mode ran the hub. Over the same runs the main process used about 5 to 6 percent of one core, compared with about 2 percent with the worker switched off. The worker process itself used about 4 to 7 percent of one core.

After the session, the reference pass reads `breathing.csv`, `eeg.csv`, and `pupil_diameter.csv` (or a Pupil Capture `pupil_positions.csv`), filters the whole recording forward and backward, and writes `coupling_reference.csv` at the same window times:

```bash
python -m sniff_sync.reference recordings/P01
```

On synthetic signals where breathing drives the delta amplitude, the live scores and this reference called breathing → electroencephalography on every window. The correlation of that score was 0.99 and the typical difference was 0.005. The clean linear-phase set, which is not what the display uses, differed from the reference by about 0.0004 on the same arrow. A simulated session whose three oscillators are not built to drive one another did not keep a stable direction: the weak scores from the live filters and from the reference did not match. That is disagreement on noise, not a reversal of a real drive.

Recency weighting is off unless the live panel turns it on. Neighbours are still searched across the whole window. When the per-reference-point values are averaged into the window score, each reference point gets an exponentially decaying weight. The newest point has weight 1, and a point one half-life older has weight one half. The half-life defaults to 10 seconds. The approximate file records whether weighting was on and what half-life was used. The reference pass reads those two columns and uses the same weighting, so a live window and its reference window stay comparable.

## Directed coupling panel

The live window has a panel titled Directed coupling. The heading on the panel is "Directed coupling (live, approximate)", because these scores use the short linear-phase filters. The panel reads `hub.latest_coupling()` on the same timer as the other plots. It does not run the neighbour search itself.

Two grids sit side by side. The left grid is conditional nonlinear interdependence. The right grid is forward forecast gain. Columns are the source and rows are the target. The diagonal is blank. Each of the six off-diagonal cells shows the score, and the larger direction of a pair is marked "stronger". Warm colours are positive scores and blue is a negative score. A weak or near-zero score is drawn faded. Hovering a cell shows the exact unsmoothed numbers for both measures, the window length, and the status.

To the right, three nodes (Breathing, Pupil diameter, and Electroencephalography) are joined by curved arrows. Arrow colour follows the sign of the forward forecast gain, warm for a positive score and blue for a negative score. Arrow thickness follows how large that score is and which direction of the pair is larger, so a weak or losing direction is faded and the dominant direction stays easy to see. Hovering an arrow shows the forward forecast gain and the conditional nonlinear interdependence for both directions of that pair.

The numbers drawn in the cells and the arrows are smoothed with an exponential moving average whose half-life is 3 seconds. A new ready result moves the picture part of the way toward that result, so the panel does not jump every second. `coupling_approximate.csv` keeps the unsmoothed worker scores. The hover text shows those unsmoothed numbers.

Until the first baseline window is ready, the panel says "Warming up" and the cells stay blank. When a window has too many missing samples, the panel says "Skipped" and includes the reason.

The control row has three settings, and each one takes effect on the next score:

- Electroencephalography signal: delta-to-alpha ratio (the default), delta (1 to 4 hertz), or alpha (8 to 12 hertz).
- Window length, from 30 to 120 seconds, default 60. This is the same length the aligned window uses.
- Recency weighting, off by default, with an exponential half-life that defaults to 10 seconds. The half-life control is available only while weighting is on.

Significance tests are not shown. The fading used for a weak score is also where a later test can fade a link it calls non-significant, without changing the layout.

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

Performance of pairwise M, conditional M and partial cross mapping on simulated data from the 2018 coupled-oscillator model is in [docs/nlid-performance-report.md](docs/nlid-performance-report.md).

## Compare the live scores with the reference

After a session that wrote `coupling_approximate.csv`:

```bash
python -m sniff_sync.compare recordings/P01
```

If `coupling_reference.csv` is not there yet, this command writes it first (the same pass as `python -m sniff_sync.reference`). It then lines the two files up by window end time. For each directed link and each measure it prints the correlation across windows and the median absolute difference. For each pair it prints the fraction of windows where the larger score points the same way. The table is `coupling_comparison.csv` and the figure is `coupling_comparison.png`, both in the session folder.

## Real-rig checklist

Do these in order on the computer that has the sensors. The headset must be an original consumer EPOC (Emotiv's original electroencephalography headset). An EPOC+ uses a different key and will not decrypt with this schedule.

1. Install this package with `pycryptodome`, and install the human-interface-device library. From the repository directory:

   ```bash
   pip install -e '.[analysis]'
   pip install hidapi
   ```

   `pycryptodome` is the decryption library (`Crypto.Cipher.AES`). `hidapi` is how the headset is opened over Universal Serial Bus. Copy `third_party/emokit/linux/epoc.rules` into the udev rules so the device is readable without root, then replug the headset.

2. Check that emokit imports on this Python before you depend on a recording:

   ```bash
   python -c 'from sniff_sync.eeg import emokit_status; print(emokit_status())'
   ```

   `import_error` should be an empty string. A message there means decryption or the reader did not load, and the hub will still write trigger rows to `eeg.csv` but no sensor values.

3. Start the other two streams. In Pupil Capture, turn on the Network API on port 50020, turn on the 3D detector (so diameter is in millimetres rather than pixels), and enable the Sniff Trigger Capture plugin. Start the Pupil recording before the first trigger. Start `getSniffs.py` so it is writing nasal pressure. Its serial-port path is hard-coded; if the device name changed, run `ls /dev/tty*` and update that path. The hub does not open the serial port. A `getSniffs.py` file uses `Unix Time` in seconds and `Nasal Pressure`. A `Sniff_polar_display.py` file is in milliseconds and is divided by 1000 on read.

4. Start the hub and point it at that pressure file:

   ```bash
   python -m sniff_sync --session recordings/P01 --sniff-csv subject_data/P01/KH_sniff.csv
   ```

   After it connects, a fresh Pupil `t` query should sit within a few milliseconds of `time.time()` plus the offset the hub printed (Pupil time minus Unix time). The link is local, so a gap of tens of milliseconds means the offset was measured badly. Restart the hub before the recording and let it measure the offset again.

5. Send one trigger from another program on the same computer:

   ```bash
   python -c 'from sniff_sync import trigger; trigger("label")'
   ```

6. Stop the hub, then confirm the three recordings share that trigger:

   ```bash
   python -m sniff_sync.check recordings/P01
   ```

   The Unix time in `breathing.csv`, `eeg.csv`, and `sniff_sync_triggers.csv` should be the same number, and the annotation should sit on the right moment in Pupil Player.

7. Recompute the windows with forward-backward filters and compare them with the live scores:

   ```bash
   python -m sniff_sync.reference recordings/P01
   python -m sniff_sync.compare recordings/P01
   ```

   The comparison writes `coupling_comparison.csv` and `coupling_comparison.png` in the session folder. Read the correlation, the median absolute difference, and the fraction of windows whose dominant direction matches.

8. Confirm the 0.51 microvolt scale on this headset before you report electroencephalography in microvolts. The saved channel values are already the 14-bit levels multiplied by 0.5151515151, which is the conversion in the vendored decoder. The library's notes call one step about 0.51 microvolts. That number has not been checked on this headset. Compare a known input, or Emotiv's own viewer, against the values in `eeg.csv`, and only then treat the file as microvolts.
