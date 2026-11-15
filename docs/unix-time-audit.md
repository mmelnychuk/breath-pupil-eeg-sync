# Checkpoint 1 — clocks in the sniff experiment scripts

Audit of [mmelnychuk/Emma_Kate_BreathExperiment](https://github.com/mmelnychuk/Emma_Kate_BreathExperiment) at commit `1618a5136bb64129f0fc8ba036734458ef6bb28c`. Read only. Nothing in that repository was modified.

Unix time here means `time.time()`: seconds since 1970-01-01 00:00:00 UTC (Coordinated Universal Time).

## Result

Every live timestamp in those scripts comes from `time.time()`. None of them use `time.perf_counter`, `time.monotonic`, `datetime` (the one import is commented out), or a PsychoPy `core.Clock`. No script stores a relative or elapsed-time clock as the event time.

One file writes that same Unix clock in milliseconds instead of seconds. That file is `Sniff_polar_display.py`.

## Scripts that save a timestamp

| File | What is stamped | Call | Unit written |
| --- | --- | --- | --- |
| `getSniffs.py` | Each nasal-pressure sample (20 Hz) | `time.time()` | seconds, column `Unix Time` |
| `Sniff_polar_display.py` | Each sample (pressure, diaphragm, phase) | `time.time() * 1000` | milliseconds, column still named `Unix Time` |
| `KORNHUBER_Exp.py` | Spacebar press | `time.time()` | seconds, column `UnixTime` |
| `mw_meditation_selfAlert.py` | Mind-wandering key press | `time.time()` | seconds, column `UnixTime` |
| `Alt_Use_Task.py` | Word onset, pause, entered response, next word, task end | `time.time()` | seconds, written as text into `AUT.txt` |

`Alt_Use_Task.py` imports `datetime` and contains a commented-out `datetime.now().strftime(...)` used only as an old filename idea. The timestamps that are actually written were changed, in the author's comments, from whole seconds to full `time.time()` precision.

`getSniffs.py` is the breathing stream. It does not detect or mark sniff onsets. It records pressure against Unix time.

## Scripts that read Unix time and do not save it

| File | Call | What happens |
| --- | --- | --- |
| `Kornhuber_Practice.py` | `time.time()` on key press | The call that would write the CSV (comma-separated values) file is commented out |
| `Kornhuber_Practice_newComp.py` | `time.time()` at start and on key press | Same: the write is commented out |
| `mw_phase_probe.py` | `time.time()` when the reply is submitted | Shown on the window label only |

## Scripts with no clock

`checkSniff.py`, `plot_sniff.py`, `Experimenter_Input.py`, `README.md`, `RunBreathExperiment.sh`, `LICENSE`, and `sniffCheckStatus.txt` do not read or write a clock. `plot_sniff.py` plots the `Nasal Pressure` column of an existing CSV and ignores time.

## Not a second clock

`Sniff_polar_display.py` also counts samples (`x += 1`) for the on-screen x-axis. That counter is not written to the CSV. The saved time column is the Unix value above, in milliseconds.

Several task scripts comment that Unix time may not match another device's clock (`UnixTime not sync'd between devices`). The value they store is still this machine's `time.time()`.
