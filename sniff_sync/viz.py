"""Live view, layout C.

Left: a polar breath-phase display in the style of Sniff_polar_display.py
(phase angle around the circle, nasal-pressure magnitude as the radius).
Right: pupil diameter for both eyes, and under it the EEG panel. The EEG
panel can hide individual channels and can show raw sensor values or
8–12 Hz band power.

Times on the pupil and EEG panels are Unix seconds. Trigger markers are
drawn on the polar plot and as vertical lines on the time panels.

The directed coupling panel is one add_panel() call along the bottom.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from sniff_sync.eeg_channels import EPOC_CHANNELS

_EEG_STEP = 80.0


def open_monitor(feed, coupling=None):
    """Open the window and block until it is closed."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Monitor(feed, coupling=coupling)
    window.show()
    app.exec()
    return window


def breath_state(pressures):
    """Latest phase (radians), radius, and colour from nasal-pressure samples.

    The phase is the angle of a Hilbert transform of the recent pressure,
    which is what Sniff_polar_display.py plots. Radius is the size of the
    latest breath relative to that window. Positive pressure is drawn red
    (inhale in that script), negative pressure blue (exhale).
    """
    values = np.asarray(pressures, dtype=float)
    if values.size < 8:
        return None
    window = values[-50:]
    centered = window - np.mean(window)
    phase = float(_hilbert_phase(centered)[-1])
    spread = float(np.std(centered))
    radius = abs(float(centered[-1])) / spread if spread > 1e-9 else 0.0
    latest = float(window[-1])
    if latest > 0.05:
        colour = "#d06048"
    elif latest < -0.05:
        colour = "#3d7ec4"
    else:
        colour = "#9a9686"
    return phase, radius, colour


def _hilbert_phase(samples):
    count = len(samples)
    spectrum = np.fft.fft(samples)
    mask = np.zeros(count)
    if count % 2 == 0:
        mask[0] = mask[count // 2] = 1
        mask[1:count // 2] = 2
    else:
        mask[0] = 1
        mask[1:(count + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * mask)
    return np.angle(analytic)


def band_power_trace(times, values, low_hz=8.0, high_hz=12.0):
    """Trailing one-second power in the 8–12 Hz band, at each sample time."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 16:
        return [], []
    step = max(1, int(times.size / 40))
    xs = []
    ys = []
    for end in range(16, times.size + 1, step):
        start_time = times[end - 1] - 1.0
        begin = int(np.searchsorted(times[:end], start_time))
        chunk_t = times[begin:end]
        chunk_v = values[begin:end]
        if chunk_t.size < 16:
            continue
        dt = float(np.median(np.diff(chunk_t)))
        if dt <= 0:
            continue
        centered = chunk_v - np.mean(chunk_v)
        spectrum = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(centered.size, d=dt)
        power = (np.abs(spectrum) ** 2) / centered.size
        band = (freqs >= low_hz) & (freqs <= high_hz)
        xs.append(float(chunk_t[-1]))
        ys.append(float(np.sum(power[band])))
    return xs, ys


class Monitor:
    """Qt window. Constructing it needs a display, or the offscreen platform."""

    def __init__(self, feed, coupling=None):
        import pyqtgraph as pg
        from PySide6 import QtCore
        from PySide6 import QtWidgets

        self.feed = feed
        self.coupling = coupling
        pg.setConfigOptions(background="#16180f", foreground="#e6e2d4", antialias=False)
        self._pg = pg
        self._QtWidgets = QtWidgets
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Sniff sync")
        self.window.resize(1280, 980)
        central = QtWidgets.QWidget()
        from PySide6 import QtGui

        central.setAutoFillBackground(True)
        palette = central.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#16180f"))
        central.setPalette(palette)
        self.window.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        self.status = QtWidgets.QLabel("Unix time on this computer, in seconds. Waiting for samples.")
        self.status.setStyleSheet("color: #e6e2d4; background: #16180f; padding: 8px;")
        outer.addWidget(self.status)

        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 3)
        self._left_layout = QtWidgets.QVBoxLayout()
        self._right_layout = QtWidgets.QVBoxLayout()
        body.addLayout(self._left_layout, 1)
        body.addLayout(self._right_layout, 1)
        self._bottom_layout = QtWidgets.QHBoxLayout()
        outer.addLayout(self._bottom_layout, 2)
        self._panels = {}

        self.polar_plot = pg.PlotWidget(title="Respiratory phase and nasal pressure")
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.setXRange(-2.2, 2.2)
        self.polar_plot.setYRange(-2.2, 2.2)
        self.polar_plot.hideAxis("bottom")
        self.polar_plot.hideAxis("left")
        self.breath_ray = self.polar_plot.plot(pen=pg.mkPen("#e6e2d4", width=2))
        self.breath_point = self.polar_plot.plot(pen=None, symbol="o", symbolSize=14, symbolBrush="#d06048")
        self.trigger_dots = self.polar_plot.plot(pen=None, symbol="t", symbolSize=12, symbolBrush="#f4f0e4")
        for angle, text in ((0, "Inhale"), (np.pi / 2, "Trough"), (np.pi, "Exhale"), (3 * np.pi / 2, "Peak")):
            label = pg.TextItem(text, color="#9a9686", anchor=(0.5, 0.5))
            label.setPos(1.8 * np.cos(angle), 1.8 * np.sin(angle))
            self.polar_plot.addItem(label)
        self.add_panel("breath", self.polar_plot, column="left")

        self.pupil_plot = pg.PlotWidget(title="Pupil diameter", axisItems={"bottom": UnixAxis(orientation="bottom")})
        self.pupil_plot.setLabel("left", "millimetres")
        self.pupil_plot.addLegend()
        self.eye0 = self.pupil_plot.plot(pen=pg.mkPen("#e0a045", width=2), name="Eye 0")
        self.eye1 = self.pupil_plot.plot(pen=pg.mkPen("#3db8c4", width=2), name="Eye 1")
        self.add_panel("pupil", self.pupil_plot, column="right")

        eeg_wrap = QtWidgets.QWidget()
        eeg_layout = QtWidgets.QVBoxLayout(eeg_wrap)
        eeg_layout.setContentsMargins(0, 0, 0, 0)
        self.eeg_plot = pg.PlotWidget(title="EEG", axisItems={"bottom": UnixAxis(orientation="bottom")})
        self.eeg_plot.setLabel("left", "sensor value")
        self.eeg_plot.setLabel("bottom", "Unix time (seconds)")
        self.eeg_plot.setXLink(self.pupil_plot)
        eeg_layout.addWidget(self.eeg_plot)
        toggles = QtWidgets.QHBoxLayout()
        self.channel_boxes = {}
        colors = ["#d08ab8", "#8fb8d0", "#d0b48a", "#a8d08a", "#d08a8a", "#b0a0d0", "#80c0b0"]
        self.eeg_curves = {}
        for index, name in enumerate(EPOC_CHANNELS):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(True)
            box.setStyleSheet("color: #e6e2d4;")
            box.toggled.connect(self.refresh)
            self.channel_boxes[name] = box
            toggles.addWidget(box)
            self.eeg_curves[name] = self.eeg_plot.plot(pen=pg.mkPen(colors[index % len(colors)], width=1), name=name)
        eeg_layout.addLayout(toggles)
        mode = QtWidgets.QHBoxLayout()
        self.raw_button = QtWidgets.QRadioButton("Raw")
        self.band_button = QtWidgets.QRadioButton("8–12 Hz power")
        self.raw_button.setChecked(True)
        for button in (self.raw_button, self.band_button):
            button.setStyleSheet("color: #e6e2d4;")
            button.toggled.connect(self.refresh)
            mode.addWidget(button)
        mode.addStretch(1)
        eeg_layout.addLayout(mode)
        self.add_panel("eeg", eeg_wrap, column="right")
        from sniff_sync.coupling_panel import make_coupling_panel

        self.coupling_panel = make_coupling_panel(coupling)
        self.add_panel("coupling", self.coupling_panel, column="bottom")

        self._lines = []
        self._seen = set()
        self._trigger_phase = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(80)
        self.refresh()

    def add_panel(self, name, widget, column="right"):
        """Place a panel. The directed-coupling view uses this same call."""
        self._panels[name] = widget
        if column == "left":
            target = self._left_layout
        elif column == "bottom":
            target = self._bottom_layout
        else:
            target = self._right_layout
        target.addWidget(widget)

    def refresh(self) -> None:
        snap = self.feed.snapshot()
        self._set_xy(self.eye0, [(t, v) for t, eye, v in snap["pupil"] if eye == 0])
        self._set_xy(self.eye1, [(t, v) for t, eye, v in snap["pupil"] if eye == 1])
        self._draw_breath(snap["breath"])
        raw = self.raw_button.isChecked()
        self.eeg_plot.setLabel("left", "sensor value" if raw else "8–12 Hz power")
        for index, name in enumerate(EPOC_CHANNELS):
            if not self.channel_boxes[name].isChecked():
                self.eeg_curves[name].setData([], [])
                continue
            times = []
            values = []
            for unix, channels in snap["eeg"]:
                if name in channels and channels[name] != "":
                    times.append(unix)
                    values.append(float(channels[name]))
            if raw:
                self.eeg_curves[name].setData(times, [value + index * _EEG_STEP for value in values])
            else:
                xs, ys = band_power_trace(times, values)
                self.eeg_curves[name].setData(xs, [value + index * _EEG_STEP for value in ys])
        for unix, label, trigger_id in snap["triggers"]:
            if trigger_id in self._seen:
                continue
            self._seen.add(trigger_id)
            state = breath_state([pressure for _t, pressure in snap["breath"]])
            if state is not None:
                phase, radius, _colour = state
                self._trigger_phase.append((radius * np.cos(phase), radius * np.sin(phase)))
            pen = self._pg.mkPen("#f4f0e4", width=1)
            for plot in (self.pupil_plot, self.eeg_plot):
                line = self._pg.InfiniteLine(pos=unix, angle=90, pen=pen, label=label if plot is self.pupil_plot else None)
                plot.addItem(line)
                self._lines.append(line)
        if self._trigger_phase:
            self.trigger_dots.setData(
                [point[0] for point in self._trigger_phase],
                [point[1] for point in self._trigger_phase],
            )
        count = len(snap["triggers"])
        self.status.setText(
            "Unix time in seconds on this computer. %d trigger%s."
            % (count, "" if count == 1 else "s")
        )
        self.coupling_panel.refresh_display()

    def _draw_breath(self, samples) -> None:
        state = breath_state([pressure for _unix, pressure in samples])
        if state is None:
            self.breath_ray.setData([], [])
            self.breath_point.setData([], [])
            return
        phase, radius, colour = state
        x = radius * np.cos(phase)
        y = radius * np.sin(phase)
        self.breath_ray.setData([0, x], [0, y])
        self.breath_point.setData([x], [y])
        self.breath_point.setSymbolBrush(colour)

    def _set_xy(self, curve, rows) -> None:
        if not rows:
            curve.setData([], [])
            return
        curve.setData([row[0] for row in rows], [row[1] for row in rows])

    def show(self) -> None:
        self.window.show()

    def close(self) -> None:
        self.timer.stop()
        self.window.close()


class UnixAxis:
    """Tick labels as clock time. The positions themselves stay Unix seconds."""

    def __new__(cls, orientation="bottom"):
        import pyqtgraph as pg

        class _Axis(pg.AxisItem):
            def tickStrings(self, values, scale, spacing):
                labels = []
                for value in values:
                    try:
                        labels.append(datetime.fromtimestamp(value).strftime("%H:%M:%S"))
                    except (OverflowError, OSError, ValueError):
                        labels.append("")
                return labels

        return _Axis(orientation=orientation)
