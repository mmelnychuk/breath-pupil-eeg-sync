"""Live view of the three streams on one Unix-time axis.

pyqtgraph is used because the traces are local and continuous. An
electroencephalography (EEG) headset sends on the order of a hundred
samples a second across fourteen channels, and pyqtgraph replaces the
points in a plot instead of redrawing a page. A browser would add
another process on a machine that already has the sensors attached.

The window shows pupil diameter (both eyes, millimetres), nasal
pressure, and the EEG channels. Triggers are vertical lines at the
same Unix time on all three plots.
"""

from __future__ import annotations

from datetime import datetime

from sniff_sync.eeg_channels import EPOC_CHANNELS

_EEG_STEP = 80.0


def open_monitor(feed):
    """Open the window and block until it is closed."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Monitor(feed)
    window.show()
    app.exec()
    return window


class Monitor:
    """Qt window. Constructing it needs a display, or the offscreen platform."""

    def __init__(self, feed):
        import pyqtgraph as pg
        from PySide6 import QtCore
        from PySide6 import QtWidgets

        self.feed = feed
        pg.setConfigOptions(background="#16180f", foreground="#e6e2d4", antialias=False)
        self._pg = pg
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Sniff sync")
        self.window.resize(1100, 820)
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self.status = QtWidgets.QLabel("Unix time on this computer. Waiting for samples.")
        self.status.setStyleSheet("color: #e6e2d4; background: #16180f; padding: 8px;")
        layout.addWidget(self.status)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)

        axis = UnixAxis(orientation="bottom")
        self.pupil_plot = self.graphics.addPlot(title="Pupil diameter", axisItems={"bottom": axis})
        self.pupil_plot.setLabel("left", "millimetres")
        self.pupil_plot.addLegend()
        self.eye0 = self.pupil_plot.plot(pen=pg.mkPen("#e0a045", width=2), name="Eye 0")
        self.eye1 = self.pupil_plot.plot(pen=pg.mkPen("#3db8c4", width=2), name="Eye 1")
        self.graphics.nextRow()
        self.breath_plot = self.graphics.addPlot(title="Nasal pressure")
        self.breath_plot.setLabel("left", "pascal")
        self.breath_plot.setXLink(self.pupil_plot)
        self.breath_curve = self.breath_plot.plot(pen=pg.mkPen("#8fbf6a", width=2))
        self.graphics.nextRow()
        self.eeg_plot = self.graphics.addPlot(title="EEG (emokit sensor values, stacked for display)")
        self.eeg_plot.setLabel("left", "stacked sensor value")
        self.eeg_plot.setLabel("bottom", "Unix time")
        self.eeg_plot.setXLink(self.pupil_plot)
        self.eeg_curves = {}
        colors = ["#d08ab8", "#8fb8d0", "#d0b48a", "#a8d08a", "#d08a8a", "#b0a0d0", "#80c0b0"]
        for index, name in enumerate(EPOC_CHANNELS):
            self.eeg_curves[name] = self.eeg_plot.plot(pen=pg.mkPen(colors[index % len(colors)], width=1), name=name)
        self._lines = []
        self._seen = set()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(80)
        self.refresh()

    def refresh(self) -> None:
        snap = self.feed.snapshot()
        self._set_xy(self.eye0, [(t, v) for t, eye, v in snap["pupil"] if eye == 0])
        self._set_xy(self.eye1, [(t, v) for t, eye, v in snap["pupil"] if eye == 1])
        self._set_xy(self.breath_curve, snap["breath"])
        for index, name in enumerate(EPOC_CHANNELS):
            rows = []
            for unix, channels in snap["eeg"]:
                if name in channels and channels[name] != "":
                    rows.append((unix, float(channels[name]) + index * _EEG_STEP))
            self._set_xy(self.eeg_curves[name], rows)
        for unix, label, trigger_id in snap["triggers"]:
            if trigger_id in self._seen:
                continue
            self._seen.add(trigger_id)
            pen = self._pg.mkPen("#f4f0e4", width=1)
            for plot in (self.pupil_plot, self.breath_plot, self.eeg_plot):
                line = self._pg.InfiniteLine(pos=unix, angle=90, pen=pen, label=label if plot is self.pupil_plot else None)
                plot.addItem(line)
                self._lines.append(line)
        count = len(snap["triggers"])
        self.status.setText(
            "Unix time on this computer. %d trigger%s. Pupil diameter, nasal pressure, and EEG."
            % (count, "" if count == 1 else "s")
        )

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
