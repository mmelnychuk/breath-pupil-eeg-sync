"""The Directed coupling panel for the live view.

The numbers drawn here are smoothed so the picture does not jump. The file
written by the worker keeps the unsmoothed scores. Significance is not shown.
``display_opacity`` already accepts a future significance flag so a
non-significant link can be faded without changing the layout.
"""

from __future__ import annotations

import math

DISPLAY_HALF_LIFE_SECONDS = 3.0

NODE_ORDER = ("breathing", "pupil diameter", "electroencephalography")
DISPLAY_NAME = {
    "breathing": "Breathing",
    "pupil diameter": "Pupil diameter",
    "electroencephalography": "Electroencephalography",
}


def display_opacity(score, significant=None):
    """How solid a link should look from its own score.

    ``significant`` is left unset until significance tests are switched on.
    When it is false, the link is faded even if the score is large.
    """
    if significant is False:
        return 0.22
    try:
        number = float(score)
    except (TypeError, ValueError):
        return 0.18
    if not math.isfinite(number):
        return 0.18
    strength = math.tanh(abs(number))
    return 0.18 + 0.82 * strength


def link_opacity(score, opposite=None, significant=None):
    """Fade a weak score, and fade the weaker direction of a pair.

    The two directions are compared by their scores. The larger one stays
    solid. The smaller one is drawn lighter, so the dominant direction can
    be read even when both numbers are far from zero. A score that was not
    tested for significance is not faded for that reason.
    """
    base = display_opacity(score, significant)
    if significant is False or opposite is None:
        return base
    try:
        left = float(score)
        right = float(opposite)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(left) or not math.isfinite(right):
        return base
    dominance = 0.5 + 0.5 * math.tanh(left - right)
    # The weaker direction keeps about a quarter of its strength.
    return base * (0.25 + 0.75 * dominance)


def _link_width(score, opposite):
    try:
        left = float(score)
    except (TypeError, ValueError):
        return 1.2
    if not math.isfinite(left):
        return 1.2
    try:
        right = float(opposite)
    except (TypeError, ValueError):
        right = 0.0
    if not math.isfinite(right):
        right = 0.0
    dominance = 0.5 + 0.5 * math.tanh(left - right)
    return 1.2 + 5.0 * dominance * math.tanh(abs(left))


def _score_color(score, opposite=None, significant=None):
    opacity = link_opacity(score, opposite, significant)
    try:
        number = float(score)
    except (TypeError, ValueError):
        number = float("nan")
    if not math.isfinite(number) or number < 0:
        red, green, blue = (90, 130, 170)
    else:
        red, green, blue = (212, 164, 84)
    return red, green, blue, opacity


class ScoreSmoother:
    """Exponential moving average of the scores that are drawn, not of the file."""

    def __init__(self, half_life_seconds=DISPLAY_HALF_LIFE_SECONDS):
        self.half_life_seconds = float(half_life_seconds)
        self._values = {"interdependence": {}, "forecast_gain": {}}
        self._seen = None

    def apply(self, result):
        identity = (
            result.get("status"),
            result.get("window_end_unix"),
            result.get("logged_unix"),
        )
        if identity == self._seen:
            return self._values
        self._seen = identity
        if result.get("status") != "ready":
            return self._values
        alpha = 1.0 - math.exp(-math.log(2.0) * 1.0 / self.half_life_seconds)
        updated = {"interdependence": {}, "forecast_gain": {}}
        for kind in ("interdependence", "forecast_gain"):
            for arrow, value in (result.get(kind) or {}).items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = float("nan")
                previous = self._values.get(kind, {}).get(arrow)
                if not math.isfinite(number):
                    updated[kind][arrow] = previous if previous is not None else float("nan")
                elif previous is None or not math.isfinite(previous):
                    updated[kind][arrow] = number
                else:
                    updated[kind][arrow] = (1.0 - alpha) * previous + alpha * number
        self._values = updated
        return self._values


def _format_exact(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    if not math.isfinite(number):
        return "not available"
    return "%.6f" % number


def _format_cell(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return "%.2f" % number


def cell_tooltip(result, source, target):
    """Exact unsmoothed values for one directed cell."""
    arrow = "%s → %s" % (source, target)
    interdependence = (result.get("interdependence") or {}).get(arrow)
    forecast = (result.get("forecast_gain") or {}).get(arrow)
    window = result.get("window_seconds")
    try:
        window_text = "%.0f seconds" % float(window)
    except (TypeError, ValueError):
        window_text = "not available"
    return (
        "Source: %s\nTarget: %s\n"
        "Conditional nonlinear interdependence: %s\n"
        "Forward forecast gain: %s\n"
        "Window length: %s\n"
        "Status: %s"
        % (
            DISPLAY_NAME[source],
            DISPLAY_NAME[target],
            _format_exact(interdependence),
            _format_exact(forecast),
            window_text,
            result.get("status") or "not available",
        )
    )


def pair_tooltip(result, source, target):
    """Both directions, and both measures, for one pair."""
    lines = ["%s and %s" % (DISPLAY_NAME[source], DISPLAY_NAME[target])]
    for left, right in ((source, target), (target, source)):
        arrow = "%s → %s" % (left, right)
        lines.append(
            "%s → %s: forward forecast gain %s, conditional nonlinear interdependence %s"
            % (
                DISPLAY_NAME[left],
                DISPLAY_NAME[right],
                _format_exact((result.get("forecast_gain") or {}).get(arrow)),
                _format_exact((result.get("interdependence") or {}).get(arrow)),
            )
        )
    return "\n".join(lines)


def make_coupling_panel(coupling):
    """Build the Directed coupling panel. ``coupling`` may be None in tests that only draw."""
    from PySide6 import QtCore, QtGui, QtWidgets

    panel = QtWidgets.QWidget()
    panel.setObjectName("directed-coupling")
    panel.setAutoFillBackground(True)
    palette = panel.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#16180f"))
    panel.setPalette(palette)
    panel.setMinimumHeight(300)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(8, 4, 8, 4)
    title = QtWidgets.QLabel("Directed coupling (live, approximate)")
    title.setStyleSheet("color: #f4f0e4; font-size: 16px; font-weight: 600;")
    layout.addWidget(title)
    status = QtWidgets.QLabel("Warming up.")
    status.setWordWrap(True)
    status.setStyleSheet("color: #e0a045; font-size: 14px; font-weight: 600;")
    layout.addWidget(status)

    controls = QtWidgets.QHBoxLayout()
    signal_label = QtWidgets.QLabel("Electroencephalography")
    signal_label.setStyleSheet("color: #e6e2d4;")
    signal_box = QtWidgets.QComboBox()
    signal_box.addItem("Delta-to-alpha ratio", "delta-to-alpha")
    signal_box.addItem("Delta (1 to 4 hertz)", "delta")
    signal_box.addItem("Alpha (8 to 12 hertz)", "alpha")
    window_label = QtWidgets.QLabel("Window length (seconds)")
    window_label.setStyleSheet("color: #e6e2d4;")
    window_box = QtWidgets.QSpinBox()
    window_box.setRange(30, 120)
    window_box.setValue(60)
    recency_box = QtWidgets.QCheckBox("Recency weighting")
    recency_box.setStyleSheet("color: #e6e2d4;")
    half_label = QtWidgets.QLabel("Half-life (seconds)")
    half_label.setStyleSheet("color: #e6e2d4;")
    half_life_box = QtWidgets.QDoubleSpinBox()
    half_life_box.setRange(1.0, 120.0)
    half_life_box.setDecimals(1)
    half_life_box.setValue(10.0)
    half_life_box.setEnabled(False)
    for widget in (signal_label, signal_box, window_label, window_box, recency_box, half_label, half_life_box):
        controls.addWidget(widget)
    controls.addStretch(1)
    layout.addLayout(controls)

    body = QtWidgets.QHBoxLayout()
    grids = QtWidgets.QHBoxLayout()
    cells = {}
    for measure, heading in (
        ("interdependence", "Conditional nonlinear interdependence"),
        ("forecast", "Forward forecast gain"),
    ):
        box = QtWidgets.QGroupBox(heading)
        box.setStyleSheet(
            "QGroupBox { color: #e6e2d4; background: #16180f; border: 1px solid #3a3f32; "
            "margin-top: 10px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        grid = QtWidgets.QGridLayout(box)
        corner = QtWidgets.QLabel("Source →\nTarget ↓")
        corner.setStyleSheet("color: #8e8a7c; font-size: 10px;")
        corner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(corner, 0, 0)
        for column, source in enumerate(NODE_ORDER, start=1):
            header = QtWidgets.QLabel(DISPLAY_NAME[source])
            header.setStyleSheet("color: #b7c4b0; font-size: 10px;")
            header.setWordWrap(True)
            header.setFixedWidth(108)
            header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(header, 0, column)
        for row, target in enumerate(NODE_ORDER, start=1):
            header = QtWidgets.QLabel(DISPLAY_NAME[target])
            header.setStyleSheet("color: #b7c4b0; font-size: 10px;")
            header.setWordWrap(True)
            header.setFixedWidth(108)
            grid.addWidget(header, row, 0)
            for column, source in enumerate(NODE_ORDER, start=1):
                label = QtWidgets.QLabel("—")
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(108, 42)
                if source == target:
                    label.setText("")
                    label.setStyleSheet("background: transparent; color: #6a675c;")
                else:
                    label.setStyleSheet("background: #2a2e22; color: #f4f0e4; border-radius: 4px;")
                grid.addWidget(label, row, column)
                cells[(source, target, measure)] = label
        grids.addWidget(box)
    body.addLayout(grids, 3)
    graph = QtWidgets.QVBoxLayout()
    graph_title = QtWidgets.QLabel("Arrows follow forward forecast gain")
    graph_title.setStyleSheet("color: #e6e2d4; font-size: 12px;")
    graph_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    graph.addWidget(graph_title)
    arrows = _build_arrow_view()
    arrows.setMinimumSize(360, 230)
    graph.addWidget(arrows, 1)
    body.addLayout(graph, 2)
    layout.addLayout(body)

    smoother = ScoreSmoother()
    state = {"result": {"status": "warming up", "reason": "", "interdependence": {}, "forecast_gain": {}}}

    def refresh_display():
        if coupling is None:
            status.setText("The coupling worker is not connected.")
            return
        result = coupling.latest()
        state["result"] = result
        text, style = _status_text(result)
        status.setText(text)
        status.setStyleSheet(style)
        smoothed = smoother.apply(result)
        _paint_cells(cells, result, smoothed)
        arrows.set_state(result, smoothed.get("forecast_gain") or {})
        arrows.update()

    def on_signal(_index):
        if coupling is None:
            return
        coupling.eeg_signal = signal_box.currentData()

    def on_window(value):
        if coupling is None:
            return
        coupling.set_window_seconds(float(value))

    def on_recency(checked):
        half_life_box.setEnabled(bool(checked))
        if coupling is None:
            return
        coupling.recency_weighting = bool(checked)

    def on_half_life(value):
        if coupling is None:
            return
        coupling.recency_half_life_seconds = float(value)

    signal_box.currentIndexChanged.connect(on_signal)
    window_box.valueChanged.connect(on_window)
    recency_box.toggled.connect(on_recency)
    half_life_box.valueChanged.connect(on_half_life)

    if coupling is not None:
        signal_box.blockSignals(True)
        index = signal_box.findData(getattr(coupling, "eeg_signal", "delta-to-alpha"))
        if index >= 0:
            signal_box.setCurrentIndex(index)
        signal_box.blockSignals(False)
        window_box.blockSignals(True)
        window_box.setValue(int(round(coupling.history.window_seconds)))
        window_box.blockSignals(False)
        recency_box.blockSignals(True)
        recency_box.setChecked(bool(getattr(coupling, "recency_weighting", False)))
        recency_box.blockSignals(False)
        half_life_box.blockSignals(True)
        half_life_box.setValue(float(getattr(coupling, "recency_half_life_seconds", 10.0)))
        half_life_box.blockSignals(False)
        half_life_box.setEnabled(recency_box.isChecked())

    panel.title_label = title
    panel.status_label = status
    panel.signal_box = signal_box
    panel.window_box = window_box
    panel.recency_box = recency_box
    panel.half_life_box = half_life_box
    panel.cells = cells
    panel.arrow_view = arrows
    panel.smoother = smoother
    panel.refresh_display = refresh_display
    panel.display_opacity = display_opacity
    return panel


def _status_text(result):
    status = result.get("status") or "warming up"
    reason = (result.get("reason") or "").strip()
    if status == "warming up":
        text = "Warming up."
        if reason:
            text = "Warming up. " + reason
        return text, "color: #e0a045; font-size: 14px; font-weight: 600;"
    if status == "skipped":
        text = "Skipped."
        if reason:
            text = "Skipped. " + reason
        return text, "color: #e07a68; font-size: 14px; font-weight: 600;"
    if status == "ready":
        return "Approximate scores for the latest window.", "color: #b7c4b0; font-size: 12px;"
    return status, "color: #e6e2d4; font-size: 12px;"


def _paint_cells(cells, result, smoothed):
    exact_inter = result.get("interdependence") or {}
    exact_forecast = result.get("forecast_gain") or {}
    smooth_inter = smoothed.get("interdependence") or {}
    smooth_forecast = smoothed.get("forecast_gain") or {}
    for source in NODE_ORDER:
        for target in NODE_ORDER:
            if source == target:
                continue
            arrow = "%s → %s" % (source, target)
            reverse = "%s → %s" % (target, source)
            tip = cell_tooltip(result, source, target)
            for measure, smooth_map, exact_map in (
                ("interdependence", smooth_inter, exact_inter),
                ("forecast", smooth_forecast, exact_forecast),
            ):
                label = cells[(source, target, measure)]
                shown = smooth_map.get(arrow, exact_map.get(arrow))
                other = smooth_map.get(reverse, exact_map.get(reverse))
                stronger = _is_stronger(shown, other)
                text = _format_cell(shown)
                if stronger and text != "—":
                    text = text + "\nstronger"
                label.setText(text)
                label.setToolTip(tip)
                red, green, blue, opacity = _score_color(shown, other, significant=None)
                label.setStyleSheet(
                    "background: rgba(%d, %d, %d, %d); color: #f4f0e4; border-radius: 4px;"
                    % (red, green, blue, int(round(opacity * 255)))
                )


def _is_stronger(shown, other):
    try:
        left = float(shown)
        right = float(other)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return left > right + 1e-6


def _build_arrow_view():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ArrowView(QtWidgets.QWidget):
        """Three nodes and the six forecast-gain arrows."""

        def __init__(self):
            super().__init__()
            self._result = {"forecast_gain": {}, "interdependence": {}}
            self._forecast = {}
            self._curves = []
            self.setMouseTracking(True)
            self.setMinimumHeight(210)

        def set_state(self, result, forecast):
            self._result = result
            self._forecast = forecast or {}

        def tooltip_for_pair(self, source, target):
            return pair_tooltip(self._result, source, target)

        def paintEvent(self, _event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            positions = self._positions()
            self._curves = []
            for source in NODE_ORDER:
                for target in NODE_ORDER:
                    if source == target:
                        continue
                    score = self._forecast.get("%s → %s" % (source, target))
                    opposite = self._forecast.get("%s → %s" % (target, source))
                    path, samples = self._arrow_path(positions[source], positions[target], source, target)
                    self._curves.append((source, target, samples))
                    red, green, blue, opacity = _score_color(score, opposite, significant=None)
                    color = QtGui.QColor(red, green, blue)
                    color.setAlpha(int(round(max(opacity, 0.0) * 255)))
                    width = _link_width(score, opposite)
                    pen = QtGui.QPen(color, width)
                    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawPath(path)
                    if samples:
                        self._draw_head(painter, samples[-2] if len(samples) > 1 else samples[0], samples[-1], color, width)
            painter.setFont(QtGui.QFont("sans-serif", 8))
            for name, center in positions.items():
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor("#2c3124"))
                painter.drawEllipse(center, 18, 18)
                painter.setPen(QtGui.QPen(QtGui.QColor("#f4f0e4")))
                if name == "pupil diameter":
                    rect = QtCore.QRectF(center.x() - 78, center.y() - 52, 156, 28)
                else:
                    rect = QtCore.QRectF(center.x() - 78, center.y() + 22, 156, 32)
                painter.drawText(
                    rect,
                    QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.TextFlag.TextWordWrap,
                    DISPLAY_NAME[name],
                )
            painter.end()

        def mouseMoveEvent(self, event):
            point = event.position() if hasattr(event, "position") else event.pos()
            best = None
            best_distance = 16.0
            for source, target, samples in self._curves:
                for sample in samples:
                    distance = math.hypot(sample.x() - point.x(), sample.y() - point.y())
                    if distance < best_distance:
                        best = (source, target)
                        best_distance = distance
            if best is None:
                self.setToolTip("")
            else:
                self.setToolTip(self.tooltip_for_pair(*best))
            super().mouseMoveEvent(event)

        def _positions(self):
            width = max(self.width(), 300)
            height = max(self.height(), 180)
            return {
                "breathing": QtCore.QPointF(78, height * 0.72),
                "pupil diameter": QtCore.QPointF(width / 2.0, 48),
                "electroencephalography": QtCore.QPointF(width - 78, height * 0.72),
            }

        def _arrow_path(self, start, end, source, target):
            # Offset the two directions of a pair onto opposite sides of the straight line.
            delta = end - start
            length = math.hypot(delta.x(), delta.y()) or 1.0
            normal = QtCore.QPointF(-delta.y() / length, delta.x() / length)
            side = 1.0 if NODE_ORDER.index(source) < NODE_ORDER.index(target) else -1.0
            control = (start + end) / 2.0 + normal * (36.0 * side)
            # Stop short of the node so the head is visible.
            trimmed_end = end - QtCore.QPointF(delta.x() / length * 28.0, delta.y() / length * 28.0)
            path = QtGui.QPainterPath(start)
            path.quadTo(control, trimmed_end)
            samples = []
            for step in range(24):
                samples.append(path.pointAtPercent(step / 23.0))
            return path, samples

        def _draw_head(self, painter, before, tip, color, width):
            angle = math.atan2(tip.y() - before.y(), tip.x() - before.x())
            length = 8.0 + width
            left = QtCore.QPointF(
                tip.x() - length * math.cos(angle - 0.45),
                tip.y() - length * math.sin(angle - 0.45),
            )
            right = QtCore.QPointF(
                tip.x() - length * math.cos(angle + 0.45),
                tip.y() - length * math.sin(angle + 0.45),
            )
            head = QtGui.QPainterPath(tip)
            head.lineTo(left)
            head.lineTo(right)
            head.closeSubpath()
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPath(head)

    return ArrowView()
