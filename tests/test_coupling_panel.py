"""The Directed coupling panel: states, scores, tooltips, and live controls."""

import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sniff_sync.coupling_panel import NODE_ORDER, ScoreSmoother, display_opacity, link_opacity
from sniff_sync.simulate import Session


class SmoothingTests(unittest.TestCase):
    def test_display_opacity_fades_a_weak_link_and_can_fade_a_non_significant_one(self):
        self.assertGreater(display_opacity(2.0), 0.8)
        self.assertLess(display_opacity(0.0), 0.25)
        self.assertLess(display_opacity(2.0, significant=False), 0.3)
        self.assertGreater(display_opacity(2.0, significant=None), display_opacity(0.05))
        self.assertGreater(link_opacity(0.8, 0.1), link_opacity(0.1, 0.8))
        self.assertGreater(link_opacity(-1.0, -40.0), link_opacity(-40.0, -1.0))
        self.assertLess(link_opacity(2.0, 0.0, significant=False), 0.3)

    def test_the_smoother_moves_only_when_a_new_result_arrives(self):
        smoother = ScoreSmoother(half_life_seconds=3.0)
        first = _ready(0.10, logged=1.0)
        second = _ready(0.90, logged=2.0)
        self.assertAlmostEqual(smoother.apply(first)["forecast_gain"]["breathing → pupil diameter"], 0.10)
        blended = smoother.apply(second)["forecast_gain"]["breathing → pupil diameter"]
        self.assertGreater(blended, 0.10)
        self.assertLess(blended, 0.90)
        again = smoother.apply(second)["forecast_gain"]["breathing → pupil diameter"]
        self.assertAlmostEqual(again, blended)


class PanelTests(unittest.TestCase):
    def test_the_panel_is_added_and_follows_the_worker(self):
        from PySide6 import QtWidgets

        from sniff_sync.viz import Monitor

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        temporary = tempfile.TemporaryDirectory()
        session = Session(temporary.name, enable_coupling=False)
        session.start()
        coupling = _StubCoupling()
        try:
            deadline = time.time() + 2
            while time.time() < deadline:
                snap = session.feed.snapshot()
                if snap["pupil"] and len(snap["breath"]) >= 8 and snap["eeg"]:
                    break
                time.sleep(0.05)
            window = Monitor(session.feed, coupling=coupling)
            window.timer.stop()
            window.show()
            app.processEvents()
            self.assertIn("coupling", window._panels)
            self.assertIs(window._panels["coupling"], window.coupling_panel)
            self.assertGreaterEqual(window._bottom_layout.indexOf(window.coupling_panel), 0)
            self.assertIn("breath", window._panels)
            self.assertIn("pupil", window._panels)
            self.assertIn("eeg", window._panels)
            title = window.coupling_panel.title_label.text()
            self.assertEqual(title, "Directed coupling (live, approximate)")
            self.assertNotIn("information transfer", title.lower())
            self.assertTrue(window.coupling_panel.status_label.text().startswith("Warming up"))
            self.assertIn("baseline", window.coupling_panel.status_label.text())
            for name in NODE_ORDER:
                self.assertEqual(window.coupling_panel.cells[(name, name, "interdependence")].text(), "")
                self.assertEqual(window.coupling_panel.cells[(name, name, "forecast")].text(), "")

            coupling.result = _ready(0.80, interdependence=0.40, reverse_forecast=0.10, reverse_interdependence=0.55)
            window.coupling_panel.refresh_display()
            forecast = window.coupling_panel.cells[("breathing", "pupil diameter", "forecast")]
            opposite = window.coupling_panel.cells[("pupil diameter", "breathing", "forecast")]
            self.assertIn("0.80", forecast.text())
            self.assertIn("stronger", forecast.text())
            self.assertNotIn("stronger", opposite.text())
            interdependence = window.coupling_panel.cells[("pupil diameter", "breathing", "interdependence")]
            self.assertIn("0.55", interdependence.text())
            self.assertIn("stronger", interdependence.text())
            tip = forecast.toolTip()
            self.assertIn("Breathing", tip)
            self.assertIn("Pupil diameter", tip)
            self.assertIn("Conditional nonlinear interdependence: 0.400000", tip)
            self.assertIn("Forward forecast gain: 0.800000", tip)
            self.assertIn("Window length: 60 seconds", tip)
            self.assertIn("Status: ready", tip)
            pair = window.coupling_panel.arrow_view.tooltip_for_pair("breathing", "pupil diameter")
            self.assertIn("forward forecast gain 0.800000", pair)
            self.assertIn("conditional nonlinear interdependence 0.400000", pair)
            self.assertIn("forward forecast gain 0.100000", pair)

            coupling.result = _ready(0.20, interdependence=0.40, reverse_forecast=0.10, reverse_interdependence=0.55, logged=2.0)
            window.coupling_panel.refresh_display()
            shown = forecast.text().split()[0]
            self.assertGreater(float(shown), 0.20)
            self.assertLess(float(shown), 0.80)
            self.assertIn("Forward forecast gain: 0.200000", forecast.toolTip())

            coupling.result = {
                "status": "skipped",
                "reason": "This window was skipped because only 10 percent of its samples are present.",
                "window_seconds": 60,
                "window_end_unix": 3.0,
                "logged_unix": 3.0,
                "interdependence": {},
                "forecast_gain": {},
            }
            window.coupling_panel.refresh_display()
            self.assertTrue(window.coupling_panel.status_label.text().startswith("Skipped."))
            self.assertIn("10 percent", window.coupling_panel.status_label.text())

            self.assertEqual(window.coupling_panel.signal_box.currentData(), "delta-to-alpha")
            self.assertEqual(window.coupling_panel.window_box.value(), 60)
            self.assertFalse(window.coupling_panel.recency_box.isChecked())
            self.assertFalse(window.coupling_panel.half_life_box.isEnabled())
            window.coupling_panel.signal_box.setCurrentIndex(window.coupling_panel.signal_box.findData("delta"))
            self.assertEqual(coupling.eeg_signal, "delta")
            window.coupling_panel.signal_box.setCurrentIndex(window.coupling_panel.signal_box.findData("alpha"))
            self.assertEqual(coupling.eeg_signal, "alpha")
            window.coupling_panel.window_box.setValue(45)
            self.assertEqual(coupling.history.window_seconds, 45.0)
            window.coupling_panel.recency_box.setChecked(True)
            self.assertTrue(coupling.recency_weighting)
            self.assertTrue(window.coupling_panel.half_life_box.isEnabled())
            window.coupling_panel.half_life_box.setValue(12.5)
            self.assertEqual(coupling.recency_half_life_seconds, 12.5)
            window.coupling_panel.recency_box.setChecked(False)
            self.assertFalse(coupling.recency_weighting)
            app.processEvents()
            window.close()
        finally:
            session.stop()
            temporary.cleanup()


class _History:
    def __init__(self):
        self.window_seconds = 60.0


class _StubCoupling:
    def __init__(self):
        self.eeg_signal = "delta-to-alpha"
        self.recency_weighting = False
        self.recency_half_life_seconds = 10.0
        self.history = _History()
        self.result = {
            "status": "warming up",
            "reason": "The baseline used to choose the embedding is not complete yet.",
            "window_seconds": 60,
            "interdependence": {},
            "forecast_gain": {},
        }

    def latest(self):
        return self.result

    def set_window_seconds(self, seconds):
        self.history.window_seconds = float(seconds)


def _ready(forecast, interdependence=0.4, reverse_forecast=0.1, reverse_interdependence=0.2, logged=1.0):
    interdependence_scores = {}
    forecast_scores = {}
    for source in NODE_ORDER:
        for target in NODE_ORDER:
            if source == target:
                continue
            interdependence_scores["%s → %s" % (source, target)] = 0.05
            forecast_scores["%s → %s" % (source, target)] = 0.05
    forecast_scores["breathing → pupil diameter"] = forecast
    forecast_scores["pupil diameter → breathing"] = reverse_forecast
    interdependence_scores["breathing → pupil diameter"] = interdependence
    interdependence_scores["pupil diameter → breathing"] = reverse_interdependence
    return {
        "status": "ready",
        "reason": "",
        "window_seconds": 60,
        "window_end_unix": logged,
        "logged_unix": logged,
        "interdependence": interdependence_scores,
        "forecast_gain": forecast_scores,
    }


if __name__ == "__main__":
    unittest.main()
