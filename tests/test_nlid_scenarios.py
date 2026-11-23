"""Ground truth for the respiration–locus coeruleus–attention oscillator.

No sensors. Each scenario and the coupling sweep is scored on seeds 0 through
19. The test records detection rates. It does not drop a seed or raise the
false-positive allowance to force a pass.
"""

import unittest

from sniff_sync.nlid.model import phase_drift, reference_coupling, reference_epsilon
from sniff_sync.nlid.validate import (
    SCENARIO_MEASURES,
    coupling_sweep,
    format_scenario_rates,
    format_sweep,
    scenario_detection_rates,
)

import numpy as np


class ModelTests(unittest.TestCase):
    def test_reference_coefficients_match_the_2018_figure(self):
        self.assertEqual(reference_epsilon(0.25), 0.33)
        self.assertEqual(reference_epsilon(0.35), 0.33)
        self.assertEqual(reference_epsilon(0.12), 0.45)
        published = reference_coupling(0.25)
        self.assertEqual(published.respiration_to_attention, 0.0)
        self.assertEqual(published.attention_to_respiration, 0.0)
        self.assertEqual(published.locus_to_attention, 0.33)

    def test_phase_equation_uses_a_coefficient_per_link(self):
        phase = np.array([0.4, -0.2, 1.1])
        frequency = np.array([0.8, 0.7, 0.5])
        matrix = np.zeros((3, 3))
        matrix[0, 1] = 0.33  # locus coeruleus → attention
        matrix[2, 1] = 0.45  # locus coeruleus → respiration
        got = phase_drift(phase, frequency, matrix)
        expected = frequency.copy()
        expected[0] += 0.33 * np.sin(phase[1] - phase[0])
        expected[2] += 0.45 * np.sin(phase[1] - phase[2])
        self.assertTrue(np.allclose(got, expected))


class ScenarioTests(unittest.TestCase):
    def test_scenario_detection_rates_across_twenty_seeds(self):
        result = scenario_detection_rates(seeds=range(20), workers=4)
        print("\n" + format_scenario_rates(result))
        self.assertEqual(result["seeds"], list(range(20)))
        self.assertEqual(result["n_seeds"], 20)
        directions = (
            ("breathing", "pupil diameter", "attention"),
            ("pupil diameter", "breathing", "attention"),
            ("pupil diameter", "attention", "breathing"),
            ("attention", "pupil diameter", "breathing"),
            ("breathing", "attention", "pupil diameter"),
            ("attention", "breathing", "pupil diameter"),
        )
        for kind in ("published", "direct", "uncoupled"):
            for frequency in (0.12, 0.25, 0.35):
                for measure in SCENARIO_MEASURES:
                    for source, target, conditioner in directions:
                        conditioned_on = "" if measure == "Andrzejak M" else conditioner
                        key = (kind, float(frequency), measure, source, target, conditioned_on)
                        self.assertIn(key, result["rates"], msg=str(key))
                        proportion = result["rates"][key]
                        self.assertEqual(proportion["tested"], 20, msg=str(key))
                        self.assertGreaterEqual(proportion["rate"], 0.0)
                        self.assertLessEqual(proportion["rate"], 1.0)
                        self.assertEqual(proportion["detected"] / proportion["tested"], proportion["rate"])

    def test_coupling_sweep_rates_across_twenty_seeds(self):
        result = coupling_sweep(seeds=range(20), workers=4)
        print("\n" + format_sweep(result))
        self.assertEqual(result["seeds"], list(range(20)))
        self.assertEqual(result["n_seeds"], 20)
        for measure, info in result["summary"].items():
            self.assertGreaterEqual(info["false_positive_rate"], 0.0)
            self.assertLessEqual(info["false_positive_rate"], 1.0)
            self.assertEqual(info["false_positives"] / info["absent_tests"], info["false_positive_rate"])
            for link, curve in info["curve"].items():
                self.assertEqual(len(curve), 7, msg=link)
                for point in curve:
                    self.assertEqual(point["tested"], 20)
                    self.assertGreaterEqual(point["rate"], 0.0)
                    self.assertLessEqual(point["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
