"""Offline nonlinear coupling among breathing, pupil diameter, and electroencephalography.

The live recording hub is not involved. This package reads files that have
already been written, or series produced by the three-oscillator simulator.
"""

from sniff_sync.nlid.analyze import analyse_triplet
from sniff_sync.nlid.model import simulate_oscillators

__all__ = ["analyse_triplet", "simulate_oscillators"]
