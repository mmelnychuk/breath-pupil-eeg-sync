"""Surrogate series and p-values for the coupling indices.

Three null series are provided. Each keeps some structure of the original
signal and destroys the time alignment that coupling would produce.

* A time-shift surrogate rotates the signal by a fraction of its length. The
  cycle shape is unchanged. For a pure, perfectly phase-locked oscillation a
  short shift is only a new constant phase offset and the interdependence can
  stay high, so shifts start after a minimum gap (by default a tenth of the
  record) and run up to half the record.
* A phase-randomised surrogate keeps the Fourier amplitudes and replaces the
  phases. The power spectrum and the linear autocorrelation stay the same.
* An iteratively amplitude-adjusted Fourier transform surrogate (Schreiber and
  Schmitz, 1996, 2000) keeps the power spectrum exactly and keeps the set of
  values almost exactly. The iteration adjusts the values, then the spectrum,
  and finishes on the spectrum so the null really does have the original
  linear autocorrelation.

The p-value is one-sided. Larger indices mean more coupling, so

    p = (1 + number of surrogates greater than or equal to the data) / (1 + number of surrogates).

Multiple-comparison corrections are Bonferroni and the Benjamini–Hochberg
false-discovery-rate procedure.
"""

from __future__ import annotations

import numpy as np


def shift_amounts(n_samples, n_surrogates, minimum_shift, generator):
    """Distinct circular shifts, each at least ``minimum_shift`` and at most half the record."""
    minimum_shift = int(minimum_shift)
    upper = max(minimum_shift + 1, int(n_samples) // 2)
    candidates = np.arange(minimum_shift, upper)
    if len(candidates) == 0:
        raise ValueError("The recording is too short for a time-shift surrogate.")
    if len(candidates) >= n_surrogates:
        return generator.choice(candidates, size=int(n_surrogates), replace=False)
    return generator.choice(candidates, size=int(n_surrogates), replace=True)


def phase_randomize(series, generator):
    """Fourier-phase randomisation. The power spectrum is unchanged."""
    values = np.asarray(series, dtype=float)
    spectrum = np.fft.rfft(values)
    amplitudes = np.abs(spectrum)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=amplitudes.shape)
    phases[0] = 0.0
    if len(values) % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(values))


def iaaft(series, generator, iterations=30):
    """Iteratively amplitude-adjusted Fourier transform surrogate.

    The result has the original power spectrum. Its value distribution is the
    original distribution up to the last spectrum adjustment.
    """
    values = np.asarray(series, dtype=float)
    amplitudes = np.abs(np.fft.rfft(values))
    ordered = np.sort(values)
    surrogate = generator.permutation(values)
    for _ in range(int(iterations)):
        spectrum = np.fft.rfft(surrogate)
        adjusted = np.fft.irfft(amplitudes * np.exp(1j * np.angle(spectrum)), n=len(values))
        surrogate = ordered[np.argsort(np.argsort(adjusted))]
    spectrum = np.fft.rfft(surrogate)
    return np.fft.irfft(amplitudes * np.exp(1j * np.angle(spectrum)), n=len(values))


def p_value_greater(observed, surrogates):
    """One-sided p-value for a statistic that grows when coupling grows."""
    null = np.asarray(surrogates, dtype=float)
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    n_greater_or_equal = int(np.sum(null >= observed))
    return (1.0 + n_greater_or_equal) / (1.0 + len(null))


def bonferroni(p_values, alpha=0.05):
    """Bonferroni-adjusted p-values, capped at 1, and which tests remain significant."""
    raw = np.asarray(p_values, dtype=float)
    adjusted = np.minimum(raw * np.sum(np.isfinite(raw)), 1.0)
    adjusted[~np.isfinite(raw)] = np.nan
    return adjusted, adjusted <= alpha


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini–Hochberg adjusted p-values and the significant tests.

    The adjusted value is the false-discovery-rate q-value, capped at 1.
    """
    raw = np.asarray(p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan)
    valid = np.isfinite(raw)
    if not np.any(valid):
        return adjusted, np.zeros(raw.shape, dtype=bool)
    order = np.argsort(raw[valid])
    ranked = raw[valid][order]
    count = len(ranked)
    q_ranked = ranked * count / (np.arange(1, count + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    place = np.flatnonzero(valid)
    adjusted[place[order]] = q_ranked
    return adjusted, adjusted <= alpha
