"""Partial cross mapping (Leng et al. 2020, Nature Communications).

Mutual cross mapping, the step this builds on, is Sugihara's convergent cross
mapping. If a cause leaves a trace in an effect, the effect's delay vectors
have neighbours whose times also carry similar values of the cause. Predicting
the cause from the effect's neighbours is therefore evidence that the cause
influences the effect. The arrow points from the predicted series (the cause)
to the series whose neighbours are used (the effect).

That prediction still lights up when the influence is only indirect, because
the intermediate signal's trace is copied onward. Partial cross mapping removes
that indirect trace. For a candidate cause X, an effect Y, and a third signal Z:

1. Cross-map Z from Y, choosing the time lag that maximises the correlation.
   The resulting series is the trace of Z that is visible in Y.
2. Cross-map X from that trace. This is the part of X that Y knows only
   because it knows Z.
3. Cross-map X from Y itself, again at its best lag. This is everything Y
   knows about X.
4. The direct-link index is the absolute partial correlation between X and
   the prediction in step 3, after the prediction in step 2 has been
   partialled out:

       ρ_D = | (r_xy − r_xz r_yz) / sqrt((1 − r_xz²)(1 − r_yz²)) |

   where x is the cause, y is the direct cross-map prediction, and z is the
   indirect cross-map prediction.

ρ_D near 0 means the effect's knowledge of the cause is accounted for by the
third signal. A larger ρ_D means a remainder that the third signal does not
explain. The plain cross-map correlation ρ_C, without partialling, is kept
beside it so an indirect link that still shows up in ordinary cross mapping
can be seen.

Leng et al. use embedding-dimension + 1 neighbours, the smallest simplex in
that dimension. Weights are the Sugihara weights: the nearest neighbour has
weight 1 and the others fall off as exp(−distance / nearest distance).

The paper excludes strongly synchronised systems. On a noise-free limit cycle
every phase is a function of every other phase, the partial correlation is
undefined, and the direct-link index is reported as missing rather than as
evidence of a direct link.
"""

from __future__ import annotations

import numpy as np

from sniff_sync.nlid.embed import delay_embed, neighbor_indices, zscore


def pearson(left, right):
    """Pearson correlation, or 0 when either series has no variance."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    left_energy = float(np.dot(a, a))
    right_energy = float(np.dot(b, b))
    if left_energy <= 1e-18 or right_energy <= 1e-18:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(left_energy * right_energy))


def partial_correlation(x_values, y_values, z_values):
    """Partial correlation of x and y with z removed. NaN if z already explains either one."""
    r_xy = pearson(x_values, y_values)
    r_xz = pearson(x_values, z_values)
    r_yz = pearson(y_values, z_values)
    denominator = np.sqrt(max(0.0, 1.0 - r_xz * r_xz) * max(0.0, 1.0 - r_yz * r_yz))
    if denominator < 1e-8:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denominator)


def _simplex_weights(squared_distances):
    distance = np.sqrt(np.maximum(squared_distances, 0.0))
    nearest = np.maximum(distance[:, :1], 1e-12)
    weights = np.exp(-distance / nearest)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


class _CrossMapLibrary:
    """Neighbours of one effect embedding, ready to predict any cause series of the same length."""

    def __init__(self, effect, dimension, delay, theiler_window):
        embedded, self.start = delay_embed(zscore(effect), int(dimension), int(delay))
        neighbor_count = int(dimension) + 1
        self.neighbor_index, squared = neighbor_indices(embedded, neighbor_count, theiler_window)
        self.weights = _simplex_weights(squared)
        self.n_vectors = len(embedded)

    def predict(self, cause):
        """Cross-map ``cause`` (same length as the original effect) at lag 0.

        The returned array is aligned with embedding rows. Entry i estimates
        the cause at original sample ``start + i``.
        """
        cause = np.asarray(cause, dtype=float)
        if len(cause) < self.start + self.n_vectors:
            raise ValueError("The cause series is shorter than the effect embedding.")
        aligned = cause[self.start : self.start + self.n_vectors]
        return np.sum(self.weights * aligned[self.neighbor_index], axis=1)


def _best_aligned_prediction(prediction, cause, start, max_lag):
    """Search lags and return the cause and its cross-map estimate on one time base.

    ``prediction[i]`` estimates ``cause[start + i]`` at lag 0. A positive lag L
    means the effect state at time t estimates the cause at time t + L, so the
    estimate of the cause at time t is ``prediction[t - start - L]``.
    """
    max_lag = int(max_lag)
    best_correlation = -1.0
    best = None
    n_vectors = len(prediction)
    for lag in range(-max_lag, max_lag + 1):
        # Estimate of cause[t] lives at prediction index (t - start - lag).
        # Valid t satisfy 0 <= t - start - lag < n_vectors and 0 <= t < len(cause).
        t_first = max(0, start + lag)
        t_last = min(len(cause), start + lag + n_vectors)
        if t_last - t_first < 20:
            continue
        pred_index = np.arange(t_first, t_last) - start - lag
        correlation = abs(pearson(cause[t_first:t_last], prediction[pred_index]))
        if correlation > best_correlation:
            best_correlation = correlation
            best = (correlation, lag, cause[t_first:t_last].copy(), prediction[pred_index].copy(), t_first)
    if best is None:
        raise ValueError("The cross-map lag search had no overlapping samples.")
    return best


class PartialCrossMapper:
    """Partial cross mapping from one cause to one effect, conditioned on a third signal.

    The expensive neighbour searches use only the effect and the conditioner, so
    a surrogate cause (a time-shifted or phase-randomised copy) is cheap to score.
    """

    def __init__(self, effect, conditioner, dimension, delay, theiler_window, max_lag):
        self.effect_library = _CrossMapLibrary(effect, dimension[0], delay[0], theiler_window)
        self.max_lag = int(max_lag)
        self._conditioner = zscore(conditioner)
        conditioner_prediction = self.effect_library.predict(self._conditioner)
        correlation, lag, observed, estimated, first = _best_aligned_prediction(
            conditioner_prediction,
            self._conditioner,
            self.effect_library.start,
            self.max_lag,
        )
        # Rebuild the indirect trace on the conditioner's own sample index so it
        # can be embedded. Missing edges stay at the series mean (zero after z-scoring).
        trace = np.zeros(len(self._conditioner))
        trace[first : first + len(estimated)] = estimated
        self.indirect_trace = trace
        self.conditioner_cross_map_correlation = correlation
        self.conditioner_lag = lag
        self.indirect_library = _CrossMapLibrary(trace, dimension[1], delay[1], theiler_window)
        self._effect_dimension = dimension
        self._effect_delay = delay

    def evaluate(self, cause):
        """Return the cross-map correlation and the direct-link index for this cause."""
        cause = zscore(np.asarray(cause, dtype=float))
        direct_prediction = self.effect_library.predict(cause)
        direct_correlation, _lag, cause_aligned, direct_aligned, first = _best_aligned_prediction(
            direct_prediction,
            cause,
            self.effect_library.start,
            self.max_lag,
        )
        indirect_prediction = self.indirect_library.predict(cause)
        _indirect_correlation, _indirect_lag, _indirect_cause, indirect_aligned, indirect_first = _best_aligned_prediction(
            indirect_prediction,
            cause,
            self.indirect_library.start,
            self.max_lag,
        )
        direct_time = np.arange(first, first + len(cause_aligned))
        indirect_time = np.arange(indirect_first, indirect_first + len(indirect_aligned))
        shared_first = max(int(direct_time[0]), int(indirect_time[0]))
        shared_last = min(int(direct_time[-1]), int(indirect_time[-1])) + 1
        if shared_last - shared_first < 20:
            return {"cross_map_correlation": direct_correlation, "direct_link_index": float("nan")}
        cause_shared = cause[shared_first:shared_last]
        direct_shared = direct_aligned[shared_first - first : shared_first - first + len(cause_shared)]
        indirect_shared = indirect_aligned[shared_first - indirect_first : shared_first - indirect_first + len(cause_shared)]
        coefficient = partial_correlation(cause_shared, direct_shared, indirect_shared)
        index = float("nan") if not np.isfinite(coefficient) else abs(float(coefficient))
        return {
            "cross_map_correlation": float(direct_correlation),
            "direct_link_index": index,
        }


def partial_cross_map(cause, effect, conditioner, dimension, delay, theiler_window, max_lag):
    """Fit and evaluate partial cross mapping once.

    ``dimension`` and ``delay`` are (effect, indirect-trace) pairs. The cause
    itself is not embedded; it is the series being predicted.
    """
    mapper = PartialCrossMapper(effect, conditioner, dimension, delay, theiler_window, max_lag)
    result = mapper.evaluate(cause)
    result["mapper"] = mapper
    return result
