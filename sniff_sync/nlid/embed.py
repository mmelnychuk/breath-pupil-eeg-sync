"""Time-delay embedding and the two usual ways to choose its parameters.

A delay vector stacks a sample with earlier samples of the same signal:

    x(t), x(t − τ), x(t − 2τ), ..., x(t − (m − 1)τ)

τ is the delay and m is the dimension. The delay is chosen from the first
minimum of the delayed mutual information (Fraser and Swinney, 1986). The
dimension is chosen from the false-nearest-neighbour fraction (Kennel, Brown,
and Abarbanel, 1992). Neighbour searches skip a Theiler window (Theiler, 1986)
so that a point is not paired with its own temporal neighbours.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def zscore(series):
    """Return a copy with mean 0 and standard deviation 1."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a single signal (a one-dimensional array).")
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError("The signal still contains missing samples; interpolate or trim it first.")
    standard_deviation = float(values.std())
    if standard_deviation < 1e-8:
        raise ValueError("The signal is constant, so it has no state-space geometry to embed.")
    return (values - float(values.mean())) / standard_deviation


def delay_embed(series, dimension, delay):
    """Build delay vectors. Row i is the state at sample i + (dimension − 1) × delay.

    Column 0 is the newest sample. The last column is the oldest sample.
    """
    values = np.asarray(series, dtype=float)
    dimension = int(dimension)
    delay = int(delay)
    if dimension < 1 or delay < 1:
        raise ValueError("Embedding dimension and delay must both be at least 1.")
    n_vectors = len(values) - (dimension - 1) * delay
    if n_vectors < 1:
        raise ValueError(
            "The signal is shorter than one delay vector "
            "(length %d, dimension %d, delay %d)." % (len(values), dimension, delay)
        )
    columns = []
    for lag_index in range(dimension):
        start = (dimension - 1 - lag_index) * delay
        columns.append(values[start : start + n_vectors])
    embedded = np.column_stack(columns)
    start_index = (dimension - 1) * delay
    return embedded, start_index


def mutual_information(a, b, bins=16):
    """Histogram mutual information, in nats, between two equal-length series."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Mutual information needs two equal series of at least two samples.")
    histogram, _, _ = np.histogram2d(left, right, bins=int(bins))
    joint = histogram / histogram.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / px_py[mask])))


def delayed_mutual_information(series, max_lag, bins=16):
    """Mutual information between the signal and itself at lags 1, 2, ..., max_lag."""
    values = np.asarray(series, dtype=float)
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    if len(values) <= max_lag + 2:
        raise ValueError("The signal is too short for the requested maximum delay.")
    scores = np.empty(max_lag, dtype=float)
    for lag in range(1, max_lag + 1):
        scores[lag - 1] = mutual_information(values[:-lag], values[lag:], bins=bins)
    return scores


def choose_delay(series, max_lag, bins=16):
    """Pick a delay: first local minimum of the delayed mutual information.

    If the curve never turns up, use the first lag where the mutual information
    has fallen below 1/e of its value at lag 1. If that also fails, use lag 1.
    """
    scores = delayed_mutual_information(series, max_lag=max_lag, bins=bins)
    for index in range(1, len(scores) - 1):
        if scores[index] <= scores[index - 1] and scores[index] < scores[index + 1]:
            return index + 1, scores
    threshold = scores[0] / np.e
    below = np.flatnonzero(scores <= threshold)
    if len(below):
        return int(below[0] + 1), scores
    return 1, scores


def neighbor_indices(vectors, neighbor_count, theiler_window):
    """k nearest neighbours, excluding the Theiler window.

    Returns integer indices of shape (n, k) and squared Euclidean distances
    of the same shape. Neighbours are ordered from nearest to farthest.
    """
    points = np.ascontiguousarray(vectors, dtype=float)
    if points.ndim != 2:
        raise ValueError("Neighbour search expects a two-dimensional array of delay vectors.")
    n_vectors, _width = points.shape
    neighbor_count = int(neighbor_count)
    theiler_window = int(theiler_window)
    if neighbor_count < 1:
        raise ValueError("The neighbour count must be at least 1.")
    if n_vectors <= neighbor_count:
        raise ValueError(
            "Need more delay vectors (%d) than neighbours (%d)." % (n_vectors, neighbor_count)
        )
    tree = cKDTree(points)
    query_count = min(n_vectors, neighbor_count + 2 * max(theiler_window, 0) + 8)
    enough = None
    valid = None
    indices = None
    distances = None
    while True:
        distances, indices = tree.query(points, k=query_count)
        if query_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        indices = indices.astype(int, copy=False)
        time_index = np.arange(n_vectors)[:, None]
        valid = np.abs(indices - time_index) > theiler_window
        enough = valid.sum(axis=1) >= neighbor_count
        if bool(np.all(enough)) or query_count == n_vectors:
            break
        query_count = min(n_vectors, max(query_count * 2, query_count + neighbor_count))
    if not bool(np.all(enough)):
        raise ValueError(
            "The Theiler window of %d samples leaves fewer than %d neighbours. "
            "Use a shorter window or a longer recording." % (theiler_window, neighbor_count)
        )
    order = np.cumsum(valid, axis=1)
    selected = valid & (order <= neighbor_count)
    out_index = indices[selected].reshape(n_vectors, neighbor_count)
    out_distance = np.square(distances[selected].reshape(n_vectors, neighbor_count))
    return out_index, out_distance


def false_nearest_neighbor_fraction(series, delay, dimension, relative_tolerance=15.0, attractor_tolerance=2.0):
    """Fraction of neighbours that are false when the dimension increases by one.

    A neighbour is false when its distance grows by more than
    ``relative_tolerance`` as the next delay is added, or when that expanded
    distance is more than ``attractor_tolerance`` times the standard deviation
    of the signal (Kennel, Brown, and Abarbanel, 1992).
    """
    values = np.asarray(series, dtype=float)
    dimension = int(dimension)
    delay = int(delay)
    embedded_now, _start_now = delay_embed(values, dimension, delay)
    embedded_next, _start_next = delay_embed(values, dimension + 1, delay)
    n_next = len(embedded_next)
    # Row j of the higher-dimensional embedding is the same time as row j+delay
    # of the lower-dimensional embedding.
    base = embedded_now[delay : delay + n_next]
    added_coordinate = embedded_next[:, -1]
    indices, squared_distance = neighbor_indices(base, neighbor_count=1, theiler_window=delay)
    nearest = indices[:, 0]
    expanded = squared_distance[:, 0] + np.square(added_coordinate - added_coordinate[nearest])
    current = np.sqrt(np.maximum(squared_distance[:, 0], 0.0))
    expanded_distance = np.sqrt(np.maximum(expanded, 0.0))
    usable = current > 1e-10
    if not np.any(usable):
        # Every neighbour lands on the same point. Nothing in the extra delay
        # coordinate marked it as false.
        return 0.0
    ratio = np.ones(len(current))
    ratio[usable] = expanded_distance[usable] / current[usable]
    scale = float(np.std(values))
    false_neighbor = (ratio > relative_tolerance) | (expanded_distance > attractor_tolerance * scale)
    return float(np.mean(false_neighbor[usable]))


def choose_dimension(series, delay, max_dimension=6, threshold=0.10):
    """Smallest dimension whose false-nearest-neighbour fraction is at most ``threshold``.

    Returns that dimension and the fraction at every dimension that was tried.
    If none falls under the threshold, returns the dimension with the smallest fraction.
    """
    fractions = []
    best_dimension = 1
    best_fraction = 1.0
    for dimension in range(1, int(max_dimension) + 1):
        fraction = false_nearest_neighbor_fraction(series, delay, dimension)
        fractions.append(fraction)
        if fraction < best_fraction:
            best_fraction = fraction
            best_dimension = dimension
        if fraction <= threshold:
            return dimension, fractions
    return best_dimension, fractions


def scale_blocks(blocks):
    """Give each signal the same total weight, whatever its embedding dimension.

    Each block is already built from a unit-variance signal. Dividing by
    sqrt(dimension) stops a higher-dimensional signal from dominating a
    concatenated distance.
    """
    scaled = []
    for block in blocks:
        array = np.asarray(block, dtype=float)
        scaled.append(array / np.sqrt(array.shape[1]))
    return scaled
