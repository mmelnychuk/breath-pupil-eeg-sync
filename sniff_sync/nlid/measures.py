"""Nearest-neighbour interdependence, and the conditional and joint extensions.

Bivariate indices follow Arnhold et al. (1999), Quian Quiroga et al. (2002)
and Andrzejak et al. (2003). For a target state x and a source state y:

    R_k(X)      mean squared distance from x to its own k nearest neighbours
    R_k(X | Y)  the same distance, but to the time indices of y's neighbours
    R(X)        mean squared distance from x to every other state outside
                the Theiler window

    similarity S                 = average of R_k(X) / R_k(X | Y)
    interdependence H            = average of ln R(X) − ln R_k(X | Y)
    normalised interdependence N = average of (R(X) − R_k(X | Y)) / R(X)
    Andrzejak interdependence M  = average of (R(X) − R_k(X | Y)) / (R(X) − R_k(X))

S, N and M are near 0 when the source tells you nothing about the target, and
near 1 when the source's neighbours are the target's neighbours. H is 0 when
the source is uninformative and grows as the source's neighbours get closer.
The arrow "source → target" means a high value of the index of the target
given the source: the source predicts the target's state. It is not, by
itself, a claim about which oscillator drives the other.

Conditional indices ask a further question. Once states have been matched on
a third signal Z, does matching the candidate source X as well make the
target Y any closer? Neighbours are taken from a pool of Z-neighbours and
then re-ranked by distance in X. This is the conditional form used here.
The same pool can also be built in the joint embedding of Y and Z, which is
the forecast form: does adding X improve a nearest-neighbour forecast of Y?

The joint index asks how well two signals together predict the third, compared
with the better of the two signals alone.
"""

from __future__ import annotations

import numpy as np

from sniff_sync.nlid.embed import delay_embed, neighbor_indices, scale_blocks, zscore


def _mean_square_radius(vectors, theiler_window):
    """Mean squared distance from each state to all states outside the Theiler window."""
    points = np.asarray(vectors, dtype=float)
    n_vectors = len(points)
    squared_norm = np.einsum("ij,ij->i", points, points)
    # Sum of squared distances from i to every j, including j = i (which is 0).
    total = n_vectors * squared_norm + squared_norm.sum() - 2.0 * points @ points.sum(axis=0)
    excluded = np.zeros(n_vectors)
    excluded_count = np.zeros(n_vectors)
    window = min(int(theiler_window), n_vectors - 1)
    for lag in range(1, window + 1):
        difference = points[:-lag] - points[lag:]
        squared = np.einsum("ij,ij->i", difference, difference)
        excluded[:-lag] += squared
        excluded[lag:] += squared
        excluded_count[:-lag] += 1
        excluded_count[lag:] += 1
    denominator = (n_vectors - 1) - excluded_count
    denominator[denominator < 1] = np.nan
    return (total - excluded) / denominator


def _neighbor_radius(vectors, neighbor_index):
    difference = vectors[neighbor_index] - vectors[:, None, :]
    return np.mean(np.einsum("ijk,ijk->ij", difference, difference), axis=1)


def _average_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    if not np.any(finite):
        return float("nan")
    scale = np.nanmedian(np.maximum(np.abs(numerator[finite]), np.abs(denominator[finite])))
    floor = 1e-8 * scale if np.isfinite(scale) and scale > 0 else 1e-18
    usable = finite & (np.abs(denominator) > floor)
    if not np.any(usable):
        if np.all(np.abs(numerator[finite]) <= max(floor, 1e-18)):
            return 1.0
        return float("nan")
    return float(np.mean(numerator[usable] / denominator[usable]))


def _bivariate_indices(target, source_neighbors, theiler_window, self_neighbors=None):
    if self_neighbors is None:
        self_neighbors, _ = neighbor_indices(target, source_neighbors.shape[1], theiler_window)
    radius_self = _neighbor_radius(target, self_neighbors)
    radius_source = _neighbor_radius(target, source_neighbors)
    radius_all = _mean_square_radius(target, theiler_window)
    log_all = np.log(np.maximum(radius_all, 1e-15))
    log_source = np.log(np.maximum(radius_source, 1e-15))
    return {
        "similarity_S": _average_ratio(radius_self, radius_source),
        "interdependence_H": float(np.nanmean(log_all - log_source)),
        "normalised_interdependence_N": _average_ratio(radius_all - radius_source, radius_all),
        "andrzejak_M": _average_ratio(radius_all - radius_source, radius_all - radius_self),
        "radius_self": radius_self,
        "radius_all": radius_all,
        "self_neighbors": self_neighbors,
    }


def embed_signal(series, dimension, delay):
    """Z-score a signal and build its delay vectors."""
    embedded, start = delay_embed(zscore(series), dimension, delay)
    return embedded, start


def bivariate_interdependence(target_series, source_series, dimension, delay, neighbor_count, theiler_window):
    """S, H, N and M for one ordered pair. The source is the predictor."""
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    source, source_start = embed_signal(source_series, dimension[1], delay[1])
    start = max(target_start, source_start)
    end = min(target_start + len(target), source_start + len(source))
    if end - start < neighbor_count + 2 * theiler_window + 5:
        raise ValueError("Not enough overlapping samples for the neighbour search.")
    target = target[start - target_start : end - target_start]
    source = source[start - source_start : end - source_start]
    source_neighbors, _ = neighbor_indices(source, neighbor_count, theiler_window)
    indices = _bivariate_indices(target, source_neighbors, theiler_window)
    indices["n_vectors"] = int(len(target))
    return indices


def _prepare_three(series, dimension, delay, neighbor_count, theiler_window, reserve_end=0):
    """Embed three series on a common time window.

    ``series`` is (target, source, conditioner). ``dimension`` and ``delay``
    are length-3 sequences in that same order.
    """
    embedded = []
    starts = []
    for values, width, lag in zip(series, dimension, delay):
        block, start = embed_signal(values, width, lag)
        embedded.append(block)
        starts.append(start)
    start = max(starts)
    end = min(item_start + len(block) for item_start, block in zip(starts, embedded)) - int(reserve_end)
    if end - start < neighbor_count + 2 * theiler_window + 5:
        raise ValueError(
            "Not enough overlapping samples after embedding. A longer recording, "
            "a shorter delay, or a lower dimension is needed."
        )
    windowed = []
    for block, item_start in zip(embedded, starts):
        windowed.append(block[start - item_start : end - item_start])
    return windowed, start, end


def conditional_interdependence(
    target_series,
    source_series,
    conditioner_series,
    dimension,
    delay,
    neighbor_count,
    pool_size,
    theiler_window,
):
    """Conditional S, H, N and M of the target given the source, after the conditioner.

    Zero means that, inside states already matched on the conditioner, matching
    the source as well does not pull the target any closer. Positive means it does.
    """
    (target, source, conditioner), _start, _end = _prepare_three(
        (target_series, source_series, conditioner_series),
        dimension,
        delay,
        neighbor_count,
        theiler_window,
    )
    pool = max(int(pool_size), int(neighbor_count) + 1)
    conditioner_neighbors, _ = neighbor_indices(conditioner, pool, theiler_window)
    self_neighbors, _ = neighbor_indices(target, neighbor_count, theiler_window)
    radius_self = _neighbor_radius(target, self_neighbors)
    radius_conditioner = _neighbor_radius(target, conditioner_neighbors[:, :neighbor_count])
    refined = _refine_pool(source, conditioner_neighbors, neighbor_count)
    radius_refined = _neighbor_radius(target, refined)
    improvement = radius_conditioner - radius_refined
    log_gap = np.log(np.maximum(radius_conditioner, 1e-15)) - np.log(np.maximum(radius_refined, 1e-15))
    ratio = radius_conditioner / np.maximum(radius_refined, 1e-15)
    return {
        "conditional_similarity_S": float(np.nanmean(ratio - 1.0)),
        "conditional_interdependence_H": float(np.nanmean(log_gap)),
        "conditional_normalised_interdependence_N": _average_ratio(improvement, radius_conditioner),
        "conditional_andrzejak_M": _average_ratio(improvement, radius_conditioner - radius_self),
        "n_vectors": int(len(target)),
    }


def _refine_pool(source, pool_index, neighbor_count):
    difference = source[pool_index] - source[:, None, :]
    squared = np.einsum("ijk,ijk->ij", difference, difference)
    chosen = np.argpartition(squared, neighbor_count - 1, axis=1)[:, :neighbor_count]
    return np.take_along_axis(pool_index, chosen, axis=1)


def prepare_pair_window(target_series, source_series, dimension, delay, neighbor_count, theiler_window):
    """Embeddings, self radii, and the source tree inputs for repeated surrogates."""
    (target, source), start, end = _align_two(target_series, source_series, dimension, delay, neighbor_count, theiler_window)
    self_neighbors, _ = neighbor_indices(target, neighbor_count, theiler_window)
    radius_self = _neighbor_radius(target, self_neighbors)
    radius_all = _mean_square_radius(target, theiler_window)
    return {
        "target": target,
        "source": source,
        "radius_self": radius_self,
        "radius_all": radius_all,
        "start": start,
        "end": end,
    }


def _align_two(target_series, source_series, dimension, delay, neighbor_count, theiler_window):
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    source, source_start = embed_signal(source_series, dimension[1], delay[1])
    start = max(target_start, source_start)
    end = min(target_start + len(target), source_start + len(source))
    if end - start < neighbor_count + 2 * theiler_window + 5:
        raise ValueError("Not enough overlapping samples for the neighbour search.")
    target = target[start - target_start : end - target_start]
    source = source[start - source_start : end - source_start]
    return (target, source), start, end


def indices_from_source_embedding(cache, source_embedding, neighbor_count, theiler_window):
    """Bivariate indices using a prepared target and a (possibly surrogate) source."""
    source_neighbors, _ = neighbor_indices(source_embedding, neighbor_count, theiler_window)
    radius_source = _neighbor_radius(cache["target"], source_neighbors)
    radius_all = cache["radius_all"]
    radius_self = cache["radius_self"]
    log_gap = np.log(np.maximum(radius_all, 1e-15)) - np.log(np.maximum(radius_source, 1e-15))
    return {
        "similarity_S": _average_ratio(radius_self, radius_source),
        "interdependence_H": float(np.nanmean(log_gap)),
        "normalised_interdependence_N": _average_ratio(radius_all - radius_source, radius_all),
        "andrzejak_M": _average_ratio(radius_all - radius_source, radius_all - radius_self),
    }


def prepare_conditional(
    target_series,
    conditioner_series,
    dimension,
    delay,
    neighbor_count,
    pool_size,
    theiler_window,
    source_dimension=1,
    source_delay=1,
):
    """Pools and radii that stay fixed while the candidate source is shuffled in time.

    ``source_dimension`` and ``source_delay`` keep the window inside the source
    embedding as well, so a source with a longer delay is not asked for states
    it does not have.
    """
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    conditioner, conditioner_start = embed_signal(conditioner_series, dimension[1], delay[1])
    source_start = (int(source_dimension) - 1) * int(source_delay)
    source_count = len(target_series) - source_start
    start = max(target_start, conditioner_start, source_start)
    end = min(target_start + len(target), conditioner_start + len(conditioner), source_start + source_count)
    if end - start < pool_size + 2 * theiler_window + 5:
        raise ValueError("Not enough overlapping samples for the conditional neighbour pool.")
    target = target[start - target_start : end - target_start]
    conditioner = conditioner[start - conditioner_start : end - conditioner_start]
    pool = max(int(pool_size), int(neighbor_count) + 1)
    conditioner_neighbors, _ = neighbor_indices(conditioner, pool, theiler_window)
    self_neighbors, _ = neighbor_indices(target, neighbor_count, theiler_window)
    radius_self = _neighbor_radius(target, self_neighbors)
    radius_conditioner = _neighbor_radius(target, conditioner_neighbors[:, :neighbor_count])
    return {
        "target": target,
        "pool_index": conditioner_neighbors,
        "radius_self": radius_self,
        "radius_conditioner": radius_conditioner,
        "start": start,
        "end": end,
        "target_start": target_start,
        "conditioner_start": conditioner_start,
    }


def conditional_from_source_embedding(cache, source_embedding, neighbor_count):
    """Conditional indices for one source embedding, given a prepared conditioner pool."""
    if len(source_embedding) != len(cache["target"]):
        raise ValueError("The source embedding is not on the same time window as the target.")
    refined = _refine_pool(source_embedding, cache["pool_index"], neighbor_count)
    radius_refined = _neighbor_radius(cache["target"], refined)
    radius_conditioner = cache["radius_conditioner"]
    radius_self = cache["radius_self"]
    improvement = radius_conditioner - radius_refined
    ratio = radius_conditioner / np.maximum(radius_refined, 1e-15)
    log_gap = np.log(np.maximum(radius_conditioner, 1e-15)) - np.log(np.maximum(radius_refined, 1e-15))
    return {
        "conditional_similarity_S": float(np.nanmean(ratio - 1.0)),
        "conditional_interdependence_H": float(np.nanmean(log_gap)),
        "conditional_normalised_interdependence_N": _average_ratio(improvement, radius_conditioner),
        "conditional_andrzejak_M": _average_ratio(improvement, radius_conditioner - radius_self),
    }


def embed_on_window(series, dimension, delay, start, end):
    """Embed a series and slice it to original-sample times ``start`` inclusive, ``end`` exclusive."""
    embedded, embedded_start = embed_signal(series, dimension, delay)
    row_start = start - embedded_start
    row_end = end - embedded_start
    if row_start < 0 or row_end > len(embedded):
        raise ValueError("The requested window is outside this signal's delay embedding.")
    return embedded[row_start:row_end]


def joint_interdependence(
    target_series,
    first_series,
    second_series,
    dimension,
    delay,
    neighbor_count,
    theiler_window,
):
    """Andrzejak M of the target given each driver, and given both drivers together.

    ``joint_gain`` is the two-driver index minus the better single-driver index.
    Positive means the two signals together match the target more closely than
    the better one alone.
    """
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    first, first_start = embed_signal(first_series, dimension[1], delay[1])
    second, second_start = embed_signal(second_series, dimension[2], delay[2])
    start = max(target_start, first_start, second_start)
    end = min(target_start + len(target), first_start + len(first), second_start + len(second))
    if end - start < neighbor_count + 2 * theiler_window + 5:
        raise ValueError("Not enough overlapping samples for the joint neighbour search.")
    target = target[start - target_start : end - target_start]
    first = first[start - first_start : end - first_start]
    second = second[start - second_start : end - second_start]
    self_neighbors, _ = neighbor_indices(target, neighbor_count, theiler_window)
    radius_self = _neighbor_radius(target, self_neighbors)
    radius_all = _mean_square_radius(target, theiler_window)

    def andrzejak(driver):
        neighbors, _ = neighbor_indices(driver, neighbor_count, theiler_window)
        radius = _neighbor_radius(target, neighbors)
        return _average_ratio(radius_all - radius, radius_all - radius_self)

    scaled_first, scaled_second = scale_blocks([first, second])
    together = np.hstack([scaled_first, scaled_second])
    from_first = andrzejak(first)
    from_second = andrzejak(second)
    from_both = andrzejak(together)
    return {
        "andrzejak_M_first": from_first,
        "andrzejak_M_second": from_second,
        "andrzejak_M_both": from_both,
        "joint_gain": from_both - max(from_first, from_second),
        "n_vectors": int(len(target)),
    }


def prediction_gain(
    target_series,
    source_series,
    conditioner_series,
    dimension,
    delay,
    neighbor_count,
    pool_size,
    theiler_window,
    horizons,
):
    """Forecast gain from adding the source inside a joint target–conditioner neighbourhood.

    Neighbours are the nearest states in the joint delay embedding of the target
    and the conditioner. The baseline forecast is the mean future of the closest
    of those neighbours. The refined forecast re-ranks the same pool by distance
    in the source and again takes the closest ``neighbor_count``. The gain at one
    horizon is 1 − (refined mean squared error) / (baseline mean squared error).
    The returned value is the largest gain across ``horizons``. Zero means the
    source does not improve the forecast. Positive means it does.
    """
    horizons = tuple(int(item) for item in horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("Forecast horizons must be positive sample counts.")
    reserve = max(horizons)
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    source, source_start = embed_signal(source_series, dimension[1], delay[1])
    conditioner, conditioner_start = embed_signal(conditioner_series, dimension[2], delay[2])
    start = max(target_start, source_start, conditioner_start)
    end = min(target_start + len(target), source_start + len(source), conditioner_start + len(conditioner))
    end -= reserve
    if end - start < pool_size + 2 * theiler_window + 5:
        raise ValueError("Not enough samples for a conditional nearest-neighbour forecast.")
    target = target[start - target_start : end - target_start]
    source = source[start - source_start : end - source_start]
    conditioner = conditioner[start - conditioner_start : end - conditioner_start]
    scaled_target, scaled_conditioner = scale_blocks([target, conditioner])
    joint = np.hstack([scaled_target, scaled_conditioner])
    pool = max(int(pool_size), int(neighbor_count) + 1)
    pool_index, _ = neighbor_indices(joint, pool, theiler_window)
    # Futures of the original (z-scored) target sample, not of the whole delay vector.
    zscored_target = zscore(target_series)
    gains = []
    for horizon in horizons:
        future = zscored_target[start + horizon : end + horizon]
        gains.append(_forecast_gain(future, pool_index, source, neighbor_count))
    best = int(np.nanargmax(gains))
    return {
        "prediction_gain": float(gains[best]),
        "prediction_horizon_samples": horizons[best],
        "prediction_gain_by_horizon": gains,
        "n_vectors": int(len(target)),
    }


def _forecast_gain(future, pool_index, source, neighbor_count):
    baseline = pool_index[:, :neighbor_count]
    refined = _refine_pool(source, pool_index, neighbor_count)

    def mean_squared_error(neighbors):
        prediction = future[neighbors].mean(axis=1)
        return float(np.mean(np.square(future - prediction)))

    baseline_error = mean_squared_error(baseline)
    refined_error = mean_squared_error(refined)
    if baseline_error <= 1e-15:
        return 0.0
    return 1.0 - refined_error / baseline_error


def prediction_gain_from_embeddings(future_by_horizon, pool_index, source_embedding, neighbor_count):
    """Same forecast gain, reusing a neighbour pool while the source changes."""
    gains = [
        _forecast_gain(future, pool_index, source_embedding, neighbor_count) for future in future_by_horizon
    ]
    best = int(np.nanargmax(gains))
    return float(gains[best]), best


def prepare_forecast(
    target_series,
    conditioner_series,
    dimension,
    delay,
    neighbor_count,
    pool_size,
    theiler_window,
    horizons,
    source_dimension=1,
    source_delay=1,
):
    """Neighbour pool of the joint target and conditioner, reused across source surrogates.

    ``dimension`` and ``delay`` are (target, conditioner). The source is embedded
    later onto the same original-sample window ``[start, end)``.
    """
    horizons = tuple(int(item) for item in horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("Forecast horizons must be positive sample counts.")
    reserve = max(horizons)
    target, target_start = embed_signal(target_series, dimension[0], delay[0])
    conditioner, conditioner_start = embed_signal(conditioner_series, dimension[1], delay[1])
    source_start = (int(source_dimension) - 1) * int(source_delay)
    source_count = len(target_series) - source_start
    start = max(target_start, conditioner_start, source_start)
    end = min(target_start + len(target), conditioner_start + len(conditioner), source_start + source_count) - reserve
    if end - start < pool_size + 2 * theiler_window + 5:
        raise ValueError("Not enough samples for a conditional nearest-neighbour forecast.")
    target = target[start - target_start : end - target_start]
    conditioner = conditioner[start - conditioner_start : end - conditioner_start]
    scaled_target, scaled_conditioner = scale_blocks([target, conditioner])
    joint = np.hstack([scaled_target, scaled_conditioner])
    pool = max(int(pool_size), int(neighbor_count) + 1)
    pool_index, _ = neighbor_indices(joint, pool, theiler_window)
    zscored_target = zscore(target_series)
    futures = [zscored_target[start + horizon : end + horizon] for horizon in horizons]
    return {
        "pool_index": pool_index,
        "futures": futures,
        "horizons": horizons,
        "start": start,
        "end": end,
    }


def forecast_from_source_embedding(cache, source_embedding, neighbor_count):
    """Best forecast gain across horizons for one source embedding."""
    gain, best = prediction_gain_from_embeddings(
        cache["futures"], cache["pool_index"], source_embedding, neighbor_count
    )
    return {
        "prediction_gain": gain,
        "prediction_horizon_samples": int(cache["horizons"][best]),
    }
