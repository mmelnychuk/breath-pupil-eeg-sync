"""Run the coupling measures on breathing, pupil diameter, and one attentional signal.

An attentional signal is either a band's smoothed amplitude or a ratio of two
bands' smoothed power. The recording entry point loops over those candidates.
This module works on arrays that are already on one uniform time grid.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sniff_sync.nlid.embed import choose_delay, choose_dimension
from sniff_sync.nlid.measures import (
    conditional_from_source_embedding,
    embed_on_window,
    forecast_from_source_embedding,
    indices_from_source_embedding,
    joint_interdependence,
    prepare_conditional,
    prepare_forecast,
    prepare_pair_window,
)
from sniff_sync.nlid.pcm import PartialCrossMapper
from sniff_sync.nlid.surrogates import (
    benjamini_hochberg,
    bonferroni,
    iaaft,
    p_value_greater,
    phase_randomize,
    shift_amounts,
)

BIVARIATE_MEASURES = (
    ("similarity_S", "similarity S"),
    ("interdependence_H", "interdependence H"),
    ("normalised_interdependence_N", "normalised interdependence N"),
    ("andrzejak_M", "Andrzejak M"),
)

CONDITIONAL_MEASURES = (
    ("conditional_similarity_S", "conditional similarity S"),
    ("conditional_interdependence_H", "conditional interdependence H"),
    ("conditional_normalised_interdependence_N", "conditional normalised interdependence N"),
    ("conditional_andrzejak_M", "conditional Andrzejak M"),
)

HEADLINE_BIVARIATE = "Andrzejak M"
HEADLINE_CONDITIONAL = "conditional Andrzejak M"
HEADLINE_PARTIAL = "direct-link index"


def is_detected(value, p_value, alpha):
    """A link counts as detected when the index is positive and the one-sided p-value is small.

    Positivity matters for the conditional indices. Their surrogate distribution
    sits below zero, because a time-shifted source is actively worse than the
    conditioner's own neighbours. A negative index that merely beats that
    baseline is not evidence that the source improved the target.
    """
    if value is None or p_value is None:
        return False
    if not np.isfinite(value) or not np.isfinite(p_value):
        return False
    return float(value) > 0.0 and float(p_value) <= float(alpha)


def choose_embedding(series, sample_rate_hz, max_dimension=6):
    """Delay from mutual information, dimension from false nearest neighbours."""
    max_lag = max(2, min(len(series) // 4, int(round(2.5 * sample_rate_hz))))
    delay, _scores = choose_delay(series, max_lag=max_lag)
    dimension, fractions = choose_dimension(series, delay, max_dimension=max_dimension)
    return {"dimension": int(dimension), "delay": int(delay), "false_nearest_fractions": fractions}


def _surrogate_series(series, kind, count, minimum_shift, generator):
    kind = kind.strip().lower().replace("_", "-")
    if kind in ("time-shift", "timeshift", "shift"):
        shifts = shift_amounts(len(series), count, minimum_shift, generator)
        return [np.roll(series, int(shift)) for shift in shifts]
    if kind in ("phase-random", "phase-randomised", "phase-randomized", "fourier"):
        return [phase_randomize(series, generator) for _ in range(count)]
    if kind in ("iaaft", "amplitude-adjusted"):
        return [iaaft(series, generator) for _ in range(count)]
    raise ValueError(
        "Surrogate kind must be time-shift, phase-random, or iaaft, not %r." % kind
    )


def _base_row(method, measure, source, target, conditioned_on, attentional_signal, value, p_value, meta):
    return {
        "method": method,
        "measure": measure,
        "source": source,
        "target": target,
        "conditioned_on": conditioned_on,
        "attentional_signal": attentional_signal,
        "value": value,
        "p_value": p_value,
        "p_value_corrected": "",
        "significant": "",
        "surrogate": meta["surrogate"],
        "surrogate_count": meta["surrogate_count"],
        "correction": meta["correction"],
        "alpha": meta["alpha"],
        "sample_rate_hz": meta["sample_rate_hz"],
        "n_samples": meta["n_samples"],
        "neighbors": meta["neighbors"],
        "conditioning_pool": meta["conditioning_pool"],
        "theiler_window_samples": meta["theiler_window_samples"],
        "theiler_window_seconds": meta["theiler_window_samples"] / meta["sample_rate_hz"],
        "embedding_delay_source": meta["embedding"][source]["delay"] if source in meta["embedding"] else "",
        "embedding_dimension_source": meta["embedding"][source]["dimension"] if source in meta["embedding"] else "",
        "embedding_delay_target": meta["embedding"][target]["delay"] if target in meta["embedding"] else "",
        "embedding_dimension_target": meta["embedding"][target]["dimension"] if target in meta["embedding"] else "",
        "embedding_delay_conditioner": (
            meta["embedding"][conditioned_on]["delay"] if conditioned_on in meta["embedding"] else ""
        ),
        "embedding_dimension_conditioner": (
            meta["embedding"][conditioned_on]["dimension"] if conditioned_on in meta["embedding"] else ""
        ),
        "prediction_horizon_samples": "",
    }


def analyse_triplet(
    signals,
    sample_rate_hz,
    attentional_signal="",
    embedding=None,
    neighbor_count=10,
    pool_size=None,
    theiler_window=None,
    n_surrogates=39,
    surrogate="time-shift",
    alpha=0.05,
    minimum_shift_fraction=0.10,
    prediction_horizons_s=(0.1, 0.2, 0.4, 0.8, 1.2),
    max_cross_map_lag_s=2.0,
    seed=0,
    emit_breathing_pupil_pair=True,
    include_prediction=True,
    include_joint=True,
    include_partial_cross_map=True,
):
    """Every ordered pair among three named series, with surrogates.

    ``signals`` maps a plain-language name to a one-dimensional array. The three
    names are arbitrary; the recording pipeline uses "breathing", "pupil diameter",
    and the name of the attentional candidate. ``embedding`` maps a name to
    ``{"dimension", "delay"}``. Missing names are chosen from the data.
    """
    names = list(signals)
    if len(names) != 3:
        raise ValueError("This analysis takes exactly three signals, got %d." % len(names))
    arrays = {name: np.asarray(signals[name], dtype=float) for name in names}
    n_samples = len(next(iter(arrays.values())))
    if any(len(values) != n_samples for values in arrays.values()):
        raise ValueError("The three signals must share one time grid and the same length.")
    if n_samples < 80:
        raise ValueError(
            "Only %d samples are left after alignment. State-space neighbours need a longer overlap."
            % n_samples
        )
    sample_rate_hz = float(sample_rate_hz)
    generator = np.random.default_rng(seed)
    resolved = {}
    for name, values in arrays.items():
        if embedding and name in embedding and embedding[name] is not None:
            spec = embedding[name]
            resolved[name] = {"dimension": int(spec["dimension"]), "delay": int(spec["delay"])}
        else:
            resolved[name] = choose_embedding(values, sample_rate_hz)
    if theiler_window is None:
        theiler_window = max((item["dimension"] - 1) * item["delay"] for item in resolved.values())
    theiler_window = max(1, int(theiler_window))
    neighbor_count = int(neighbor_count)
    pool_size = int(pool_size) if pool_size else max(5 * neighbor_count, neighbor_count + 1)
    minimum_shift = max(theiler_window + 1, int(n_samples * float(minimum_shift_fraction)))
    horizons = sorted({max(1, int(round(float(seconds) * sample_rate_hz))) for seconds in prediction_horizons_s})
    max_lag = max(1, int(round(float(max_cross_map_lag_s) * sample_rate_hz)))
    max_lag = min(max_lag, max(1, n_samples // 5))
    meta = {
        "surrogate": surrogate,
        "surrogate_count": int(n_surrogates),
        "correction": "none",
        "alpha": float(alpha),
        "sample_rate_hz": sample_rate_hz,
        "n_samples": n_samples,
        "neighbors": neighbor_count,
        "conditioning_pool": pool_size,
        "theiler_window_samples": theiler_window,
        "embedding": resolved,
    }
    rows = []
    attention_names = set()
    if attentional_signal:
        attention_names.add(attentional_signal)

    def dim_delay(name):
        return resolved[name]["dimension"], resolved[name]["delay"]

    surrogates = {
        name: _surrogate_series(arrays[name], surrogate, int(n_surrogates), minimum_shift, generator)
        for name in names
    }

    # Bivariate nearest-neighbour indices.
    for source in names:
        for target in names:
            if source == target:
                continue
            involves_attention = source in attention_names or target in attention_names or not attention_names
            breathing_pupil = attention_names and source not in attention_names and target not in attention_names
            if breathing_pupil and not emit_breathing_pupil_pair:
                continue
            source_dim, source_delay = dim_delay(source)
            target_dim, target_delay = dim_delay(target)
            cache = prepare_pair_window(
                arrays[target],
                arrays[source],
                (target_dim, source_dim),
                (target_delay, source_delay),
                neighbor_count,
                theiler_window,
            )
            observed = indices_from_source_embedding(cache, cache["source"], neighbor_count, theiler_window)
            null_rows = []
            for copy in surrogates[source]:
                copied = embed_on_window(copy, source_dim, source_delay, cache["start"], cache["end"])
                null_rows.append(indices_from_source_embedding(cache, copied, neighbor_count, theiler_window))
            signal_label = "" if breathing_pupil else attentional_signal
            for key, label in BIVARIATE_MEASURES:
                nulls = [item[key] for item in null_rows]
                rows.append(
                    _base_row(
                        "nearest-neighbour interdependence",
                        label,
                        source,
                        target,
                        "",
                        signal_label,
                        observed[key],
                        p_value_greater(observed[key], nulls),
                        meta,
                    )
                )

    # Conditional indices, forecast gain, and partial cross mapping.
    for source in names:
        for target in names:
            if source == target:
                continue
            conditioner = next(name for name in names if name not in (source, target))
            target_dim, target_delay = dim_delay(target)
            source_dim, source_delay = dim_delay(source)
            conditioner_dim, conditioner_delay = dim_delay(conditioner)
            cache = prepare_conditional(
                arrays[target],
                arrays[conditioner],
                (target_dim, conditioner_dim),
                (target_delay, conditioner_delay),
                neighbor_count,
                pool_size,
                theiler_window,
                source_dimension=source_dim,
                source_delay=source_delay,
            )
            observed_source = embed_on_window(arrays[source], source_dim, source_delay, cache["start"], cache["end"])
            observed = conditional_from_source_embedding(cache, observed_source, neighbor_count)
            null_rows = []
            for copy in surrogates[source]:
                copied = embed_on_window(copy, source_dim, source_delay, cache["start"], cache["end"])
                null_rows.append(conditional_from_source_embedding(cache, copied, neighbor_count))
            for key, label in CONDITIONAL_MEASURES:
                nulls = [item[key] for item in null_rows]
                rows.append(
                    _base_row(
                        "conditional nearest-neighbour interdependence",
                        label,
                        source,
                        target,
                        conditioner,
                        attentional_signal,
                        observed[key],
                        p_value_greater(observed[key], nulls),
                        meta,
                    )
                )

            if include_prediction:
                forecast = prepare_forecast(
                    arrays[target],
                    arrays[conditioner],
                    (target_dim, conditioner_dim),
                    (target_delay, conditioner_delay),
                    neighbor_count,
                    pool_size,
                    theiler_window,
                    horizons,
                    source_dimension=source_dim,
                    source_delay=source_delay,
                )
                observed_source = embed_on_window(
                    arrays[source], source_dim, source_delay, forecast["start"], forecast["end"]
                )
                observed_gain = forecast_from_source_embedding(forecast, observed_source, neighbor_count)
                null_gains = []
                null_horizons = []
                for copy in surrogates[source]:
                    copied = embed_on_window(copy, source_dim, source_delay, forecast["start"], forecast["end"])
                    scored = forecast_from_source_embedding(forecast, copied, neighbor_count)
                    null_gains.append(scored["prediction_gain"])
                    null_horizons.append(scored["prediction_horizon_samples"])
                row = _base_row(
                    "conditional nearest-neighbour forecast",
                    "prediction gain",
                    source,
                    target,
                    conditioner,
                    attentional_signal,
                    observed_gain["prediction_gain"],
                    p_value_greater(observed_gain["prediction_gain"], null_gains),
                    meta,
                )
                row["prediction_horizon_samples"] = observed_gain["prediction_horizon_samples"]
                rows.append(row)

            if include_partial_cross_map:
                mapper = PartialCrossMapper(
                    arrays[target],
                    arrays[conditioner],
                    (target_dim, conditioner_dim),
                    (target_delay, conditioner_delay),
                    theiler_window,
                    max_lag,
                )
                observed_map = mapper.evaluate(arrays[source])
                null_correlation = []
                null_direct = []
                for copy in surrogates[source]:
                    scored = mapper.evaluate(copy)
                    null_correlation.append(scored["cross_map_correlation"])
                    null_direct.append(scored["direct_link_index"])
                rows.append(
                    _base_row(
                        "partial cross mapping",
                        "cross-map correlation",
                        source,
                        target,
                        conditioner,
                        attentional_signal,
                        observed_map["cross_map_correlation"],
                        p_value_greater(observed_map["cross_map_correlation"], null_correlation),
                        meta,
                    )
                )
                rows.append(
                    _base_row(
                        "partial cross mapping",
                        "direct-link index",
                        source,
                        target,
                        conditioner,
                        attentional_signal,
                        observed_map["direct_link_index"],
                        p_value_greater(observed_map["direct_link_index"], null_direct),
                        meta,
                    )
                )

    if include_joint:
        for target in names:
            drivers = [name for name in names if name != target]
            dims = [dim_delay(target)[0], dim_delay(drivers[0])[0], dim_delay(drivers[1])[0]]
            delays = [dim_delay(target)[1], dim_delay(drivers[0])[1], dim_delay(drivers[1])[1]]
            observed = joint_interdependence(
                arrays[target],
                arrays[drivers[0]],
                arrays[drivers[1]],
                dims,
                delays,
                neighbor_count,
                theiler_window,
            )
            # Shift both drivers together, so their mutual timing stays and only
            # their alignment with the target is broken.
            shifts = shift_amounts(n_samples, int(n_surrogates), minimum_shift, generator)
            null_gains = []
            null_both = []
            for shift in shifts:
                scored = joint_interdependence(
                    arrays[target],
                    np.roll(arrays[drivers[0]], int(shift)),
                    np.roll(arrays[drivers[1]], int(shift)),
                    dims,
                    delays,
                    neighbor_count,
                    theiler_window,
                )
                null_gains.append(scored["joint_gain"])
                null_both.append(scored["andrzejak_M_both"])
            both_names = "%s and %s" % (drivers[0], drivers[1])
            row = _base_row(
                "joint nearest-neighbour interdependence",
                "joint gain of Andrzejak M",
                both_names,
                target,
                "",
                attentional_signal,
                observed["joint_gain"],
                p_value_greater(observed["joint_gain"], null_gains),
                meta,
            )
            rows.append(row)
            row = _base_row(
                "joint nearest-neighbour interdependence",
                "Andrzejak M from both",
                both_names,
                target,
                "",
                attentional_signal,
                observed["andrzejak_M_both"],
                p_value_greater(observed["andrzejak_M_both"], null_both),
                meta,
            )
            rows.append(row)
    return rows, resolved


def apply_correction(rows, correction, alpha):
    """Correct p-values within each measure, across every direction and attentional signal."""
    correction = (correction or "none").strip().lower()
    families = {}
    for index, row in enumerate(rows):
        families.setdefault(row["measure"], []).append(index)
    for measure, indices in families.items():
        raw = np.array([rows[index]["p_value"] for index in indices], dtype=float)
        if correction in ("none", ""):
            adjusted = raw
            rejected = np.array([is_detected(rows[index]["value"], rows[index]["p_value"], alpha) for index in indices])
            for index, flag in zip(indices, rejected):
                rows[index]["correction"] = "none"
                rows[index]["p_value_corrected"] = ""
                rows[index]["significant"] = "yes" if flag else "no"
            continue
        if correction in ("bonferroni",):
            adjusted, _rejected = bonferroni(raw, alpha)
        elif correction in ("fdr", "benjamini-hochberg", "bh"):
            adjusted, _rejected = benjamini_hochberg(raw, alpha)
        else:
            raise ValueError("Correction must be none, bonferroni, or fdr, not %r." % correction)
        for index, q_value in zip(indices, adjusted):
            rows[index]["correction"] = correction
            rows[index]["p_value_corrected"] = q_value
            rows[index]["significant"] = "yes" if is_detected(rows[index]["value"], q_value, alpha) else "no"
    return rows


def write_results_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("There are no results to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stored = {}
            for key, value in row.items():
                if isinstance(value, float):
                    stored[key] = "" if not np.isfinite(value) else "%.6g" % value
                else:
                    stored[key] = value
            writer.writerow(stored)
    return path


def lookup(rows, measure, source, target, conditioned_on=""):
    """Return the first matching row, or None."""
    for row in rows:
        if row["measure"] != measure:
            continue
        if row["source"] != source or row["target"] != target:
            continue
        if (row["conditioned_on"] or "") != (conditioned_on or ""):
            continue
        return row
    return None
