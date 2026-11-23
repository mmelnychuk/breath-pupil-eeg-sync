"""Ground-truth runs for the three-oscillator model.

The scenarios are the published model (no direct breathing–attention link),
the same model with that link added in both directions, and a model with
every coefficient at zero. A sweep then turns each directed coefficient on
by itself, from zero up to about twice the 0.33 reference.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sniff_sync.nlid.analyze import analyse_triplet, is_detected
from sniff_sync.nlid.model import (
    LINK_NAMES,
    OBSERVABLE_OF_LINK,
    Coupling,
    reference_epsilon,
    scenario_coupling,
    simulate_oscillators,
)

PHASE_NOISE = 0.55
SAMPLE_RATE_HZ = 10.0
HEADLINE_MEASURES = (
    "Andrzejak M",
    "conditional Andrzejak M",
    "direct-link index",
    "prediction gain",
)
OBSERVABLES = ("breathing", "pupil diameter", "attention")
LINK_OF_OBSERVABLE = {pair: name for name, pair in OBSERVABLE_OF_LINK.items()}


def scenario_duration(respiratory_hz):
    """At least 180 seconds, and at least about 50 respiratory cycles."""
    return max(180.0, 50.0 / float(respiratory_hz))


def embedding_for(respiratory_hz, sample_rate_hz=SAMPLE_RATE_HZ):
    """Quarter-cycle delay and dimension 3, with a Theiler window of one cycle.

    These slow oscillators are close to a closed loop, so three delays a
    quarter of a cycle apart are enough. The window stops a state being
    matched to the samples it was built from.
    """
    delay = max(1, int(round(sample_rate_hz / (4.0 * respiratory_hz))))
    theiler = max((3 - 1) * delay, int(round(sample_rate_hz / respiratory_hz)))
    embedding = {name: {"dimension": 3, "delay": delay} for name in OBSERVABLES}
    return embedding, theiler


def run_scenario(
    kind,
    respiratory_hz=0.25,
    seed=1,
    n_surrogates=19,
    phase_noise=PHASE_NOISE,
    include_prediction=False,
    include_joint=False,
    duration_s=None,
):
    """Simulate one scenario and return the analysis rows."""
    coupling = scenario_coupling(kind, respiratory_hz)
    duration_s = scenario_duration(respiratory_hz) if duration_s is None else float(duration_s)
    simulated = simulate_oscillators(
        coupling,
        respiratory_hz=respiratory_hz,
        duration_s=duration_s,
        sample_rate_hz=SAMPLE_RATE_HZ,
        phase_noise=phase_noise,
        seed=seed,
    )
    embedding, theiler = embedding_for(respiratory_hz)
    signals = {name: simulated[name] for name in OBSERVABLES}
    rows, _resolved = analyse_triplet(
        signals,
        sample_rate_hz=SAMPLE_RATE_HZ,
        attentional_signal="attention",
        embedding=embedding,
        neighbor_count=8,
        pool_size=40,
        theiler_window=theiler,
        n_surrogates=n_surrogates,
        surrogate="time-shift",
        seed=seed,
        include_prediction=include_prediction,
        include_joint=include_joint,
        include_partial_cross_map=True,
        minimum_shift_fraction=0.15,
    )
    return rows


def _third(source, target):
    return next(name for name in OBSERVABLES if name not in (source, target))


def _evaluate_cell(spec):
    """One sweep cell. Top-level so a process pool can call it."""
    link = spec["link"]
    epsilon = float(spec["epsilon"])
    coefficients = {name: 0.0 for name in LINK_NAMES}
    coefficients[link] = epsilon
    respiratory_hz = float(spec["respiratory_hz"])
    simulated = simulate_oscillators(
        Coupling(**coefficients),
        respiratory_hz=respiratory_hz,
        duration_s=float(spec["duration_s"]),
        sample_rate_hz=SAMPLE_RATE_HZ,
        phase_noise=float(spec["phase_noise"]),
        seed=int(spec["seed"]),
    )
    embedding, theiler = embedding_for(respiratory_hz)
    rows, _resolved = analyse_triplet(
        {name: simulated[name] for name in OBSERVABLES},
        sample_rate_hz=SAMPLE_RATE_HZ,
        attentional_signal="attention",
        embedding=embedding,
        neighbor_count=8,
        pool_size=40,
        theiler_window=theiler,
        n_surrogates=int(spec["n_surrogates"]),
        surrogate="time-shift",
        seed=int(spec["seed"]),
        include_prediction=False,
        include_joint=False,
        include_partial_cross_map=True,
        minimum_shift_fraction=0.15,
        emit_breathing_pupil_pair=True,
    )
    kept = []
    for row in rows:
        if row["measure"] not in ("Andrzejak M", "conditional Andrzejak M", "direct-link index"):
            continue
        if row["measure"] == "Andrzejak M" and row["conditioned_on"]:
            continue
        source, target = row["source"], row["target"]
        if (source, target) not in LINK_OF_OBSERVABLE:
            continue
        direction = LINK_OF_OBSERVABLE[(source, target)]
        detected = is_detected(row["value"], row["p_value"], spec["alpha"])
        kept.append(
            {
                "link_varied": link,
                "epsilon": epsilon,
                "seed": int(spec["seed"]),
                "measure": row["measure"],
                "direction": direction,
                "value": float(row["value"]) if row["value"] == row["value"] else float("nan"),
                "p_value": float(row["p_value"]) if row["p_value"] == row["p_value"] else float("nan"),
                "detected": bool(detected),
                "absent": direction != link or epsilon == 0.0,
                "pure_null": epsilon == 0.0,
            }
        )
    return kept


# A link is called reliably detected only when this fraction of seeds passes.
# The full rate at every coefficient is always reported as well. 0.80 was
# chosen before looking at the sweep, as a pre-set bar for "usually detected",
# not as a bar the results were later required to clear.
RELIABLE_DETECTION_RATE = 0.80


def _proportion(records):
    count = len(records)
    if count == 0:
        return {"rate": float("nan"), "detected": 0, "tested": 0}
    detected = sum(1 for record in records if record["detected"])
    return {"rate": detected / count, "detected": detected, "tested": count}


def coupling_sweep(
    epsilons=(0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9),
    respiratory_hz=0.25,
    n_surrogates=19,
    phase_noise=PHASE_NOISE,
    duration_s=120.0,
    seeds=range(20),
    alpha=0.05,
    workers=4,
):
    """Vary one directed coefficient at a time, on every seed.

    Every other coefficient stays zero. Detection rate is the proportion of
    seeds on which that direction is detected. The false-positive rate is the
    proportion of tests whose coefficient was zero and which were still
    detected. "Starts detecting" means the smallest positive coefficient whose
    detection rate is at least ``RELIABLE_DETECTION_RATE``. If none is, the
    link is reported as not reaching that rate.
    """
    seeds = [int(seed) for seed in seeds]
    specs = []
    for seed in seeds:
        for link in LINK_NAMES:
            for epsilon in epsilons:
                specs.append(
                    {
                        "link": link,
                        "epsilon": float(epsilon),
                        "respiratory_hz": float(respiratory_hz),
                        "n_surrogates": int(n_surrogates),
                        "phase_noise": float(phase_noise),
                        "duration_s": float(duration_s),
                        "seed": seed,
                        "alpha": float(alpha),
                    }
                )
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            nested = list(pool.map(_evaluate_cell, specs, chunksize=2))
    else:
        nested = [_evaluate_cell(spec) for spec in specs]
    records = [item for batch in nested for item in batch]
    summary = {}
    for measure in ("Andrzejak M", "conditional Andrzejak M", "direct-link index"):
        curves = {}
        thresholds = {}
        for link in LINK_NAMES:
            curve = []
            threshold = None
            for epsilon in epsilons:
                matched = [
                    record
                    for record in records
                    if record["measure"] == measure
                    and record["link_varied"] == link
                    and record["direction"] == link
                    and record["epsilon"] == float(epsilon)
                ]
                proportion = _proportion(matched)
                proportion["epsilon"] = float(epsilon)
                curve.append(proportion)
                if float(epsilon) > 0.0 and threshold is None and proportion["rate"] >= RELIABLE_DETECTION_RATE:
                    threshold = float(epsilon)
            curves[link] = curve
            thresholds[link] = threshold
        absent = [record for record in records if record["measure"] == measure and record["absent"]]
        pure_null = [record for record in absent if record["pure_null"]]
        while_other_on = [record for record in absent if not record["pure_null"]]
        summary[measure] = {
            "curve": curves,
            "threshold_at_80_percent": thresholds,
            "false_positive_rate": _proportion(absent)["rate"],
            "false_positives": _proportion(absent)["detected"],
            "absent_tests": _proportion(absent)["tested"],
            "pure_null": _proportion(pure_null),
            "absent_while_other_link_on": _proportion(while_other_on),
            "reference_epsilon": reference_epsilon(respiratory_hz),
        }
    return {
        "records": records,
        "summary": summary,
        "epsilons": [float(value) for value in epsilons],
        "seeds": seeds,
        "n_seeds": len(seeds),
    }


def format_sweep(result):
    """Detection rate at each coefficient, and false-positive rates, across seeds."""
    summary = result["summary"] if "summary" in result and "curve" in next(iter(result["summary"].values())) else result
    # Accept either the full return value or the summary dict.
    if "summary" in result and "n_seeds" in result:
        summary = result["summary"]
        n_seeds = result["n_seeds"]
    else:
        n_seeds = None
    lines = []
    if n_seeds is not None:
        lines.append("Coupling sweep across %d seeds. Other coefficients are zero." % n_seeds)
        lines.append(
            "A link 'starts detecting' at the smallest positive coefficient where the "
            "detection rate reaches %.0f%%. Rates below that are reported as well."
            % (100 * RELIABLE_DETECTION_RATE)
        )
    for measure, info in summary.items():
        lines.append(
            "%s: false-positive rate %.3f (%d / %d absent-link tests)"
            % (measure, info["false_positive_rate"], info["false_positives"], info["absent_tests"])
        )
        if "pure_null" in info:
            pure = info["pure_null"]
            other = info["absent_while_other_link_on"]
            lines.append(
                "  of which the all-zero model is %.3f (%d / %d), and an absent direction while one other link is on is %.3f (%d / %d)"
                % (
                    pure["rate"],
                    pure["detected"],
                    pure["tested"],
                    other["rate"],
                    other["detected"],
                    other["tested"],
                )
            )
        curves = info.get("curve") or {}
        thresholds = info.get("threshold_at_80_percent") or info.get("threshold") or {}
        for link, curve in curves.items():
            source, target = OBSERVABLE_OF_LINK[link]
            lines.append("  %s → %s" % (source, target))
            for point in curve:
                lines.append(
                    "    coefficient %.2f: detection rate %.3f (%d / %d seeds)"
                    % (point["epsilon"], point["rate"], point["detected"], point["tested"])
                )
            threshold = thresholds.get(link)
            if threshold is None:
                lines.append("    does not reach %.0f%% detection by the top of the sweep" % (100 * RELIABLE_DETECTION_RATE))
            else:
                lines.append("    reaches %.0f%% detection at coefficient %.2f" % (100 * RELIABLE_DETECTION_RATE, threshold))
    return "\n".join(lines)


SCENARIO_MEASURES = (
    "Andrzejak M",
    "conditional Andrzejak M",
    "direct-link index",
    "prediction gain",
)


def _scenario_job(spec):
    """One seed of one scenario. Returns detection flags, not the full table."""
    rows = run_scenario(
        spec["kind"],
        respiratory_hz=spec["frequency"],
        seed=spec["seed"],
        n_surrogates=spec["n_surrogates"],
        phase_noise=spec["phase_noise"],
        include_prediction=True,
        include_joint=False,
    )
    kept = []
    for row in rows:
        if row["measure"] not in SCENARIO_MEASURES:
            continue
        if row["measure"] == "Andrzejak M" and row["conditioned_on"]:
            continue
        value = row["value"]
        p_value = row["p_value"]
        kept.append(
            {
                "kind": spec["kind"],
                "frequency": float(spec["frequency"]),
                "seed": int(spec["seed"]),
                "measure": row["measure"],
                "source": row["source"],
                "target": row["target"],
                "conditioned_on": row["conditioned_on"] or "",
                "value": float(value) if value == value else float("nan"),
                "p_value": float(p_value) if p_value == p_value else float("nan"),
                "detected": bool(is_detected(value, p_value, spec["alpha"])),
            }
        )
    return kept


def scenario_detection_rates(
    frequencies=(0.12, 0.25, 0.35),
    kinds=("published", "direct", "uncoupled"),
    seeds=range(20),
    n_surrogates=19,
    phase_noise=PHASE_NOISE,
    alpha=0.05,
    workers=4,
):
    """Detection rate of every directed test, as a proportion of seeds.

    Seeds are 0, 1, 2, ... in order. No seed is dropped because its result
    was inconvenient. A detection is a positive index with p ≤ alpha.
    """
    seeds = [int(seed) for seed in seeds]
    specs = []
    for kind in kinds:
        for frequency in frequencies:
            for seed in seeds:
                specs.append(
                    {
                        "kind": kind,
                        "frequency": float(frequency),
                        "seed": seed,
                        "n_surrogates": int(n_surrogates),
                        "phase_noise": float(phase_noise),
                        "alpha": float(alpha),
                    }
                )
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            nested = list(pool.map(_scenario_job, specs, chunksize=1))
    else:
        nested = [_scenario_job(spec) for spec in specs]
    records = [item for batch in nested for item in batch]
    cells = {}
    for record in records:
        key = (
            record["kind"],
            record["frequency"],
            record["measure"],
            record["source"],
            record["target"],
            record["conditioned_on"],
        )
        cells.setdefault(key, []).append(record)
    rates = {}
    for key, group in cells.items():
        proportion = _proportion(group)
        finite = [record["value"] for record in group if record["value"] == record["value"]]
        proportion["mean_value"] = float(np.mean(finite)) if finite else float("nan")
        rates[key] = proportion
    return {
        "rates": rates,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_surrogates": int(n_surrogates),
        "phase_noise": float(phase_noise),
        "alpha": float(alpha),
        "frequencies": [float(value) for value in frequencies],
        "kinds": list(kinds),
    }


def format_scenario_rates(result):
    """One line per direction: detection rate and the mean index across seeds."""
    lines = [
        "Scenario detection rates across %d seeds (phase noise %.2f, %d time-shift surrogates, alpha %.2f)."
        % (result["n_seeds"], result["phase_noise"], result["n_surrogates"], result["alpha"]),
        "Detected means a positive index and p ≤ alpha. Rates are detected seeds / seeds.",
    ]
    order = (
        ("breathing", "pupil diameter", "attention"),
        ("pupil diameter", "breathing", "attention"),
        ("pupil diameter", "attention", "breathing"),
        ("attention", "pupil diameter", "breathing"),
        ("breathing", "attention", "pupil diameter"),
        ("attention", "breathing", "pupil diameter"),
    )
    for kind in result["kinds"]:
        for frequency in result["frequencies"]:
            lines.append(
                "%s at %.2f Hz (reference coefficient %.2f)"
                % (kind, frequency, reference_epsilon(frequency))
            )
            for measure in SCENARIO_MEASURES:
                lines.append("  %s" % measure)
                for source, target, conditioner in order:
                    conditioned_on = "" if measure == "Andrzejak M" else conditioner
                    key = (kind, float(frequency), measure, source, target, conditioned_on)
                    proportion = result["rates"].get(key)
                    if proportion is None:
                        lines.append("    %s → %s: not computed" % (source, target))
                        continue
                    given = (" given %s" % conditioned_on) if conditioned_on else ""
                    lines.append(
                        "    %s → %s%s: rate %.3f (%d / %d), mean index %.3f"
                        % (
                            source,
                            target,
                            given,
                            proportion["rate"],
                            proportion["detected"],
                            proportion["tested"],
                            proportion["mean_value"],
                        )
                    )
    return "\n".join(lines)


def format_scenario(kind, respiratory_hz, rows, measures=None):
    measures = measures or ("Andrzejak M", "conditional Andrzejak M", "direct-link index")
    lines = ["%s at %.2f Hz (reference coefficient %.2f)" % (kind, respiratory_hz, reference_epsilon(respiratory_hz))]
    for row in rows:
        if row["measure"] not in measures:
            continue
        if row["measure"] == "Andrzejak M" and row["conditioned_on"]:
            continue
        flag = "detected" if is_detected(row["value"], row["p_value"], 0.05) else "not detected"
        given = (" given %s" % row["conditioned_on"]) if row["conditioned_on"] else ""
        value = row["value"]
        p_value = row["p_value"]
        value_text = "nan" if not np.isfinite(value) else "%.3f" % value
        p_text = "nan" if not np.isfinite(p_value) else "%.3f" % p_value
        lines.append(
            "  %-28s %s → %s%s: %s, p=%s, %s"
            % (row["measure"], row["source"], row["target"], given, value_text, p_text, flag)
        )
    return "\n".join(lines)
