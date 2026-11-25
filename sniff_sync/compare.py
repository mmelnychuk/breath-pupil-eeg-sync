"""Compare live coupling scores with the post-session reference.

The live file is approximate. It uses the short linear-phase filters.
The reference file uses forward-backward filtering of the whole recording.
This command lines the two up by window end time and reports, for every
directed link, how well the scores agree.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from sniff_sync.coupling import (
    APPROXIMATE_NAME,
    PAIRS,
    SIGNAL_NAMES,
    _arrow_column,
)

REFERENCE_NAME = "coupling_reference.csv"
COMPARISON_TABLE = "coupling_comparison.csv"
COMPARISON_FIGURE = "coupling_comparison.png"
MEASURES = (
    ("interdependence", "Conditional nonlinear interdependence"),
    ("forecast_gain", "Forward forecast gain"),
)
TIE = 1e-6


def compare_session(session):
    """Write the comparison table and figure. Return the summary.

    If the reference file is not in the session yet, the reference pass is
    run first. The approximate file must already exist.
    """
    root = Path(session)
    approximate = root / APPROXIMATE_NAME
    if not approximate.is_file():
        raise ValueError(
            "No approximate coupling file in %s. Run a live session first so the window times exist."
            % root
        )
    reference = root / REFERENCE_NAME
    if not reference.is_file():
        from sniff_sync.reference import write_reference

        print("The reference file is not in the session yet. Writing it first.")
        write_reference(root)
    matched = _matched_windows(_load(approximate), _load(reference))
    if not matched:
        raise ValueError(
            "No window end time is ready in both %s and %s." % (APPROXIMATE_NAME, REFERENCE_NAME)
        )
    summary = _summarise(matched)
    _write_table(root / COMPARISON_TABLE, summary)
    _write_figure(root / COMPARISON_FIGURE, matched, summary)
    summary["table"] = str(root / COMPARISON_TABLE)
    summary["figure"] = str(root / COMPARISON_FIGURE)
    summary["windows"] = len(matched)
    return summary


def _load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _end_key(row):
    text = (row.get("window_end_unix") or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return round(value, 6)


def _matched_windows(approximate_rows, reference_rows):
    reference_by_end = {}
    for row in reference_rows:
        if (row.get("status") or "") != "ready":
            continue
        key = _end_key(row)
        if key is not None:
            reference_by_end[key] = row
    matched = []
    for row in approximate_rows:
        if (row.get("status") or "") != "ready":
            continue
        key = _end_key(row)
        other = reference_by_end.get(key)
        if other is not None:
            matched.append((row, other))
    matched.sort(key=lambda pair: _end_key(pair[0]))
    return matched


def _cell(row, measure, source, target):
    text = (row.get(_arrow_column(measure, source, target)) or "").strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return value


def _correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(finite)) < 2:
        return float("nan")
    if float(np.std(left[finite])) < 1e-12 or float(np.std(right[finite])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _median_absolute_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return float("nan")
    return float(np.median(np.abs(left[finite] - right[finite])))


def _dominant(forward, backward):
    if not math.isfinite(forward) or not math.isfinite(backward):
        return None
    margin = forward - backward
    if abs(margin) <= TIE:
        return "neither"
    if margin > 0:
        return "forward"
    return "backward"


def _summarise(matched):
    links = []
    pairs = []
    for measure, title in MEASURES:
        for source in SIGNAL_NAMES:
            for target in SIGNAL_NAMES:
                if source == target:
                    continue
                live = [_cell(row, measure, source, target) for row, _other in matched]
                reference = [_cell(other, measure, source, target) for _row, other in matched]
                links.append(
                    {
                        "measure": measure,
                        "measure_title": title,
                        "source": source,
                        "target": target,
                        "windows": int(np.sum(np.isfinite(live) & np.isfinite(reference))),
                        "correlation": _correlation(live, reference),
                        "median_absolute_difference": _median_absolute_difference(live, reference),
                    }
                )
        for left, right in PAIRS:
            agreed = 0
            counted = 0
            for live_row, reference_row in matched:
                live_label = _dominant(
                    _cell(live_row, measure, left, right),
                    _cell(live_row, measure, right, left),
                )
                reference_label = _dominant(
                    _cell(reference_row, measure, left, right),
                    _cell(reference_row, measure, right, left),
                )
                if live_label is None or reference_label is None:
                    continue
                counted += 1
                if live_label == reference_label:
                    agreed += 1
            pairs.append(
                {
                    "measure": measure,
                    "measure_title": title,
                    "left": left,
                    "right": right,
                    "windows": counted,
                    "direction_matches": agreed,
                    "direction_match_fraction": (agreed / counted) if counted else float("nan"),
                }
            )
    return {"links": links, "pairs": pairs}


def _write_table(path, summary):
    columns = [
        "kind",
        "measure",
        "source",
        "target",
        "windows",
        "correlation",
        "median_absolute_difference",
        "direction_matches",
        "direction_match_fraction",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in summary["links"]:
            writer.writerow(
                {
                    "kind": "link",
                    "measure": item["measure"],
                    "source": item["source"],
                    "target": item["target"],
                    "windows": item["windows"],
                    "correlation": _number(item["correlation"]),
                    "median_absolute_difference": _number(item["median_absolute_difference"]),
                    "direction_matches": "",
                    "direction_match_fraction": "",
                }
            )
        for item in summary["pairs"]:
            writer.writerow(
                {
                    "kind": "pair",
                    "measure": item["measure"],
                    "source": item["left"],
                    "target": item["right"],
                    "windows": item["windows"],
                    "correlation": "",
                    "median_absolute_difference": "",
                    "direction_matches": item["direction_matches"],
                    "direction_match_fraction": _number(item["direction_match_fraction"]),
                }
            )


def _number(value):
    if value is None or not math.isfinite(float(value)):
        return ""
    return "%.6f" % float(value)


def _write_figure(path, matched, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ends = [_end_key(row) for row, _other in matched]
    origin = ends[0] if ends else 0.0
    times = [end - origin for end in ends]
    arrows = [
        (source, target)
        for source in SIGNAL_NAMES
        for target in SIGNAL_NAMES
        if source != target
    ]
    figure, axes = plt.subplots(len(arrows), len(MEASURES), figsize=(11, 14), sharex=True)
    for column, (measure, title) in enumerate(MEASURES):
        for row_index, (source, target) in enumerate(arrows):
            axis = axes[row_index, column]
            live = [_cell(row, measure, source, target) for row, _other in matched]
            reference = [_cell(other, measure, source, target) for _row, other in matched]
            axis.plot(times, live, color="#c47b2b", linewidth=1.4, label="Live")
            axis.plot(times, reference, color="#2f6f9f", linestyle="--", linewidth=1.2, label="Reference")
            axis.grid(True, alpha=0.3)
            axis.set_ylabel("Score", fontsize=8)
            if row_index == 0:
                axis.set_title(title, fontsize=11)
            axis.text(
                0.01,
                0.92,
                "%s → %s" % (source, target),
                transform=axis.transAxes,
                fontsize=8,
                va="top",
            )
            if row_index == 0 and column == 0:
                axis.legend(fontsize=7, loc="lower right")
    axes[-1, 0].set_xlabel("Seconds after the first compared window")
    axes[-1, 1].set_xlabel("Seconds after the first compared window")
    figure.suptitle("Live approximate scores and the forward-backward reference")
    footer = "\n".join(_pair_sentence(summary, measure) for measure, _title in MEASURES)
    figure.text(0.01, 0.01, footer, fontsize=8, va="bottom")
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _pair_sentence(summary, measure):
    title = dict(MEASURES)[measure]
    parts = []
    for item in summary["pairs"]:
        if item["measure"] != measure:
            continue
        fraction = item["direction_match_fraction"]
        if math.isfinite(fraction):
            text = "%.0f percent" % (100.0 * fraction)
        else:
            text = "not defined"
        parts.append("%s and %s: %s" % (item["left"], item["right"], text))
    return "%s. Direction match: %s" % (title, "; ".join(parts))


def format_summary(summary) -> str:
    """A plain-text table of correlations, differences, and direction matches."""
    lines = ["Compared %d ready windows." % summary["windows"], ""]
    for measure, title in MEASURES:
        lines.append(title)
        lines.append(
            "%-28s %-28s %12s %14s %10s"
            % ("Source", "Target", "Correlation", "Median |diff|", "Windows")
        )
        for item in summary["links"]:
            if item["measure"] != measure:
                continue
            lines.append(
                "%-28s %-28s %12s %14s %10d"
                % (
                    item["source"],
                    item["target"],
                    _shown(item["correlation"]),
                    _shown(item["median_absolute_difference"]),
                    item["windows"],
                )
            )
        lines.append("")
        lines.append("Dominant direction of each pair (the larger score).")
        for item in summary["pairs"]:
            if item["measure"] != measure:
                continue
            fraction = item["direction_match_fraction"]
            if item["windows"] and math.isfinite(fraction):
                agreement = "%d of %d windows (%.0f percent)" % (
                    item["direction_matches"],
                    item["windows"],
                    100.0 * fraction,
                )
            else:
                agreement = "not defined"
            lines.append("  %s and %s: %s" % (item["left"], item["right"], agreement))
        lines.append("")
    lines.append("Table: %s" % summary["table"])
    lines.append("Figure: %s" % summary["figure"])
    return "\n".join(lines)


def _shown(value):
    if value is None or not math.isfinite(float(value)):
        return "not defined"
    return "%.4f" % float(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare coupling_approximate.csv with coupling_reference.csv. "
            "The reference file is written first when it is missing."
        )
    )
    parser.add_argument("session", help="Session directory written by the hub.")
    args = parser.parse_args(argv)
    try:
        summary = compare_session(args.session)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
