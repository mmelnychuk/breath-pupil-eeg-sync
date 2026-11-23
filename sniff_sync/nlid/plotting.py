"""Summary figure: coupling strength and which directions pass the surrogate test.

Rows are attentional signals (a band's power, or a ratio of two bands). Columns
are the six directed pairs among breathing, pupil diameter, and that attentional
signal. A star means the index is positive and the p-value, after any requested
correction, is at most the alpha level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DIRECTION_ORDER = (
    "breathing → pupil diameter",
    "pupil diameter → breathing",
    "breathing → attentional signal",
    "attentional signal → breathing",
    "pupil diameter → attentional signal",
    "attentional signal → pupil diameter",
)


def _role(name, attentional_signal):
    if attentional_signal and name == attentional_signal:
        return "attentional signal"
    return name


def _direction(row):
    source = _role(row["source"], row["attentional_signal"])
    target = _role(row["target"], row["attentional_signal"])
    return "%s → %s" % (source, target)


def _matrix(rows, measure, signals):
    values = np.full((len(signals), len(DIRECTION_ORDER)), np.nan)
    stars = np.zeros((len(signals), len(DIRECTION_ORDER)), dtype=bool)
    index = {direction: column for column, direction in enumerate(DIRECTION_ORDER)}
    signal_index = {name: row for row, name in enumerate(signals)}
    shared = {}
    for row in rows:
        if row["measure"] != measure:
            continue
        direction = _direction(row)
        if direction not in index:
            continue
        signal = row["attentional_signal"] or ""
        if signal == "":
            shared[direction] = row
            continue
        if signal not in signal_index:
            continue
        values[signal_index[signal], index[direction]] = row["value"]
        stars[signal_index[signal], index[direction]] = row["significant"] == "yes"
    for direction, row in shared.items():
        column = index[direction]
        if np.all(~np.isfinite(values[:, column])):
            values[:, column] = row["value"]
            stars[:, column] = row["significant"] == "yes"
    return values, stars


def write_summary_figure(rows, path, alpha=0.05):
    """Write a PNG with pairwise M, conditional M, and the partial-cross-mapping direct-link index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = []
    for row in rows:
        name = row.get("attentional_signal") or ""
        if name and name not in signals:
            signals.append(name)
    if not signals:
        raise ValueError("No attentional signals to plot.")
    panels = (
        ("Andrzejak M", "Pairwise interdependence (Andrzejak M)", "sequential"),
        ("conditional Andrzejak M", "Conditional interdependence given the third signal (Andrzejak M)", "diverging"),
        ("direct-link index", "Partial cross mapping direct-link index", "sequential"),
    )
    n_rows = len(signals)
    fig_height = max(4.5, 0.38 * n_rows + 2.2)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fig_height), squeeze=False)
    short_columns = (
        "breathing → pupil",
        "pupil → breathing",
        "breathing → attention",
        "attention → breathing",
        "pupil → attention",
        "attention → pupil",
    )
    for axis, (measure, title, scale) in zip(axes[0], panels):
        values, stars = _matrix(rows, measure, signals)
        finite = values[np.isfinite(values)]
        if scale == "diverging":
            limit = 0.2 if len(finite) == 0 else max(0.05, float(np.nanmax(np.abs(finite))))
            image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        else:
            upper = 0.2 if len(finite) == 0 else max(0.05, float(np.nanmax(finite)))
            image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=upper)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(len(short_columns)))
        axis.set_xticklabels(short_columns, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(range(n_rows))
        axis.set_yticklabels(signals, fontsize=8)
        annotate = n_rows <= 16
        for row_index in range(n_rows):
            for column in range(values.shape[1]):
                value = values[row_index, column]
                if not np.isfinite(value):
                    continue
                text = "%.2f" % value if annotate else ""
                if stars[row_index, column]:
                    text = (text + " *").strip()
                if text:
                    color = "white" if scale == "sequential" and value > 0.6 * (upper if scale == "sequential" else limit) else "black"
                    axis.text(column, row_index, text, ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Directed coupling. A star means a positive index with p ≤ %.3f%s."
        % (alpha, " after correction" if rows and rows[0].get("correction") not in ("", "none") else ""),
        fontsize=12,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
