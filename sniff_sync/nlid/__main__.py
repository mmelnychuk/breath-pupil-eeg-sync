"""Command line: python -m sniff_sync.nlid recordings/P01

Writes a table of directed coupling indices and a summary figure. The live
recording hub is not started and its files are only read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Nearest-neighbour coupling among breathing, pupil diameter, and "
            "electroencephalography band-power ratios."
        )
    )
    parser.add_argument("session", help="Session directory with breathing.csv and eeg.csv")
    parser.add_argument("--pupil", default="", help="Pupil Core export. Searched inside the session if omitted.")
    parser.add_argument(
        "--pupil-offset",
        type=float,
        default=None,
        help="Pupil time minus Unix time, when the export is still on Pupil's clock.",
    )
    parser.add_argument("--sample-rate", type=float, default=16.0, help="Common grid in hertz (10–20 is the intended range).")
    parser.add_argument("--channels", default="", help="Comma-separated electroencephalography channels to average.")
    parser.add_argument(
        "--group",
        default="frontal",
        help="Named channel average used when --channels is omitted. Default: frontal (AF3, AF4, F3, F4).",
    )
    parser.add_argument(
        "--signals",
        default="",
        help="Comma-separated attentional signals to test. Default: every band power and every power ratio.",
    )
    parser.add_argument("--output", default="", help="Directory for the table and the figure. Default: SESSION/nlid")
    parser.add_argument("--surrogates", type=int, default=99, help="Surrogate count for each directed test.")
    parser.add_argument(
        "--surrogate",
        default="time-shift",
        choices=("time-shift", "phase-random", "iaaft"),
        help="Null series. time-shift keeps the waveform, phase-random and iaaft keep the spectrum.",
    )
    parser.add_argument(
        "--correction",
        default="none",
        choices=("none", "bonferroni", "fdr"),
        help="Multiple-comparison correction across attentional signals and directions, within each measure.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--neighbors", type=int, default=10, help="Nearest neighbours k for the interdependence indices.")
    parser.add_argument("--pool", type=int, default=0, help="Conditioning pool size. Default: 5 times --neighbors.")
    parser.add_argument("--theiler", type=int, default=0, help="Theiler window in samples. Default: the longest embedding window.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-blink", type=float, default=0.5, help="Longest pupil gap, in seconds, filled by interpolation.")
    parser.add_argument("--envelope-lowpass", type=float, default=2.0, help="Low-pass cutoff in hertz for band-power envelopes.")
    return parser


def main(argv=None):
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "The coupling analysis needs numpy, scipy, and matplotlib. Install them with:\n"
            "  pip install -e '.[analysis]'",
            file=sys.stderr,
        )
        return 1
    from sniff_sync.nlid.analyze import analyse_triplet, apply_correction, write_results_csv
    from sniff_sync.nlid.plotting import write_summary_figure
    from sniff_sync.nlid.preprocess import load_session

    args = _build_parser().parse_args(argv)
    channels = [item.strip() for item in args.channels.split(",") if item.strip()] or None
    signal_names = [item.strip() for item in args.signals.split(",") if item.strip()] or None
    try:
        assembled = load_session(
            args.session,
            pupil_path=args.pupil or None,
            pupil_offset=args.pupil_offset,
            sample_rate_hz=args.sample_rate,
            channels=channels,
            group=args.group,
            max_blink_s=args.max_blink,
            envelope_lowpass_hz=args.envelope_lowpass,
            attentional_names=signal_names,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    for warning in assembled["warnings"]:
        print("Warning:", warning)
    output = Path(args.output) if args.output else Path(args.session) / "nlid"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    embeddings = {}
    names = list(assembled["attentional_signals"])
    print(
        "Analysing %d attentional signals on %d samples at %.2f Hz, with %d %s surrogates."
        % (len(names), len(assembled["breathing"]), assembled["sample_rate_hz"], args.surrogates, args.surrogate)
    )
    for index, name in enumerate(names):
        print("  %s" % name, flush=True)
        part, resolved = analyse_triplet(
            {
                "breathing": assembled["breathing"],
                "pupil diameter": assembled["pupil diameter"],
                name: assembled["attentional_signals"][name],
            },
            sample_rate_hz=assembled["sample_rate_hz"],
            attentional_signal=name,
            neighbor_count=args.neighbors,
            pool_size=args.pool or None,
            theiler_window=args.theiler or None,
            n_surrogates=args.surrogates,
            surrogate=args.surrogate,
            alpha=args.alpha,
            seed=args.seed,
            emit_breathing_pupil_pair=(index == 0),
        )
        rows.extend(part)
        embeddings[name] = {key: {"dimension": value["dimension"], "delay": value["delay"]} for key, value in resolved.items()}
    apply_correction(rows, args.correction, args.alpha)
    table_path = write_results_csv(rows, output / "interdependence.csv")
    figure_path = write_summary_figure(rows, output / "summary.png", alpha=args.alpha)
    parameters = {
        "sample_rate_hz": assembled["sample_rate_hz"],
        "n_samples": int(len(assembled["breathing"])),
        "duration_s": float(len(assembled["breathing"]) / assembled["sample_rate_hz"]),
        "pupil_file": assembled.get("pupil_file", ""),
        "warnings": assembled["warnings"],
        "surrogate": args.surrogate,
        "surrogate_count": args.surrogates,
        "correction": args.correction,
        "alpha": args.alpha,
        "neighbors": args.neighbors,
        "embeddings": embeddings,
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2))
    n_significant = sum(row["significant"] == "yes" for row in rows)
    print("Wrote %s" % table_path)
    print("Wrote %s" % figure_path)
    print("%d of %d tests are marked significant at alpha %.3f." % (n_significant, len(rows), args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
