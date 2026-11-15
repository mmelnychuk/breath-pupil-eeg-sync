"""Confirm the same triggers were stored in all three recordings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _rows(path: Path) -> dict:
    found = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            kind = (row.get("kind") or "").strip()
            if kind == "sample":
                continue
            trigger_id = (row.get("trigger_id") or "").strip()
            if not trigger_id or trigger_id == "trigger_id":
                continue
            label = (row.get("label") or row.get("Trigger") or "").strip()
            unix = (row.get("unix_time") or row.get("Unix Time") or row.get("timestamp") or "").strip()
            if not unix:
                continue
            found[trigger_id] = {"unix_time": unix, "label": label}
    return found


def compare_session(directory) -> dict:
    root = Path(directory)
    paths = {
        "breathing": root / "breathing.csv",
        "eeg": root / "eeg.csv",
        "pupil": root / "pupil_recording" / "sniff_sync_triggers.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"ok": False, "compared": 0, "problems": ["missing file: %s" % ", ".join(missing)]}
    tables = {name: _rows(path) for name, path in paths.items()}
    ids = set()
    for table in tables.values():
        ids.update(table)
    problems = []
    max_dt = 0.0
    compared = 0
    for trigger_id in sorted(ids):
        present = {name: tables[name].get(trigger_id) for name in tables}
        absent = [name for name, row in present.items() if row is None]
        if absent:
            problems.append("%s missing from %s" % (trigger_id, ", ".join(absent)))
            continue
        stamps = [present[name]["unix_time"] for name in ("breathing", "eeg", "pupil")]
        labels = {present[name]["label"] for name in present}
        if len(labels) != 1:
            problems.append("%s labels differ: %s" % (trigger_id, sorted(labels)))
        if len(set(stamps)) != 1:
            numbers = [float(item) for item in stamps]
            max_dt = max(max_dt, max(numbers) - min(numbers))
            problems.append("%s unix times differ: %s" % (trigger_id, stamps))
        else:
            compared += 1
    return {
        "ok": not problems and compared > 0,
        "compared": compared,
        "problems": problems,
        "max_abs_dt_s": max_dt,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check that trigger Unix times match across the three recordings.")
    parser.add_argument("session", help="Session directory written by the hub")
    args = parser.parse_args(argv)
    result = compare_session(args.session)
    if result["ok"]:
        print("OK. %d trigger(s) share the same Unix time in breathing, EEG, and Pupil." % result["compared"])
        return 0
    print("Triggers do not match.")
    for problem in result["problems"]:
        print(" -", problem)
    return 1


if __name__ == "__main__":
    sys.exit(main())
