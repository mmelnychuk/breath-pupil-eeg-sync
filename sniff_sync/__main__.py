"""Run the hub on this computer.

    python -m sniff_sync --simulate
    python -m sniff_sync --session recordings/today --sniff-csv subject_data/P01/KH_sniff.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

from sniff_sync.eeg import EmokitHeadset
from sniff_sync.simulate import Session
from sniff_sync.timefmt import unix_seconds
from sniff_sync.trigger import DEFAULT_PORT


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Local trigger hub, EEG reader, and live view.")
    parser.add_argument("--session", default="recordings/session")
    parser.add_argument("--simulate", action="store_true", help="Fake all three streams. No headset, no glasses.")
    parser.add_argument("--sniff-csv", default="", help="getSniffs.py file to tail. Unix Time in seconds.")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0, help="Stop after this many seconds. 0 runs until interrupted.")
    args = parser.parse_args(argv)
    session_dir = Path(args.session)
    if args.simulate:
        session = Session(session_dir, port=DEFAULT_PORT)
        session.start()
        print("Hub listening on tcp://127.0.0.1:%s" % session.hub.port, flush=True)
        print("Session files are in %s" % session.directory, flush=True)
        print('From another program: from sniff_sync import trigger; trigger("label")', flush=True)
        window = None
        if not args.no_window:
            if not os.environ.get("DISPLAY"):
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6 import QtWidgets

            from sniff_sync.viz import Monitor

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            window = Monitor(session.feed, coupling=session.hub.coupling)
            window.show()
        try:
            if args.seconds > 0 and window is None:
                time.sleep(args.seconds)
            elif window is not None:
                deadline = time.time() + args.seconds if args.seconds > 0 else None
                while deadline is None or time.time() < deadline:
                    app.processEvents()
                    time.sleep(0.02)
            else:
                while True:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            if window is not None:
                window.close()
            session.stop()
        return 0

    from sniff_sync.feed import Feed
    from sniff_sync.hub import Hub

    feed = Feed()
    hub = Hub(session_dir, feed=feed)
    hub.start()
    os.environ["SNIFF_SYNC_PORT"] = str(hub.port)
    print("Hub listening on tcp://127.0.0.1:%s" % hub.port)
    if hub.pupil.connected:
        print("Pupil offset (Pupil time minus Unix time): %.6f seconds" % hub.pupil.offset)
    else:
        print("Pupil Capture is not on localhost:50020. Breathing and EEG logs will still get each trigger.")
    headset = EmokitHeadset()
    if headset.open():
        print("Emotiv headset opened through emokit.")
    else:
        print("No Emotiv stream (%s). Trigger markers will still be written to eeg.csv." % (headset.error or "headset not found"))
    stop = False

    import threading

    threads = []
    if args.sniff_csv:
        threads.append(
            threading.Thread(
                target=_tail_sniff,
                args=(Path(args.sniff_csv), feed, hub.history, lambda: stop),
                daemon=True,
            )
        )

    def pump():
        while not stop:
            if headset._headset is None:
                time.sleep(0.05)
                continue
            try:
                sample = headset.read()
            except Exception:
                time.sleep(0.05)
                continue
            if sample and sample["channels"]:
                hub.eeg.write_sample(sample["unix_time"], sample["channels"], counter=sample["counter"])
                feed.add_eeg(sample["unix_time"], sample["channels"])
                hub.history.add_eeg(sample["unix_time"], sample["channels"])
            time.sleep(0.002)

    threads.append(threading.Thread(target=pump, daemon=True))
    for thread in threads:
        thread.start()
    try:
        if args.no_window or not os.environ.get("DISPLAY"):
            while True:
                time.sleep(0.2)
        from sniff_sync.viz import open_monitor

        open_monitor(feed, coupling=hub.coupling)
    except KeyboardInterrupt:
        pass
    finally:
        stop = True
        headset.close()
        hub.close()
    return 0


def _tail_sniff(path: Path, feed, history, stopped) -> None:
    """Follow a getSniffs.py CSV. Times are stored as Unix seconds."""
    while not path.exists() and not stopped():
        time.sleep(0.1)
    if not path.exists():
        return
    with open(path, newline="") as handle:
        handle.seek(0, os.SEEK_END)
        while not stopped():
            position = handle.tell()
            line = handle.readline()
            if not line:
                time.sleep(0.05)
                handle.seek(position)
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 2 or parts[0] in ("Unix Time", "unix_time"):
                continue
            try:
                stamp = unix_seconds(parts[0])
                pressure = float(parts[1])
                feed.add_breath(stamp, pressure)
                history.add_breath(stamp, pressure)
            except ValueError:
                continue


if __name__ == "__main__":
    sys.exit(main())
