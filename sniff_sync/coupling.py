"""Live coupling scores, computed in a separate process.

About once a second the hub sends the latest aligned window. Neighbour
searches run in the other process, so they do not hold Python's global
interpreter lock against the readers or the live view. If a result is not
ready when the next window is due, that window is skipped rather than queued.
Live results are approximate: they use the short linear-phase filters.
The post-session reference is a different file.
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import threading
import time
from pathlib import Path

import numpy as np

from sniff_sync.nlid.analyze import choose_embedding
from sniff_sync.nlid.measures import conditional_interdependence, prediction_gain
from sniff_sync.nlid.preprocess import longest_finite_run

OUTPUT_RATE_HZ = 10.0
DEFAULT_WINDOW_SECONDS = 60.0
NEIGHBOR_COUNT = 8
POOL_SIZE = 40
HORIZON_SECONDS = 0.4
HORIZON_SAMPLES = 4
# A baseline needs a long stretch of real samples before the embedding is fixed.
BASELINE_SECONDS = 20.0
MINIMUM_VALID_FRACTION = 0.80

SIGNAL_NAMES = ("breathing", "pupil diameter", "electroencephalography")
EEG_SIGNALS = {
    "delta-to-alpha": "delta_to_alpha",
    "delta": "delta_envelope",
    "alpha": "alpha_envelope",
}

PAIRS = (
    ("breathing", "pupil diameter"),
    ("breathing", "electroencephalography"),
    ("pupil diameter", "electroencephalography"),
)

APPROXIMATE_NAME = "coupling_approximate.csv"


def _longest_valid_seconds(mask):
    start, stop = longest_finite_run(np.asarray(mask, dtype=bool))
    return (stop - start) / OUTPUT_RATE_HZ


def _finite_series(values, mask):
    """The longest stretch where ``mask`` is true. Missing samples are not passed to the embedding."""
    start, stop = longest_finite_run(np.asarray(mask, dtype=bool))
    piece = np.asarray(values, dtype=float)[start:stop]
    if piece.size == 0 or not np.all(np.isfinite(piece)):
        return None
    return piece


def score_triplet(series, embedding, neighbor_count=NEIGHBOR_COUNT, pool_size=POOL_SIZE, horizon_samples=HORIZON_SAMPLES):
    """Conditional interdependence and forecast gain for all six directions.

    ``series`` maps the three plain names to one-dimensional arrays on the
    same 10 hertz grid. ``embedding`` maps each name to delay and dimension,
    both in samples. The forecast horizon is fixed; it is not searched.
    """
    names = SIGNAL_NAMES
    theiler = 1
    for name in names:
        spec = embedding[name]
        theiler = max(theiler, (int(spec["dimension"]) - 1) * int(spec["delay"]))
    interdependence = {}
    forecast = {}
    for source in names:
        for target in names:
            if source == target:
                continue
            conditioner = next(name for name in names if name not in (source, target))
            arrow = "%s → %s" % (source, target)
            dimension = (
                int(embedding[target]["dimension"]),
                int(embedding[source]["dimension"]),
                int(embedding[conditioner]["dimension"]),
            )
            delay = (
                int(embedding[target]["delay"]),
                int(embedding[source]["delay"]),
                int(embedding[conditioner]["delay"]),
            )
            conditional = conditional_interdependence(
                series[target],
                series[source],
                series[conditioner],
                dimension,
                delay,
                neighbor_count,
                pool_size,
                theiler,
            )
            gained = prediction_gain(
                series[target],
                series[source],
                series[conditioner],
                dimension,
                delay,
                neighbor_count,
                pool_size,
                theiler,
                (int(horizon_samples),),
            )
            interdependence[arrow] = float(conditional["conditional_andrzejak_M"])
            forecast[arrow] = float(gained["prediction_gain"])
    return interdependence, forecast, theiler


def direction_summary(interdependence, forecast):
    """For each pair, which way is stronger and by how much, for both measures."""
    summary = {}
    for left, right in PAIRS:
        key = "%s and %s" % (left, right)
        summary[key] = {
            "interdependence_stronger": _stronger(interdependence, left, right),
            "interdependence_margin": _margin(interdependence, left, right),
            "forecast_stronger": _stronger(forecast, left, right),
            "forecast_margin": _margin(forecast, left, right),
        }
    return summary


def _margin(scores, left, right):
    forward = scores.get("%s → %s" % (left, right))
    backward = scores.get("%s → %s" % (right, left))
    if forward is None or backward is None:
        return float("nan")
    if not math.isfinite(forward) or not math.isfinite(backward):
        return float("nan")
    return float(forward - backward)


def _stronger(scores, left, right):
    forward = "%s → %s" % (left, right)
    backward = "%s → %s" % (right, left)
    margin = _margin(scores, left, right)
    if not math.isfinite(margin) or abs(margin) <= 1e-6:
        return "neither"
    return forward if margin > 0 else backward


def _estimate_embedding(series_by_name):
    embedding = {}
    for name, values in series_by_name.items():
        chosen = choose_embedding(values, OUTPUT_RATE_HZ)
        embedding[name] = {"delay": int(chosen["delay"]), "dimension": int(chosen["dimension"])}
    return embedding


class _WorkerMemory:
    """Embedding chosen once, then held. Lives only inside the worker process."""

    def __init__(self):
        self.embedding = None
        self.source = ""
        self.eeg_signal = ""

    def compute(self, job):
        started = time.perf_counter()
        result = _blank_result(job)
        result["compute_seconds"] = 0.0
        try:
            result = self._compute(job)
        except Exception as exc:
            result = _blank_result(job)
            result["status"] = "skipped"
            result["reason"] = "The coupling update failed: %s." % exc
        result["compute_seconds"] = time.perf_counter() - started
        return result

    def _compute(self, job):
        result = _blank_result(job)
        values = job.get("values") or {}
        incoming = job.get("valid")
        mask = np.asarray([] if incoming is None else incoming, dtype=bool)
        if mask.size == 0 or any(name not in values or len(values[name]) != mask.size for name in ("breathing", "pupil", "eeg")):
            result["status"] = "warming up"
            result["reason"] = "The aligned window does not have all three signals yet."
            return result
        fraction = float(np.mean(mask)) if mask.size else 0.0
        result["valid_fraction"] = fraction
        long_enough = _longest_valid_seconds(mask) >= BASELINE_SECONDS
        usable = fraction >= MINIMUM_VALID_FRACTION and long_enough
        hand = job.get("hand_embedding")
        eeg_signal = job.get("eeg_signal") or "delta-to-alpha"
        if self.embedding is not None and eeg_signal != self.eeg_signal and not hand:
            # The electroencephalography trace changed, so the old embedding does not apply.
            self.embedding = None
            self.source = ""
        if hand:
            self.embedding = {name: {"delay": int(hand[name]["delay"]), "dimension": int(hand[name]["dimension"])} for name in SIGNAL_NAMES}
            self.source = "hand"
            self.eeg_signal = eeg_signal
        if self.embedding is None:
            if not usable:
                result["status"] = "warming up"
                result["reason"] = (
                    "The baseline used to choose the embedding is not complete yet. "
                    "It needs at least %.0f seconds with at least %.0f percent of the samples present."
                    % (BASELINE_SECONDS, 100.0 * MINIMUM_VALID_FRACTION)
                )
                return result
            series = _series_map(values, mask)
            if series is None:
                result["status"] = "warming up"
                result["reason"] = "The baseline stretch still has a missing sample, so the embedding was not chosen."
                return result
            self.embedding = _estimate_embedding(series)
            self.source = "baseline"
            self.eeg_signal = eeg_signal
        result["embedding"] = {name: dict(spec) for name, spec in self.embedding.items()}
        result["embedding_source"] = self.source
        if not usable:
            result["status"] = "skipped"
            if fraction < MINIMUM_VALID_FRACTION:
                result["reason"] = (
                    "This window was skipped because only %.0f percent of its samples are present. "
                    "At least %.0f percent is required."
                    % (100.0 * fraction, 100.0 * MINIMUM_VALID_FRACTION)
                )
            else:
                result["reason"] = (
                    "This window was skipped because its longest complete stretch is shorter than %.0f seconds."
                    % BASELINE_SECONDS
                )
            return result
        series = _series_map(values, mask)
        if series is None:
            result["status"] = "skipped"
            result["reason"] = "This window was skipped because the complete stretch still has a missing sample."
            return result
        interdependence, forecast, theiler = score_triplet(
            series,
            self.embedding,
            neighbor_count=int(job.get("neighbor_count", NEIGHBOR_COUNT)),
            pool_size=int(job.get("pool_size", POOL_SIZE)),
            horizon_samples=int(job.get("horizon_samples", HORIZON_SAMPLES)),
        )
        result["status"] = "ready"
        result["reason"] = ""
        result["theiler_window_samples"] = int(theiler)
        result["interdependence"] = interdependence
        result["forecast_gain"] = forecast
        result["directions"] = direction_summary(interdependence, forecast)
        return result


def _series_map(values, mask):
    breathing = _finite_series(values["breathing"], mask)
    pupil = _finite_series(values["pupil"], mask)
    eeg = _finite_series(values["eeg"], mask)
    if breathing is None or pupil is None or eeg is None:
        return None
    # The three stretches must be the same samples. longest_finite_run on the
    # joint mask already selected one stretch, and each series was cut with it.
    count = min(len(breathing), len(pupil), len(eeg))
    if count < int(BASELINE_SECONDS * OUTPUT_RATE_HZ):
        return None
    return {
        "breathing": breathing[:count],
        "pupil diameter": pupil[:count],
        "electroencephalography": eeg[:count],
    }


def _blank_result(job):
    return {
        "label": "approximate",
        "status": "warming up",
        "reason": "",
        "window_end_unix": job.get("window_end_unix"),
        "window_seconds": job.get("window_seconds"),
        "eeg_signal": job.get("eeg_signal") or "delta-to-alpha",
        "eeg_channels": list(job.get("eeg_channels") or []),
        "filter_profile": job.get("filter_profile") or "live",
        "embedding": None,
        "embedding_source": "",
        "neighbor_count": int(job.get("neighbor_count", NEIGHBOR_COUNT)),
        "pool_size": int(job.get("pool_size", POOL_SIZE)),
        "horizon_seconds": HORIZON_SECONDS,
        "horizon_samples": int(job.get("horizon_samples", HORIZON_SAMPLES)),
        "surrogates": bool(job.get("surrogates", False)),
        "interdependence": {},
        "forecast_gain": {},
        "directions": {},
        "valid_fraction": float(job.get("valid_fraction") or 0.0),
        "compute_seconds": 0.0,
        "delays_seconds": dict(job.get("delays_seconds") or {}),
        "theiler_window_samples": "",
    }


def worker_main(job_queue, result_queue):
    """Process entry. ``None`` on the job queue asks the process to stop."""
    memory = _WorkerMemory()
    while True:
        job = job_queue.get()
        if job is None:
            return
        result_queue.put(memory.compute(job))


class LiveCoupling:
    """Sends windows to the worker and keeps the latest result.

    The scheduler is a thread so the hub's own loops are not blocked.
    The neighbour search is a process. A window that is still being scored
    when the next second arrives is not queued.
    """

    def __init__(self, history, session_directory, eeg_signal="delta-to-alpha"):
        if eeg_signal not in EEG_SIGNALS:
            raise ValueError(
                "The electroencephalography signal must be one of %s, not %r."
                % (", ".join(EEG_SIGNALS), eeg_signal)
            )
        self.history = history
        self.session_directory = Path(session_directory)
        self.eeg_signal = eeg_signal
        self.neighbor_count = NEIGHBOR_COUNT
        self.pool_size = POOL_SIZE
        self.horizon_samples = HORIZON_SAMPLES
        self.surrogates = False
        self.hand_embedding = None
        self._context = multiprocessing.get_context("spawn")
        self._jobs = None
        self._results = None
        self._process = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = _blank_result({"eeg_signal": eeg_signal, "window_seconds": history.window_seconds})
        self._latest["reason"] = "The coupling worker has not reported yet."
        self._busy = False
        self._log = None
        self._writer = None

    def set_embedding(self, embedding) -> None:
        """Fix the delay and dimension by hand. Keys are the three signal names.

        Each value is ``{"delay": samples, "dimension": samples}``. Passing
        ``None`` returns to choosing them from the baseline.
        """
        if embedding is None:
            self.hand_embedding = None
            return
        chosen = {}
        for name in SIGNAL_NAMES:
            if name not in embedding:
                raise ValueError("The hand-set embedding needs %r." % name)
            spec = embedding[name]
            delay = int(spec["delay"])
            dimension = int(spec["dimension"])
            if delay < 1 or dimension < 1:
                raise ValueError("Delay and dimension must both be at least 1.")
            chosen[name] = {"delay": delay, "dimension": dimension}
        self.hand_embedding = chosen

    def set_window_seconds(self, seconds: float) -> None:
        self.history.set_window_seconds(seconds)

    def start(self) -> None:
        if self._process is not None:
            return
        self.session_directory.mkdir(parents=True, exist_ok=True)
        path = self.session_directory / APPROXIMATE_NAME
        new_file = not path.exists() or path.stat().st_size == 0
        self._log = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._log, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new_file:
            self._writer.writeheader()
            self._log.flush()
        self._jobs = self._context.Queue(maxsize=1)
        self._results = self._context.Queue()
        self._stop.clear()
        self._busy = False
        self._process = self._context.Process(
            target=worker_main,
            name="coupling-worker",
            args=(self._jobs, self._results),
            daemon=True,
        )
        self._process.start()
        self._thread = threading.Thread(target=self._loop, name="coupling-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._jobs is not None and self._process is not None and self._process.is_alive():
            try:
                self._jobs.put_nowait(None)
            except Exception:
                pass
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None
            self._writer = None

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def _loop(self) -> None:
        next_time = time.monotonic()
        while not self._stop.is_set():
            self._drain()
            now = time.monotonic()
            if now < next_time:
                self._stop.wait(0.05)
                continue
            next_time = now + 1.0
            if self._busy:
                continue
            try:
                payload = self._payload()
            except Exception as exc:
                self._remember(_blank_result({}), "The aligned window could not be built: %s." % exc)
                continue
            try:
                self._jobs.put_nowait(payload)
                self._busy = True
            except Exception:
                # The worker still holds the previous window. Do not queue another.
                continue

    def _payload(self) -> dict:
        window = self.history.resample()
        field = EEG_SIGNALS[self.eeg_signal]
        eeg = getattr(window, field)
        valid = np.asarray(window.valid, dtype=bool)
        end = float(window.time_seconds[-1]) if window.time_seconds.size else None
        return {
            "values": {
                "breathing": np.asarray(window.breathing, dtype=float),
                "pupil": np.asarray(window.pupil_diameter, dtype=float),
                "eeg": np.asarray(eeg, dtype=float),
            },
            "valid": valid,
            "valid_fraction": float(np.mean(valid)) if valid.size else 0.0,
            "window_end_unix": end,
            "window_seconds": float(self.history.window_seconds),
            "eeg_signal": self.eeg_signal,
            "eeg_channels": list(self.history.eeg_channels),
            "filter_profile": window.filter_profile,
            "neighbor_count": self.neighbor_count,
            "pool_size": self.pool_size,
            "horizon_samples": self.horizon_samples,
            "surrogates": bool(self.surrogates),
            "hand_embedding": self.hand_embedding,
            "delays_seconds": dict(window.delays_seconds),
        }

    def _drain(self) -> None:
        if self._results is None:
            return
        while True:
            try:
                result = self._results.get_nowait()
            except Exception:
                return
            self._busy = False
            result["logged_unix"] = time.time()
            self._write(result)
            with self._lock:
                self._latest = result

    def _remember(self, result, reason) -> None:
        result["status"] = "skipped"
        result["reason"] = reason
        result["logged_unix"] = time.time()
        self._write(result)
        with self._lock:
            self._latest = result

    def _write(self, result) -> None:
        if self._writer is None:
            return
        self._writer.writerow(_log_row(result))
        self._log.flush()


def _arrow_column(measure, source, target):
    return "%s__%s_to_%s" % (
        measure,
        source.replace(" ", "_"),
        target.replace(" ", "_"),
    )


def _direction_column(kind, left, right):
    return "direction_%s_%s_%s" % (kind, left.replace(" ", "_"), right.replace(" ", "_"))


LOG_COLUMNS = (
    [
        "label",
        "status",
        "reason",
        "logged_unix",
        "window_end_unix",
        "window_seconds",
        "eeg_signal",
        "eeg_channels",
        "filter_profile",
        "embedding_source",
        "neighbor_count",
        "pool_size",
        "horizon_samples",
        "surrogates",
        "valid_fraction",
        "compute_seconds",
        "theiler_window_samples",
    ]
    + ["embedding_delay_%s" % name.replace(" ", "_") for name in SIGNAL_NAMES]
    + ["embedding_dimension_%s" % name.replace(" ", "_") for name in SIGNAL_NAMES]
    + [
        _arrow_column(measure, source, target)
        for measure in ("interdependence", "forecast_gain")
        for source in SIGNAL_NAMES
        for target in SIGNAL_NAMES
        if source != target
    ]
    + [_direction_column(kind, left, right) for kind in ("interdependence", "forecast") for left, right in PAIRS]
)


def _log_row(result) -> dict:
    row = {column: "" for column in LOG_COLUMNS}
    row["label"] = result.get("label") or "approximate"
    row["status"] = result.get("status") or ""
    row["reason"] = result.get("reason") or ""
    row["logged_unix"] = _number(result.get("logged_unix"))
    row["window_end_unix"] = _number(result.get("window_end_unix"))
    row["window_seconds"] = _number(result.get("window_seconds"))
    row["eeg_signal"] = result.get("eeg_signal") or ""
    channels = result.get("eeg_channels") or []
    row["eeg_channels"] = " ".join(channels)
    row["filter_profile"] = result.get("filter_profile") or ""
    row["embedding_source"] = result.get("embedding_source") or ""
    row["neighbor_count"] = result.get("neighbor_count", "")
    row["pool_size"] = result.get("pool_size", "")
    row["horizon_samples"] = result.get("horizon_samples", "")
    row["surrogates"] = "yes" if result.get("surrogates") else "no"
    row["valid_fraction"] = _number(result.get("valid_fraction"))
    row["compute_seconds"] = _number(result.get("compute_seconds"))
    row["theiler_window_samples"] = result.get("theiler_window_samples", "")
    embedding = result.get("embedding") or {}
    for name in SIGNAL_NAMES:
        spec = embedding.get(name) or {}
        row["embedding_delay_%s" % name.replace(" ", "_")] = spec.get("delay", "")
        row["embedding_dimension_%s" % name.replace(" ", "_")] = spec.get("dimension", "")
    for measure, key in (("interdependence", "interdependence"), ("forecast_gain", "forecast_gain")):
        scores = result.get(key) or {}
        for source in SIGNAL_NAMES:
            for target in SIGNAL_NAMES:
                if source == target:
                    continue
                row[_arrow_column(measure, source, target)] = _number(scores.get("%s → %s" % (source, target)))
    directions = result.get("directions") or {}
    for left, right in PAIRS:
        item = directions.get("%s and %s" % (left, right)) or {}
        for kind, stronger_key, margin_key in (
            ("interdependence", "interdependence_stronger", "interdependence_margin"),
            ("forecast", "forecast_stronger", "forecast_margin"),
        ):
            column = _direction_column(kind, left, right)
            stronger = item.get(stronger_key) or ""
            row[column] = "" if not stronger else "%s (%s)" % (stronger, _number(item.get(margin_key)))
    return row


def _number(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return "%.6f" % number
