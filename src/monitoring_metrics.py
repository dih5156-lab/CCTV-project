"""Bounded process-local timings and read-only durable backlog measurements."""

import math
import shutil
import time
from collections import Counter, deque
from datetime import datetime
from threading import Lock

from .monitoring_contract import require_string
from .monitoring_snapshot import check_source, pending_counts, source_rows, utc_text


def _epoch(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            return None
        result = parsed.timestamp()
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


class CommitObservations:
    """Post-commit wall-clock observations, never reconstructed from transaction timestamps."""

    PHASES = (
        "ingest_commit_completed_at",
        "normalize_completed_at",
        "consumer_ack_persisted_at",
    )

    def __init__(self, *, max_samples=1000, wall_clock=time.time, clock_sync="unknown"):
        if type(max_samples) is not int or not 1 <= max_samples <= 10000:
            raise ValueError("max_samples must be 1..10000")
        if clock_sync not in {"confirmed", "unknown"}:
            raise ValueError("clock_sync must be confirmed or unknown")
        self.samples = deque(maxlen=max_samples)
        self.wall_clock, self.clock_sync = wall_clock, clock_sync
        self.total = 0
        self.lock = Lock()

    def committed(self, phase, ingest_id, *, received_at, delivery_id=None):
        if phase not in self.PHASES:
            raise ValueError("Unknown commit phase")
        now = self.wall_clock()
        received = _epoch(received_at)
        quality, elapsed = "unknown", None
        if not math.isfinite(now) or received is None or now < received:
            quality = "invalid"
        elif self.clock_sync == "confirmed":
            quality, elapsed = "valid", (now - received) * 1000
        sample = {
            "phase": phase,
            "ingest_id": ingest_id,
            "delivery_id": delivery_id,
            "received_at": received_at,
            "commit_observed_at": utc_text(now) if math.isfinite(now) else None,
            "clock_sync": self.clock_sync,
            "quality": quality,
            "elapsed_ms": elapsed,
        }
        with self.lock:
            self.samples.append(sample)
            self.total += 1

    def report(self):
        with self.lock:
            rows, total = list(self.samples), self.total
        acknowledgements = [
            row for row in rows if row["phase"] == "consumer_ack_persisted_at"
        ]
        qualities = Counter(row["quality"] for row in acknowledgements)
        values = sorted(
            row["elapsed_ms"] for row in acknowledgements if row["quality"] == "valid"
        )

        def percentile(fraction):
            return (
                values[max(0, math.ceil(len(values) * fraction) - 1)]
                if values
                else None
            )

        return {
            "lifetime": "current_process",
            "window": "last_samples",
            "max_samples": self.samples.maxlen,
            "total_seen": total,
            "evicted": total - len(rows),
            "samples": rows,
            "latency": {
                "name": "received_to_ack_commit",
                "clock": "wall",
                "clock_sync": self.clock_sync,
                "valid": qualities["valid"],
                "invalid": qualities["invalid"],
                "unknown": qualities["unknown"],
                "p50_ms": percentile(0.5),
                "p95_ms": percentile(0.95),
                "max_ms": values[-1] if values else None,
            },
        }


class MonitoringTimings:
    STAGES = ("consumer_lookup", "mqtt_publish", "consumer_ack_commit")

    def __init__(self, *, max_samples=1000, monotonic=time.monotonic):
        if type(max_samples) is not int or not 1 <= max_samples <= 10000:
            raise ValueError("max_samples must be 1..10000")
        self.monotonic, self.max_samples = monotonic, max_samples
        self.samples = {stage: deque(maxlen=max_samples) for stage in self.STAGES}
        self.totals = Counter()
        self.lock = Lock()

    def measure(self, stage, operation):
        if stage not in self.samples:
            raise ValueError("Unknown monitoring timing stage")
        start = self.monotonic()
        outcome = "raised"
        try:
            value = operation()
            outcome = "returned"
            return value
        finally:
            end = self.monotonic()
            elapsed = (end - start) * 1000
            valid = (
                math.isfinite(start)
                and math.isfinite(end)
                and math.isfinite(elapsed)
                and elapsed >= 0
            )
            with self.lock:
                self.samples[stage].append(
                    (start, end, elapsed if valid else None, outcome)
                )
                self.totals[stage] += 1

    def report(self):
        result = {}
        with self.lock:
            for stage, samples in self.samples.items():
                values = sorted(row[2] for row in samples if row[2] is not None)

                def percentile(fraction):
                    return (
                        values[max(0, math.ceil(len(values) * fraction) - 1)]
                        if values
                        else None
                    )

                result[stage] = {
                    "unit": "ms",
                    "clock": "monotonic",
                    "lifetime": "current_process",
                    "window": "last_samples",
                    "max_samples": self.max_samples,
                    "window_start_monotonic": (
                        samples[0][0]
                        if samples and math.isfinite(samples[0][0])
                        else None
                    ),
                    "window_end_monotonic": (
                        samples[-1][1]
                        if samples and math.isfinite(samples[-1][1])
                        else None
                    ),
                    "valid": len(values),
                    "invalid": len(samples) - len(values),
                    "raised": sum(row[3] == "raised" for row in samples),
                    "total_seen": self.totals[stage],
                    "evicted": self.totals[stage] - len(samples),
                    "p50_ms": percentile(0.5),
                    "p95_ms": percentile(0.95),
                    "max_ms": values[-1] if values else None,
                }
        return result


def source_metrics(
    store,
    binding,
    *,
    namespace=None,
    now=None,
    clock_sync="unknown",
    window_seconds=3600,
):
    now = time.time() if now is None else now
    if (
        not math.isfinite(now)
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("finite now and positive window required")
    if clock_sync not in {"confirmed", "unknown"}:
        raise ValueError("clock_sync must be confirmed or unknown")
    if namespace is not None:
        require_string(namespace)
    states = Counter(
        {
            state: 0
            for state in (
                "pending",
                "awaiting_ack",
                "uncertain",
                "quarantined",
                "blocked",
                "stored",
            )
        }
    )
    age = {"valid": 0, "invalid": 0, "unknown": 0, "oldest_seconds": None}
    published_retries, active_leases, stored_in_window = 0, 0, 0
    with store._connect() as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        check_source(connection, binding)
        pending = pending_counts(connection, binding, namespace)
        for row in source_rows(connection, binding, namespace):
            states[row["state"]] += 1
            published_retries += max(0, row["publish_attempts"] - 1)
            active_leases += bool(
                row["lease_until"] is not None and row["lease_until"] > now
            )
            if (
                row["acked_at"] is not None
                and now - window_seconds <= row["acked_at"] <= now
            ):
                stored_in_window += 1
            if row["state"] == "stored":
                continue
            received = _epoch(row["received_at"])
            if received is None or now < received:
                age["invalid"] += 1
            elif clock_sync == "unknown":
                age["unknown"] += 1
            else:
                age["valid"] += 1
                seconds = now - received
                age["oldest_seconds"] = max(age["oldest_seconds"] or 0, seconds)
        attempt_errors = dict(
            connection.execute(
                "SELECT a.error,count(*) FROM monitoring_attempts a JOIN delivery_jobs j USING(job_id) "
                "WHERE j.destination=? AND (? IS NULL OR j.source_namespace=?) AND a.error IN "
                "('ack_mismatch','delivery_conflict','record_conflict') GROUP BY a.error",
                (binding.destination, namespace, namespace),
            ).fetchall()
        )
        arrivals, invalid_arrival_times = 0, 0
        for row in connection.execute(
            "SELECT received_at FROM ingest_receipts WHERE (? IS NULL OR source_namespace=?)",
            (namespace, namespace),
        ):
            received = _epoch(row[0])
            if received is None or received > now:
                invalid_arrival_times += 1
            elif received >= now - window_seconds:
                arrivals += 1
    return {
        "source_namespace": namespace,
        "destination": binding.destination,
        "measured_at": utc_text(now),
        "clock_sync": clock_sync,
        "states": dict(states),
        **pending,
        "unconfirmed_age": age,
        "active_leases": active_leases,
        "publish_retries_total": published_retries,
        "ack_invalid_total": attempt_errors.get("ack_mismatch", 0),
        "conflict_total": sum(
            attempt_errors.get(code, 0)
            for code in ("record_conflict", "delivery_conflict")
        ),
        "window_seconds": window_seconds,
        "ingest_count_in_window": arrivals if clock_sync == "confirmed" else None,
        "consumer_ack_recorded_in_window": (
            stored_in_window if clock_sync == "confirmed" else None
        ),
        "invalid_arrival_times": invalid_arrival_times,
        "ingest_per_hour": (
            arrivals * 3600 / window_seconds if clock_sync == "confirmed" else None
        ),
        "consumer_ack_recorded_per_hour": (
            stored_in_window * 3600 / window_seconds
            if clock_sync == "confirmed"
            else None
        ),
        "database_bytes": store.path.stat().st_size,
        "filesystem_free_bytes": shutil.disk_usage(store.path.parent).free,
        "not_measured": [
            "ingest_commit_completed_at",
            "normalize_completed_at",
            "consumer_ack_persisted_at",
            "received_to_ack_commit",
            "displayed_at",
            "consumer_duplicate_attempts",
        ],
    }
