"""Read-time sensor age; never rewrite records, ACKs, or measurement quality."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone


def _epoch(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.utcoffset() is not None else None
    except (ValueError, TypeError, AttributeError, OverflowError, OSError):
        return None


@dataclass(frozen=True)
class SensorFreshness:
    stale_after_seconds: float | None = None
    clock_sync: str = "unknown"

    def __post_init__(self):
        threshold = self.stale_after_seconds
        if threshold is not None and (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("stale_after_seconds must be positive and finite")
        if self.clock_sync not in ("confirmed", "unknown"):
            raise ValueError("clock_sync must be confirmed or unknown")

    def _at(self, observed, quality, as_of):
        result = {"status": "unknown", "age_seconds": None, "reason": None}
        if quality == "missing":
            reason = "observed_at_missing"
        elif quality != "valid" or observed is None:
            reason = "observed_at_invalid"
        elif as_of is None:
            reason = "reference_time_invalid"
        elif self.clock_sync != "confirmed":
            reason = "clock_sync_unknown"
        elif observed > as_of:
            reason = "future_observed_at"
        else:
            result["age_seconds"] = as_of - observed
            if self.stale_after_seconds is None:
                reason = "threshold_unset"
            elif result["age_seconds"] > self.stale_after_seconds:
                result["status"], reason = "stale", "age_exceeded"
            else:
                result["status"], reason = "fresh", "within_threshold"
        result["reason"] = reason
        return result

    def describe(self, item, *, now):
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("finite evaluation time required")
        evaluated_at = datetime.fromtimestamp(now, timezone.utc).isoformat()
        record = item.get("record")
        if item.get("validation_error") or not isinstance(record, dict):
            status, reason = "unknown", "unverifiable_record"
        else:
            if record.get("message_type") != "sensor_observation":
                record = record.get("observation")
            if (
                not isinstance(record, dict)
                or record.get("message_type") != "sensor_observation"
            ):
                status, reason = "not_applicable", "not_sensor_observation"
            else:
                status, reason = None, None
        if reason:
            current = {"status": status, "age_seconds": None, "reason": reason}
            arrival = dict(current)
        else:
            observed = _epoch(record.get("observed_at"))
            quality = record.get("observed_at_quality", "missing")
            current = self._at(observed, quality, now)
            received = _epoch(item.get("received_at"))
            # A future arrival cannot establish a trustworthy historical status.
            arrival = self._at(
                observed,
                quality,
                received if received is not None and received <= now else None,
            )
        return {
            **current,
            "evaluated_at": evaluated_at,
            "stale_after_seconds": self.stale_after_seconds,
            "clock_sync": self.clock_sync,
            "on_arrival": {**arrival, "received_at": item.get("received_at")},
        }
