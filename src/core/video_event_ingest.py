"""Durable video event intake; memory queues carry receipts, recovery only records."""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, fields

from ..canonical_event import get_payload_occurred_at
from ..event_ingest_store import EventIngestStore
from ..event_validation import EventValidationError, validate_event_identity
from ..video_observation import build_video_observation
from ._event_publish import normalize_event_queue_item
from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)
NAMESPACE = "video_detection"
CONVERTER_VERSION = "video-v1"


@dataclass(frozen=True)
class VideoReceipt:
    ingest_id: str


class VideoEventIngest:
    def __init__(self, store, *, backend, topic_prefix):
        self.store = store
        self.backend = backend
        self.topic_prefix = topic_prefix
        self._live_receipts = set()
        self._lock = threading.RLock()
        self._next_recovery = 0.0

    @classmethod
    def from_env(cls, *, backend, topic_prefix):
        path = os.environ.get("AIOT_INGEST_DB", "").strip()
        if not path:
            return None
        if os.environ.get("MQTT_EVENT_OUTBOX_DB"):
            raise ValueError(
                "Common ingest and legacy MQTT outbox replay cannot run together"
            )
        return cls(EventIngestStore(path), backend=backend, topic_prefix=topic_prefix)

    def accept(self, queue_item, camera_name):
        is_detection = isinstance(queue_item, DetectionEvent)
        # Capture dataclass fields without invoking to_dict or canonical conversion.
        payload = (
            {
                field.name: getattr(queue_item, field.name)
                for field in fields(queue_item)
            }
            if is_detection
            else queue_item
        )
        raw = {
            "kind": "detection" if is_detection else "dict",
            "payload": payload,
            "camera_name": camera_name,
            "backend": self.backend,
        }
        with self._lock:
            ingest_id = self.store.accept_event_json(
                NAMESPACE, raw, destination=f"mqtt:{self.topic_prefix}", busy_retries=1
            )
            self._live_receipts.add(ingest_id)
        return VideoReceipt(ingest_id)

    def defer(self, reference):
        with self._lock:
            self._live_receipts.discard(reference.ingest_id)

    def _normalize(self, receipt):
        if receipt["raw_format"] != "json":
            raise EventValidationError([{"field": "raw", "code": "non_json_value"}])
        raw = receipt["raw"]
        if not isinstance(raw["payload"], dict):
            raise EventValidationError([{"field": "raw", "code": "invalid_object"}])
        if raw["kind"] == "detection":
            arguments = dict(raw["payload"])
            if arguments.get("metadata") is not None and not isinstance(
                arguments["metadata"], dict
            ):
                raise EventValidationError(
                    [{"field": "metadata", "code": "invalid_object"}]
                )
            arguments["event_type"] = EventType(arguments["event_type"])
            data = DetectionEvent(**arguments).to_dict()
        else:
            data = dict(raw["payload"])
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise EventValidationError(
                [{"field": "metadata", "code": "invalid_object"}]
            )
        camera_id = data.get(
            "camera_id", (metadata or {}).get("camera_id", raw["camera_name"])
        )
        occurred_at = next(
            (
                data[key]
                for key in ("occurred_at", "timestamp", "received_at", "queued_at")
                if data.get(key) is not None
            ),
            None,
        )
        validate_event_identity(data, device_id=camera_id, occurred_at=occurred_at)
        event_type = data.get("type") or data.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise EventValidationError(
                [{"field": "type", "code": "invalid_event_type"}]
            )
        data["camera_id"] = camera_id
        # Canonical legacy fallback uses truthiness; preserve epoch zero explicitly.
        data["occurred_at"] = get_payload_occurred_at(
            {"occurred_at": "0" if occurred_at == 0 else occurred_at}
        )
        event, event_type, camera_id = normalize_event_queue_item(
            data, default_backend=raw["backend"]
        )
        if raw["kind"] == "detection":
            event["observation"] = build_video_observation(
                data,
                namespace=NAMESPACE,
                source_record_id=event["event_id"],
                device_id=camera_id,
                received_at=receipt["received_at"],
            )
        topic_prefix = receipt["destination"].removeprefix("mqtt:")
        return event, f"{topic_prefix}/{camera_id}/{event_type}"

    def _prepare(self, ingest_id, *, recovery):
        receipt = self.store.get_event_receipt(ingest_id)
        if receipt["stage"] != "pending":
            return None
        try:
            event, topic = self._normalize(receipt)
            json.dumps(event, allow_nan=False)
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ) as error:
            errors = (
                error.errors
                if isinstance(error, EventValidationError)
                else [{"field": "raw", "code": type(error).__name__}]
            )
            self.store.reject_event(
                ingest_id, errors, converter_version=CONVERTER_VERSION
            )
            return None
        _, job_id = self.store.record_event(
            ingest_id, event, converter_version=CONVERTER_VERSION, recovery=recovery
        )
        return (event, topic, job_id) if job_id else None

    def publish(self, reference, *, mqtt_publish, event_publisher):
        try:
            with self._lock:
                prepared = self._prepare(reference.ingest_id, recovery=False)
            if prepared is None:
                return False
            event, topic, job_id = prepared
            if not self.store.begin_live_delivery(job_id):
                return False
            submitted, error_name = False, None
            try:
                if mqtt_publish is not None:
                    submitted = mqtt_publish(topic, event) is not False
                else:
                    submitted = bool(event_publisher.publish_event(event))
                return submitted
            except Exception as error:
                error_name = type(error).__name__
                raise
            finally:
                self.store.finish_live_delivery(
                    job_id,
                    submitted=submitted,
                    error=(
                        error_name if submitted else error_name or "submission_failed"
                    ),
                )
        except Exception as error:
            self.store.retry_event(reference.ingest_id, type(error).__name__)
            raise
        finally:
            self.defer(reference)

    def recover_pending(self):
        with self._lock:
            for ingest_id in self.store.pending_events(NAMESPACE):
                if ingest_id in self._live_receipts:
                    continue
                try:
                    self._prepare(ingest_id, recovery=True)
                except Exception as error:
                    self.store.retry_event(ingest_id, type(error).__name__)
                    logger.warning(
                        "Video receipt pending recovery: %s (%s)",
                        ingest_id,
                        type(error).__name__,
                    )

    def poll_recovery(self):
        if time.monotonic() < self._next_recovery:
            return
        self._next_recovery = time.monotonic() + 1.0
        try:
            self.recover_pending()
        except Exception:
            logger.exception("Video receipt recovery remains pending")
