"""영상 원본을 먼저 저장하고 오케스트레이션 결과는 별도 중앙 ACK 큐로 전달한다."""

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
    def __init__(
        self, store, *, backend, topic_prefix, edge_context=None, session_id=None
    ):
        self.store = store
        self.backend = backend
        self.topic_prefix = topic_prefix
        self.edge_context = edge_context
        self.session_id = session_id
        if edge_context is not None:
            if backend != "deepstream" or not session_id:
                raise ValueError("edge results require a supervised DeepStream session")
            edge_context.snapshot(session_id)
        self._live_receipts = set()
        self._lock = threading.RLock()
        self._next_recovery = 0.0

    @classmethod
    def from_env(cls, *, backend, topic_prefix):
        path = os.environ.get("AIOT_INGEST_DB", "").strip()
        context_path = os.environ.get("EDGE_RESULT_CONTEXT_DB", "").strip()
        if not path:
            if context_path:
                raise ValueError("edge results require AIOT_INGEST_DB")
            return None
        if os.environ.get("MQTT_EVENT_OUTBOX_DB"):
            raise ValueError(
                "Common ingest and legacy MQTT outbox replay cannot run together"
            )
        context = None
        if context_path:
            from ..services.edge_result_context import EdgeResultContext

            context = EdgeResultContext(context_path)
        try:
            return cls(
                EventIngestStore(path),
                backend=backend,
                topic_prefix=topic_prefix,
                edge_context=context,
                session_id=os.environ.get("EDGE_SERVICE_SESSION_ID"),
            )
        except BaseException:
            if context is not None:
                context.close()
            raise

    def stamp(self, queue_item, camera_name):
        if self.edge_context is None:
            return
        is_detection = isinstance(queue_item, DetectionEvent)
        metadata = dict(
            (queue_item.metadata if is_detection else queue_item.get("metadata")) or {}
        )
        camera_id = (
            metadata.get("camera_id", camera_name)
            if is_detection
            else queue_item.get("camera_id", metadata.get("camera_id", camera_name))
        )
        stamp = metadata.get("edge_context")
        if stamp is None:
            stamp = self.edge_context.capture(
                self.session_id,
                event_id=None if is_detection else queue_item.get("event_id"),
            )
        self.edge_context.check_capture(stamp)
        if (
            stamp["edge"]["session_id"] != self.session_id
            or stamp["edge"]["camera_id"] != camera_id
        ):
            raise ValueError("edge capture does not match this camera session")
        if (
            not is_detection
            and queue_item.get("event_id", stamp["event_id"]) != stamp["event_id"]
        ):
            raise ValueError("edge capture event identity changed")
        metadata["edge_context"] = stamp
        if is_detection:
            queue_item.metadata = metadata
        else:
            queue_item["metadata"] = metadata

    def accept(self, queue_item, camera_name):
        self.stamp(queue_item, camera_name)
        is_detection = isinstance(queue_item, DetectionEvent)
        # 정규화 전에 원본과 최초 접수 담당 정보를 함께 저장한다.
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
        if self.edge_context is not None:
            stamp = (metadata or {}).get("edge_context")
            self.edge_context.check_capture(stamp)
            data["event_id"] = stamp["event_id"]
            data["received_at"] = receipt["received_at"]
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
        # removeprefix는 Python 3.9부터 제공되므로 B의 3.8에서도 같은 접두어만 제거한다.
        destination = receipt["destination"]
        topic_prefix = (
            destination[5:] if destination.startswith("mqtt:") else destination
        )
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
        if self.edge_context is not None:
            # 중앙 송신 큐 commit 뒤 원본 접수를 완료해야 중간 종료 시 재처리가 가능하다.
            stamp = receipt["raw"]["payload"]["metadata"]["edge_context"]
            self.edge_context.enqueue(
                {"kind": "event", "event": event, "edge": stamp["edge"]}
            )
        _, job_id = self.store.record_event(
            ingest_id, event, converter_version=CONVERTER_VERSION, recovery=recovery
        )
        return (event, topic, job_id) if job_id else None

    def publish(self, reference, *, mqtt_publish, event_publisher):
        try:
            if self.edge_context is not None and mqtt_publish is not None:
                raise ValueError(
                    "edge results cannot bypass the central ownership gate"
                )
            with self._lock:
                prepared = self._prepare(reference.ingest_id, recovery=False)
            if prepared is None:
                return False
            event, topic, job_id = prepared
            if not self.store.begin_live_delivery(job_id):
                return False
            submitted, error_name = False, None
            try:
                if self.edge_context is not None:
                    # 이 단계의 성공은 로컬 접수만 뜻하며 중앙 ACK는 감독자가 별도로 확인한다.
                    submitted = True
                elif mqtt_publish is not None:
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
