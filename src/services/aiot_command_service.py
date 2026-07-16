from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Protocol

from src.aiot.command_store import CommandStore
from src.aiot.contracts import (
    AiQueryRequest,
    CommandValidationError,
    FetchMediaRequest,
    build_command_result,
    parse_ai_query_request,
    parse_fetch_media_request,
)


logger = logging.getLogger(__name__)


class ResultOutbox(Protocol):
    def store_result(
        self, request_id: str, payload: Mapping[str, Any], last_error: str
    ) -> None: ...


class AiotCommandService:
    def __init__(
        self,
        *,
        command_store: CommandStore,
        query_service: Any,
        media_uploader: Any,
        resolve_match: Callable[[str], Any],
        publish_result: Callable[[Mapping[str, Any]], bool],
        result_outbox: ResultOutbox,
        max_results: int = 20,
        metrics: Optional[Any] = None,
    ):
        self.command_store = command_store
        self.query_service = query_service
        self.media_uploader = media_uploader
        self.resolve_match = resolve_match
        self.publish_result = publish_result
        self.result_outbox = result_outbox
        self.max_results = max_results
        self.metrics = metrics

    def handle(self, payload: Mapping[str, Any]) -> None:
        request_id = str(payload.get("request_id") or "")
        message_type = str(payload.get("message_type") or "")
        if self.metrics is not None:
            self.metrics.commands_received.labels(message_type=message_type or "unknown").inc()
        try:
            request = self._parse(payload, message_type)
        except CommandValidationError as exc:
            if self.metrics is not None:
                self.metrics.commands_rejected.labels(reason="invalid_request").inc()
            if request_id:
                self._publish(
                    request_id,
                    build_command_result(
                        request_id, "failed", error_code="invalid_request", message=str(exc)
                    ),
                )
            return

        claim = self.command_store.claim(
            request.request_id, message_type, request.expires_at
        )
        if not claim.is_new:
            if self.metrics is not None:
                self.metrics.commands_duplicate.inc()
            record = self.command_store.get(request.request_id)
            if record and record.result_payload:
                self._publish(request.request_id, record.result_payload)
            return

        self._set_and_publish(request.request_id, "accepted")
        self._set_and_publish(request.request_id, "running")
        try:
            result = self._execute(request)
            completed = build_command_result(
                request.request_id,
                "completed",
                parent_request_id=getattr(request, "parent_request_id", None),
                **result,
            )
            self.command_store.update(request.request_id, "completed", completed)
            self._publish(request.request_id, completed)
        except Exception as exc:
            logger.warning("AIoT command failed request_id=%s: %s", request_id, exc)
            failed = build_command_result(
                request.request_id, "failed", error_code="execution_failed", message=str(exc)
            )
            self.command_store.update(request.request_id, "failed", failed)
            self._publish(request.request_id, failed)

    def _parse(
        self, payload: Mapping[str, Any], message_type: str
    ) -> AiQueryRequest | FetchMediaRequest:
        now = datetime.now(timezone.utc)
        if message_type == "ai_query_request":
            return parse_ai_query_request(payload, now=now, max_results=self.max_results)
        if message_type == "fetch_media_request":
            return parse_fetch_media_request(payload, now=now)
        raise CommandValidationError("invalid message_type")

    def _execute(
        self, request: AiQueryRequest | FetchMediaRequest
    ) -> dict[str, Any]:
        if isinstance(request, AiQueryRequest):
            started = time.monotonic()
            if self.metrics is not None:
                self.metrics.query_inflight.inc()
            try:
                matches = self.query_service.search(request)
                if self.metrics is not None:
                    self.metrics.query_matches.labels(
                        search_mode=request.search_mode
                    ).inc(len(matches))
                    self.metrics.query_duration.labels(
                        search_mode=request.search_mode, status="completed"
                    ).observe(time.monotonic() - started)
                return {"matches": matches}
            finally:
                if self.metrics is not None:
                    self.metrics.query_inflight.dec()
        if self.media_uploader is None:
            raise RuntimeError("media uploader is disabled")
        uploads = self.media_uploader.upload(request, self.resolve_match)
        return {
            "uploads": [
                {
                    "match_id": item.match_id,
                    "bytes_uploaded": item.bytes_uploaded,
                    "sha256": item.sha256,
                    "status": item.status,
                }
                for item in uploads
            ]
        }

    def _set_and_publish(self, request_id: str, status: str) -> None:
        self.command_store.update(request_id, status)
        self._publish(request_id, build_command_result(request_id, status))

    def _publish(self, request_id: str, payload: Mapping[str, Any]) -> None:
        try:
            published = self.publish_result(payload)
        except Exception as exc:
            published = False
            error = str(exc)
        else:
            error = "publisher returned false"
        if not published:
            self.result_outbox.store_result(request_id, payload, error)
