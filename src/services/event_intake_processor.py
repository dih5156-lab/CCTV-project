"""수신 이벤트를 Action Layer 실행에 사용할 형태로 준비한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..canonical_event import (
    canonicalize_event_payload,
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_event_type,
)


@dataclass(frozen=True)
class PreparedEvent:
    """정규화와 사이트 설정 반영이 끝난 이벤트 정보다."""

    payload: Dict[str, Any]
    camera_id: str
    event_type: str
    mode: Any
    site_id: str | None
    confidence: float | None
    confidence_threshold: float | None

    @property
    def is_filtered(self) -> bool:
        """신뢰도 기준 미달로 조치를 생략해야 하는지 반환한다."""
        return (
            self.confidence_threshold is not None
            and self.confidence is not None
            and self.confidence < self.confidence_threshold
        )


class EventIntakeProcessor:
    """수신 이벤트 정규화·사이트 설정 반영·신뢰도 판정을 담당한다."""

    def __init__(self, *, site_registry, manual_mode) -> None:
        """사이트 설정 조회기와 수동 모드 값을 보관한다."""
        self._site_registry = site_registry
        self._manual_mode = manual_mode

    def prepare(self, payload: Dict[str, Any]) -> PreparedEvent:
        """이벤트 payload를 실행 분기에 필요한 공통 정보로 변환한다."""
        normalized = canonicalize_event_payload(
            payload, source="action-layer", source_type="action"
        )
        camera_id = get_payload_camera_id(normalized)
        mode, site_id = self._site_registry.resolve_mode(camera_id)
        settings = self._site_registry.resolve_action_settings(camera_id)
        event_type = get_payload_event_type(normalized).lower()
        threshold = settings.get("confidence_threshold")
        confidence = get_payload_confidence(normalized)

        display_message = settings.get("display_message") or ""
        tts_message = settings.get("tts_message") or ""
        if display_message or tts_message:
            normalized = dict(normalized)
            event_payload = dict(normalized.get("event") or {})
            if display_message:
                event_payload["display_message"] = display_message
                event_payload.setdefault("message", display_message)
            if tts_message:
                event_payload["tts_message"] = tts_message
            normalized["event"] = event_payload

        return PreparedEvent(
            payload=normalized,
            camera_id=camera_id,
            event_type=event_type,
            mode=mode,
            site_id=site_id,
            confidence=confidence,
            confidence_threshold=threshold,
        )
