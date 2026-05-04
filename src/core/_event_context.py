"""DetectionEvent를 후처리 컨텍스트로 변환하는 유틸리티."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .events import DetectionEvent


def events_to_nearby_objects(events: Iterable[DetectionEvent]) -> List[Dict[str, Any]]:
    """외형 분석 nearby_objects 입력 포맷으로 변환한다."""
    return [
        {
            "class_name": event.class_name or event.event_type.value,
            "event_type": event.event_type.value,
            "x": event.x,
            "y": event.y,
            "width": event.width,
            "height": event.height,
            "confidence": event.confidence,
            "metadata": dict(event.metadata or {}),
        }
        for event in events
    ]
