"""DeepStream과 독립적인 낙상 보조판정 정책."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .events import DetectionEvent, EventType


def should_confirm_fall_with_aux(
    event: DetectionEvent,
    *,
    confirm_borderline: bool,
    aux_enabled: bool,
    max_fall_score: Optional[float],
    detector_score_threshold: float,
) -> bool:
    """이 이벤트를 발행 전에 보조 모델로 확인해야 하는지 반환한다."""
    if event.event_type != EventType.FALL_DETECTED or not confirm_borderline or not aux_enabled:
        return False
    score = float((event.metadata or {}).get("fall_score", 0.0))
    upper_score = detector_score_threshold if max_fall_score is None else max_fall_score
    return score <= float(upper_score)


def aux_result_confirms_fall(
    event_payload: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    compare_veto_enabled: bool,
    compare_veto_min_fall_score: float,
) -> bool:
    """보조 모델 결과가 최종 발행 가능한 낙상인지 반환한다."""
    if result.get("status") != "ok" or result.get("confirmed") is not True:
        return False
    metadata = event_payload.get("metadata") or {}
    compare_result = result.get("compare_model") or {}
    vetoed = (
        compare_veto_enabled
        and float(metadata.get("fall_score", 0.0)) >= float(compare_veto_min_fall_score)
        and compare_result.get("status") == "ok"
        and compare_result.get("confirmed") is False
    )
    return not vetoed
