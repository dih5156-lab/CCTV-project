"""DetectionEvent 후처리 컨텍스트 관련 유틸리티."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..services.appearance_conditions import AppearanceConditionStore
from .events import DetectionEvent

logger = logging.getLogger(__name__)


def events_to_nearby_objects(events: Iterable[DetectionEvent]) -> List[Dict[str, Any]]:
    """외형 분석 nearby_objects 입력 포맷으로 변환한다."""
    return [
        {
            "class_name": event.class_name or event.event_type.value,
            "event_type": event.event_type.value,
            **event.bbox_dict(),
            "confidence": event.confidence,
            "metadata": dict(event.metadata or {}),
        }
        for event in events
    ]


def refresh_appearance_conditions(
    *,
    appearance_enabled_default: bool,
    camera_ai_flags: Dict[str, Dict[str, bool]],
    appearance: Any,
    appearance_db_path: Path,
    current_mtime: Optional[float],
    checked_at: float,
    refresh_sec: float,
    now_monotonic: float,
) -> Tuple[float, Optional[float]]:
    """외형 조건 DB를 주기적으로 새로 읽고 갱신된 상태값을 반환한다."""
    if not appearance_enabled_default and not any(
        flags.get("use_appearance") for flags in camera_ai_flags.values()
    ):
        return checked_at, current_mtime

    if now_monotonic - checked_at < refresh_sec:
        return checked_at, current_mtime
    checked_at = now_monotonic

    try:
        stat = appearance_db_path.stat()
    except FileNotFoundError:
        if appearance.conditions:
            appearance.set_conditions([])
        return checked_at, None
    except OSError as exc:
        logger.debug("외형 조건 DB stat 실패: %s", exc)
        return checked_at, current_mtime

    if current_mtime == stat.st_mtime:
        return checked_at, current_mtime

    conditions = AppearanceConditionStore(appearance_db_path).list_all()
    appearance.set_conditions(conditions)
    return checked_at, stat.st_mtime


def log_appearance_capability_hints(
    *,
    logged_cameras: Set[str],
    camera_name: str,
    flags: Dict[str, bool],
    backend_name: str,
    pphuman_sgie_enabled: bool,
    pphuman_config_exists: bool,
    yolo_labels: Sequence[str],
    bag_classes: Set[str],
    face_recognizer_enabled: bool,
    helmet_enabled: bool,
) -> None:
    """외형 검색 가능 여부를 카메라별로 1회 로그로 남긴다."""
    if camera_name in logged_cameras:
        return

    pphuman_sgie_active = (
        pphuman_sgie_enabled
        and flags.get("use_appearance", False)
        and pphuman_config_exists
    )
    bag_labels = sorted(
        label for label in yolo_labels if str(label).strip().lower() in bag_classes
    )
    gender_ready = bool(flags.get("use_face")) and face_recognizer_enabled
    helmet_ready = bool(flags.get("use_helmet")) and helmet_enabled
    bag_ready = bool(bag_labels) or backend_name != "hsv"

    logger.info(
        "[%s] 외형 검색 컨텍스트: backend=%s, pphuman_sgie=%s, gender_ready=%s, helmet_ready=%s, bag_ready=%s",
        camera_name,
        backend_name,
        pphuman_sgie_active,
        gender_ready,
        helmet_ready,
        bag_ready,
    )

    if not gender_ready:
        logger.warning(
            "[%s] use_face가 꺼져 있거나 얼굴 인식이 비활성화되어 gender 값이 비어 있을 수 있습니다.",
            camera_name,
        )

    if not helmet_ready:
        logger.warning(
            "[%s] use_helmet 또는 DS_HELMET_ENABLED가 꺼져 있어 has_helmet 검색값이 채워지지 않을 수 있습니다.",
            camera_name,
        )

    if not bag_ready:
        logger.warning(
            "[%s] 현재 backend=%s 이고 bag class labels=%s 이라 backpack/handbag/suitcase 값이 채워지기 어렵습니다.",
            camera_name,
            backend_name,
            ",".join(bag_labels) if bag_labels else "none",
        )

    logged_cameras.add(camera_name)
