"""객체 추적 ID 관리 — ObjectTracker.

YOLO track() 결과에서 ID를 추출하고, track ID가 없는 경우
IoU 기반 bbox 매칭으로 임시 ID를 일관되게 유지한다.
추론 스레드 간 공유 없이 AIAnalyzer 인스턴스마다 독립적으로 소유한다.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from ._constants import (
    _TEMP_TRACK_ID_END,
    _TEMP_TRACK_ID_START,
    _TEMP_TRACK_MAX_AREA_RATIO_DELTA,
    _TEMP_TRACK_MAX_CENTER_RATIO,
    _TEMP_TRACK_MIN_IOU,
    _TEMP_TRACK_TTL_SEC,
)
from ._yolo_helpers import bbox_iou_from_coords, center_distance_ratio, extract_track_id

logger = logging.getLogger(__name__)


class ObjectTracker:
    """임시 track ID 할당 및 bbox 기반 매칭.

    YOLO 트래커가 ID를 제공하지 않을 때 직전 프레임 bbox와의 IoU·중심 거리로
    동일 객체인지 판별하여 안정적인 임시 ID를 부여한다.
    """

    def __init__(self) -> None:
        self._next_id: int = _TEMP_TRACK_ID_START
        self._cache:   Dict[int, Dict] = {}

    # ── 공개 API ──────────────────────────────────────────────────────

    def resolve_id(
        self,
        box,
        x1: int,
        y1: int,
        width: int,
        height: int,
        track_group: str,
        now_ts: Optional[float] = None,
    ) -> int:
        """YOLO track ID가 없을 때 최근 bbox와 매칭해 안정적인 임시 ID를 유지한다."""
        track_id = extract_track_id(box)
        if track_id is not None:
            return track_id

        now_ts = time.time() if now_ts is None else now_ts
        self._cleanup(now_ts)

        bbox     = (x1, y1, width, height)
        area     = max(width, 0) * max(height, 0)
        best_id:    Optional[int]  = None
        best_score: float          = -1.0

        for cached_id, state in self._cache.items():
            if state["group"] != track_group:
                continue
            cached_bbox  = state["bbox"]
            iou          = bbox_iou_from_coords(bbox, cached_bbox)
            cdist        = center_distance_ratio(bbox, cached_bbox)
            cached_area  = max(cached_bbox[2], 0) * max(cached_bbox[3], 0)
            area_delta   = abs(area - cached_area) / max(area, cached_area, 1)

            if iou < _TEMP_TRACK_MIN_IOU and cdist > _TEMP_TRACK_MAX_CENTER_RATIO:
                continue
            if area_delta > _TEMP_TRACK_MAX_AREA_RATIO_DELTA:
                continue

            score = iou - (cdist * 0.1) - (area_delta * 0.05)
            if score > best_score:
                best_score = score
                best_id    = cached_id

        if best_id is None:
            best_id = self._allocate()

        self._cache[best_id] = {"group": track_group, "bbox": bbox, "last_seen": now_ts}
        return best_id

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _allocate(self) -> int:
        """단조 증가 임시 ID를 발급한다 (충돌 확률 최소화)."""
        track_id     = self._next_id
        self._next_id += 1
        if self._next_id > _TEMP_TRACK_ID_END:
            self._next_id = _TEMP_TRACK_ID_START
        return track_id

    def _cleanup(self, now_ts: float) -> None:
        """만료된 캐시 항목을 제거한다."""
        expired = [
            tid for tid, state in self._cache.items()
            if now_ts - float(state["last_seen"]) > _TEMP_TRACK_TTL_SEC
        ]
        for tid in expired:
            self._cache.pop(tid, None)
