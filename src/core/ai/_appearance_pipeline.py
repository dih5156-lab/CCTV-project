"""외형 분석 파이프라인 전담 모듈."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from ..events import DetectionEvent, EventType
from ._appearance_analyzer import AppearanceAnalyzer

if TYPE_CHECKING:
    from ...services.appearance_log import AppearanceLog

logger = logging.getLogger(__name__)


class AppearancePipeline:
    """외형 속성 추출, 로그 저장, 조건 매칭을 담당한다."""

    def __init__(self, appearance: AppearanceAnalyzer, crop_dir: Path) -> None:
        self._appearance = appearance
        self._crop_dir = crop_dir
        self._crop_dir.mkdir(parents=True, exist_ok=True)
        self._appearance_cooldown: Dict[str, float] = {}
        self._appearance_log: Optional["AppearanceLog"] = None

    def ensure_log(self) -> None:
        """AppearanceLog를 지연 초기화한다."""
        if self._appearance_log is None:
            try:
                from ...services.appearance_log import AppearanceLog

                self._appearance_log = AppearanceLog()
            except Exception:
                logger.warning("AppearanceLog 초기화 실패 – DB 기록 비활성화")
                self._appearance_log = False  # type: ignore[assignment]

    def save_person_crop(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        camera_id: str,
        track_id: Optional[int],
        ts: float,
    ) -> Optional[str]:
        """person bbox 영역을 JPEG로 저장하고 경로를 반환한다."""
        try:
            frame_h, frame_w = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            file_name = f"{camera_id}_{track_id}_{int(ts * 1000)}.jpg"
            file_path = self._crop_dir / file_name
            cv2.imwrite(str(file_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return str(file_path)
        except Exception:
            logger.debug("person crop 저장 실패", exc_info=True)
            return None

    def _build_face_meta_map(
        self, face_events: List[DetectionEvent]
    ) -> Dict[int, Dict]:
        face_meta_map: Dict[int, Dict] = {}
        for face_event in face_events:
            if face_event.metadata and face_event.object_id is not None:
                face_meta_map[face_event.object_id] = face_event.metadata
        return face_meta_map

    def _log_person_appearance(
        self,
        frame: np.ndarray,
        person: DetectionEvent,
        now: float,
        camera_id: str,
        nearby_objects: List[Dict],
        face_meta: Dict,
    ) -> None:
        log_key = f"_applog_{person.object_id}"
        last_log = self._appearance_cooldown.get(log_key, 0.0)
        if now - last_log < 3.0:
            return

        attrs = self._appearance.extract_attributes(
            frame,
            person.x,
            person.y,
            person.width,
            person.height,
            nearby_objects=nearby_objects,
        )
        gender = face_meta.get("gender")
        age_group = face_meta.get("age_group")
        face_name = face_meta.get("face_name")

        logger.info(
            "[외형] track=%s  상의=%s  하의=%s  모자=%s  백팩=%s  핸드백=%s  캐리어=%s  성별=%s  나이=%s",
            person.object_id,
            attrs.get("upper_color", "?"),
            attrs.get("lower_color", "?"),
            attrs.get("hat_color", "?"),
            attrs.get("has_backpack", False),
            attrs.get("has_handbag", False),
            attrs.get("has_suitcase", False),
            gender or "?",
            age_group or "?",
        )
        self._appearance_cooldown[log_key] = now

        crop_path = self.save_person_crop(
            frame, person.x, person.y, person.width, person.height, camera_id, person.object_id, now
        )

        if self._appearance_log:
            self._appearance_log.insert(
                camera_id=camera_id,
                track_id=person.object_id,
                upper_color=attrs.get("upper_color"),
                lower_color=attrs.get("lower_color"),
                hat_color=attrs.get("hat_color"),
                has_backpack=bool(attrs.get("has_backpack")),
                has_handbag=bool(attrs.get("has_handbag")),
                has_suitcase=bool(attrs.get("has_suitcase")),
                gender=gender,
                age_group=age_group,
                face_name=face_name,
                crop_path=crop_path,
                bbox_x=person.x,
                bbox_y=person.y,
                bbox_w=person.width,
                bbox_h=person.height,
                timestamp=now,
            )

    def run_matching(
        self,
        frame: np.ndarray,
        person_events: List[DetectionEvent],
        camera_id: Optional[str] = None,
        cooldown: float = 5.0,
        nearby_objects: Optional[List[Dict]] = None,
    ) -> List[DetectionEvent]:
        """외형 조건 매칭 이벤트를 생성한다."""
        if frame is None or not person_events:
            return []

        appearance_events: List[DetectionEvent] = []
        now = time.time()

        for person in person_events:
            matches = self._appearance.find_matches(
                frame,
                person.x,
                person.y,
                person.width,
                person.height,
                camera_id=camera_id,
                nearby_objects=nearby_objects,
            )
            for match in matches:
                cooldown_key = f"{person.object_id}:{match['condition_id']}"
                last_ts = self._appearance_cooldown.get(cooldown_key, 0.0)
                if now - last_ts < cooldown:
                    continue
                self._appearance_cooldown[cooldown_key] = now

                appearance_events.append(
                    DetectionEvent(
                        event_type=EventType.APPEARANCE_MATCH,
                        x=person.x,
                        y=person.y,
                        width=person.width,
                        height=person.height,
                        confidence=match["score"],
                        timestamp=now,
                        object_id=person.object_id,
                        class_name="person",
                        metadata={
                            "condition_id": match["condition_id"],
                            "condition_name": match["condition_name"],
                            "upper_color": match["attributes"]["upper_color"],
                            "lower_color": match["attributes"]["lower_color"],
                            "hat_color": match["attributes"].get("hat_color", "unknown"),
                            "has_backpack": match["attributes"].get("has_backpack", False),
                            "has_handbag": match["attributes"].get("has_handbag", False),
                            "has_suitcase": match["attributes"].get("has_suitcase", False),
                            "match_score": match["score"],
                        },
                    )
                )

        stale = [
            key for key, timestamp in self._appearance_cooldown.items()
            if now - timestamp > cooldown * 5
        ]
        for key in stale:
            self._appearance_cooldown.pop(key, None)

        if appearance_events:
            logger.info("외형 매칭: %d건 발생", len(appearance_events))
        return appearance_events

    def run(
        self,
        frame: np.ndarray,
        person_events: List[DetectionEvent],
        face_events: List[DetectionEvent],
        *,
        camera_id: Optional[str],
        use_appearance: bool,
        nearby_objects: Optional[List[Dict]] = None,
    ) -> List[DetectionEvent]:
        """외형 속성 추출, 로그 저장, 조건 매칭을 순서대로 수행한다."""
        if not use_appearance or not person_events:
            return []

        resolved_camera_id = camera_id or "unknown"
        now = time.time()
        nearby = nearby_objects or []
        self.ensure_log()
        face_meta_map = self._build_face_meta_map(face_events)

        for person in person_events:
            self._log_person_appearance(
                frame,
                person,
                now,
                resolved_camera_id,
                nearby,
                face_meta_map.get(person.object_id, {}),
            )

        if not self._appearance.conditions:
            return []

        return self.run_matching(
            frame,
            person_events,
            camera_id=resolved_camera_id,
            nearby_objects=nearby,
        )
