"""YOLO 객체 탐지와 속성 분석 실행 순서 전담 모듈."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

from ..events import DetectionEvent

if TYPE_CHECKING:
    from .analyzer import AIAnalyzer

logger = logging.getLogger(__name__)


class ObjectDetectionPipeline:
    """사람/낙상/헬멧/얼굴/외형 속성 추론을 순서대로 실행한다."""

    def __init__(self, analyzer: "AIAnalyzer") -> None:
        self._analyzer = analyzer

    def run(
        self,
        frame,
        *,
        use_helmet: bool,
        use_pose: bool,
        use_person: bool,
        use_face: bool,
        use_appearance: bool,
        camera_id: str | None,
    ) -> List[DetectionEvent]:
        """프레임에 대한 전체 YOLO+속성 추론을 수행한다."""
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        person_events, fall_events = self.detect_primary_people(
            frame,
            use_pose=use_pose,
            use_person=use_person,
        )
        helmet_events = self.detect_helmet_events(
            frame,
            person_events,
            fall_events,
            use_helmet=use_helmet,
        )
        face_events = (
            self._analyzer._run_face_recognition(frame, person_events)
            if use_face and person_events
            else []
        )
        appearance_events = self._analyzer._run_appearance_pipeline(
            frame,
            person_events,
            face_events,
            camera_id=camera_id,
            use_appearance=use_appearance,
            nearby_objects=self._analyzer._build_appearance_nearby_objects(
                getattr(self._analyzer, "_last_bag_objects", []),
                helmet_events,
            ),
        )

        return fall_events + person_events + face_events + helmet_events + appearance_events

    def detect_primary_people(
        self,
        frame: np.ndarray,
        *,
        use_pose: bool,
        use_person: bool,
    ) -> tuple[List[DetectionEvent], List[DetectionEvent]]:
        """포즈 모델 우선, 없으면 일반 YOLO person 모델로 사람을 감지한다."""
        analyzer = self._analyzer
        person_events: List[DetectionEvent] = []
        fall_events: List[DetectionEvent] = []

        if use_pose and analyzer.pose_model:
            person_events, fall_events = analyzer._run_pose_full_frame(frame)
            logger.debug(
                "포즈 모델(전체 프레임): 사람 %d명, 낙상 %d건",
                len(person_events),
                len(fall_events),
            )
            return person_events, fall_events

        if use_person:
            if analyzer.person_model:
                analyzer._last_bag_objects = []
                person_events = analyzer._run_single_model(
                    analyzer.person_model,
                    frame,
                    model_type="person",
                )
                logger.debug("사람 모델(폴백): %d 감지됨", len(person_events))
            elif not analyzer._person_warning_shown:
                logger.warning("포즈 모델과 사람 모델이 모두 없어 사람 감지가 불가합니다.")
                analyzer._person_warning_shown = True

        return person_events, fall_events

    def detect_helmet_events(
        self,
        frame: np.ndarray,
        person_events: List[DetectionEvent],
        fall_events: List[DetectionEvent],
        *,
        use_helmet: bool,
    ) -> List[DetectionEvent]:
        """사람 ROI 상단에 헬멧 모델을 적용한다."""
        analyzer = self._analyzer
        if not use_helmet:
            return []

        if not analyzer.helmet_model:
            if not analyzer._helmet_warning_shown:
                logger.warning("헬멧 모델이 로드되지 않았습니다.")
                analyzer._helmet_warning_shown = True
            return []

        if not person_events:
            return []

        fallen_ids = {event.object_id for event in fall_events}
        standing_people = [
            person for person in person_events if person.object_id not in fallen_ids
        ]
        if fallen_ids:
            logger.debug("낙상자 %d명 헬멧 탐지 제외", len(fallen_ids))
        if not standing_people:
            return []

        raw_events = analyzer._run_helmet_on_person_rois(frame, standing_people)
        logger.debug(
            "헬멧 모델: %d 감지됨 (threshold=%s)",
            len(raw_events),
            getattr(analyzer, "helmet_threshold", analyzer.confidence_threshold),
        )
        return analyzer._filter_helmet_boxes(raw_events)
