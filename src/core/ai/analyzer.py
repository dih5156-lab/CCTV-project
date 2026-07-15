"""AIAnalyzer — 멀티 모델 AI 분석 시스템 오케스트레이터.

구성:
- 포즈 모델(yolov8-pose): 사람 탐지 + 낙상 감지 (전체 프레임)
- 사람 모델(yolov8n):     포즈 모델 없을 때 fallback
- 헬멧 모델:              사람 머리 ROI 기반 헬멧 감지

내부적으로 아래 전담 컴포넌트를 사용한다:
  ObjectTracker  — track ID 관리  (_object_tracker.py)
  FallDetector   — 낙상·사람 검증 (_fall_detector.py)
  _yolo_helpers  — YOLO 결과 추출 유틸리티
  _constants     — 공유 상수·_MODEL_IMGSZ·_IMGSZ_LOCK
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...utils.geometry import boxes_overlap, is_helmet_worn
from ..events import DetectionEvent, EventType
from ._appearance_analyzer import BAG_CLASSES, AppearanceAnalyzer
from ._appearance_pipeline import AppearancePipeline
from ._constants import (
    _IMGSZ_LOCK,
    _MODEL_IMGSZ,
    DEFAULT_IMAGE_SIZE_HELMET,
    DEFAULT_IMAGE_SIZE_POSE,
    DEFAULT_IOU_THRESHOLD,
    DUPLICATE_IOU_THRESHOLD,
    HEAD_REGION_RATIO,
    MAX_HELMET_ASPECT_RATIO,
    MAX_HELMET_HEIGHT,
    MAX_HELMET_WIDTH,
    MIN_HELMET_SIZE,
    MIN_PERSON_HEIGHT,
    MIN_PERSON_WIDTH,
    PERSON_DUPLICATE_IOU_THRESHOLD,
)
from ._face_recognition_pipeline import FaceRecognitionPipeline
from ._fall_detector import FallDetector
from ._falldata_aux import FallDataAuxVerifier
from ._object_detection_pipeline import ObjectDetectionPipeline
from ._object_tracker import ObjectTracker
from ._yolo_helpers import (
    detect_engine_imgsz,
    extract_bbox,
    extract_confidence,
    extract_keypoints,
    generate_temp_id,
)

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

# Jetson Orin cuDNN 안정화 — "GET was unable to find an engine" 에러 방지
try:
    import torch
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass


class AIAnalyzer:
    """멀티 모델 AI 분석 오케스트레이터.

    모델 로딩·설정 관리, YOLO 추론 실행, 헬멧/낙상/얼굴 이벤트 생성을
    모두 담당하되, 낙상 감지/사람 검증은 FallDetector에,
    track ID 관리는 ObjectTracker에 위임한다.
    """

    # ── 클래스 매핑 상수 ──────────────────────────────────────────────
    _HELMET_CLASS_MAP: Dict[str, str] = {
        "helmet_missing": "head",
        "no_helmet":      "head",
        "helmet":         "helmet",
        "helmet_wearing": "helmet",
        "head":           "head",
    }
    _COMMON_CLASS_MAP: Dict[str, str] = {
        "danger_zone":      "danger_zone",
        "unsafe_behavior":  "unsafe_behavior",
        "unsafe":           "unsafe_behavior",
        "person":           "person",
        "face_recognized":  "face_recognized",
        "face_unknown":     "face_unknown",
    }
    _CLASS_MAP: Dict[str, str] = {**_HELMET_CLASS_MAP, **_COMMON_CLASS_MAP}

    # ── 초기화 ────────────────────────────────────────────────────────

    def __init__(
        self,
        model_path: Optional[str] = None,          # 하위 호환성: pose_model_path 별칭
        helmet_model_path: Optional[str] = None,
        person_model_path: Optional[str] = None,
        pose_model_path:   Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        fall_height_ratio: float = 0.3,
        appearance_backend: str = "hsv",
        appearance_model_path: Optional[str] = None,
        appearance_label_map_path: Optional[str] = None,
        appearance_runtime: str = "auto",
        appearance_input_size: int = 224,
        appearance_score_threshold: float = 0.5,
        appearance_bbox_expand_ratio: float = 0.15,
    ) -> None:
        # 하위 호환성: model_path → pose_model_path
        if model_path and not pose_model_path:
            pose_model_path = model_path

        self.helmet_model_path = helmet_model_path
        self.person_model_path = person_model_path
        self.pose_model_path   = pose_model_path
        self.confidence_threshold = confidence_threshold
        self.device            = device
        self.fall_height_ratio = fall_height_ratio

        # 모델 객체
        self.helmet_model = None
        self.person_model = None
        self.pose_model   = None

        # 내부 상태
        self.last_load_errors: List[Tuple[str, str]] = []
        self._person_warning_shown  = False
        self._helmet_warning_shown  = False
        self._last_bag_objects: List[Dict] = []

        # 전담 컴포넌트
        self._tracker      = ObjectTracker()
        self._fall         = FallDetector(fall_height_ratio)
        self._falldata_aux = FallDataAuxVerifier()
        self._face_recognizer = None
        self._face_pipeline = FaceRecognitionPipeline(lambda: self.face_recognizer)
        self._appearance   = AppearanceAnalyzer(
            backend_name=appearance_backend,
            backend_model_path=appearance_model_path,
            backend_label_map_path=appearance_label_map_path,
            backend_runtime=appearance_runtime,
            backend_device=self.device,
            backend_input_size=appearance_input_size,
            backend_score_threshold=appearance_score_threshold,
            bbox_expand_ratio=appearance_bbox_expand_ratio,
            color_model_path=os.environ.get("APPEARANCE_COLOR_MODEL_PATH"),
            color_label_map_path=os.environ.get("APPEARANCE_COLOR_LABEL_MAP_PATH"),
            color_input_size=int(os.environ.get("APPEARANCE_COLOR_INPUT_SIZE", "160")),
            color_score_threshold=float(os.environ.get("APPEARANCE_COLOR_SCORE_THRESHOLD", "0.75")),
        )
        self._crop_dir = Path(
            os.environ.get("APPEARANCE_CROP_DIR", "data/runtime/appearance_crops")
        )
        self._appearance_pipeline = AppearancePipeline(
            self._appearance,
            self._crop_dir,
            save_crops=os.environ.get("APPEARANCE_SAVE_CROPS", "").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        self._object_pipeline = ObjectDetectionPipeline(self)

        if YOLO is None:
            raise ImportError("ultralytics 패키지가 필요합니다 (`pip install ultralytics`)")

        self.load_models()

    @property
    def face_recognizer(self):
        """얼굴 인식 엔진은 실제 사용 시점에 초기화한다."""
        if self._face_recognizer is None:
            from ...utils.face_recognition import FaceRecognitionEngine

            self._face_recognizer = FaceRecognitionEngine(device=self.device)
        return self._face_recognizer

    # ── 모델 관리 ─────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        """단일 YOLO 모델 로드."""
        if YOLO is None:
            raise RuntimeError("YOLO 라이브러리를 찾을 수 없습니다 (ultralytics 설치 필요).")
        if not model_path:
            return None

        p = Path(model_path)
        if not p.exists():
            base = Path(p.name)
            if base.exists():
                model_path = str(base)
            else:
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        try:
            model = YOLO(model_path)
            try:
                model.to(self.device)
            except Exception:
                pass

            # ultralytics 버전과 .pt 버전 불일치 시 fuse() AttributeError 패치
            inner = getattr(model, "model", None)
            if inner is not None:
                _orig_fuse = getattr(inner, "fuse", None)
                if callable(_orig_fuse):
                    def _safe_fuse(verbose=True, _orig=_orig_fuse):
                        try:
                            return _orig(verbose=verbose)
                        except AttributeError as _e:
                            logger.debug("fuse() 건너뜀 (버전 불일치 무시): %s", _e)
                            return inner
                    inner.fuse = _safe_fuse

            logger.info("모델 로드 성공: %s (device=%s)", model_path, self.device)
            return model
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다 ({model_path}): {exc}")
        except Exception as exc:
            raise RuntimeError(f"모델 로드 실패 ({model_path}): {exc}")

    def _try_load(self, name: str, path: Optional[str]) -> None:
        """단일 모델 로드를 시도하고 결과를 인스턴스 속성에 저장."""
        attr = f"{name}_model"
        if not path:
            logger.warning("%s 모델 경로가 지정되지 않음", name)
            return
        try:
            setattr(self, attr, self._load_model(path))
            logger.info("%s 모델 로드 완료: %s", name, path)
        except Exception as exc:
            setattr(self, attr, None)
            self.last_load_errors.append((name, str(exc)))
            logger.warning("%s 모델 로드 실패: %s", name, exc)

    def load_models(self) -> None:
        """헬멧·포즈 모델을 우선 로드하고 필요 시에만 person 모델을 로드한다."""
        self.last_load_errors.clear()
        self._try_load("helmet", self.helmet_model_path)
        self._try_load("pose",   self.pose_model_path)
        if self.pose_model is None:
            self._try_load("person", self.person_model_path)
        elif self.person_model_path:
            logger.info("pose 모델이 활성화되어 person 모델 로드는 건너뜁니다.")

        if not any([self.helmet_model, self.person_model, self.pose_model]):
            logger.error("로드된 모델이 없습니다. 경로/라이브러리/파일을 확인하세요.")
        else:
            logger.info(
                "로드된 모델: Helmet=%s, Person=%s, Pose=%s",
                bool(self.helmet_model), bool(self.person_model), bool(self.pose_model),
            )

        # TRT .engine 파일 imgsz 자동 감지
        _MODEL_IMGSZ["helmet"] = detect_engine_imgsz(self.helmet_model, DEFAULT_IMAGE_SIZE_HELMET)
        _MODEL_IMGSZ["pose"]   = detect_engine_imgsz(self.pose_model,   DEFAULT_IMAGE_SIZE_POSE)
        logger.info(
            "imgsz 설정 → helmet=%d, pose=%d, person=%d",
            _MODEL_IMGSZ["helmet"], _MODEL_IMGSZ["pose"], _MODEL_IMGSZ["person"],
        )

    def get_loaded_model_names(self) -> Dict[str, Optional[Dict[int, str]]]:
        """로드된 모델 클래스명 조회 (디버깅용)."""
        res: Dict[str, Optional[Dict[int, str]]] = {"helmet": None, "person": None, "pose": None}
        for k in ("helmet", "person", "pose"):
            m = getattr(self, f"{k}_model", None)
            if m:
                try:
                    res[k] = getattr(m, "names", None)
                except Exception:
                    pass
        return res

    def set_device(self, device: str = "cpu") -> None:
        """디바이스 설정 및 모델 이동."""
        self.device = device
        for m in (self.helmet_model, self.person_model, self.pose_model):
            if m is not None:
                try:
                    m.to(device)
                except Exception as exc:
                    logger.warning("디바이스 설정 실패: %s", exc)
        logger.info("디바이스 설정 완료: %s", device)

    def update_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 업데이트."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"임계값은 0.0~1.0 사이여야 합니다 (입력값: {threshold})")
        self.confidence_threshold = threshold
        logger.info("신뢰도 임계값 업데이트: %s", threshold)

    # ── 클래스 매핑 ───────────────────────────────────────────────────

    def _map_class_to_event_type(self, class_name: str, model_type: str) -> EventType:
        """클래스명을 EventType으로 매핑."""
        if not class_name:
            return EventType.OTHER
        normalized   = class_name.lower().strip().replace(" ", "_")
        mapped_str   = self._CLASS_MAP.get(normalized)

        if model_type == "helmet":
            return {
                "head":             EventType.HEAD,
                "helmet":           EventType.HELMET,
                "danger_zone":      EventType.DANGER_ZONE,
                "unsafe_behavior":  EventType.UNSAFE_BEHAVIOR,
                "person":           EventType.PERSON,
            }.get(mapped_str, EventType.OTHER)

        if model_type == "person":
            return EventType.PERSON if normalized == "person" else EventType.OTHER

        return EventType.OTHER

    def _threshold_for_model(self, model_type: str) -> float:
        if model_type == "helmet":
            return getattr(self, "helmet_threshold", self.confidence_threshold)
        if model_type == "person":
            return getattr(self, "person_threshold", self.confidence_threshold)
        return self.confidence_threshold

    # ── YOLO 추론 공통 ────────────────────────────────────────────────

    def _run_single_model(
        self,
        model,
        frame,
        use_tracking: bool = True,
        model_type: str = "unknown",
    ) -> List[DetectionEvent]:
        """단일 YOLO 모델 결과를 DetectionEvent 리스트로 변환."""
        events: List[DetectionEvent] = []
        if model is None or frame is None:
            return events

        conf_threshold = self._threshold_for_model(model_type)
        with _IMGSZ_LOCK:
            imgsz = _MODEL_IMGSZ.get(model_type, DEFAULT_IMAGE_SIZE_HELMET)

        try:
            if use_tracking:
                results = model.track(
                    frame, conf=conf_threshold, iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=imgsz, verbose=False, persist=True,
                )
            else:
                results = model.predict(
                    frame, conf=conf_threshold, iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=imgsz, verbose=False,
                )
        except Exception as exc:
            logger.error("모델 추론 실패 (%s): %s", model_type, exc, exc_info=True)
            return events

        logger.debug("[%s] 추론 완료: %d개 결과", model_type, len(results))

        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", None) or {}
            if boxes is None:
                continue
            logger.debug("[%s] 감지된 박스: %d개", model_type, len(boxes))

            for box in boxes:
                bbox = extract_bbox(box)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                width  = x2 - x1
                height = y2 - y1

                conf = extract_confidence(box)

                cls_idx: Optional[int] = None
                try:
                    cls_t = box.cls[0]
                    cls_idx = int(cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else cls_t)
                except (ValueError, TypeError, IndexError):
                    pass

                class_name: Optional[str] = None
                if cls_idx is not None and isinstance(names, (dict, list)):
                    class_name = (
                        names.get(cls_idx) if isinstance(names, dict)
                        else (names[cls_idx] if cls_idx < len(names) else None)
                    )

                if model_type == "person":
                    lower_cls = (class_name or "").lower()
                    if lower_cls in BAG_CLASSES:
                        self._last_bag_objects.append({
                            "class_name": lower_cls,
                            "x": x1, "y": y1,
                            "width": width, "height": height,
                            "confidence": conf,
                        })
                        continue
                    if lower_cls != "person":
                        continue

                event_type = self._map_class_to_event_type(class_name or "", model_type)
                if event_type == EventType.OTHER:
                    continue

                track_id = self._tracker.resolve_id(
                    box, x1, y1, width, height,
                    track_group=f"{model_type}:{event_type.value}",
                )

                events.append(DetectionEvent(
                    event_type=event_type,
                    x=x1, y=y1, width=width, height=height,
                    confidence=conf,
                    timestamp=time.time(),
                    object_id=track_id,
                    class_idx=cls_idx,
                    class_name=class_name,
                ))

        return events

    # ── 포즈 모델 추론 ────────────────────────────────────────────────

    def _run_pose_full_frame(self, frame) -> Tuple[List[DetectionEvent], List[DetectionEvent]]:
        """포즈 모델을 전체 프레임에서 실행 → (person_events, fall_events)."""
        person_events: List[DetectionEvent] = []
        fall_events:   List[DetectionEvent] = []

        if frame is None or self.pose_model is None:
            return person_events, fall_events

        conf_threshold = getattr(self, "pose_threshold", self.confidence_threshold)
        with _IMGSZ_LOCK:
            imgsz = _MODEL_IMGSZ.get("pose", DEFAULT_IMAGE_SIZE_POSE)

        try:
            results = self.pose_model.track(
                frame, conf=conf_threshold, iou=DEFAULT_IOU_THRESHOLD,
                imgsz=imgsz, verbose=False, persist=True,
            )
        except Exception as exc:
            logger.error("포즈 모델 전체 프레임 추론 실패: %s", exc, exc_info=True)
            return person_events, fall_events

        for result in results:
            boxes     = getattr(result, "boxes",     None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None:
                continue

            for idx, box in enumerate(boxes):
                bbox = extract_bbox(box)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                width  = x2 - x1
                height = y2 - y1

                if width < MIN_PERSON_WIDTH or height < MIN_PERSON_HEIGHT:
                    logger.debug("사람 bbox 크기 미달 거부: %sx%s", width, height)
                    continue

                conf = extract_confidence(box)
                if conf < conf_threshold:
                    logger.debug("저신뢰도 ghost track 거부: conf=%.2f", conf)
                    continue

                track_id = self._tracker.resolve_id(
                    box, x1, y1, width, height, track_group="pose:person",
                )

                # 낙상 감지를 사람 검증보다 먼저 실행
                # (누워있는 사람은 기립 자세 검증을 건너뛰어야 함)
                is_fallen    = False
                kpts_for_fall = None
                person_keypoints = None
                if keypoints is not None:
                    _kpts_tmp = extract_keypoints(keypoints, idx)
                    person_keypoints = _kpts_tmp.tolist() if _kpts_tmp is not None else None
                    is_fallen = self._fall.detect(keypoints, idx, width, height)
                    if is_fallen:
                        kpts_for_fall = person_keypoints

                if not is_fallen:
                    # 기립 자세 검증 (옷걸이·의류 오탐 제거)
                    if keypoints is not None and not self._fall.validate_person(keypoints, idx):
                        continue
                    # 어깨 위치 검증 (FallDetector로 위임)
                    if keypoints is not None and not self._fall.validate_shoulder_position(
                        keypoints, idx, y1, height
                    ):
                        continue

                person_events.append(DetectionEvent(
                    event_type=EventType.PERSON,
                    x=x1, y=y1, width=width, height=height,
                    confidence=conf,
                    timestamp=time.time(),
                    object_id=track_id,
                    class_idx=0,
                    class_name="person",
                    keypoints=person_keypoints,
                ))

                if is_fallen:
                    fall_events.append(DetectionEvent(
                        event_type=EventType.FALL_DETECTED,
                        x=x1, y=y1, width=width, height=height,
                        confidence=conf,
                        timestamp=time.time(),
                        object_id=track_id,
                        class_idx=0,
                        class_name="person",
                        keypoints=kpts_for_fall,
                    ))

        person_events = self._remove_duplicates(
            person_events, iou_threshold=PERSON_DUPLICATE_IOU_THRESHOLD
        )
        logger.debug(
            "포즈 전체 프레임: 사람 %d명 (중복 제거 후), 낙상 %d건",
            len(person_events), len(fall_events),
        )
        return person_events, fall_events

    # ── 헬멧 검출 ────────────────────────────────────────────────────

    def _run_helmet_on_person_rois(
        self, frame, person_events: List[DetectionEvent]
    ) -> List[DetectionEvent]:
        """사람 ROI 상단 비율 영역에만 헬멧 모델을 실행하고 좌표 복원."""
        if frame is None or not person_events:
            return []

        frame_h, frame_w = frame.shape[:2]
        helmet_events: List[DetectionEvent] = []

        for person in person_events:
            x1     = max(int(person.x), 0)
            y1     = max(int(person.y), 0)
            x2     = min(int(person.x + person.width), frame_w)
            head_h = int(person.height * HEAD_REGION_RATIO)
            y2     = min(int(person.y + max(head_h, 1)), frame_h)

            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_events = self._run_single_model(
                self.helmet_model, roi, use_tracking=False, model_type="helmet"
            )
            for ev in roi_events:
                ev.x = int(ev.x) + x1
                ev.y = int(ev.y) + y1
            helmet_events.extend(roi_events)

        return helmet_events

    def _filter_helmet_boxes(self, helmet_events: List[DetectionEvent]) -> List[DetectionEvent]:
        """헬멧 박스 필터링: 크기·종횡비·중복 제거."""
        candidates: List[DetectionEvent] = []
        passthrough: List[DetectionEvent] = []

        for event in helmet_events:
            if event.event_type not in (EventType.HELMET, EventType.HEAD):
                passthrough.append(event)
                continue
            if not (
                MIN_HELMET_SIZE <= event.width  <= MAX_HELMET_WIDTH and
                MIN_HELMET_SIZE <= event.height <= MAX_HELMET_HEIGHT
            ):
                logger.debug("헬멧 크기 거부: %sx%s", event.width, event.height)
                continue
            aspect = max(event.width, event.height) / max(min(event.width, event.height), 1)
            if aspect > MAX_HELMET_ASPECT_RATIO:
                logger.debug("헬멧 종횡비 거부: %.2f", aspect)
                continue
            candidates.append(event)

        filtered = self._remove_duplicates(candidates)
        logger.debug("헬멧 필터링: %d → %d", len(candidates), len(filtered))
        return filtered + passthrough

    def _remove_duplicates(
        self,
        events: List[DetectionEvent],
        iou_threshold: float = DUPLICATE_IOU_THRESHOLD,
    ) -> List[DetectionEvent]:
        """IoU 기준 중복 박스 제거 (높은 신뢰도 우선)."""
        if len(events) <= 1:
            return events
        sorted_events = sorted(events, key=lambda e: e.confidence, reverse=True)
        keep: List[DetectionEvent] = []
        for event in sorted_events:
            if not any(
                boxes_overlap(event, kept, threshold=iou_threshold) for kept in keep
            ):
                keep.append(event)
        return keep

    # ── 얼굴 인식 ────────────────────────────────────────────────────

    def _run_face_recognition(
        self,
        frame,
        person_events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        """사람 ROI 상단에서 얼굴 검출/인식을 수행한다."""
        return self._face_pipeline.run(frame, person_events)

    # ── 외형 로그 DB / 크롭 ──────────────────────────────────────────

    def _save_person_crop(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        camera_id: str,
        track_id: Optional[int],
        ts: float,
    ) -> Optional[str]:
        """person bbox 영역을 JPEG로 저장하고 상대 경로를 반환한다."""
        return self._appearance_pipeline.save_person_crop(
            frame, x, y, w, h, camera_id, track_id, ts
        )

    @staticmethod
    def _build_appearance_nearby_objects(
        bag_objects: Optional[List[Dict]],
        helmet_events: Optional[List[DetectionEvent]],
    ) -> List[Dict]:
        """외형 분석에 필요한 주변 객체 문맥을 합친다."""
        nearby: List[Dict] = list(bag_objects or [])
        for event in helmet_events or []:
            if event.event_type != EventType.HELMET:
                continue
            nearby.append({
                "class_name": str(event.class_name or event.event_type.value).lower(),
                "event_type": event.event_type.value,
                "x": event.x,
                "y": event.y,
                "width": event.width,
                "height": event.height,
                "confidence": event.confidence,
                "metadata": dict(event.metadata or {}),
            })
        return nearby

    # ── 공개 API — 이벤트 분류 ───────────────────────────────────────

    def split_events(
        self, events: List[DetectionEvent]
    ) -> Tuple[List[DetectionEvent], List[DetectionEvent], List[DetectionEvent]]:
        """이벤트를 (사람, 헬멧, 기타)로 분리."""
        persons = [ev for ev in events if ev.event_type == EventType.PERSON]
        helmets = [ev for ev in events if ev.event_type in (EventType.HELMET, EventType.HEAD)]
        others  = [
            ev for ev in events
            if ev.event_type not in (EventType.PERSON, EventType.HELMET, EventType.HEAD)
        ]
        return persons, helmets, others

    def check_helmet_compliance(
        self,
        events: List[DetectionEvent],
        persons: Optional[List[DetectionEvent]] = None,
        helmets: Optional[List[DetectionEvent]] = None,
    ) -> List[Dict]:
        """사람/헬멧 매칭으로 착용 여부 판단."""
        if persons is None or helmets is None:
            persons, helmets, _ = self.split_events(events)

        helmet_bboxes = [
            {"x": h.x, "y": h.y, "width": h.width, "height": h.height}
            for h in helmets
        ]
        return [
            {
                "person":    person,
                "is_wearing": is_helmet_worn(
                    {"x": person.x, "y": person.y, "width": person.width, "height": person.height},
                    helmet_bboxes,
                ),
            }
            for person in persons
        ]

    # ── 공개 API — 종합 추론 ─────────────────────────────────────────

    def run_inference(
        self,
        frame,
        use_helmet: bool = True,
        use_pose:   bool = True,
        use_person: bool = False,
        use_face:   bool = False,
        use_appearance: bool = False,
        camera_id: Optional[str] = None,
    ) -> List[DetectionEvent]:
        """프레임에 대한 종합 추론을 수행한다.

        우선순위: 낙상(최우선) → 사람 → 얼굴 → 헬멧
        낙상이 감지된 사람은 헬멧 탐지 대상에서 제외한다.
        """
        self._falldata_aux.add_frame(frame)
        return self._object_pipeline.run(
            frame,
            use_helmet=use_helmet,
            use_pose=use_pose,
            use_person=use_person,
            use_face=use_face,
            use_appearance=use_appearance,
            camera_id=camera_id,
        )

    def _build_face_meta_map(
        self, face_events: List[DetectionEvent]
    ) -> Dict[int, Dict]:
        """얼굴 이벤트 메타데이터를 track_id 기준으로 정리한다."""
        return self._appearance_pipeline._build_face_meta_map(face_events)

    def _run_appearance_pipeline(
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
        return self._appearance_pipeline.run(
            frame,
            person_events,
            face_events,
            camera_id=camera_id,
            use_appearance=use_appearance,
            nearby_objects=nearby_objects,
        )

    def run_inference_with_compliance(
        self,
        frame,
        use_helmet:       bool = True,
        use_pose:         bool = True,
        check_compliance: bool = True,
        use_person:       bool = False,
        use_face:         bool = False,
        use_appearance:   bool = False,
        camera_id: Optional[str] = None,
    ) -> Dict[str, List]:
        """이벤트 리스트와 헬메 착용 준수 여부를 함께 반환."""
        events = self.run_inference(
            frame,
            use_helmet=use_helmet,
            use_pose=use_pose,
            use_person=use_person,
            use_face=use_face,
            use_appearance=use_appearance,
            camera_id=camera_id,
        )
        compliance: List[Dict] = []
        if check_compliance:
            persons, helmets, _ = self.split_events(events)
            if persons and helmets:
                compliance = self.check_helmet_compliance(events, persons=persons, helmets=helmets)
        return {"events": events, "compliance": compliance}

    # ── 하위 호환 위임 메서드 (테스트·외부 코드 호환성 유지) ──────────

    def _detect_fall_from_keypoints(
        self, keypoints, idx: int, bbox_width: int, bbox_height: int
    ) -> bool:
        """FallDetector.detect() 위임 — 기존 호출부 호환성 유지."""
        return self._fall.detect(keypoints, idx, bbox_width, bbox_height)

    def _validate_person_keypoints(self, keypoints, idx: int) -> bool:
        """FallDetector.validate_person() 위임 — 기존 호출부 호환성 유지."""
        return self._fall.validate_person(keypoints, idx)

    def _generate_temp_id(self, x: int, y: int, w: int, h: int) -> int:
        """generate_temp_id() 위임 — 기존 호출부 호환성 유지."""
        return generate_temp_id(x, y, w, h)
