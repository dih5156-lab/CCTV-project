"""AI 분석 모듈 — 하위 호환성 유지 심(shim).

실제 구현은 src/core/ai/ 패키지로 분리되었습니다:
  src/core/ai/analyzer.py        — AIAnalyzer (오케스트레이터)
  src/core/ai/_constants.py      — 공유 상수·_MODEL_IMGSZ·_IMGSZ_LOCK
  src/core/ai/_yolo_helpers.py   — YOLO 결과 추출 유틸리티
  src/core/ai/_object_tracker.py — ObjectTracker (track ID 관리)
  src/core/ai/_fall_detector.py  — FallDetector (낙상 감지·사람 검증)

이 파일은 기존 import 경로(from .ai_analysis import AIAnalyzer)를 유지하기 위해
남겨두었습니다.
"""

# ── 핵심 클래스 및 공유 상태 재내보내기 ──────────────────────────────
from .ai.analyzer import AIAnalyzer  # noqa: F401
from .ai._constants import (          # noqa: F401
    _MODEL_IMGSZ,
    _IMGSZ_LOCK,
    DEFAULT_IMAGE_SIZE_HELMET,
    DEFAULT_IMAGE_SIZE_POSE,
    DEFAULT_IMAGE_SIZE_PERSON,
    DEFAULT_IOU_THRESHOLD,
    MAX_HELMET_WIDTH,
    MAX_HELMET_HEIGHT,
    MIN_HELMET_SIZE,
    MAX_HELMET_ASPECT_RATIO,
    DUPLICATE_IOU_THRESHOLD,
    PERSON_DUPLICATE_IOU_THRESHOLD,
    HEAD_REGION_RATIO,
    MIN_PERSON_WIDTH,
    MIN_PERSON_HEIGHT,
    MIN_KEYPOINT_CONFIDENCE,
    FALL_ANGLE_HORIZONTAL,
    FALL_ANGLE_INVERTED,
    MIN_HIP_CONFIDENCE,
    FALL_KEYPOINT_SPAN_RATIO,
    SHOULDER_TOP_MIN_RATIO,
    _TEMP_TRACK_ID_START,
    _TEMP_TRACK_ID_END,
    _TEMP_TRACK_TTL_SEC,
    _TEMP_TRACK_MIN_IOU,
    _TEMP_TRACK_MAX_CENTER_RATIO,
    _TEMP_TRACK_MAX_AREA_RATIO_DELTA,
    _FACE_TRACK_COOLDOWN_SEC,
)
from .ai._yolo_helpers import (        # noqa: F401
    detect_engine_imgsz as _detect_engine_imgsz,
    age_to_group as _age_to_group,
)
# 테스트에서 patch("src.core.ai_analysis.YOLO") 로 참조하므로 재내보내기
from .ai.analyzer import YOLO  # noqa: F401

# 순환 참조 방지를 위해 events를 먼저 import
from .events import EventType, DetectionEvent  # noqa: F401
from ..utils.geometry import is_helmet_worn, boxes_overlap  # noqa: F401
from ..utils.face_recognition import FaceRecognitionEngine  # noqa: F401

