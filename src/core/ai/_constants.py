"""AI 분석 모듈 공유 상수 및 글로벌 설정.

모든 수치 임계값·전역 상태(_MODEL_IMGSZ, _IMGSZ_LOCK)를 한 곳에 모아
변경 시 파급 범위를 최소화한다.
"""

import threading
from typing import Dict

# ── 헬멧 감지 임계값 ──────────────────────────────────────────────────
MAX_HELMET_WIDTH        = 300   # 헬멧 최대 너비 (px)
MAX_HELMET_HEIGHT       = 300   # 헬멧 최대 높이 (px)
MIN_HELMET_SIZE         = 15    # 최소 감지 크기 (px)
MAX_HELMET_ASPECT_RATIO = 2.0   # 헬멧 최대 가로세로 비율
DUPLICATE_IOU_THRESHOLD        = 0.3   # 헬멧 이벤트 중복 제거 임계값 (후처리)
PERSON_DUPLICATE_IOU_THRESHOLD = 0.4   # 사람 이벤트 중복 제거 임계값
HEAD_REGION_RATIO              = 0.35  # 헬멧 검증용 머리 영역 비율 (사람 상단 35%)

# ── 사람 탐지 최소 크기 ───────────────────────────────────────────────
MIN_PERSON_WIDTH  = 30   # px
MIN_PERSON_HEIGHT = 60   # px

# ── 키포인트 감지 임계값 ──────────────────────────────────────────────
MIN_KEYPOINT_CONFIDENCE = 0.3   # 0.2 → 0.3: 낮은 값은 배경 키포인트 할루시네이션 유발

# ── 낙상 감지 임계값 ─────────────────────────────────────────────────
FALL_ANGLE_HORIZONTAL    = 40    # 어깨-엉덩이 벡터 수평 각도 임계값 (°)
FALL_ANGLE_INVERTED      = 140   # 역방향 수평 각도 임계값 (°)
MIN_HIP_CONFIDENCE       = 0.3   # 엉덩이 키포인트 최소 신뢰도
MIN_LEG_CONFIDENCE       = 0.3   # 무릎/발목 키포인트 최소 신뢰도
FALL_KEYPOINT_SPAN_RATIO = 0.4   # 키포인트 수직 분산 / bbox 높이 비율 임계값

# 어깨 위치 검증 — 어깨가 bbox 상단에서 이 비율 이상 아래에 위치해야 사람으로 인정
SHOULDER_TOP_MIN_RATIO = 0.15

# ── 임시 트랙 ID ─────────────────────────────────────────────────────
_TEMP_TRACK_ID_START          = 1_500_000_000
_TEMP_TRACK_ID_END            = 1_999_999_999
_TEMP_TRACK_TTL_SEC           = 2.0    # 캐시 만료 시간 (초)
_TEMP_TRACK_MIN_IOU           = 0.35   # IoU 매칭 최소 임계값
_TEMP_TRACK_MAX_CENTER_RATIO  = 0.6    # 중심 거리 비율 최대 임계값
_TEMP_TRACK_MAX_AREA_RATIO_DELTA = 0.75  # 면적 비율 차 최대 임계값

# ── 얼굴 인식 ────────────────────────────────────────────────────────
_FACE_TRACK_COOLDOWN_SEC = 2.0  # 동일 객체에 대한 얼굴 인식 재실행 억제 간격 (초)

# ── YOLO 모델 설정 ───────────────────────────────────────────────────
# ※ 고정 shape TensorRT .engine 파일은 컴파일 시 imgsz와 정확히 일치해야 함
#   dynamic profile 엔진과 .pt 파일은 runtime imgsz 조정 가능
DEFAULT_IMAGE_SIZE_HELMET = 320   # 폴백: .engine 자동 감지 실패 시 (원래 480)
DEFAULT_IMAGE_SIZE_POSE   = 320   # 폴백: .engine 자동 감지 실패 시 (원래 416)
DEFAULT_IMAGE_SIZE_PERSON = 640   # .pt 파일 사용, 640 유지
DEFAULT_IOU_THRESHOLD     = 0.45  # YOLO NMS 임계값 (모델 추론 단계)

# model_type → imgsz 매핑 테이블 (로드 후 _detect_engine_imgsz()로 자동 갱신됨)
# AdaptiveGovernor(백그라운드 스레드)와 추론 스레드 간 읽기/쓰기 보호
_MODEL_IMGSZ: Dict[str, int] = {
    "helmet": DEFAULT_IMAGE_SIZE_HELMET,
    "pose":   DEFAULT_IMAGE_SIZE_POSE,
    "person": DEFAULT_IMAGE_SIZE_PERSON,
}
_IMGSZ_LOCK: threading.Lock = threading.Lock()
