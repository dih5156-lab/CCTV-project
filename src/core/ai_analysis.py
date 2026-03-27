"""AI 분석 모듈: YOLO 기반 멀티 모델 객체 탐지

포즈 모델(yolov8-pose)이 있으면 전체 프레임에서 사람 탐지 + 낙상 감지를
한 번의 추론으로 처리한다. (person 모델 불필요)
포즈 모델이 없을 때만 person 모델을 fallback으로 사용한다.
헬멧 모델은 항상 사람 머리 ROI 기반으로 동작한다.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

# 순환 참조 방지를 위해 events를 먼저 import
from .events import EventType, DetectionEvent
from ..utils.geometry import is_helmet_worn, boxes_overlap
from ..utils.face_recognition import FaceRecognitionEngine

logger = logging.getLogger(__name__)


# ==================================================================
# 상수 정의
# ==================================================================


# 헬멧 감지 임계값
MAX_HELMET_WIDTH = 300  # 헬멧 최대 너비
MAX_HELMET_HEIGHT = 300  # 헬멧 최대 높이
MIN_HELMET_SIZE = 15  # 최소 감지 크기
MAX_HELMET_ASPECT_RATIO = 2.0  # 헬멧 최대 가로세로 비율
DUPLICATE_IOU_THRESHOLD = 0.3       # 헬멧 이벤트 중복 제거 임계값 (후처리 단계)
PERSON_DUPLICATE_IOU_THRESHOLD = 0.4  # 사람 이벤트 중복 제거 임계값
HEAD_REGION_RATIO = 0.35  # 헬멧 검증용 머리 영역 비율 (사람 상단 35%)

# 사람 탐지 최소 크기 (픽셀) — 너무 작은 박스는 잡음/오탐 가능성 높음
MIN_PERSON_WIDTH = 30
MIN_PERSON_HEIGHT = 60


# 키포인트 감지 임계값
MIN_KEYPOINT_CONFIDENCE = 0.3  # 0.2 → 0.3: 낮은 값은 옷/배경 키포인트 할루시네이션이 통과할 수 있음

# 낙상 감지 임계값 — 사용처: _detect_fall_from_keypoints()
FALL_ANGLE_HORIZONTAL = 40   # 수평 각도 임계값 (도): 어깨-엉덩이 벡터가 이 이하이면 낙상 (방법 1) — 30→40 확장
FALL_ANGLE_INVERTED = 140    # 역방향 수평 각도 임계값 (도): 왼쪽으로 누운 경우 (방법 1) — 150→140 확장
MIN_HIP_CONFIDENCE = 0.3     # 엉덩이 키포인트 최소 신뢰도 (방법 1·2)
FALL_KEYPOINT_SPAN_RATIO = 0.4  # 방법 4: 키포인트 수직 분산 / bbox 높이 비율 임계값 (이하이면 낙상으로 판정)

# 어깨 위치 검증: 어깨가 bbox 상단에서 이 비율 이상 아래에 위치해야 사람으로 인정
# 실제 사람은 머리가 위에 있으므로 어깨는 상단 15% 이상 아래
# 옷걸이·행거 의류는 어깨 키포인트가 bbox 최상단 근처에 찍힘
SHOULDER_TOP_MIN_RATIO = 0.15


# YOLO 모델 설정
# ※ TensorRT .engine 파일은 컴파일 시 고정된 imgsz와 정확히 일치해야 함
#   .pt 파일은 runtime에 자동 리사이즈되므로 640 사용 가능
# ※ 로드 후 _detect_engine_imgsz()로 실제 입력 크기를 자동 감지하여 덮어씀
DEFAULT_IMAGE_SIZE_HELMET = 480  # 폴백: .engine 자동 감지 실패 시 사용
DEFAULT_IMAGE_SIZE_POSE = 416    # 폴백: .engine 자동 감지 실패 시 사용
DEFAULT_IMAGE_SIZE_PERSON = 640  # .pt 파일 사용, 640 유지
DEFAULT_IOU_THRESHOLD = 0.45  # YOLO NMS 임계값 (모델 추론 단계)

# model_type → imgsz 매핑 테이블 (로드 후 _apply_engine_imgsz()로 자동 갱신됨)
_MODEL_IMGSZ: Dict[str, int] = {
    "helmet": DEFAULT_IMAGE_SIZE_HELMET,
    "pose":   DEFAULT_IMAGE_SIZE_POSE,
    "person": DEFAULT_IMAGE_SIZE_PERSON,
}

_TEMP_TRACK_ID_START = 1_500_000_000
_TEMP_TRACK_ID_END = 1_999_999_999
_TEMP_TRACK_TTL_SEC = 2.0
_TEMP_TRACK_MIN_IOU = 0.35
_TEMP_TRACK_MAX_CENTER_RATIO = 0.6
_TEMP_TRACK_MAX_AREA_RATIO_DELTA = 0.75
_FACE_TRACK_COOLDOWN_SEC = 2.0


def _detect_engine_imgsz(model, fallback: int) -> int:
    """로드된 YOLO 모델에서 실제 입력 이미지 크기를 자동 감지한다.

    TensorRT .engine 파일은 컴파일 시 입력 shape이 고정되므로
    ultralytics가 노출하는 메타데이터에서 imgsz를 읽어 인적 오류를 방지한다.
    .pt 파일이거나 감지 실패 시 fallback 값을 반환한다.

    탐색 우선순위:
      1. model.model.imgsz  (ultralytics TensorRT 래퍼)
      2. model.overrides["imgsz"]  (저장된 학습 설정)
      3. 첫 번째 바인딩의 입력 shape  (tensorrt.ICudaEngine 직접 접근)
    """
    if model is None:
        return fallback
    try:
        # 1순위: ultralytics TensorRT 래퍼가 imgsz 속성 노출
        inner = getattr(model, "model", None)
        imgsz_attr = getattr(inner, "imgsz", None)
        if imgsz_attr is not None:
            if isinstance(imgsz_attr, (list, tuple)):
                size = int(imgsz_attr[0])
            else:
                size = int(imgsz_attr)
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (model.model.imgsz): %d", size)
                return size
    except Exception:
        pass
    try:
        # 2순위: overrides 딕셔너리 (학습/내보내기 시 저장된 값)
        overrides = getattr(model, "overrides", {}) or {}
        val = overrides.get("imgsz")
        if val is not None:
            if isinstance(val, (list, tuple)):
                size = int(val[0])
            else:
                size = int(val)
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (overrides): %d", size)
                return size
    except Exception:
        pass
    try:
        # 3순위: TensorRT ICudaEngine 직접 접근 (tensorrt 패키지 필요)
        import tensorrt as trt  # type: ignore
        engine = getattr(getattr(model, "model", None), "engine", None)
        if engine is not None and isinstance(engine, trt.ICudaEngine):
            binding_name = engine.get_binding_name(0)
            shape = engine.get_binding_shape(binding_name)
            # shape: (batch, channels, H, W) 또는 (batch, H, W)
            size = int(shape[-1])  # 마지막 차원 = width = height (정사각형 가정)
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (TRT binding[0]): %d", size)
                return size
    except Exception:
        pass
    logger.debug("엔진 imgsz 자동 감지 실패 → 폴백 값 사용: %d", fallback)
    return fallback

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Jetson Orin nvgpu cuDNN 안정화 설정
# "GET was unable to find an engine" 에러 방지
try:
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass


class AIAnalyzer:
    """멀티 모델 AI 분석 시스템
    
    모델 구성:
    - 포즈 모델 (yolov8n-pose): 사람 탐지 + 낙상 감지 (전체 프레임, 기본)
    - 사람 모델 (yolov8n): 포즈 모델 없을 때 fallback 전용 (선택)
    - 헬멧 모델(커스텀): 헬멧 감지 (사람 머리 ROI 내부)

    주요 메서드 위치 안내:
    ┌─────────────────────────────────────────────────────────────────┐
    │ [낙상 감지]  _detect_fall_from_keypoints()  ← '낙상 감지 (포즈 기반)' 섹션  │
    │ [포즈 추론]  _run_pose_full_frame()          ← '모델 추론 메소드' 섹션       │
    │ [사람 검증]  _validate_person_keypoints()    ← '포즈 기반 사람 검증' 섹션    │
    │ [헬멧 추론]  _run_helmet_on_person_rois()    ← '모델 추론 메소드' 섹션       │
    └─────────────────────────────────────────────────────────────────┘
    """
    # 클래스 매핑 상수 (순환 import 방지를 위해 문자열로 저장)
    _HELMET_CLASS_MAP: Dict[str, str] = {
        "helmet_missing": "head",
        "no_helmet": "head",
        "helmet": "helmet",
        "helmet_wearing": "helmet",
        "head": "head",
    }
    _COMMON_CLASS_MAP: Dict[str, str] = {
        "danger_zone": "danger_zone",
        "unsafe_behavior": "unsafe_behavior",
        "unsafe": "unsafe_behavior",
        "person": "person",
        "face_recognized": "face_recognized",
        "face_unknown": "face_unknown",
    }
    _CLASS_MAP: Dict[str, str] = {**_HELMET_CLASS_MAP, **_COMMON_CLASS_MAP}

    def __init__(
        self,
        model_path: Optional[str] = None,  # 하위 호환성: model_path가 제공되면 pose_model_path로 사용
        helmet_model_path: Optional[str] = None,
        person_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,  # 포즈 모델 경로 (사람 + 키포인트 감지)
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        fall_height_ratio: float = 0.3, # 낙상 감지 높이 비율 (0.0~1.0, 낮을수록 엄격)
    ):
        # 하위 호환성: model_path가 제공되면 pose_model_path로 사용
        if model_path and not pose_model_path:
            pose_model_path = model_path
        
        # 모델 경로 및 설정
        self.helmet_model_path = helmet_model_path
        self.person_model_path = person_model_path
        self.pose_model_path = pose_model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.fall_height_ratio = fall_height_ratio

        # 모델 객체 초기화
        self.helmet_model = None
        self.person_model = None
        self.pose_model = None  # YOLOv8-pose 모델

        # 기타 상태
        self.last_load_errors = []
        self._person_warning_shown = False
        self._helmet_warning_shown = False
        self._next_temp_track_id = _TEMP_TRACK_ID_START
        self._temp_track_cache: Dict[int, Dict[str, object]] = {}
        self.face_recognizer = FaceRecognitionEngine()
        self._face_identity_cache: Dict[int, Dict[str, object]] = {}

        # YOLO 라이브러리 확인
        if YOLO is None:
            logger.error("ultralytics 패키지가 설치되지 않았습니다. `pip install ultralytics`를 실행하세요.")
            raise ImportError("ultralytics 패키지가 필요합니다")
        
        # 모델 동기 로딩
        self.load_models()

    # ====================
    # 공개 API 메소드
    # ====================

    def run_helmet_model(self, frame) -> List[DetectionEvent]:
        """헬멧 모델로 추론 실행"""
        return self._run_single_model(self.helmet_model, frame, model_type="helmet")

    def run_person_model(self, frame) -> List[DetectionEvent]:
        """사람 모델로 추론 실행.

        기본 운영 경로는 pose 모델이며, person 모델은 pose 모델이 없을 때만
        제한적으로 fallback 용도로 사용한다.
        """
        return self._run_single_model(self.person_model, frame, model_type="person")
    
    # ====================
    # 모델 관리
    # ====================

    def _load_model(self, model_path: str):
        """단일 YOLO 모델 로드"""
        if YOLO is None:
            raise RuntimeError("YOLO 라이브러리를 찾을 수 없습니다 (ultralytics 설치 필요).")

        if not model_path:
            return None

        # 파일 존재 확인 (상대/절대)
        if not os.path.exists(model_path):
            # basename으로 시도
            basename = os.path.basename(model_path)
            if os.path.exists(basename):
                model_path = basename
            else:
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        try:
            model = YOLO(model_path)
            # device 설정 (ultralytics YOLO 객체는 .to()를 가지고 있음)
            try:
                model.to(self.device)
            except Exception:
                # 일부 ultralytics 버전에서는 to() 불필요하거나 다르게 동작
                pass

            logger.info("모델 로드 성공: %s (device=%s)", model_path, self.device)
            return model
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다 ({model_path}): {exc}")
        except Exception as exc:
            raise RuntimeError(f"모델 로드 실패 ({model_path}): {exc}")

    def _try_load(self, name: str, path: Optional[str]) -> None:
        """단일 모델 로드를 시도하고 결과를 인스턴스 속성에 저장한다."""
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
        """헬멧과 pose 모델을 우선 로드하고 필요 시에만 person fallback을 로드한다."""
        self.last_load_errors.clear()
        self._try_load("helmet", self.helmet_model_path)
        self._try_load("pose", self.pose_model_path)
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

        # TensorRT .engine 파일의 실제 입력 크기를 자동 감지하여 _MODEL_IMGSZ 갱신
        # .pt 파일은 자동 리사이즈되므로 감지 실패 시 기존 기본값 유지
        _MODEL_IMGSZ["helmet"] = _detect_engine_imgsz(
            self.helmet_model, DEFAULT_IMAGE_SIZE_HELMET
        )
        _MODEL_IMGSZ["pose"] = _detect_engine_imgsz(
            self.pose_model, DEFAULT_IMAGE_SIZE_POSE
        )
        # person 모델은 .pt 전용이므로 자동 감지 불필요 (640 고정)
        logger.info(
            "imgsz 설정 → helmet=%d, pose=%d, person=%d",
            _MODEL_IMGSZ["helmet"], _MODEL_IMGSZ["pose"], _MODEL_IMGSZ["person"],
        )

    def get_loaded_model_names(self) -> Dict[str, Optional[Dict[int, str]]]:
        """로드된 모델의 클래스명 조회 (디버깅용)"""
        res = {"helmet": None, "person": None, "pose": None}
        if self.helmet_model:
            try:
                res["helmet"] = getattr(self.helmet_model, "names", None)
            except Exception:
                res["helmet"] = None
        if self.person_model:
            try:
                res["person"] = getattr(self.person_model, "names", None)
            except Exception:
                res["person"] = None
        if self.pose_model:
            try:
                res["pose"] = getattr(self.pose_model, "names", None)
            except Exception:
                res["pose"] = None
        return res

    # ====================
    # 설정 메소드
    # ====================

    def set_device(self, device: str = "cpu") -> None:
        """디바이스 설정 (cpu 또는 cuda). 모델이 이미 로드된 경우 .to() 시도"""
        self.device = device
        for m in (self.helmet_model, self.person_model, self.pose_model):
            if m is not None:
                try:
                    m.to(device)
                except Exception as exc:
                    logger.warning("디바이스 설정 실패: %s", exc)
        logger.info("디바이스 설정 완료: %s", device)

    def update_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 업데이트"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"임계값은 0.0~1.0 사이여야 합니다 (입력값: {threshold})")
        
        self.confidence_threshold = threshold
        logger.info("신뢰도 임계값 업데이트: %s", threshold)

    # ====================
    # YOLO 결과 추출 매핑
    # ====================
    def _map_class_to_event_type(self, class_name: str, model_type: str) -> EventType:
        """클래스명을 EventType으로 매핑
        
        매개변수:
            class_name: YOLO 모델 클래스명
            model_type: 모델 타입("helmet", "pose")
            
        반환:
            매핑된 EventType
        """
        if not class_name:
            return EventType.OTHER

        normalized = class_name.lower().strip().replace(" ", "_")

        if model_type == "helmet":
            mapped_str = self._CLASS_MAP.get(normalized)
            mapped_event_types = {
                "head": EventType.HEAD,
                "helmet": EventType.HELMET,
                "danger_zone": EventType.DANGER_ZONE,
                "unsafe_behavior": EventType.UNSAFE_BEHAVIOR,
                "person": EventType.PERSON,
            }
            return mapped_event_types.get(mapped_str, EventType.OTHER)

        if model_type == "person":
            return EventType.PERSON if normalized == "person" else EventType.OTHER
        
        # 포즈 모델은 _run_pose_full_frame에서 직접 EventType을 결정
        return EventType.OTHER

    def _threshold_for_model(self, model_type: str) -> float:
        if model_type == "helmet":
            return getattr(self, "helmet_threshold", self.confidence_threshold)
        if model_type == "person":
            return getattr(self, "person_threshold", self.confidence_threshold)
        return self.confidence_threshold

    # ====================
    # YOLO 결과 추출 메소드
    # ====================
    def _extract_bbox(self, box) -> Optional[Tuple[int, int, int, int]]:
        """YOLO box에서 bbox 좌표 추출 (x1, y1, x2, y2)"""
        try:
            xyxy_tensor = box.xyxy[0]
            if hasattr(xyxy_tensor, "cpu"):
                xyxy = xyxy_tensor.cpu().numpy().astype(int)
            else:
                xyxy = np.array(xyxy_tensor).astype(int)
            return int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("bbox 추출 실패: %s", exc)
            return None

    def _extract_confidence(self, box) -> float:
        """YOLO box에서 신뢰도 추출"""
        try:
            conf_tensor = box.conf[0]
            if hasattr(conf_tensor, "cpu"):
                return float(conf_tensor.cpu().numpy())
            else:
                return float(conf_tensor)
        except (ValueError, TypeError, IndexError):
            return 0.0

    def _extract_keypoints(self, keypoints, idx: int) -> Optional[np.ndarray]:
        """YOLO pose 결과에서 포인트 배열 추출 (N, 3) - [x, y, confidence]"""
        try:
            if hasattr(keypoints, "data"):
                kpts = keypoints.data[idx]
                if hasattr(kpts, "cpu"):
                    return kpts.cpu().numpy()
                return kpts
            elif hasattr(keypoints, "xy"):
                kpts_xy = keypoints.xy[idx]
                kpts_conf = keypoints.conf[idx]
                if hasattr(kpts_xy, "cpu"):
                    kpts_xy = kpts_xy.cpu().numpy()
                    kpts_conf = kpts_conf.cpu().numpy()
                return np.column_stack([kpts_xy, kpts_conf])
            return None
        except Exception as exc:
            logger.debug("포인트 추출 실패: %s", exc)
            return None

    # ====================
    # 유틸리티 메소드
    # ====================

    def _extract_track_id(self, box) -> Optional[int]:
        """YOLOv8 track() 결과에서 추적 ID 추출
        
        매개변수:
            box: YOLO box 객체
            
        반환:
            추적 ID (없으면 None)
        """
        if not hasattr(box, 'id') or box.id is None:
            return None
        
        try:
            track_id = box.id[0]
            if hasattr(track_id, 'cpu'):
                return int(track_id.cpu().numpy())
            return int(track_id)
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            logger.debug("추적 ID 추출 실패: %s", exc)
            return None

    def _allocate_temp_track_id(self) -> int:
        """충돌 가능성을 낮추기 위해 단조 증가 임시 ID를 발급한다."""
        track_id = self._next_temp_track_id
        self._next_temp_track_id += 1
        if self._next_temp_track_id > _TEMP_TRACK_ID_END:
            self._next_temp_track_id = _TEMP_TRACK_ID_START
        return track_id

    @staticmethod
    def _bbox_iou_from_coords(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area1 = max(0, bbox1[2]) * max(0, bbox1[3])
        area2 = max(0, bbox2[2]) * max(0, bbox2[3])
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _center_distance_ratio(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        c1x = bbox1[0] + (bbox1[2] / 2.0)
        c1y = bbox1[1] + (bbox1[3] / 2.0)
        c2x = bbox2[0] + (bbox2[2] / 2.0)
        c2y = bbox2[1] + (bbox2[3] / 2.0)
        dist = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
        scale = max(
            1.0,
            bbox1[2],
            bbox1[3],
            bbox2[2],
            bbox2[3],
        )
        return dist / scale

    def _cleanup_temp_track_cache(self, now_ts: float) -> None:
        expired_ids = [
            track_id
            for track_id, state in self._temp_track_cache.items()
            if now_ts - float(state["last_seen"]) > _TEMP_TRACK_TTL_SEC
        ]
        for track_id in expired_ids:
            self._temp_track_cache.pop(track_id, None)

    def _resolve_object_id(
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
        track_id = self._extract_track_id(box)
        if track_id is not None:
            return track_id

        now_ts = time.time() if now_ts is None else now_ts
        self._cleanup_temp_track_cache(now_ts)

        bbox = (x1, y1, width, height)
        area = max(width, 0) * max(height, 0)
        best_track_id: Optional[int] = None
        best_score = -1.0

        for cached_track_id, state in self._temp_track_cache.items():
            if state["group"] != track_group:
                continue

            cached_bbox = state["bbox"]
            iou = self._bbox_iou_from_coords(bbox, cached_bbox)
            center_ratio = self._center_distance_ratio(bbox, cached_bbox)
            cached_area = max(cached_bbox[2], 0) * max(cached_bbox[3], 0)
            area_ratio_delta = abs(area - cached_area) / max(area, cached_area, 1)

            if iou < _TEMP_TRACK_MIN_IOU and center_ratio > _TEMP_TRACK_MAX_CENTER_RATIO:
                continue
            if area_ratio_delta > _TEMP_TRACK_MAX_AREA_RATIO_DELTA:
                continue

            score = iou - (center_ratio * 0.1) - (area_ratio_delta * 0.05)
            if score > best_score:
                best_score = score
                best_track_id = cached_track_id

        if best_track_id is None:
            best_track_id = self._allocate_temp_track_id()

        self._temp_track_cache[best_track_id] = {
            "group": track_group,
            "bbox": bbox,
            "last_seen": now_ts,
        }
        return best_track_id

    def _filter_helmet_boxes(self, helmet_events: List) -> List:
        """헬멧 박스 필터링: 크기, 종횡비, 위치 + 중복 제거"""

        helmet_candidates: List[DetectionEvent] = []
        passthrough: List[DetectionEvent] = []

        for event in helmet_events:
            if event.event_type not in (EventType.HELMET, EventType.HEAD):
                # Danger zone 등 부수적인 클래스는 그대로 유지
                passthrough.append(event)
                continue

            if not (
                MIN_HELMET_SIZE <= event.width <= MAX_HELMET_WIDTH
                and MIN_HELMET_SIZE <= event.height <= MAX_HELMET_HEIGHT
            ):
                logger.debug("헬멧 크기 거부: %sx%s", event.width, event.height)
                continue

            aspect_ratio = max(event.width, event.height) / max(min(event.width, event.height), 1)
            if aspect_ratio > MAX_HELMET_ASPECT_RATIO:
                logger.debug("헬멧 종횡비 거부: %.2f (너무 길쭉함)", aspect_ratio)
                continue

            helmet_candidates.append(event)

        filtered = self._remove_duplicates(helmet_candidates)
        logger.debug(
            "헬멧 필터링: %d -> %d (크기/종횡비/중복 제거)",
            len(helmet_candidates),
            len(filtered),
        )
        return filtered + passthrough
    
    def _remove_duplicates(self, events: List, iou_threshold: float = DUPLICATE_IOU_THRESHOLD) -> List:
        """중복 박스 제거 - IoU 기준 박스가 겹치는 경우 높은 신뢰도만 유지
        
        매개변수:
            events: 감지 이벤트 리스트
            iou_threshold: IoU 기준
            
        반환:
            중복 제거된 감지 이벤트 리스트
        """
        if len(events) <= 1:
            return events
        
        # 신뢰도 기준 내림차순 정렬
        sorted_events = sorted(events, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        for event in sorted_events:
            # 이미 유지된 이벤트와 IoU 계산하여 중복 여부 판단
            is_duplicate = any(
                boxes_overlap(event, kept_event, threshold=iou_threshold)
                for kept_event in keep
            )
            
            if not is_duplicate:
                keep.append(event)
        
        return keep

    # ====================
    # 모델 추론 메소드
    # ====================

    def _run_single_model(self, model, frame, use_tracking: bool = True, model_type: str = "unknown") -> List:
        """단일 YOLO 모델 결과를 DetectionEvent 리스트로 변환"""
        events: List[DetectionEvent] = []
        if model is None or frame is None:
            return events

        # 모델 타입에 따라 신뢰도 임계값 선택
        conf_threshold = self._threshold_for_model(model_type)

        try:
            if use_tracking:
                results = model.track(
                    frame,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=_MODEL_IMGSZ.get(model_type, DEFAULT_IMAGE_SIZE_HELMET),
                    verbose=False,
                    persist=True  # 추적 결과 유지 (ID 추출 위해)
                )
            else:
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=_MODEL_IMGSZ.get(model_type, DEFAULT_IMAGE_SIZE_HELMET),
                    verbose=False,
                )
        except Exception as exc:
            logger.error("모델 추론 실패 (%s): %s", model_type, exc, exc_info=True)
            return events

        logger.debug("[%s] 추론 완료: %s개 결과", model_type, len(results))
        
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", None) or {}
            
            if boxes is None:
                logger.debug("[%s] boxes 없음", model_type)
                continue
            
            logger.debug("[%s] 감지된 박스: %s개", model_type, len(boxes))

            # boxes는 box 객체의 컬렉션
            for box in boxes:
                # 바운딩 박스 추출
                bbox = self._extract_bbox(box)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # 신뢰도 추출
                conf = self._extract_confidence(box)

                # 클래스 인덱스 추출
                cls_idx = None
                try:
                    cls_tensor = box.cls[0]
                    if hasattr(cls_tensor, "cpu"):
                        cls_idx = int(cls_tensor.cpu().numpy())
                    else:
                        cls_idx = int(cls_tensor)
                except (ValueError, TypeError, IndexError):
                    cls_idx = None

                class_name = None
                if cls_idx is not None and isinstance(names, (dict, list)):
                    if isinstance(names, dict):
                        class_name = names.get(cls_idx)
                    elif cls_idx < len(names):
                        class_name = names[cls_idx]
                    else:
                        class_name = None

                # 사람 모델은 person 클래스만 사용 (다른 객체는 제외)
                if model_type == "person":
                    if not class_name or class_name.lower() != "person":
                        continue

                event_type = self._map_class_to_event_type(
                                    class_name or "",
                                    model_type=model_type
                                )
                
                # OTHER 이벤트는 제외 (필요시 제외)

                if event_type == EventType.OTHER:
                    continue

                track_id = self._resolve_object_id(
                    box,
                    x1,
                    y1,
                    width,
                    height,
                    track_group=f"{model_type}:{event_type.value}",
                )
                
                ev = DetectionEvent(
                    event_type=event_type,
                    x=x1,
                    y=y1,
                    width=width,
                    height=height,
                    confidence=conf,
                    timestamp=time.time(),
                    object_id=track_id,
                    class_idx=cls_idx,
                )
                events.append(ev)

        return events

    def _run_helmet_on_person_rois(self, frame, person_events: List) -> List:
        """사람 ROI(머리 비율) 영역에만 헬멧 모델을 실행하고 좌표 복원"""
        if frame is None or not person_events:
            return []

        frame_h, frame_w = frame.shape[:2]
        helmet_events: List = []

        for person in person_events:
            x1 = max(int(person.x), 0)
            y1 = max(int(person.y), 0)
            x2 = min(int(person.x + person.width), frame_w)
            head_h = int(person.height * HEAD_REGION_RATIO)
            y2 = min(int(person.y + max(head_h, 1)), frame_h)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_events = self._run_single_model(self.helmet_model, roi, use_tracking=False, model_type="helmet")

            for ev in roi_events:
                ev.x = int(ev.x) + x1
                ev.y = int(ev.y) + y1

            helmet_events.extend(roi_events)

        return helmet_events

    def _run_pose_full_frame(self, frame) -> Tuple[List, List]:
        """포즈 모델을 전체 프레임에서 실행하여 사람 이벤트와 낙상 이벤트를 동시에 반환

        yolov8-pose는 사람 bbox + 키포인트를 함께 출력하므로
        person 모델 없이도 사람 탐지 + 낙상 감지를 한 번의 추론으로 처리합니다.

        반환:
            (person_events, fall_events)
        """
        person_events: List[DetectionEvent] = []
        fall_events: List[DetectionEvent] = []

        if frame is None or self.pose_model is None:
            return person_events, fall_events

        conf_threshold = getattr(self, "pose_threshold", self.confidence_threshold)

        try:
            results = self.pose_model.track(
                frame,
                conf=conf_threshold,
                iou=DEFAULT_IOU_THRESHOLD,
                imgsz=_MODEL_IMGSZ.get("pose", DEFAULT_IMAGE_SIZE_POSE),
                verbose=False,
                persist=True,
            )
        except Exception as exc:
            logger.error("포즈 모델 전체 프레임 추론 실패: %s", exc, exc_info=True)
            return person_events, fall_events

        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)

            if boxes is None:
                continue

            for idx, box in enumerate(boxes):
                bbox = self._extract_bbox(box)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # 최소 크기 필터: 너무 작은 박스는 잡음/오탐 가능성 높음
                if width < MIN_PERSON_WIDTH or height < MIN_PERSON_HEIGHT:
                    logger.debug("사람 bbox 크기 미달 거부: %sx%s", width, height)
                    continue

                conf = self._extract_confidence(box)

                # persist=True 트래킹으로 ghost track이 저신뢰도로 유지될 수 있음
                # 직접 신뢰도 게이트를 적용하여 ghost track 필터링
                if conf < conf_threshold:
                    logger.debug(
                        "저신뢰도 ghost track 거부: conf=%.2f < threshold=%.2f", conf, conf_threshold
                    )
                    continue

                track_id = self._resolve_object_id(
                    box,
                    x1,
                    y1,
                    width,
                    height,
                    track_group="pose:person",
                )

                # 낙상 감지를 먼저 실행: 누워있는 사람은 기립 자세 전제의 검증을 건너뛰어야 함
                # (_validate_person_keypoints의 코>어깨>엉덩이 Y순서 검증이
                #  수평으로 누운 자세에서는 항상 실패하므로 낙상 여부를 먼저 판단)
                is_fallen = False
                kpts_for_fall = None
                if keypoints is not None:
                    is_fallen = self._detect_fall_from_keypoints(keypoints, idx, width, height)
                    if is_fallen:
                        _kpts_tmp = self._extract_keypoints(keypoints, idx)
                        kpts_for_fall = _kpts_tmp.tolist() if _kpts_tmp is not None else None

                # 기립 자세 검증은 낙상 상태가 아닐 때만 적용
                # (누워있는 사람을 옷걸이 오탐으로 잘못 거부하지 않도록)
                if not is_fallen:
                    # 키포인트 기반 사람 검증 (노이즈 bbox 제거)
                    if keypoints is not None and not self._validate_person_keypoints(keypoints, idx):
                        continue

                    # 어깨 위치 검증: 어깨가 bbox 상단에 너무 가까우면 옷걸이·행거 의류 오탐
                    if keypoints is not None:
                        _kpts = self._extract_keypoints(keypoints, idx)
                        if _kpts is not None and len(_kpts) > 6:
                            _ls_conf = _kpts[5][2]
                            _rs_conf = _kpts[6][2]
                            _shoulder_ys = []
                            if _ls_conf > MIN_KEYPOINT_CONFIDENCE:
                                _shoulder_ys.append(_kpts[5][1])
                            if _rs_conf > MIN_KEYPOINT_CONFIDENCE:
                                _shoulder_ys.append(_kpts[6][1])
                            if _shoulder_ys:
                                _avg_shoulder_y = sum(_shoulder_ys) / len(_shoulder_ys)
                                _ratio = (_avg_shoulder_y - y1) / max(height, 1)
                                if _ratio < SHOULDER_TOP_MIN_RATIO:
                                    logger.debug(
                                        "어깨 bbox 상단 치우침 거부(옷걸이 오탐): ratio=%.2f", _ratio
                                    )
                                    continue

                person_ev = DetectionEvent(
                    event_type=EventType.PERSON,
                    x=x1,
                    y=y1,
                    width=width,
                    height=height,
                    confidence=conf,
                    timestamp=time.time(),
                    object_id=track_id,
                    class_idx=0,
                )
                person_events.append(person_ev)

                if is_fallen:
                    fall_events.append(
                        DetectionEvent(
                            event_type=EventType.FALL_DETECTED,
                            x=x1,
                            y=y1,
                            width=width,
                            height=height,
                            confidence=conf,
                            timestamp=time.time(),
                            object_id=track_id,
                            class_idx=0,
                            keypoints=kpts_for_fall,
                        )
                    )

        # 사후 중복 제거: YOLO NMS 이후에도 부분 겹침 박스가 남는 경우 처리
        person_events = self._remove_duplicates(person_events, iou_threshold=PERSON_DUPLICATE_IOU_THRESHOLD)
        logger.debug(
            "포즈 전체 프레임: 사람 %d명 (중복 제거 후), 낙상 %d건",
            len(person_events), len(fall_events),
        )
        return person_events, fall_events


    # ====================
    # 포즈 기반 사람 검증
    # ====================
    def _validate_person_keypoints(self, keypoints, idx: int) -> bool:
        """키포인트 신뢰도 및 수직 체형 순서로 실제 사람인지 검증

        두 가지 검사를 순차 수행:
        1. 주요 키포인트(코·어깨·엉덩이) 중 2개 이상 신뢰도 충족
        2. 감지된 키포인트가 해부학적 수직 순서를 만족 (코 < 어깨 < 엉덩이)
           이를 위반하면 옷걸이·의류 오탐으로 판단
        """
        try:
            kpts = self._extract_keypoints(keypoints, idx)
            if kpts is None:
                return True  # 키포인트 추출 실패 시

            # COCO 키포인트: 0-코, 5-왼쪽어깨, 6-오른쪽어깨, 11-왼쪽엉덩이, 12-오른쪽엉덩이
            nose_conf = kpts[0][2] if len(kpts) > 0 else 0
            left_shoulder_conf = kpts[5][2] if len(kpts) > 5 else 0
            right_shoulder_conf = kpts[6][2] if len(kpts) > 6 else 0
            left_hip_conf = kpts[11][2] if len(kpts) > 11 else 0
            right_hip_conf = kpts[12][2] if len(kpts) > 12 else 0

            has_nose = nose_conf > MIN_KEYPOINT_CONFIDENCE
            has_shoulder = (left_shoulder_conf > MIN_KEYPOINT_CONFIDENCE or
                            right_shoulder_conf > MIN_KEYPOINT_CONFIDENCE)
            has_hip = (left_hip_conf > MIN_KEYPOINT_CONFIDENCE or
                       right_hip_conf > MIN_KEYPOINT_CONFIDENCE)

            # 검사 1: 주요 키포인트 2개 이상 필요
            valid_keypoints = sum([has_nose, has_shoulder, has_hip])
            if valid_keypoints < 2:
                logger.debug("키포인트 부족: nose=%s, shoulder=%s, hip=%s", has_nose, has_shoulder, has_hip)
                return False

            # 검사 2: 수직 순서 검증 (코 위 → 어깨 → 엉덩이 아래)
            # 이미지 좌표계: y 값이 클수록 아래 → nose_y < shoulder_y < hip_y 이어야 함
            if has_nose and has_shoulder:
                nose_y = kpts[0][1]
                shoulder_y_vals = []
                if left_shoulder_conf > MIN_KEYPOINT_CONFIDENCE:
                    shoulder_y_vals.append(kpts[5][1])
                if right_shoulder_conf > MIN_KEYPOINT_CONFIDENCE:
                    shoulder_y_vals.append(kpts[6][1])
                min_shoulder_y = min(shoulder_y_vals)
                if nose_y >= min_shoulder_y:
                    # 코가 어깨보다 아래에 있음 → 옷걸이/의류 오탐
                    logger.debug("수직 순서 위반(코>=어깨): nose_y=%.1f, shoulder_y=%.1f", nose_y, min_shoulder_y)
                    return False

            if has_shoulder and has_hip:
                shoulder_y_vals = []
                if left_shoulder_conf > MIN_KEYPOINT_CONFIDENCE:
                    shoulder_y_vals.append(kpts[5][1])
                if right_shoulder_conf > MIN_KEYPOINT_CONFIDENCE:
                    shoulder_y_vals.append(kpts[6][1])
                hip_y_vals = []
                if left_hip_conf > MIN_KEYPOINT_CONFIDENCE:
                    hip_y_vals.append(kpts[11][1])
                if right_hip_conf > MIN_KEYPOINT_CONFIDENCE:
                    hip_y_vals.append(kpts[12][1])
                avg_shoulder_y = sum(shoulder_y_vals) / len(shoulder_y_vals)
                avg_hip_y = sum(hip_y_vals) / len(hip_y_vals)
                if avg_shoulder_y >= avg_hip_y:
                    # 어깨가 엉덩이보다 아래·같은 위치 → 자세 비정상
                    logger.debug("수직 순서 위반(어깨>=엉덩이): shoulder_y=%.1f, hip_y=%.1f", avg_shoulder_y, avg_hip_y)
                    return False

            # 검사 3: 얼굴(코)도 없고 무릎·발목도 없으면 옷걸이/행거 의류 오탐
            # 실제 사람은 반드시 얼굴 또는 다리(무릎·발목) 키포인트 중 하나는 감지됨
            # 옷걸이에 걸린 패딩은 코도 없고 다리도 없음
            has_lower_leg = (
                (len(kpts) > 13 and kpts[13][2] > MIN_KEYPOINT_CONFIDENCE) or  # 왼쪽 무릎
                (len(kpts) > 14 and kpts[14][2] > MIN_KEYPOINT_CONFIDENCE) or  # 오른쪽 무릎
                (len(kpts) > 15 and kpts[15][2] > MIN_KEYPOINT_CONFIDENCE) or  # 왼쪽 발목
                (len(kpts) > 16 and kpts[16][2] > MIN_KEYPOINT_CONFIDENCE)     # 오른쪽 발목
            )
            if not has_nose and not has_lower_leg:
                logger.debug("얼굴(코)·다리 키포인트 모두 부재: 옷걸이/의류 오탐 판단")
                return False

            return True
        except Exception as exc:
            logger.debug("키포인트 검증 실패: %s", exc)
            return True

    # ==========================
    # 낙상 감지 (포즈 기반)
    # ==========================

    def _detect_fall_from_keypoints(self, keypoints, idx: int, bbox_width: int, bbox_height: int) -> bool:
        """COCO 키포인트를 이용한 포즈 기반 낙상 감지

        4가지 방법 중 하나라도 낙상으로 판단되면 True 반환.
        사용 상수: FALL_ANGLE_HORIZONTAL, FALL_ANGLE_INVERTED, MIN_HIP_CONFIDENCE,
                   FALL_KEYPOINT_SPAN_RATIO

        방법 1 (어깨-엉덩이 각도):
            어깨 중심~엉덩이 중심 벡터가 수평(±40°) 이내이면 낙상 판정.
            한쪽 엉덩이 키포인트만 있어도 적용(OR 조건). 코 불필요.

        방법 2 (다리가 머리 위):
            무릎 또는 발목 y좌표가 코 y좌표보다 작으면(이미지 좌표계: 위쪽) 낙상 판정.
            코 키포인트가 유효할 때만 적용.

        방법 3 (bbox 가로비율 + 머리 위치):
            bbox 가로 > 세로×1.8 이고 코가 bbox 높이 일정 비율(fall_height_ratio) 아래에
            있으면 낙상 판정. 코 키포인트가 유효할 때만 적용.

        방법 4 (키포인트 수직 분산 비율):
            감지된 모든 키포인트의 y 범위가 bbox 높이의 FALL_KEYPOINT_SPAN_RATIO 미만이고
            bbox가 어느 정도 가로로 넓으면(> 높이×1.3) 낙상 판정.
            hip·nose 불필요 → 어깨만 감지돼도 작동.
        """
        kpts = self._extract_keypoints(keypoints, idx)
        if kpts is None:
            return False

        try:
            
            # COCO 키포인트: 0-코, 5-왼쪽어깨, 6-오른쪽어깨
            #                 11-왼쪽엉덩이, 12-오른쪽엉덩이, 13-왼쪽무릎, 14-오른쪽무릎
            #                 15-왼쪽발목, 16-오른쪽발목
            nose = kpts[0][:2]
            left_shoulder = kpts[5][:2]
            right_shoulder = kpts[6][:2]
            left_hip = kpts[11][:2]
            right_hip = kpts[12][:2]
            left_knee = kpts[13][:2]
            right_knee = kpts[14][:2]
            left_ankle = kpts[15][:2]
            right_ankle = kpts[16][:2]
            
            # 신뢰도 확인
            # - 어깨 키포인트 중 하나 이상 필수
            # - 코(nose)는 선택: 완전히 누우면 카메라에서 안 보여 신뢰도 낮아질 수 있음
            nose_valid = kpts[0][2] >= MIN_KEYPOINT_CONFIDENCE
            left_shoulder_valid = kpts[5][2] >= MIN_KEYPOINT_CONFIDENCE
            right_shoulder_valid = kpts[6][2] >= MIN_KEYPOINT_CONFIDENCE

            if not left_shoulder_valid and not right_shoulder_valid:
                return False

            # 방법 1: 어깨-엉덩이 벡터 각도 (single hip 지원)
            # 코 키포인트 불필요 → 완전히 바닥에 누워도 감지 가능
            # 한쪽 엉덩이만 감지되어도 판정 가능 (OR 조건으로 recall 향상)
            left_hip_valid = kpts[11][2] >= MIN_HIP_CONFIDENCE
            right_hip_valid = kpts[12][2] >= MIN_HIP_CONFIDENCE
            if left_hip_valid or right_hip_valid:
                shoulder_xs, shoulder_ys = [], []
                if left_shoulder_valid:
                    shoulder_xs.append(left_shoulder[0])
                    shoulder_ys.append(left_shoulder[1])
                if right_shoulder_valid:
                    shoulder_xs.append(right_shoulder[0])
                    shoulder_ys.append(right_shoulder[1])
                shoulder_center = np.array([sum(shoulder_xs) / len(shoulder_xs),
                                            sum(shoulder_ys) / len(shoulder_ys)])
                hip_xs, hip_ys = [], []
                if left_hip_valid:
                    hip_xs.append(left_hip[0])
                    hip_ys.append(left_hip[1])
                if right_hip_valid:
                    hip_xs.append(right_hip[0])
                    hip_ys.append(right_hip[1])
                hip_center = np.array([sum(hip_xs) / len(hip_xs),
                                       sum(hip_ys) / len(hip_ys)])

                # 수평과 수직 각도 계산
                body_vector = hip_center - shoulder_center
                angle = np.abs(np.arctan2(body_vector[1], body_vector[0]) * 180 / np.pi)

                # 거의 수평면에 있는 것으로 간주
                # 0-40도: 오른쪽으로 누움, 140-180도: 왼쪽으로 누움
                if angle < FALL_ANGLE_HORIZONTAL or angle > FALL_ANGLE_INVERTED:
                    return True

            # 방법 2: 무릎이나 발목이 머리보다 높은 경우 (코가 감지된 경우에만)
            if nose_valid:
                valid_knees = [left_knee[1] if kpts[13][2] > MIN_HIP_CONFIDENCE else float('inf'),
                               right_knee[1] if kpts[14][2] > MIN_HIP_CONFIDENCE else float('inf')]
                valid_ankles = [left_ankle[1] if kpts[15][2] > MIN_HIP_CONFIDENCE else float('inf'),
                                right_ankle[1] if kpts[16][2] > MIN_HIP_CONFIDENCE else float('inf')]

                knee_y_min = min(valid_knees)
                ankle_y_min = min(valid_ankles)
                head_y = nose[1]

                if (knee_y_min != float('inf') and knee_y_min < head_y) or \
                   (ankle_y_min != float('inf') and ankle_y_min < head_y):
                    return True

            # 방법 3: 바운딩 박스 가로 비율 + 머리 위치 (코가 감지된 경우에만)
            # bbox 비율 임계값 2.0 → 1.8 완화: 가로:세로 비율 1.8 이상이면 누울 가능성 높음
            if nose_valid and bbox_width > bbox_height * 1.8 and nose[1] > bbox_height * self.fall_height_ratio:
                return True

            # 방법 4: 키포인트 수직 분산 비율 (hip·nose 없이도 감지)
            # 서 있는 사람: 코~발목까지 키포인트가 bbox 높이를 대부분 채움 (span_ratio≈1.0)
            # 누운 사람:   모든 키포인트가 좁은 y 범위에 밀집 (span_ratio < FALL_KEYPOINT_SPAN_RATIO)
            # bbox_width > bbox_height * 1.3: 수평으로 넓은 bbox에만 적용 (직립 좁은 bbox 오탐 방지)
            if bbox_height > 0 and bbox_width > bbox_height * 1.3:
                all_detected_ys = [
                    kpts[ki][1] for ki in range(min(len(kpts), 17))
                    if kpts[ki][2] >= MIN_KEYPOINT_CONFIDENCE
                ]
                if len(all_detected_ys) >= 3:
                    keypoint_y_span = max(all_detected_ys) - min(all_detected_ys)
                    span_ratio = keypoint_y_span / bbox_height
                    if span_ratio < FALL_KEYPOINT_SPAN_RATIO:
                        return True

            return False
            
        except Exception as exc:
            logger.debug("낙상 감지 keypoint 처리 실패(idx=%s): %s", idx, exc, exc_info=True)
            return False
    
    # ====================
    # 공개 API: 이벤트 분류
    # ====================

    def split_events(self, events: List) -> Tuple[List, List, List]:
        """이벤트를 사람, 헬멧, 기타 카테고리로 분리
        
        매개변수

            events: 이벤트 리스트
            
        반환값
            (사람 이벤트 리스트, 헬멧 이벤트 리스트, 기타 이벤트 리스트)
        """
        
        persons = [ev for ev in events if ev.event_type == EventType.PERSON]
        helmets = [ev for ev in events if ev.event_type in (EventType.HELMET, EventType.HEAD)]
        others = [ev for ev in events if ev.event_type not in (EventType.PERSON, EventType.HELMET, EventType.HEAD)]
        
        return persons, helmets, others
    
    def check_helmet_compliance(
        self,
        events: List,
        persons: Optional[List] = None,
        helmets: Optional[List] = None,
    ) -> List[Dict]:
        """사람/헬멧 매칭으로 착용 여부 판단"""

        if persons is None or helmets is None:
            persons, helmets, _ = self.split_events(events)

        # 헬멧 이벤트를 bbox dict로 변환 (is_helmet_worn 함수에서 사용)
        helmet_bboxes = [
            {'x': h.x, 'y': h.y, 'width': h.width, 'height': h.height}
            for h in helmets
        ]
        
        results = []
        for person in persons:
            person_bbox = {
                'x': person.x,
                'y': person.y,
                'width': person.width,
                'height': person.height
            }
            
            # is_helmet_worn 함수 호출 (IoU, overlap, 중심점 기반)
            wearing = is_helmet_worn(person_bbox, helmet_bboxes)
            
            results.append({
                "person": person,
                "is_wearing": wearing
            })

        return results

    # ====================
    # 공개 API: 종합 추론 인터페이스
    # ====================

    def run_inference(
        self,
        frame,
        use_helmet: bool = True,
        use_pose: bool = True,
        use_person: bool = False,
        use_face: bool = False,
    ) -> List:
        """
        프레임에 대한 종합 추론을 수행하고 헬멧 착용 여부를 판단

        우선순위: 낙상(최우선) → 사람 → 헬멧
        낙상이 감지된 사람은 헬멧 탐지를 수행하지 않는다.

        매개변수
            frame: 입력 프레임
            use_helmet: 헬멧 모델 사용 여부
            use_pose: pose 모델 사용 여부 (사람 탐지 + 낙상 감지, 기본 경로)
            use_person: pose 모델이 없을 때 person 모델 fallback 허용 여부

        반환값
            이벤트 리스트 — 낙상 이벤트를 맨 앞에 배치
        """

        if frame is None or not isinstance(frame, np.ndarray):
            return []

        # 결과 초기화
        person_events: List[DetectionEvent] = []
        fall_events: List[DetectionEvent] = []
        small_helmet_events: List[DetectionEvent] = []
        face_events: List[DetectionEvent] = []

        # 포즈 모델 우선: 전체 프레임 한 번 추론으로 사람 탐지 + 낙상 감지 동시 처리
        if use_pose and self.pose_model:
            person_events, fall_events = self._run_pose_full_frame(frame)
            logger.debug(
                "포즈 모델(전체 프레임): 사람 %s명, 낙상 %s건 감지됨",
                len(person_events), len(fall_events),
            )
        elif use_person:
            # 포즈 모델이 없을 때만 person 모델 fallback
            if self.person_model:
                person_events = self._run_single_model(self.person_model, frame, model_type="person")
                logger.debug("사람 모델(fallback): %s 감지됨", len(person_events))
            elif not self._person_warning_shown:
                logger.warning("포즈 모델과 사람 모델이 모두 없어 사람 감지가 불가합니다.")
                self._person_warning_shown = True

        # 헬멧 모델 (낙상자 제외한 기립 사람 ROI 기반)
        # 낙상이 감지된 사람은 위험 상태이므로 헬멧 탐지를 수행하지 않는다.
        if use_helmet and self.helmet_model and person_events:
            fallen_ids = {ev.object_id for ev in fall_events}
            standing_persons = [p for p in person_events if p.object_id not in fallen_ids]
            if fallen_ids:
                logger.debug(
                    "낙상자 %d명 헬멧 탐지 제외 (object_ids=%s)",
                    len(fallen_ids), fallen_ids,
                )
            if standing_persons:
                helmet_events = self._run_helmet_on_person_rois(frame, standing_persons)
                logger.debug(
                    "헬멧 모델: %d 감지됨 (threshold=%s)",
                    len(helmet_events),
                    getattr(self, "helmet_threshold", self.confidence_threshold),
                )
                small_helmet_events = self._filter_helmet_boxes(helmet_events)
        elif use_helmet and not self.helmet_model and not self._helmet_warning_shown:
            logger.warning("헬멧 모델이 로드되지 않았습니다.")
            self._helmet_warning_shown = True

        if use_face and person_events:
            face_events = self._run_face_recognition(frame, person_events)

        # 최종 반환: 낙상(최우선) → 사람 → 헬멧
        return fall_events + person_events + face_events + small_helmet_events

    def _run_face_recognition(
        self,
        frame,
        person_events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        """사람 ROI 상단에서 얼굴 검출/인식을 수행한다."""
        if frame is None or not person_events or not self.face_recognizer.enabled:
            return []

        face_events: List[DetectionEvent] = []
        now = time.time()

        for person in person_events:
            object_id = person.object_id
            if object_id is None:
                continue

            cached = self._face_identity_cache.get(object_id)
            if cached and now - float(cached.get("timestamp", 0.0)) < _FACE_TRACK_COOLDOWN_SEC:
                cached_event = cached.get("event")
                if cached_event is not None:
                    face_events.append(cached_event)
                continue

            results = self.face_recognizer.detect_and_recognize(
                frame,
                {
                    "x": person.x,
                    "y": person.y,
                    "width": person.width,
                    "height": person.height,
                },
            )
            if not results:
                self._face_identity_cache.pop(object_id, None)
                continue

            best = max(results, key=lambda item: item.confidence)
            event_type = EventType.FACE_RECOGNIZED if best.matched else EventType.FACE_UNKNOWN
            event = DetectionEvent(
                event_type=event_type,
                x=best.bbox["x"],
                y=best.bbox["y"],
                width=best.bbox["width"],
                height=best.bbox["height"],
                confidence=best.confidence,
                timestamp=now,
                object_id=object_id,
                metadata={
                    "person_object_id": object_id,
                    "face_name": best.label,
                    "face_score": round(best.confidence, 4),
                    "recognizer": "opencv_haar_baseline",
                },
            )
            face_events.append(event)
            self._face_identity_cache[object_id] = {"timestamp": now, "event": event}

        stale_ids = [
            obj_id for obj_id, item in self._face_identity_cache.items()
            if now - float(item.get("timestamp", 0.0)) > (_FACE_TRACK_COOLDOWN_SEC * 5)
        ]
        for obj_id in stale_ids:
            self._face_identity_cache.pop(obj_id, None)

        return face_events

    def run_inference_with_compliance(
        self,
        frame,
        use_helmet: bool = True,
        use_pose: bool = True,
        check_compliance: bool = True,
        use_person: bool = False,
        use_face: bool = False,
    ) -> Dict[str, List]:
        """이벤트 리스트와 헬멧 착용 준수 여부를 함께 반환"""
        events = self.run_inference(
            frame,
            use_helmet=use_helmet,
            use_pose=use_pose,
            use_person=use_person,
            use_face=use_face,
        )

        compliance: List[Dict] = []
        if check_compliance:
            persons, helmets, _ = self.split_events(events)
            if persons and helmets:
                compliance = self.check_helmet_compliance(
                    events,
                    persons=persons,
                    helmets=helmets,
                )

        return {"events": events, "compliance": compliance}

