"""AI 분석 모듈: YOLO 기반 멀티 모델 객체 탐지

상향식 검증 흐름으로 사람(검증) → 낙상(포즈) → 헬멧 모델을 순차 적용
"""

import os
import time
import logging
import hashlib
from typing import List, Dict, Optional, Tuple

import numpy as np

# 순환 참조 방지를 위해 events를 먼저 import
from .events import EventType, DetectionEvent
from ..utils.geometry import is_helmet_worn, boxes_overlap

logger = logging.getLogger(__name__)


# ====================
# 상수 정의
# ====================


# 헬멧 감지 임계값
MAX_HELMET_WIDTH = 300  # 헬멧 최대 너비
MAX_HELMET_HEIGHT = 300  # 헬멧 최대 높이
MIN_HELMET_SIZE = 15  # 최소 감지 크기
MAX_HELMET_ASPECT_RATIO = 2.0  # 헬멧 최대 가로세로 비율
DUPLICATE_IOU_THRESHOLD = 0.3  # 검증된 이벤트 중복 제거 임계값 (후처리 단계)
HEAD_REGION_RATIO = 0.35  # 헬멧 검증용 머리 영역 비율 (사람 상단 35%)


# 키포인트 감지 임계값
MIN_KEYPOINT_CONFIDENCE = 0.2
FALL_ANGLE_HORIZONTAL = 30  # 수평 각도 임계값 (도)
FALL_ANGLE_INVERTED = 150  # 역방향 수평 각도 임계값 (도)
MIN_HIP_CONFIDENCE = 0.3  # 엉덩이 키포인트 최소 신뢰도


# YOLO 모델 설정
DEFAULT_IMAGE_SIZE_HELMET = 640  # 헬멧 감지 개선
DEFAULT_IMAGE_SIZE_POSE = 640  # 기본 해상도
DEFAULT_IMAGE_SIZE_PERSON = 800  # 사람 감지 - 원거리 사람 단기 위해 800으로 업그레이드
DEFAULT_IOU_THRESHOLD = 0.45  # YOLO NMS 임계값 (모델 추론 단계)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class AIAnalyzer:
    """멀티 모델 AI 분석 시스템
    
    모델 구성:
    - 사람 모델 (yolov8n): 사람 감지 전용
    - 포즈 모델 (yolov8n-pose): 낙상 감지 (사람 ROI 내부)
    - 헬멧 모델(커스텀): 헬멧 감지 (사람 머리 ROI 내부)
    """

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
        # 클래스 매핑 (순환 import 방지를 위해 문자열로 저장)
        self.HELMET_CLASS_MAPPING_STR = {
            "helmet_missing": "head",
            "no_helmet": "head",
            "helmet": "helmet",
            "helmet_wearing": "helmet",
            "head": "head",
        }

        self.COMMON_CLASS_MAPPING_STR = {
            "danger_zone": "danger_zone",
            "unsafe_behavior": "unsafe_behavior",
            "unsafe": "unsafe_behavior",
            "person": "person",
        }

        # 병합된 클래스 매핑
        self.CLASS_MAPPING_STR = {**self.HELMET_CLASS_MAPPING_STR, **self.COMMON_CLASS_MAPPING_STR}
        
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

        # YOLO 라이브러리 확인
        if YOLO is None:
            logger.error("ultralytics 패키지가 설치되지 않았습니다. `pip install ultralytics`를 실행하세요.")
            raise ImportError("ultralytics 패키지가 필요합니다")
        
        # 모델 동기 로딩
        self.load_models()

    # ====================
    # 공개 API 메소드
    # ====================

    def run_helmet_model(self, frame):
        """헬멧 모델로 추론 실행"""
        return self._run_single_model(self.helmet_model, frame, model_type="helmet")

    def run_person_model(self, frame):
        """사람 모델로 추론 실행"""
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
        except FileNotFoundError as e:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다 ({model_path}): {e}")
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패 ({model_path}): {e}")

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
        """헬멧, Person, Pose 모델 로드"""
        self.last_load_errors.clear()
        self._try_load("helmet", self.helmet_model_path)
        self._try_load("person", self.person_model_path)
        self._try_load("pose", self.pose_model_path)

        if not any([self.helmet_model, self.person_model, self.pose_model]):
            logger.error("로드된 모델이 없습니다. 경로/라이브러리/파일을 확인하세요.")
        else:
            logger.info(
                "로드된 모델: Helmet=%s, Person=%s, Pose=%s",
                bool(self.helmet_model), bool(self.person_model), bool(self.pose_model),
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
                except Exception as e:
                    logger.warning("디바이스 설정 실패: %s", e)
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
    def _map_class_to_event_type(self, class_name: str, model_type: str):
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
            mapped_str = self.CLASS_MAPPING_STR.get(normalized)
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
        
        # 포즈 모델은 EventType을 _run_pose_on_person_rois에서 직접 정의
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
        except (ValueError, TypeError, IndexError) as e:
            logger.debug("bbox 추출 실패: %s", e)
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
        except Exception as e:
            logger.debug("포인트 추출 실패: %s", e)
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
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logger.debug("추적 ID 추출 실패: %s", e)
            return None
    
    def _generate_temp_id(self, x1: int, y1: int, width: int, height: int) -> int:
        """추적 실패 시 bbox 기반 임시 ID 생성
        
        임시 ID는 객체가 추적되지 않을 때 사용 (임시 기반 추적)
        범위: 1500000000 ~ 1999999999 (일반 추적 ID와 충돌 최소화)
        """
        # 중심 좌표 계산 (50픽셀 단위 그리드로 묶음)
        center_x = (x1 + width // 2) // 50
        center_y = (y1 + height // 2) // 50
        wq = max(width, 0) // 10
        hq = max(height, 0) // 10
        payload = f"{center_x}:{center_y}:{wq}:{hq}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=4).digest()
        return 1_500_000_000 + (int.from_bytes(digest, "big") % 500_000_000)

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
                    imgsz=DEFAULT_IMAGE_SIZE_PERSON if model_type == "person" else DEFAULT_IMAGE_SIZE_HELMET,
                    verbose=False,
                    persist=True  # 추적 결과 유지 (ID 추출 위해)
                )
            else:
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=DEFAULT_IMAGE_SIZE_PERSON if model_type == "person" else DEFAULT_IMAGE_SIZE_HELMET,
                    verbose=False,
                )
        except Exception as e:
            logger.error("모델 추론 실패 (%s): %s", model_type, e, exc_info=True)
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
                    class_name = names.get(cls_idx, None) if isinstance(names, dict) else (names[cls_idx] if cls_idx < len(names) else None)

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

                # YOLOv8 track()에서 ID 추출
                track_id = self._extract_track_id(box)
                
                # 추적 실패 시 ID 생성 (bbox 기반)
                if track_id is None:
                    track_id = self._generate_temp_id(x1, y1, width, height)
                
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
                ev.x += x1
                ev.y += y1

            helmet_events.extend(roi_events)

        return helmet_events

    def _run_pose_on_person_rois(self, frame, person_events: List) -> List:
        """사람 ROI 영역에서 포즈 모델을 실행하고 좌표 복원 (ROI 기반 방식)"""
        if frame is None or not person_events or self.pose_model is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        conf_threshold = getattr(self, 'pose_threshold', self.confidence_threshold)
        fall_events: List = []

        for person in person_events:
            x1 = max(int(person.x), 0)
            y1 = max(int(person.y), 0)
            x2 = min(int(person.x + person.width), frame_w)
            y2 = min(int(person.y + person.height), frame_h)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            try:
                results = self.pose_model.predict(
                    roi,
                    conf=conf_threshold,
                    iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=DEFAULT_IMAGE_SIZE_POSE,
                    verbose=False,
                )
            except Exception as e:
                logger.debug("Pose ROI 추론 실패: %s", e)
                continue

            best_fall = None
            for result in results:
                boxes = getattr(result, "boxes", None)
                keypoints = getattr(result, "keypoints", None)

                if boxes is None:
                    continue

                for idx, box in enumerate(boxes):
                    # 바운딩 박스 추출
                    bbox = self._extract_bbox(box)
                    if bbox is None:
                        continue
                    bx1, by1, bx2, by2 = bbox
                    width = bx2 - bx1
                    height = by2 - by1

                    # 신뢰도 추출
                    conf = self._extract_confidence(box)

                    if keypoints is not None:
                        is_real_person = self._validate_person_keypoints(keypoints, idx)
                        if not is_real_person:
                            continue

                    is_fallen = False
                    keypoints_data = None
                    if keypoints is not None:
                        is_fallen = self._detect_fall_from_keypoints(keypoints, idx, width, height)

                        if is_fallen:
                            kpts = self._extract_keypoints(keypoints, idx)
                            if kpts is not None:
                                # ROI 좌표를 전체 프레임 좌표로 변환
                                kpts[:, 0] += x1
                                kpts[:, 1] += y1
                                keypoints_data = kpts.tolist()

                    if not is_fallen:
                        continue

                    if best_fall is None or conf > best_fall["confidence"]:
                        best_fall = {
                            "confidence": conf,
                            "keypoints": keypoints_data,
                        }

            if best_fall is not None:
                fall_events.append(
                    DetectionEvent(
                        event_type=EventType.FALL_DETECTED,
                        x=person.x,
                        y=person.y,
                        width=person.width,
                        height=person.height,
                        confidence=best_fall["confidence"],
                        timestamp=time.time(),
                        object_id=person.object_id,
                        class_idx=0,
                        keypoints=best_fall["keypoints"],
                    )
                )

        return fall_events

    # ====================
    # 포즈 기반 사람 검증
    # ====================
    def _validate_person_keypoints(self, keypoints, idx: int) -> bool:
        """키포인트 신뢰도를 확인하여 실제 사람인지 검증"""
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
            
            # 최소 기준: 코 OR (어깨 1개 + 엉덩이 1개)
            has_nose = nose_conf > MIN_KEYPOINT_CONFIDENCE
            has_shoulder = (left_shoulder_conf > MIN_KEYPOINT_CONFIDENCE or 
                          right_shoulder_conf > MIN_KEYPOINT_CONFIDENCE)
            has_hip = (left_hip_conf > MIN_KEYPOINT_CONFIDENCE or 
                      right_hip_conf > MIN_KEYPOINT_CONFIDENCE)
            
            # 최소 1개의 주요 키포인트가 있어야 사람으로 간주
            valid_keypoints = sum([has_nose, has_shoulder, has_hip])
            
            if valid_keypoints < 1:
                logger.debug("키포인트 부족: nose=%s, shoulder=%s, hip=%s", has_nose, has_shoulder, has_hip)
                return False
            
            return True
        except Exception as e:
            logger.debug("키포인트 검증 실패: %s", e)
            return True
    
    def _detect_fall_from_keypoints(self, keypoints, idx: int, bbox_width: int, bbox_height: int) -> bool:
        """포즈 기반 낙상 감지"""
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
            
            # 신뢰도 확인 (최소 신뢰도 이상일 경우에만 사용)
            if kpts[0][2] < MIN_KEYPOINT_CONFIDENCE or kpts[5][2] < MIN_KEYPOINT_CONFIDENCE or kpts[6][2] < MIN_KEYPOINT_CONFIDENCE:
                return False
            
            # 방법 1: 전후 평면 자세 (누워있음) - 어깨-엉덩이 각도
            if kpts[11][2] > MIN_HIP_CONFIDENCE and kpts[12][2] > MIN_HIP_CONFIDENCE:
                shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                           (left_shoulder[1] + right_shoulder[1]) / 2])
                hip_center = np.array([(left_hip[0] + right_hip[0]) / 2,
                                      (left_hip[1] + right_hip[1]) / 2])
                
                # 수평과 수직 각도 계산
                body_vector = hip_center - shoulder_center
                angle = np.abs(np.arctan2(body_vector[1], body_vector[0]) * 180 / np.pi)
                
                # 거의 수평면에 있는 것으로 간주
                # 0-30도: 오른쪽으로 누움, 150-180도: 왼쪽으로 누움
                if angle < FALL_ANGLE_HORIZONTAL or angle > FALL_ANGLE_INVERTED:
                    return True
            
            # 방법 2: 무릎이나 발목이 머리보다 높은 경우 (머리보다 위에 있는 경우)
            valid_knees = [left_knee[1] if kpts[13][2] > MIN_HIP_CONFIDENCE else float('inf'),
                          right_knee[1] if kpts[14][2] > MIN_HIP_CONFIDENCE else float('inf')]
            valid_ankles = [left_ankle[1] if kpts[15][2] > MIN_HIP_CONFIDENCE else float('inf'),
                           right_ankle[1] if kpts[16][2] > MIN_HIP_CONFIDENCE else float('inf')]
            
            knee_y_min = min(valid_knees)
            ankle_y_min = min(valid_ankles)
            head_y = nose[1]
            
            # 무릎이나 발목이 머리보다 높은 경우
            if (knee_y_min != float('inf') and knee_y_min < head_y) or \
               (ankle_y_min != float('inf') and ankle_y_min < head_y):
                return True
            
            # 방법 3 : 바운딩 박스가 비율이 가로가 세로보다 크코, 머리 위치가 낮은경우 (낮은자세 / 2배 이상)
            if bbox_width > bbox_height * 2 and nose[1] > bbox_height * self.fall_height_ratio:
                return True
            return False
            
        except Exception as e:
            logger.debug("낙상 감지 keypoint 처리 실패(idx=%s): %s", idx, e, exc_info=True)
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
        use_person: bool = True,
    ) -> List:
        """
        프레임에 대한 종합 추론을 수행하고 헬멧 착용 여부를 판단
        
        매개변수
            frame: 입력 프레임
            use_helmet: 헬멧 모델 사용 여부
            use_pose: pose 모델 사용 여부 (낙상 감지)
            use_person: person 모델 사용 여부
            
        반환값
            이벤트 리스트 (사람 이벤트, 헬멧 이벤트, 기타 이벤트)
        """
        
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        # 결과 초기화
        person_events: List[DetectionEvent] = []
        fall_events: List[DetectionEvent] = []
        small_helmet_events: List[DetectionEvent] = []

        # 사람 모델
        if use_person:
            if self.person_model:
                person_events = self._run_single_model(self.person_model, frame, model_type="person")
                logger.debug("사람 모델: %s 감지됨", len(person_events))
            elif not self._person_warning_shown:
                logger.warning("사람 모델이 비활성화되어 사람 감지가 불가합니다.")
                self._person_warning_shown = True

        # 포즈 모델 (사람 ROI 기반 낙상 감지)
        if use_pose and self.pose_model and person_events:
            fall_events = self._run_pose_on_person_rois(frame, person_events)

        # 헬멧 모델 (사람 ROI 기반)
        if use_helmet and self.helmet_model and person_events:
            helmet_events = self._run_helmet_on_person_rois(frame, person_events)
            logger.debug(
                "헬멧 모델: %d 감지됨 (threshold=%s)",
                len(helmet_events),
                getattr(self, "helmet_threshold", self.confidence_threshold),
            )
            small_helmet_events = self._filter_helmet_boxes(helmet_events)
        elif use_helmet and not self.helmet_model and not self._helmet_warning_shown:
            logger.warning("헬멧 모델이 로드되지 않았습니다.")
            self._helmet_warning_shown = True
                
        # 최종 반환: 사람 + 헬멧 박스 + 기타 이벤트 (각각 리스트로)
        return person_events + fall_events + small_helmet_events

    def run_inference_with_compliance(
        self,
        frame,
        use_helmet: bool = True,
        use_pose: bool = True,
        check_compliance: bool = True,
        use_person: bool = True,
    ) -> Dict[str, List]:
        """이벤트 리스트와 헬멧 착용 준수 여부를 함께 반환"""
        events = self.run_inference(
            frame,
            use_helmet=use_helmet,
            use_pose=use_pose,
            use_person=use_person,
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

