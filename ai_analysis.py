"""
ai_analysis.py - 멀티 모델 객체 탐지 시스템
설명: YOLO 기반 헬멧/낙상/사람 감지 및 헬멧 착용 여부 판단
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from utils.bbox_utils import is_helmet_worn
from utils.geometry import boxes_overlap
from events import EventType, DetectionEvent

# 로깅 설정
logger = logging.getLogger(__name__)

# Detection Constants
MAX_HELMET_WIDTH = 1000
MAX_HELMET_HEIGHT = 1000
MAX_HELMET_BODY_SIZE = 150  # 헬멧 박스 최대 크기 (너무 크면 사람 전체를 헬멧으로 오감지)
MIN_HELMET_SIZE = 15
DUPLICATE_IOU_THRESHOLD = 0.2
HEAD_REGION_RATIO = 0.35  # 헬멧 검증용 상단 영역 비율 (사람 상단 35%)

# Keypoint Detection Constants
MIN_KEYPOINT_CONFIDENCE = 0.2
FALL_ANGLE_HORIZONTAL = 30  # 수평 각도 임계값 (도)
FALL_ANGLE_INVERTED = 150  # 역방향 수평 각도 임계값 (도)
MIN_HIP_CONFIDENCE = 0.3  # 엉덩이 관절 최소 신뢰도

# Model Constants
DEFAULT_IMAGE_SIZE_HELMET = 640
DEFAULT_IMAGE_SIZE_POSE = 1280
DEFAULT_IOU_THRESHOLD = 0.3

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class AIAnalyzer:
    """
    Multi-model AI analysis system for helmet and fall detection.
    
    Uses YOLOv8-pose for person detection with keypoint-based fall detection.
    """

    HELMET_CLASS_MAPPING = {
        "helmet_missing": EventType.HELMET_MISSING,
        "no_helmet": EventType.HELMET_MISSING,
        "helmet": EventType.HELMET_WEARING,
        "helmet_wearing": EventType.HELMET_WEARING,
        "on_helmet": EventType.HELMET_WEARING,
    }

    FALL_CLASS_MAPPING = {
        "not_fall": EventType.NOT_FALL,
        "fall_detected": EventType.FALL_DETECTED,
        "fall": EventType.FALL_DETECTED,
        "falling": EventType.FALL_DETECTED,
        "stand": EventType.NOT_FALL,
    }

    COMMON_CLASS_MAPPING = {
        "danger_zone": EventType.DANGER_ZONE,
        "unsafe_behavior": EventType.UNSAFE_BEHAVIOR,
        "unsafe": EventType.UNSAFE_BEHAVIOR,
        "person": EventType.PERSON,
    }

    # 병합된 매핑 (내부에서 사용)
    CLASS_MAPPING = {**HELMET_CLASS_MAPPING, **FALL_CLASS_MAPPING, **COMMON_CLASS_MAPPING}

    def __init__(
        self,
        model_path: Optional[str] = None,  # 기존 호환성용
        helmet_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,  # YOLOv8-pose (사람 + 관절)
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        fall_angle_threshold: float = 0.45,
        fall_height_ratio: float = 0.3,
    ):
        # 기존 호환성: model_path가 주어지면 pose_model_path로 사용
        if model_path and not pose_model_path:
            pose_model_path = model_path
        
        self.helmet_model_path = helmet_model_path
        self.pose_model_path = pose_model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_height_ratio = fall_height_ratio

        # 모델 객체 (별도 보관)
        self.helmet_model = None
        self.pose_model = None  # YOLOv8-pose 모델
        self.current_model_type = None

        # 편의: 마지막 로드 에러 메시지
        self.last_load_errors = []
        self.compliance_result = []  # 헬멧 착용 판단 결과

        # ultralytics가 설치되어 있나 체크
        if YOLO is None:
            logger.error("ultralytics 패키지가 설치되어 있지 않습니다. `pip install ultralytics` 후 사용하세요.")
            raise ImportError("ultralytics 패키지가 필요합니다")
        
        # 모델 자동 로딩
        self.load_models()

    def run_helmet_model(self, frame):
        self.current_model_type = "helmet"
        return self._run_single_model(self.helmet_model, frame)

    def run_pose_model(self, frame):
        """Pose 모델 실행 (사람 + 관절 감지)"""
        self.current_model_type = "pose"
        return self._run_pose_model(self.pose_model, frame)

    # ---------------------------
    # 모델 로딩 관련
    # ---------------------------
    def _load_model(self, model_path: str):
        """단일 모델 로딩 (YOLO 객체 반환)"""
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

            logger.info(f"모델 로드 성공: {model_path} (device={self.device})")
            return model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다 ({model_path}): {e}")
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패 ({model_path}): {e}")

    def load_models(self) -> None:
        """헬멧/Pose 모델 로드"""
        self.last_load_errors.clear()
        
        # 헬멧 모델 로드
        if self.helmet_model_path:
            try:
                self.helmet_model = self._load_model(self.helmet_model_path)
                logger.info(f"헬멧 모델 로드 완료: {self.helmet_model_path}")
            except Exception as e:
                self.helmet_model = None
                self.last_load_errors.append(("helmet", str(e)))
                logger.warning(f"헬멧 모델 로드 실패: {e}")
        else:
            logger.warning("헬멧 모델 경로가 지정되지 않았습니다")

        # Pose 모델 로드 (사람 + 관절 감지)
        if self.pose_model_path:
            try:
                self.pose_model = self._load_model(self.pose_model_path)
                logger.info(f"Pose 모델 로드 완료: {self.pose_model_path}")
            except Exception as e:
                self.pose_model = None
                self.last_load_errors.append(("pose", str(e)))
                logger.warning(f"Pose 모델 로드 실패: {e}")

        if not any([self.helmet_model, self.pose_model]):
            logger.error("모델이 로드되지 않았습니다. 경로/라이브러리/파일을 확인하세요.")
        else:
            logger.info(f"로드된 모델: 헬멧={bool(self.helmet_model)}, Pose={bool(self.pose_model)}")

    def get_loaded_model_names(self) -> Dict[str, Optional[Dict[int, str]]]:
        """
        로드된 모델별 클래스 이름 조회 (디버깅용)
        반환값 예: {"helmet": {0: "helmet_wearing", 1: "helmet_missing"}, "pose": {0: "person"}}
        """
        res = {"helmet": None, "pose": None}
        if self.helmet_model:
            try:
                res["helmet"] = getattr(self.helmet_model, "names", None)
            except Exception:
                res["helmet"] = None
        if self.pose_model:
            try:
                res["pose"] = getattr(self.pose_model, "names", None)
            except Exception:
                res["pose"] = None
        return res

    # ---------------------------
    # 장치 / threshold 설정
    # ---------------------------
    def set_device(self, device: str = "cpu") -> None:
        """디바이스 설정 (cpu 또는 cuda) — 이미 로드된 모델이 있으면 .to() 시도"""
        self.device = device
        for m in (self.helmet_model, self.pose_model):
            if m is not None:
                try:
                    m.to(device)
                except Exception as e:
                    logger.warning(f"디바이스 설정 실패: {e}")
        logger.info(f"디바이스 설정: {device}")

    def update_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 업데이트"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"임계값은 0.0 ~ 1.0 사이여야 합니다 (입력값: {threshold})")
        
        self.confidence_threshold = threshold
        logger.info(f"신뢰도 임계값 업데이트: {threshold}")

    # ---------------------------
    # 유틸리티: 클래스 매핑
    # ---------------------------
    def _map_class_to_event_type(self, class_name: str, model_type: str) -> EventType:
        if not class_name:
            return EventType.OTHER

        normalized = class_name.lower().strip().replace(" ", "_")

        if model_type == "helmet":
            return self.HELMET_CLASS_MAPPING.get(normalized, EventType.OTHER)

        if model_type == "fall":
            return self.FALL_CLASS_MAPPING.get(normalized, EventType.OTHER)

        if model_type == "person":
            if normalized == "person":
                return EventType.PERSON
            return EventType.OTHER

        return EventType.OTHER

    # ---------------------------
    # 유틸리티 헬퍼 메서드
    # ---------------------------
    def _extract_track_id(self, box) -> Optional[int]:
        """YOLOv8 track() 결과에서 tracking ID 추출
        
        Args:
            box: YOLO box 객체
            
        Returns:
            tracking ID (없으면 None)
        """
        if not hasattr(box, 'id') or box.id is None:
            return None
        
        try:
            track_id = box.id[0]
            if hasattr(track_id, 'cpu'):
                return int(track_id.cpu().numpy())
            return int(track_id)
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logger.debug(f"Track ID 추출 실패: {e}")
            return None
    
    def _generate_temp_id(self, x1: int, y1: int, width: int, height: int) -> int:
        """Tracking 실패 시 bbox 기반 임시 ID 생성
        
        동일 위치의 객체는 비슷한 ID를 가짐 (위치 기반 추적)
        범위: 1000000 ~ 9999999 (정상 track ID와 구분)
        """
        # 중심점 기반 해시 (50 픽셀 그리드로 그룹화)
        center_x = (x1 + width // 2) // 50
        center_y = (y1 + height // 2) // 50
        size_hash = (width * height) // 1000  # 크기도 고려
        
        # 해시 생성 (충돌 최소화)
        temp_id = (center_x * 1000 + center_y * 100 + size_hash) % 8999999 + 1000000
        return temp_id

    def _remove_duplicates(self, events: List[DetectionEvent], iou_threshold: float = DUPLICATE_IOU_THRESHOLD) -> List[DetectionEvent]:
        """중복 박스 제거 - IoU가 높은 박스들 중 confidence 높은 것만 남김
        
        Args:
            events: 감지된 이벤트 리스트
            iou_threshold: IoU 임계값 (기본: 0.2)
            
        Returns:
            중복 제거된 이벤트 리스트
        """
        if len(events) <= 1:
            return events
        
        # confidence 높은 순으로 정렬
        sorted_events = sorted(events, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        for event in sorted_events:
            # 이미 선택된 박스들과 겹치는지 확인
            is_duplicate = any(
                boxes_overlap(event, kept_event, threshold=iou_threshold)
                for kept_event in keep
            )
            
            if not is_duplicate:
                keep.append(event)
        
        return keep

    # ---------------------------
    # 단일 모델 추론 헬퍼
    # ---------------------------
    def _run_single_model(self, model, frame) -> List[DetectionEvent]:
        """단일 YOLO 모델 결과를 DetectionEvent 리스트로 변환"""
        import numpy as _np

        events: List[DetectionEvent] = []
        if model is None or frame is None:
            return events

        # 모델별 confidence threshold 선택
        if self.current_model_type == "helmet":
            conf_threshold = getattr(self, 'helmet_threshold', self.confidence_threshold)
        elif self.current_model_type == "fall":
            conf_threshold = getattr(self, 'fall_threshold', self.confidence_threshold)
        elif self.current_model_type == "person":
            conf_threshold = getattr(self, 'person_threshold', self.confidence_threshold)
        else:
            conf_threshold = self.confidence_threshold

        # Run YOLO: ultralytics 모델 호출 (track=True로 객체 ID 자동 할당)
        try:
            results = model.track(
                frame, 
                conf=conf_threshold, 
                iou=DEFAULT_IOU_THRESHOLD, 
                imgsz=DEFAULT_IMAGE_SIZE_HELMET, 
                verbose=False, 
                persist=True, 
                tracker="bytetrack.yaml"
            )
        except Exception as e:
            logger.error(f"모델 추론 실패 ({self.current_model_type}): {e}")
            return events

        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", None) or {}
            if boxes is None:
                continue

            # boxes는 iterable of box objects
            for box in boxes:
                try:
                    # xyxy와 conf, cls 추출 (tensor -> numpy)
                    xyxy_tensor = box.xyxy[0]
                    if hasattr(xyxy_tensor, "cpu"):
                        xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    else:
                        xyxy = np.array(xyxy_tensor).astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    width = x2 - x1
                    height = y2 - y1
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"bbox 추출 실패: {e}")
                    continue

                # confidence
                try:
                    conf_tensor = box.conf[0]
                    if hasattr(conf_tensor, "cpu"):
                        conf = float(conf_tensor.cpu().numpy())
                    else:
                        conf = float(conf_tensor)
                except (ValueError, TypeError, IndexError):
                    conf = 0.0

                # class index & name
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

                # 사람 모델에서는 person 클래스만 허용 (다른 객체 필터링)
                if self.current_model_type == "person":
                    if not class_name or class_name.lower() != "person":
                        continue

                event_type = self._map_class_to_event_type(
                                    class_name or "",
                                    model_type=self.current_model_type
                                )
                
                # OTHER 이벤트 제외 (화면에 표시 안 함)
                if event_type == EventType.OTHER:
                    continue

                # YOLOv8 track()에서 ID 추출
                track_id = self._extract_track_id(box)
                
                # tracking 실패 시 임시 ID 생성 (bbox 기반)
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
        
    def _run_pose_model(self, model, frame) -> List[DetectionEvent]:
        """
        YOLOv8-pose 모델 추론 (사람 + 관절 감지)
        관절 정보를 이용해 낙상 여부를 판단
        """
        events: List[DetectionEvent] = []
        if model is None or frame is None:
            return events
        
        # Pose 모델 confidence threshold
        conf_threshold = getattr(self, 'pose_threshold', self.confidence_threshold)
        
        try:
            results = model.track(
                frame, 
                conf=conf_threshold, 
                iou=DEFAULT_IOU_THRESHOLD, 
                imgsz=DEFAULT_IMAGE_SIZE_POSE, 
                verbose=False, 
                persist=True, 
                tracker="bytetrack.yaml"
            )
        except Exception as e:
            logger.error(f"Pose 모델 추론 실패: {e}")
            return events
        
        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)  # 관절 정보
            
            if boxes is None:
                continue
            
            for idx, box in enumerate(boxes):
                try:
                    # bbox 추출
                    xyxy_tensor = box.xyxy[0]
                    if hasattr(xyxy_tensor, "cpu"):
                        xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    else:
                        xyxy = np.array(xyxy_tensor).astype(int)
                    
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # confidence
                    conf_tensor = box.conf[0]
                    if hasattr(conf_tensor, "cpu"):
                        conf = float(conf_tensor.cpu().numpy())
                    else:
                        conf = float(conf_tensor)
                    
                    # YOLOv8 track()에서 ID 추출
                    track_id = self._extract_track_id(box)
                    
                    # tracking 실패 시 임시 ID 생성 (bbox 기반)
                    if track_id is None:
                        track_id = self._generate_temp_id(x1, y1, width, height)
                    
                    # 낙상 여부 판단 (관절 정보 이용)
                    is_fallen = False
                    keypoints_data = None
                    if keypoints is not None:
                        is_fallen = self._detect_fall_from_keypoints(keypoints, idx, width, height, y1)
                        
                        # 낙상 감지 시 keypoints 데이터 추출 (시각화용)
                        if is_fallen:
                            try:
                                if hasattr(keypoints, "data"):
                                    kpts = keypoints.data[idx].cpu().numpy() if hasattr(keypoints.data[idx], "cpu") else keypoints.data[idx]
                                elif hasattr(keypoints, "xy"):
                                    kpts_xy = keypoints.xy[idx].cpu().numpy() if hasattr(keypoints.xy[idx], "cpu") else keypoints.xy[idx]
                                    kpts_conf = keypoints.conf[idx].cpu().numpy() if hasattr(keypoints.conf[idx], "cpu") else keypoints.conf[idx]
                                    kpts = np.column_stack([kpts_xy, kpts_conf])
                                else:
                                    kpts = None
                                
                                if kpts is not None:
                                    keypoints_data = kpts.tolist()  # numpy -> list로 변환
                            except:
                                pass
                    
                    # 이벤트 타입 결정
                    event_type = EventType.FALL_DETECTED if is_fallen else EventType.PERSON
                    
                    ev = DetectionEvent(
                        event_type=event_type,
                        x=x1,
                        y=y1,
                        width=width,
                        height=height,
                        confidence=conf,
                        timestamp=time.time(),
                        object_id=track_id,  # YOLOv8 track ID
                        class_idx=0,  # person class
                        keypoints=keypoints_data,  # 낙상 시에만 keypoints 저장
                    )
                    events.append(ev)
                    
                except Exception as e:
                    continue
        
        # 중복 person/fall 박스 제거 (IoU 높은 박스 중 confidence 높은 것만)
        events = self._remove_duplicates(events)
        
        return events
    

    
    def _detect_fall_from_keypoints(self, keypoints, idx: int, bbox_width: int, bbox_height: int, bbox_y1: int) -> bool:
        """
        관절 정보를 이용한 낙상 감지
        
        Args:
            keypoints: YOLO pose keypoints 객체
            idx: 현재 박스 인덱스
            bbox_width: 바운딩 박스 너비
            bbox_height: 바운딩 박스 높이
            bbox_y1: 바운딩 박스 상단 Y 좌표
            
        Returns:
            낙상 여부 (True/False)
        """
        try:
            # keypoints 데이터 추출 (N, 17, 3) - [x, y, confidence]
            if hasattr(keypoints, "data"):
                kpts = keypoints.data[idx].cpu().numpy() if hasattr(keypoints.data[idx], "cpu") else keypoints.data[idx]
            elif hasattr(keypoints, "xy"):
                kpts_xy = keypoints.xy[idx].cpu().numpy() if hasattr(keypoints.xy[idx], "cpu") else keypoints.xy[idx]
                kpts_conf = keypoints.conf[idx].cpu().numpy() if hasattr(keypoints.conf[idx], "cpu") else keypoints.conf[idx]
                kpts = np.column_stack([kpts_xy, kpts_conf])
            else:
                return False
            
            # COCO keypoints: 0-nose, 5-left_shoulder, 6-right_shoulder,
            #                 11-left_hip, 12-right_hip, 13-left_knee, 14-right_knee,
            #                 15-left_ankle, 16-right_ankle
            nose = kpts[0][:2]
            left_shoulder = kpts[5][:2]
            right_shoulder = kpts[6][:2]
            left_hip = kpts[11][:2]
            right_hip = kpts[12][:2]
            left_knee = kpts[13][:2]
            right_knee = kpts[14][:2]
            left_ankle = kpts[15][:2]
            right_ankle = kpts[16][:2]
            
            # 신뢰도 확인 (최소 신뢰도 이상만 사용)
            if kpts[0][2] < MIN_KEYPOINT_CONFIDENCE or kpts[5][2] < MIN_KEYPOINT_CONFIDENCE or kpts[6][2] < MIN_KEYPOINT_CONFIDENCE:
                return False
            
            # 방법 1: 완전히 수평인 자세 (누운 자세) - 어깨-엉덩이 각도
            if kpts[11][2] > MIN_HIP_CONFIDENCE and kpts[12][2] > MIN_HIP_CONFIDENCE:
                shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                           (left_shoulder[1] + right_shoulder[1]) / 2])
                hip_center = np.array([(left_hip[0] + right_hip[0]) / 2,
                                      (left_hip[1] + right_hip[1]) / 2])
                
                # 수평선과의 각도 계산
                body_vector = hip_center - shoulder_center
                angle = np.abs(np.arctan2(body_vector[1], body_vector[0]) * 180 / np.pi)
                
                # 거의 수평이면 낙상 판정
                # 0-30도: 오른쪽으로 누움, 150-180도: 왼쪽으로 누움
                if angle < FALL_ANGLE_HORIZONTAL or angle > FALL_ANGLE_INVERTED:
                    return True
            
            # 방법 2: 무릎과 발목이 머리보다 높이 있으면 넘어진 것 (다리가 위로)
            if (kpts[13][2] > MIN_HIP_CONFIDENCE or kpts[14][2] > MIN_HIP_CONFIDENCE) and \
               (kpts[15][2] > MIN_HIP_CONFIDENCE or kpts[16][2] > MIN_HIP_CONFIDENCE):
                knee_y_min = min(left_knee[1], right_knee[1]) if kpts[13][2] > 0.3 and kpts[14][2] > 0.3 else \
                             (left_knee[1] if kpts[13][2] > 0.3 else right_knee[1])
                ankle_y_min = min(left_ankle[1], right_ankle[1]) if kpts[15][2] > 0.3 and kpts[16][2] > 0.3 else \
                              (left_ankle[1] if kpts[15][2] > 0.3 else right_ankle[1])
                head_y = nose[1]
                
                # 무릎이나 발목이 머리보다 위에 있으면 넘어진 것
                if knee_y_min < head_y or ankle_y_min < head_y:
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def split_events(self, events: List[DetectionEvent]) -> Tuple[List[DetectionEvent], List[DetectionEvent], List[DetectionEvent]]:
        """이벤트를 사람, 헬멧, 기타로 분류
        
        Args:
            events: 전체 이벤트 리스트
            
        Returns:
            (사람 이벤트, 헬멧 이벤트, 기타 이벤트) 튜플
        """
        persons = [ev for ev in events if ev.event_type == EventType.PERSON]
        helmets = [ev for ev in events if ev.event_type in (EventType.HELMET_WEARING, EventType.HELMET_MISSING)]
        others = [ev for ev in events if ev.event_type not in (EventType.PERSON, EventType.HELMET_WEARING, EventType.HELMET_MISSING)]
        
        return persons, helmets, others
    
    def check_helmet_compliance(self, events: List[DetectionEvent]) -> List[Dict]:
        """
        사람 객체와 헬멧 객체를 매칭하여 착용 여부 판단
        헬멧은 사람의 상단 35% 영역에서만 인정
        """
        persons, helmets, _ = self.split_events(events)
        
        # 헬멧 bbox 필터링: 사람 머리 영역(상단 35%)에 있고 적절한 크기인 것만 사용
        valid_helmets = []
        for h in helmets:
            # 1. 헬멧 박스가 너무 크면 제외 (몸 전체를 헬멧으로 잘못 감지)
            if h.height > MAX_HELMET_BODY_SIZE or h.width > MAX_HELMET_BODY_SIZE:
                logger.debug(f"헬멧 박스 너무 큼 제외: {h.width}x{h.height}")
                continue
            
            # 2. 헬멧이 너무 작으면 제외
            if h.height < MIN_HELMET_SIZE or h.width < MIN_HELMET_SIZE:
                logger.debug(f"헬멧 박스 너무 작음 제외: {h.width}x{h.height}")
                continue
                
            # 3. 사람 bbox와 비교하여 상단 35% 영역에 있는지 확인
            helmet_valid = False
            for person in persons:
                person_top = person.y
                person_height = person.height
                head_region_bottom = person_top + (person_height * HEAD_REGION_RATIO)
                
                helmet_center_y = h.y + (h.height / 2)
                
                # 헬멧 중심이 사람의 상단 영역에 있으면 유효
                if person_top <= helmet_center_y <= head_region_bottom:
                    # 추가 검증: 헬멧 박스가 사람 박스 너비의 60% 이하인지 확인
                    if h.width <= person.width * 0.6:
                        helmet_valid = True
                        break
            
            if helmet_valid:
                valid_helmets.append({
                    'x': h.x,
                    'y': h.y,
                    'width': h.width,
                    'height': h.height
                })
            else:
                logger.debug(f"헬멧 박스 머리 영역 밖: center_y={h.y + h.height/2}")

        logger.debug(f"헬멧 필터링: {len(helmets)}개 → {len(valid_helmets)}개 유효")
        
        results = []

        for person in persons:
            person_bbox = {
                'x': person.x,
                'y': person.y,
                'width': person.width,
                'height': person.height
            }

            wearing = is_helmet_worn(person_bbox, valid_helmets)

            results.append({
                "person": person,
                "is_wearing": wearing
            })

        return results

    # ---------------------------
    # 공용 추론 인터페이스
    # ---------------------------
    def run_inference(
        self,
        frame,
        use_helmet: bool = True,
        use_pose: bool = True,
        check_compliance: bool = True,
    ) -> List[DetectionEvent]:
        """
        프레임 추론 및 헬멧 착용 여부 판단
        
        Args:
            frame: 입력 프레임
            use_helmet: 헬멧 모델 사용 여부
            use_pose: Pose 모델 사용 여부 (사람 + 낙상 감지)
            check_compliance: 헬멧 착용 여부 판단
            
        Returns: 사람+헬멧+낙상 이벤트 리스트
        """
        if frame is None or not isinstance(frame, (np.ndarray,)):
            return []

        person_and_fall_events = []
        helmet_events = []
        
        # Pose 모델 (사람 + 낙상 감지)
        if use_pose and self.pose_model:
            self.current_model_type = "pose"
            person_and_fall_events = self._run_pose_model(self.pose_model, frame)
            logger.debug(f"Pose 모델: {len(person_and_fall_events)}개 감지")

        # 헬멧 모델 (착용 여부 판단용, 화면에 표시)
        if use_helmet and self.helmet_model:
            self.current_model_type = "helmet"
            helmet_events = self._run_single_model(self.helmet_model, frame)
            logger.debug(f"헬멧 모델: {len(helmet_events)}개 감지 (threshold={getattr(self, 'helmet_threshold', self.confidence_threshold)})")
        elif use_helmet and not self.helmet_model:
            logger.warning("헬멧 모델이 로드되지 않았습니다")
        
        # 헬멧 박스 크기 필터링 (너무 큰 박스 제외)
        small_helmet_events = []
        if helmet_events:
            for h in helmet_events:
                if h.height <= MAX_HELMET_HEIGHT and h.width <= MAX_HELMET_WIDTH:
                    small_helmet_events.append(h)
        
        # 중복 헬멧 박스 제거 (IoU > 0.3인 박스 중 confidence 높은 것만 남김)
        small_helmet_events = self._remove_duplicates(small_helmet_events)
        logger.debug(f"헬멧 필터링 후: {len(small_helmet_events)}개 (크기/중복 제거)")
        
        # 사람 이벤트만 추출 (낙상은 제외)
        person_events = [e for e in person_and_fall_events if e.event_type == EventType.PERSON]
                
        # 헬멧 착용 여부 판단 (person과 helmet 둘 다 있을 때만)
        if check_compliance and person_events and small_helmet_events:
            all_events = person_events + small_helmet_events
            compliance_results = self.check_helmet_compliance(all_events)
            self.compliance_result = compliance_results
                
        # 화면 표시용: 사람 + 헬멧 박스 + 낙상 모두 반환
        return person_and_fall_events + small_helmet_events

    # 기존 코드와의 호환성 (processor.py 등에서 _run_inference 호출 시 동작)
    def _run_inference(self, frame):
        """하위 호환용 래퍼"""
        return self.run_inference(
            frame, 
            use_helmet=bool(self.helmet_model),
            use_pose=bool(self.pose_model)
        )

