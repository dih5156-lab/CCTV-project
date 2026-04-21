"""외형 속성 분석 — AppearanceAnalyzer.

person bbox를 상/하체로 분리한 뒤 HSV 히스토그램 기반으로
주요 색상을 추출하고, 사용자 등록 조건과 매칭한다.

지원 속성:
- upper_color: 상의 색상 (HSV)
- lower_color: 하의 색상 (HSV)
- has_helmet:  헬멧 착용 여부
- helmet_color: 헬멧 영역 색상 (HSV, 헬멧 검출 시에만)
- has_backpack: 백팩 소지 여부 (YOLO COCO class 24)
- has_handbag:  핸드백 소지 여부 (YOLO COCO class 26)

설계 원칙:
- CPU 전용 (OpenCV HSV) → GPU 부담 없음
- 조명 보정(CLAHE) 적용으로 야간·역광 대응
- 조건별 독립 평가 → 색상·소지품 조합 필터링 가능
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ._attribute_backend import AttributeCrop, AttributeBackend
from ._attribute_backends import build_attribute_backend

logger = logging.getLogger(__name__)

# ── HSV 색상 범위 매핑 ────────────────────────────────────────────────
# 각 색상에 대한 (lower, upper) HSV 범위 리스트
# red는 HSV hue가 0 근처에서 wrap-around 하므로 두 범위 사용

_COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red":    [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "orange": [((11, 70, 50), (25, 255, 255))],
    "yellow": [((26, 70, 50), (34, 255, 255))],
    "green":  [((35, 70, 50), (85, 255, 255))],
    "blue":   [((86, 70, 50), (130, 255, 255))],
    "purple": [((131, 70, 50), (169, 255, 255))],
    "white":  [((0, 0, 180), (180, 30, 255))],
    "black":  [((0, 0, 0), (180, 255, 50))],
    "gray":   [((0, 0, 51), (180, 30, 179))],
}

# 외형 조건 매칭 기본 임계값
DEFAULT_MATCH_THRESHOLD = 0.8

# 머리 영역 비율 (person bbox 상단 15%)
_HEAD_REGION_RATIO = 0.15

# 좌우 배경 제거를 위한 수평 마진 비율 (양쪽 각 30% 제거 → 중앙 40% 사용)
_HORIZONTAL_MARGIN = 0.30

# 상의/하의 색상 샘플링 밴드.
# 단순 45/55 분할 대신 중심 구간만 사용해 팔, 가방, 긴 외투, 발쪽 배경이
# 색상 판별에 섞이는 문제를 줄인다.
_UPPER_SAMPLE_START_RATIO = 0.18
_UPPER_SAMPLE_END_RATIO = 0.42
_LOWER_SAMPLE_START_RATIO = 0.58
_LOWER_SAMPLE_END_RATIO = 0.90

_POSE_KEYPOINT_MIN_CONFIDENCE = 0.30
_MIN_FULL_BODY_COVERAGE_RATIO = 0.75
_MIN_UPPER_BODY_COVERAGE_RATIO = 0.35
_MIN_COLOR_DOMINANCE_RATIO = 0.20
_MIN_HELMET_COLOR_DOMINANCE_RATIO = 0.45

# YOLO COCO 클래스 매핑 — 가방/소지품
BAG_CLASSES: Dict[str, str] = {
    "backpack": "has_backpack",
    "handbag":  "has_handbag",
    "suitcase": "has_suitcase",
}
HELMET_CLASSES = {"helmet", "helmet_wearing", "hardhat"}

# 가방 bbox 중심이 사람 bbox 주변 이 비율 이내일 때 소유로 판정
_BAG_PROXIMITY_RATIO = 0.5

# 최소 crop 크기 (px) — 너무 작으면 색상 분석 신뢰도 저하
_MIN_CROP_SIZE = 20


class AppearanceAnalyzer:
    """HSV 기반 외형 속성 추출 및 조건 매칭.

    AIAnalyzer에서 person 이벤트 생성 후 호출되며,
    등록된 조건(상의 색상, 하의 색상 등)이 일치하면
    APPEARANCE_MATCH 이벤트를 생성하도록 한다.
    """

    def __init__(
        self,
        *,
        backend: Optional[AttributeBackend] = None,
        backend_name: str = "hsv",
        backend_model_path: Optional[str] = None,
        backend_label_map_path: Optional[str] = None,
        backend_runtime: str = "auto",
        backend_device: str = "cpu",
        backend_input_size: int = 224,
        backend_score_threshold: float = 0.5,
        bbox_expand_ratio: float = 0.15,
    ) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._conditions: List[Dict] = []
        self._backend = backend or build_attribute_backend(
            backend_name,
            model_path=backend_model_path,
            label_map_path=backend_label_map_path,
            runtime=backend_runtime,
            device=backend_device,
            input_size=backend_input_size,
            score_threshold=backend_score_threshold,
        )
        self._bbox_expand_ratio = max(0.0, float(bbox_expand_ratio))

    # ── 조건 관리 ─────────────────────────────────────────────────────

    @property
    def conditions(self) -> List[Dict]:
        """등록된 외형 조건 목록."""
        return list(self._conditions)

    def set_conditions(self, conditions: List[Dict]) -> None:
        """외형 조건을 교체한다 (API에서 호출)."""
        validated = []
        for cond in conditions:
            entry: Dict = {
                "id": cond.get("id", ""),
                "name": cond.get("name", ""),
                "upper_color": cond.get("upper_color"),
                "lower_color": cond.get("lower_color"),
                "has_helmet": cond.get("has_helmet"),
                "helmet_color": cond.get("helmet_color"),
                "has_backpack": cond.get("has_backpack"),
                "has_handbag": cond.get("has_handbag"),
                "has_suitcase": cond.get("has_suitcase"),
                "threshold": float(cond.get("threshold", DEFAULT_MATCH_THRESHOLD)),
                "cameras": cond.get("cameras"),
                "enabled": cond.get("enabled", True),
            }
            validated.append(entry)
        self._conditions = validated
        logger.info("외형 조건 %d건 등록됨", len(validated))

    def add_condition(self, condition: Dict) -> Dict:
        """단일 외형 조건을 추가한다."""
        entry: Dict = {
            "id": condition.get("id", f"cond_{len(self._conditions) + 1}"),
            "name": condition.get("name", ""),
            "upper_color": condition.get("upper_color"),
            "lower_color": condition.get("lower_color"),
            "has_helmet": condition.get("has_helmet"),
            "helmet_color": condition.get("helmet_color"),
            "has_backpack": condition.get("has_backpack"),
            "has_handbag": condition.get("has_handbag"),
            "has_suitcase": condition.get("has_suitcase"),
            "threshold": float(condition.get("threshold", DEFAULT_MATCH_THRESHOLD)),
            "cameras": condition.get("cameras"),
            "enabled": condition.get("enabled", True),
        }
        self._conditions.append(entry)
        logger.info("외형 조건 추가: %s (%s)", entry["id"], entry["name"])
        return entry

    def remove_condition(self, condition_id: str) -> bool:
        """ID로 외형 조건을 제거한다."""
        before = len(self._conditions)
        self._conditions = [c for c in self._conditions if c["id"] != condition_id]
        removed = len(self._conditions) < before
        if removed:
            logger.info("외형 조건 제거: %s", condition_id)
        return removed

    def get_enabled_conditions(
        self, camera_id: Optional[str] = None,
    ) -> List[Dict]:
        """활성화된 조건만 반환한다. camera_id가 주어지면 해당 카메라로 필터링."""
        result = []
        for cond in self._conditions:
            if not cond.get("enabled", True):
                continue
            cams = cond.get("cameras")
            if cams and camera_id and camera_id not in cams:
                continue
            result.append(cond)
        return result

    # ── 속성 추출 ─────────────────────────────────────────────────────

    def extract_attributes(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        nearby_objects: Optional[List[Dict]] = None,
        keypoints: Optional[List[List[float]]] = None,
    ) -> Dict[str, object]:
        """person bbox에서 색상 속성을 추출하고, 주변 객체로 소지품을 판별한다."""
        frame_h, frame_w = frame.shape[:2]

        # bbox 클리핑
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + width, frame_w)
        y2 = min(y + height, frame_h)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < _MIN_CROP_SIZE or crop_h < _MIN_CROP_SIZE:
            return {
                "upper_color": "unknown", "lower_color": "unknown",
                "has_helmet": False,
                "helmet_color": "unknown",
                "has_backpack": False, "has_handbag": False, "has_suitcase": False,
            }

        crop = frame[y1:y2, x1:x2]

        # 좌우 배경 제거 — 중앙 영역만 사용
        margin_px = int(crop_w * _HORIZONTAL_MARGIN)
        if crop_w - 2 * margin_px >= _MIN_CROP_SIZE:
            crop = crop[:, margin_px:crop_w - margin_px]

        # 머리 영역 (상단 15%)
        head_h = max(int(crop_h * _HEAD_REGION_RATIO), 5)
        head_region = crop[:head_h, :]

        visibility = self._estimate_region_visibility(
            x=x,
            y=y,
            width=width,
            height=height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            head_h=head_h,
            crop_h=crop_h,
            keypoints=keypoints,
        )

        upper, lower = self._split_body_regions(
            crop,
            crop_h=crop_h,
            head_h=head_h,
            frame_x1=x1 + margin_px,
            frame_y1=y1,
            keypoints=keypoints,
        )

        # 소지품 판별
        bag_attrs = self._detect_bags(x, y, width, height, nearby_objects)
        has_helmet = self._has_helmet_evidence(nearby_objects)

        attrs = {
            "upper_color": self._dominant_color(upper) if visibility["upper_visible"] else "unknown",
            "lower_color": self._dominant_color(lower) if visibility["lower_visible"] else "unknown",
            "has_helmet": has_helmet,
            "helmet_color": self._dominant_color(
                head_region,
                min_ratio=_MIN_HELMET_COLOR_DOMINANCE_RATIO,
                allow_low_signal_fallback=False,
            ) if visibility["hat_visible"] and has_helmet else "unknown",
            **bag_attrs,
        }
        return self._merge_backend_attributes(
            attrs,
            frame,
            x,
            y,
            width,
            height,
        )

    def _expand_bbox(self, x: int, y: int, width: int, height: int) -> Dict[str, int]:
        """속성 분석용 person bbox를 약간 확장한다."""
        if self._bbox_expand_ratio <= 0.0:
            return {"x": x, "y": y, "width": width, "height": height}
        pad_x = int(width * self._bbox_expand_ratio)
        pad_top = int(height * self._bbox_expand_ratio)
        pad_bottom = int(height * (self._bbox_expand_ratio * 0.5))
        return {
            "x": x - pad_x,
            "y": y - pad_top,
            "width": width + (pad_x * 2),
            "height": height + pad_top + pad_bottom,
        }

    def _merge_backend_attributes(
        self,
        attrs: Dict[str, object],
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Dict[str, object]:
        """추가 속성 모델 결과를 현재 속성과 병합한다."""
        backend_attrs = self._backend.predict(
            AttributeCrop(frame=frame, **self._expand_bbox(x, y, width, height))
        )
        if not backend_attrs:
            return attrs
        merged = dict(attrs)
        for key, value in backend_attrs.items():
            if value in (None, "", "unknown"):
                continue
            merged[key] = value
        merged["attribute_backend"] = getattr(self._backend, "backend_name", "unknown")
        return merged

    def _estimate_region_visibility(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        head_h: int,
        crop_h: int,
        keypoints: Optional[List[List[float]]],
    ) -> Dict[str, bool]:
        """모자/상의/하의가 실제로 보이는지 보수적으로 추정한다."""
        top_clipped = y1 > y
        bottom_clipped = y2 < (y + height)
        coverage_ratio = crop_h / max(float(height), 1.0)

        result = {
            "hat_visible": (not top_clipped) and crop_h >= head_h + 5,
            "upper_visible": coverage_ratio >= _MIN_UPPER_BODY_COVERAGE_RATIO,
            "lower_visible": (not bottom_clipped) and coverage_ratio >= _MIN_FULL_BODY_COVERAGE_RATIO,
        }

        if not keypoints:
            return result

        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
        except (TypeError, ValueError):
            return result

        if kpts.ndim != 2 or kpts.shape[1] < 3:
            return result

        def has_visible(indices: List[int]) -> bool:
            for idx in indices:
                if len(kpts) <= idx or float(kpts[idx][2]) < _POSE_KEYPOINT_MIN_CONFIDENCE:
                    continue
                px = float(kpts[idx][0])
                py = float(kpts[idx][1])
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return True
            return False

        has_face = has_visible([0, 1, 2, 3, 4])
        has_shoulders = has_visible([5, 6])
        has_hips = has_visible([11, 12])
        has_lower_joints = has_visible([13, 14, 15, 16])

        result["hat_visible"] = has_face
        result["upper_visible"] = has_shoulders
        result["lower_visible"] = has_hips and has_lower_joints
        return result

    def _split_body_regions(
        self,
        crop: np.ndarray,
        *,
        crop_h: int,
        head_h: int,
        frame_x1: int,
        frame_y1: int,
        keypoints: Optional[List[List[float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """상의/하의 분석용 ROI를 반환한다.

        키포인트가 있으면 어깨-엉덩이-무릎 기준으로 동적으로 자르고,
        없거나 신뢰도가 낮으면 비율 기반 폴백을 사용한다.
        """
        pose_split = self._split_body_regions_from_keypoints(
            crop,
            crop_h=crop_h,
            head_h=head_h,
            frame_x1=frame_x1,
            frame_y1=frame_y1,
            keypoints=keypoints,
        )
        if pose_split is not None:
            return pose_split

        upper_start = max(head_h, int(crop_h * _UPPER_SAMPLE_START_RATIO))
        upper_end = max(upper_start + 1, int(crop_h * _UPPER_SAMPLE_END_RATIO))
        lower_start = max(upper_end + 1, int(crop_h * _LOWER_SAMPLE_START_RATIO))
        lower_end = max(lower_start + 1, int(crop_h * _LOWER_SAMPLE_END_RATIO))

        upper = crop[upper_start:upper_end, :]
        lower = crop[lower_start:lower_end, :]
        return upper, lower

    @staticmethod
    def _mean_keypoint_axis(kpts: np.ndarray, indices: List[int], axis: int) -> Optional[float]:
        values = [
            float(kpts[idx][axis])
            for idx in indices
            if len(kpts) > idx and float(kpts[idx][2]) >= _POSE_KEYPOINT_MIN_CONFIDENCE
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _split_body_regions_from_keypoints(
        self,
        crop: np.ndarray,
        *,
        crop_h: int,
        head_h: int,
        frame_x1: int,
        frame_y1: int,
        keypoints: Optional[List[List[float]]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """포즈 키포인트 기반으로 상/하체 ROI를 계산한다."""
        if not keypoints:
            return None

        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if kpts.ndim != 2 or kpts.shape[1] < 3:
            return None

        shoulder_y = self._mean_keypoint_axis(kpts, [5, 6], axis=1)
        hip_y = self._mean_keypoint_axis(kpts, [11, 12], axis=1)
        knee_y = self._mean_keypoint_axis(kpts, [13, 14], axis=1)
        center_x = self._mean_keypoint_axis(kpts, [5, 6, 11, 12], axis=0)
        shoulder_span = None
        hip_span = None

        if len(kpts) > 6 and float(kpts[5][2]) >= _POSE_KEYPOINT_MIN_CONFIDENCE and float(kpts[6][2]) >= _POSE_KEYPOINT_MIN_CONFIDENCE:
            shoulder_span = abs(float(kpts[6][0]) - float(kpts[5][0]))
        if len(kpts) > 12 and float(kpts[11][2]) >= _POSE_KEYPOINT_MIN_CONFIDENCE and float(kpts[12][2]) >= _POSE_KEYPOINT_MIN_CONFIDENCE:
            hip_span = abs(float(kpts[12][0]) - float(kpts[11][0]))

        if shoulder_y is None or hip_y is None or hip_y <= shoulder_y:
            return None

        crop_local_h = crop.shape[0]
        crop_local_w = crop.shape[1]
        shoulder_local = shoulder_y - frame_y1
        hip_local = hip_y - frame_y1
        knee_local = knee_y - frame_y1 if knee_y is not None else None

        torso_h = max(hip_local - shoulder_local, 1.0)
        upper_start = max(head_h, int(shoulder_local + torso_h * 0.12))
        upper_end = min(crop_local_h, max(upper_start + 1, int(hip_local - torso_h * 0.10)))

        if knee_local is not None and knee_local > hip_local:
            thigh_h = max(knee_local - hip_local, 1.0)
            lower_start = max(
                upper_end + 1,
                int(max(hip_local + thigh_h * 0.45, knee_local - thigh_h * 0.25)),
            )
            lower_end = min(crop_local_h, max(lower_start + 1, int(knee_local + thigh_h * 0.20)))
        else:
            lower_start = max(upper_end + 1, int(crop_h * _LOWER_SAMPLE_START_RATIO))
            lower_end = min(crop_local_h, max(lower_start + 1, int(crop_h * _LOWER_SAMPLE_END_RATIO)))

        if upper_end - upper_start < 5 or lower_end - lower_start < 5:
            return None

        x_start = 0
        x_end = crop_local_w
        if center_x is not None:
            center_local_x = center_x - frame_x1
            body_span = max(v for v in (shoulder_span, hip_span, crop_local_w * 0.30) if v is not None)
            half_width = max(int(body_span * 0.38), int(crop_local_w * 0.18))
            x_start = max(0, int(center_local_x - half_width))
            x_end = min(crop_local_w, int(center_local_x + half_width))
            if x_end - x_start < 5:
                x_start = 0
                x_end = crop_local_w

        upper = crop[upper_start:upper_end, x_start:x_end]
        lower = crop[lower_start:lower_end, x_start:x_end]
        if upper.size == 0 or lower.size == 0:
            return None
        return upper, lower

    @staticmethod
    def _has_helmet_evidence(nearby_objects: Optional[List[Dict]]) -> bool:
        """주변 객체 정보에 헬멧 근거가 있을 때만 True."""
        if not nearby_objects:
            return False
        for obj in nearby_objects:
            class_name = str(obj.get("class_name", "")).lower().strip()
            if class_name in HELMET_CLASSES:
                return True
        return False

    @staticmethod
    def _build_skin_mask(hsv: np.ndarray) -> np.ndarray:
        """피부색 영역 마스크를 생성한다 (제외 용도)."""
        # 일반적인 피부색 HSV 범위
        mask = cv2.inRange(
            hsv,
            np.array((0, 30, 80), dtype=np.uint8),
            np.array((25, 170, 255), dtype=np.uint8),
        )
        # 약간 더 붉은 피부톤
        mask |= cv2.inRange(
            hsv,
            np.array((165, 30, 80), dtype=np.uint8),
            np.array((180, 170, 255), dtype=np.uint8),
        )
        return mask

    def _dominant_color(
        self,
        region: np.ndarray,
        *,
        min_ratio: float = _MIN_COLOR_DOMINANCE_RATIO,
        allow_low_signal_fallback: bool = True,
    ) -> str:
        """HSV 히스토그램에서 가장 비율이 높은 색상명을 반환한다.

        피부색 픽셀은 제외하여 옷 색상만 분석한다.
        """
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return "unknown"

        # 조명 보정: L 채널에 CLAHE 적용
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        corrected = cv2.merge([l_ch, a_ch, b_ch])
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2HSV)

        # 피부색 마스크 생성 — 피부 영역을 분석에서 제외
        skin_mask = self._build_skin_mask(hsv)
        clothing_mask = cv2.bitwise_not(skin_mask)
        total = float(cv2.countNonZero(clothing_mask))
        if total < 50:
            if not allow_low_signal_fallback:
                return "unknown"
            # 피부 제외 후 남은 픽셀이 너무 적으면 전체 사용
            clothing_mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
            total = float(hsv.shape[0] * hsv.shape[1])
        if total == 0:
            return "unknown"

        best_color = "unknown"
        best_ratio = 0.0

        for color_name, ranges in _COLOR_RANGES.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                color_mask |= cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
            # 피부 영역을 제외한 색상 매칭
            combined = cv2.bitwise_and(color_mask, clothing_mask)
            ratio = float(cv2.countNonZero(combined)) / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name

        if best_ratio < min_ratio:
            logger.debug("색상 분석 근거 부족: %s (비율=%.2f < %.2f)", best_color, best_ratio, min_ratio)
            return "unknown"

        logger.debug("색상 분석 결과: %s (비율=%.2f)", best_color, best_ratio)
        return best_color

    # ── 소지품(가방) 판별 ──────────────────────────────────────────

    @staticmethod
    def _detect_bags(
        px: int,
        py: int,
        pw: int,
        ph: int,
        nearby_objects: Optional[List[Dict]] = None,
    ) -> Dict[str, bool]:
        """주변 YOLO 객체 중 사람 근처 가방류를 판별한다."""
        result = {"has_backpack": False, "has_handbag": False, "has_suitcase": False}
        if not nearby_objects:
            return result

        person_cx = px + pw / 2
        person_cy = py + ph / 2

        for obj in nearby_objects:
            cls_name = obj.get("class_name", "")
            attr_key = BAG_CLASSES.get(cls_name)
            if attr_key is None:
                continue

            ox = obj.get("x", 0)
            oy = obj.get("y", 0)
            ow = obj.get("width", 0)
            oh = obj.get("height", 0)
            obj_cx = ox + ow / 2
            obj_cy = oy + oh / 2

            # 사람 bbox 대각선 길이 기준 근접도 판별
            diag = max((pw ** 2 + ph ** 2) ** 0.5, 1)
            dist = ((person_cx - obj_cx) ** 2 + (person_cy - obj_cy) ** 2) ** 0.5

            if dist / diag < _BAG_PROXIMITY_RATIO:
                result[attr_key] = True
                logger.debug("소지품 감지: %s (거리비=%.2f)", cls_name, dist / diag)

        return result

    # ── 조건 매칭 ─────────────────────────────────────────────────────

    def match_conditions(
        self,
        attributes: Dict[str, object],
        condition: Dict,
    ) -> float:
        """추출된 속성과 단일 조건의 매칭 점수를 반환한다 (0.0 ~ 1.0)."""
        checks = 0
        matches = 0

        # 색상 조건
        for color_key in ("upper_color", "lower_color", "helmet_color"):
            if condition.get(color_key):
                checks += 1
                if attributes.get(color_key) == condition[color_key]:
                    matches += 1

        # bool 조건
        for bool_key in ("has_helmet", "has_backpack", "has_handbag", "has_suitcase"):
            if condition.get(bool_key) is not None:
                checks += 1
                if bool(attributes.get(bool_key)) == bool(condition[bool_key]):
                    matches += 1

        if checks == 0:
            return 0.0
        return matches / checks

    def find_matches(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        camera_id: Optional[str] = None,
        nearby_objects: Optional[List[Dict]] = None,
        keypoints: Optional[List[List[float]]] = None,
    ) -> List[Dict]:
        """사람 bbox에서 속성을 추출하고, 매칭되는 조건 목록을 반환한다.

        Returns:
            매칭된 조건 리스트. 각 항목은 condition + score + attributes를 포함.
        """
        conditions = self.get_enabled_conditions(camera_id)
        if not conditions:
            return []

        attributes = self.extract_attributes(
            frame, x, y, width, height, nearby_objects=nearby_objects, keypoints=keypoints,
        )
        if (
            attributes["upper_color"] == "unknown"
            and attributes["lower_color"] == "unknown"
            and attributes["helmet_color"] == "unknown"
            and not bool(attributes.get("has_helmet"))
            and not bool(attributes.get("has_backpack"))
            and not bool(attributes.get("has_handbag"))
            and not bool(attributes.get("has_suitcase"))
        ):
            return []

        results = []
        for cond in conditions:
            score = self.match_conditions(attributes, cond)
            threshold = cond.get("threshold", DEFAULT_MATCH_THRESHOLD)
            if score >= threshold:
                results.append({
                    "condition_id": cond["id"],
                    "condition_name": cond["name"],
                    "score": round(score, 4),
                    "attributes": attributes,
                })

        return results
