"""외형 속성 분석 — AppearanceAnalyzer.

person bbox를 상/하체로 분리한 뒤 HSV 히스토그램 기반으로
주요 색상을 추출하고, 사용자 등록 조건과 매칭한다.

지원 속성:
- upper_color: 상의 색상 (HSV)
- lower_color: 하의 색상 (HSV)
- hat_color:   머리 영역 색상 (HSV, 모자 추정)
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

# 상/하체 분리 비율 (상체: 상위 45%, 하체: 하위 55%)
_UPPER_BODY_RATIO = 0.45

# 머리 영역 비율 (person bbox 상단 15%)
_HEAD_REGION_RATIO = 0.15

# 좌우 배경 제거를 위한 수평 마진 비율 (양쪽 각 30% 제거 → 중앙 40% 사용)
_HORIZONTAL_MARGIN = 0.30

# YOLO COCO 클래스 매핑 — 가방/소지품
BAG_CLASSES: Dict[str, str] = {
    "backpack": "has_backpack",
    "handbag":  "has_handbag",
    "suitcase": "has_suitcase",
}

# 가방 bbox와 사람 bbox 겹침 판정 최소 IoU
_BAG_OVERLAP_MIN_IOU = 0.05
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

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._conditions: List[Dict] = []

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
                "hat_color": cond.get("hat_color"),
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
            "hat_color": condition.get("hat_color"),
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
                "hat_color": "unknown",
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

        # 상/하체 분리
        mid = int(crop_h * _UPPER_BODY_RATIO)
        upper = crop[head_h:mid, :]  # 머리 아래 ~ 중간
        lower = crop[mid:, :]

        # 소지품 판별
        bag_attrs = self._detect_bags(x, y, width, height, nearby_objects)

        return {
            "upper_color": self._dominant_color(upper),
            "lower_color": self._dominant_color(lower),
            "hat_color": self._dominant_color(head_region),
            **bag_attrs,
        }

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

    def _dominant_color(self, region: np.ndarray) -> str:
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
        for color_key in ("upper_color", "lower_color", "hat_color"):
            if condition.get(color_key):
                checks += 1
                if attributes.get(color_key) == condition[color_key]:
                    matches += 1

        # 소지품 조건 (bool)
        for bag_key in ("has_backpack", "has_handbag", "has_suitcase"):
            if condition.get(bag_key) is not None:
                checks += 1
                if bool(attributes.get(bag_key)) == bool(condition[bag_key]):
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
    ) -> List[Dict]:
        """사람 bbox에서 속성을 추출하고, 매칭되는 조건 목록을 반환한다.

        Returns:
            매칭된 조건 리스트. 각 항목은 condition + score + attributes를 포함.
        """
        conditions = self.get_enabled_conditions(camera_id)
        if not conditions:
            return []

        attributes = self.extract_attributes(
            frame, x, y, width, height, nearby_objects=nearby_objects,
        )
        if attributes["upper_color"] == "unknown" and attributes["lower_color"] == "unknown":
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
