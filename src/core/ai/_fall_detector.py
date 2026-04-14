"""포즈 키포인트 기반 낙상 감지 및 사람 검증 — FallDetector.

AIAnalyzer에서 낙상/검증 로직만 분리하여 단독 테스트 및 재사용이 가능하다.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._constants import (
    MIN_KEYPOINT_CONFIDENCE,
    MIN_HIP_CONFIDENCE,
    FALL_ANGLE_HORIZONTAL,
    FALL_ANGLE_INVERTED,
    FALL_KEYPOINT_SPAN_RATIO,
    SHOULDER_TOP_MIN_RATIO,
)
from ._yolo_helpers import extract_keypoints

logger = logging.getLogger(__name__)


class FallDetector:
    """COCO 키포인트를 이용한 낙상 감지 및 사람 자세 검증.

    낙상 감지:
        4가지 방법(어깨-엉덩이 각도, 다리가 머리 위, bbox 가로비율,
        키포인트 수직 분산) 중 하나라도 성립하면 낙상으로 판정.

    사람 검증:
        키포인트 신뢰도 수 + 해부학적 수직 순서(코 > 어깨 > 엉덩이)로
        옷걸이·의류 오탐을 걸러낸다.
    """

    def __init__(self, fall_height_ratio: float = 0.3) -> None:
        self.fall_height_ratio = fall_height_ratio

    # ── 공개 API ──────────────────────────────────────────────────────

    def detect(self, keypoints, idx: int, bbox_width: int, bbox_height: int) -> bool:
        """낙상 여부를 반환한다 (True = 낙상)."""
        kpts = extract_keypoints(keypoints, idx)
        if kpts is None:
            return False
        try:
            return self._check_fall(kpts, bbox_width, bbox_height)
        except Exception as exc:
            logger.debug("낙상 감지 키포인트 처리 실패(idx=%s): %s", idx, exc, exc_info=True)
            return False

    def validate_person(self, keypoints, idx: int) -> bool:
        """실제 사람인지 키포인트로 검증한다 (False = 오탐 의심)."""
        try:
            kpts = extract_keypoints(keypoints, idx)
            if kpts is None:
                return True  # 키포인트 추출 실패 시 통과
            return self._check_person(kpts)
        except Exception as exc:
            logger.debug("키포인트 검증 실패: %s", exc)
            return True

    # ── 낙상 감지 로직 ────────────────────────────────────────────────

    def _check_fall(self, kpts: np.ndarray, bbox_w: int, bbox_h: int) -> bool:
        """낙상 4-방법 판정."""
        # COCO: 0-코, 5-왼쪽어깨, 6-오른쪽어깨
        #        11-왼쪽엉덩이, 12-오른쪽엉덩이
        #        13-왼쪽무릎, 14-오른쪽무릎, 15-왼쪽발목, 16-오른쪽발목
        nose             = kpts[0][:2]
        left_shoulder    = kpts[5][:2]
        right_shoulder   = kpts[6][:2]
        left_hip         = kpts[11][:2]
        right_hip        = kpts[12][:2]
        left_knee        = kpts[13][:2]
        right_knee       = kpts[14][:2]
        left_ankle       = kpts[15][:2]
        right_ankle      = kpts[16][:2]

        nose_valid         = kpts[0][2]  >= MIN_KEYPOINT_CONFIDENCE
        left_shoulder_v    = kpts[5][2]  >= MIN_KEYPOINT_CONFIDENCE
        right_shoulder_v   = kpts[6][2]  >= MIN_KEYPOINT_CONFIDENCE
        left_hip_v         = kpts[11][2] >= MIN_HIP_CONFIDENCE
        right_hip_v        = kpts[12][2] >= MIN_HIP_CONFIDENCE

        # 어깨 키포인트가 최소 하나 있어야 함
        if not left_shoulder_v and not right_shoulder_v:
            return False

        # 방법 1: 어깨-엉덩이 벡터 각도
        if left_hip_v or right_hip_v:
            shoulder_xs, shoulder_ys = [], []
            if left_shoulder_v:
                shoulder_xs.append(left_shoulder[0]); shoulder_ys.append(left_shoulder[1])
            if right_shoulder_v:
                shoulder_xs.append(right_shoulder[0]); shoulder_ys.append(right_shoulder[1])
            sc = np.array([sum(shoulder_xs) / len(shoulder_xs), sum(shoulder_ys) / len(shoulder_ys)])

            hip_xs, hip_ys = [], []
            if left_hip_v:
                hip_xs.append(left_hip[0]); hip_ys.append(left_hip[1])
            if right_hip_v:
                hip_xs.append(right_hip[0]); hip_ys.append(right_hip[1])
            hc = np.array([sum(hip_xs) / len(hip_xs), sum(hip_ys) / len(hip_ys)])

            body_vec = hc - sc
            angle    = np.abs(np.arctan2(body_vec[1], body_vec[0]) * 180 / np.pi)
            if angle < FALL_ANGLE_HORIZONTAL or angle > FALL_ANGLE_INVERTED:
                return True

        # 방법 2: 무릎/발목이 코보다 높은 경우
        if nose_valid:
            _inf = float("inf")
            knee_y_min  = min(
                kpts[13][1] if kpts[13][2] > MIN_HIP_CONFIDENCE else _inf,
                kpts[14][1] if kpts[14][2] > MIN_HIP_CONFIDENCE else _inf,
            )
            ankle_y_min = min(
                kpts[15][1] if kpts[15][2] > MIN_HIP_CONFIDENCE else _inf,
                kpts[16][1] if kpts[16][2] > MIN_HIP_CONFIDENCE else _inf,
            )
            head_y = nose[1]
            if (knee_y_min  != _inf and knee_y_min  < head_y) or \
               (ankle_y_min != _inf and ankle_y_min < head_y):
                return True

        # 방법 3: bbox 가로 비율 + 코 위치
        if nose_valid and bbox_w > bbox_h * 1.8 and nose[1] > bbox_h * self.fall_height_ratio:
            return True

        # 방법 4: 키포인트 수직 분산 비율
        if bbox_h > 0 and bbox_w > bbox_h * 1.3:
            ys_valid = [
                kpts[ki][1] for ki in range(min(len(kpts), 17))
                if kpts[ki][2] >= MIN_KEYPOINT_CONFIDENCE
            ]
            if len(ys_valid) >= 3:
                span_ratio = (max(ys_valid) - min(ys_valid)) / bbox_h
                if span_ratio < FALL_KEYPOINT_SPAN_RATIO:
                    return True

        return False

    # ── 사람 검증 로직 ────────────────────────────────────────────────

    def _check_person(self, kpts: np.ndarray) -> bool:
        """키포인트 신뢰도 및 해부학적 수직 순서 검증."""
        # COCO: 0-코, 5-왼어깨, 6-오른어깨, 11-왼엉덩이, 12-오른엉덩이
        nose_conf  = kpts[0][2]  if len(kpts) > 0  else 0.0
        ls_conf    = kpts[5][2]  if len(kpts) > 5  else 0.0
        rs_conf    = kpts[6][2]  if len(kpts) > 6  else 0.0
        lh_conf    = kpts[11][2] if len(kpts) > 11 else 0.0
        rh_conf    = kpts[12][2] if len(kpts) > 12 else 0.0

        has_nose     = nose_conf > MIN_KEYPOINT_CONFIDENCE
        has_shoulder = (ls_conf > MIN_KEYPOINT_CONFIDENCE or rs_conf > MIN_KEYPOINT_CONFIDENCE)
        has_hip      = (lh_conf > MIN_KEYPOINT_CONFIDENCE or rh_conf > MIN_KEYPOINT_CONFIDENCE)

        # 검사 1: 주요 키포인트 2개 이상 필요
        if sum([has_nose, has_shoulder, has_hip]) < 2:
            logger.debug("키포인트 부족: nose=%s, shoulder=%s, hip=%s", has_nose, has_shoulder, has_hip)
            return False

        # 검사 2: 수직 순서 (y 좌표계: 위로 갈수록 값이 작음)
        if has_nose and has_shoulder:
            nose_y = kpts[0][1]
            sh_ys  = [kpts[5][1] for _ in [()] if ls_conf > MIN_KEYPOINT_CONFIDENCE] + \
                     [kpts[6][1] for _ in [()] if rs_conf > MIN_KEYPOINT_CONFIDENCE]
            sh_ys  = ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) + \
                     ([kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else [])
            if sh_ys and nose_y >= min(sh_ys):
                logger.debug("수직 순서 위반(코>=어깨): nose_y=%.1f, shoulder_y=%.1f", nose_y, min(sh_ys))
                return False

        if has_shoulder and has_hip:
            sh_ys  = ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) + \
                     ([kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else [])
            hip_ys = ([kpts[11][1]] if lh_conf > MIN_KEYPOINT_CONFIDENCE else []) + \
                     ([kpts[12][1]] if rh_conf > MIN_KEYPOINT_CONFIDENCE else [])
            if sh_ys and hip_ys:
                avg_sh  = sum(sh_ys)  / len(sh_ys)
                avg_hip = sum(hip_ys) / len(hip_ys)
                if avg_sh >= avg_hip:
                    logger.debug("수직 순서 위반(어깨>=엉덩이): shoulder_y=%.1f, hip_y=%.1f", avg_sh, avg_hip)
                    return False

        # 검사 3: 얼굴(코)도 없고 다리 키포인트도 없으면 옷걸이 오탐 의심
        has_lower_leg = any(
            len(kpts) > ki and kpts[ki][2] > MIN_KEYPOINT_CONFIDENCE
            for ki in (13, 14, 15, 16)
        )
        if not has_nose and not has_lower_leg:
            logger.debug("얼굴(코)·다리 키포인트 모두 부재: 옷걸이/의류 오탐 판단")
            return False

        return True

    def validate_shoulder_position(
        self, keypoints, idx: int, bbox_y1: int, bbox_height: int
    ) -> bool:
        """어깨가 bbox 상단에 치우쳐 있으면 False (옷걸이 오탐 거부).

        어깨의 평균 y 좌표가 bbox 상단으로부터 ``SHOULDER_TOP_MIN_RATIO``
        미만 위치에 있으면 옷걸이·의류 오탐으로 간주한다.
        """
        try:
            kpts = extract_keypoints(keypoints, idx)
            if kpts is None or len(kpts) <= 6:
                return True
            ls_conf = kpts[5][2]
            rs_conf = kpts[6][2]
            sh_ys = (
                ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) +
                ([kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else [])
            )
            if not sh_ys:
                return True
            avg_sh_y = sum(sh_ys) / len(sh_ys)
            ratio = (avg_sh_y - bbox_y1) / max(bbox_height, 1)
            if ratio < SHOULDER_TOP_MIN_RATIO:
                logger.debug("어깨 bbox 상단 치우침 거부(옷걸이 오탐): ratio=%.2f", ratio)
                return False
            return True
        except Exception as exc:
            logger.debug("어깨 위치 검증 실패: %s", exc)
            return True