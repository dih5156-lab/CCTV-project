"""포즈 키포인트 기반 낙상 감지 및 사람 검증 — FallDetector.

AIAnalyzer에서 낙상/검증 로직만 분리하여 단독 테스트 및 재사용이 가능하다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._constants import (
    FALL_ANGLE_HORIZONTAL,
    FALL_ANGLE_INVERTED,
    FALL_KEYPOINT_SPAN_RATIO,
    MIN_HIP_CONFIDENCE,
    MIN_KEYPOINT_CONFIDENCE,
    MIN_LEG_CONFIDENCE,
    SHOULDER_TOP_MIN_RATIO,
)
from ._yolo_helpers import extract_keypoints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallScore:
    """낙상 후보 점수와 판정 근거."""

    score: float
    reasons: tuple[str, ...]


class FallDetector:
    """COCO 키포인트를 이용한 낙상 감지 및 사람 자세 검증.

    낙상 감지:
        어깨-엉덩이 각도, 다리가 머리 위, bbox 가로비율,
        키포인트 수직 분산 등 단일 프레임 신호를 점수화해 판정한다.

    사람 검증:
        키포인트 신뢰도 수 + 해부학적 수직 순서(코 > 어깨 > 엉덩이)로
        옷걸이·의류 오탐을 걸러낸다.
    """

    def __init__(
        self,
        fall_height_ratio: float = 0.3,
        *,
        angle_horizontal: float = FALL_ANGLE_HORIZONTAL,
        angle_inverted: float = FALL_ANGLE_INVERTED,
        bbox_aspect_ratio: float = 1.8,
        span_bbox_aspect_ratio: float = 1.3,
        span_ratio: float = FALL_KEYPOINT_SPAN_RATIO,
        score_threshold: float = 3.0,
        enable_folded_pose: bool = False,
        folded_pose_max_span_ratio: float = 0.30,
        suppress_sitting_like_pose: bool = False,
        sitting_like_aspect_ratio: float = 1.45,
        min_keypoint_confidence: float = MIN_KEYPOINT_CONFIDENCE,
        min_hip_confidence: float = MIN_HIP_CONFIDENCE,
        min_leg_confidence: float = MIN_LEG_CONFIDENCE,
    ) -> None:
        self.fall_height_ratio = fall_height_ratio
        self.angle_horizontal = angle_horizontal
        self.angle_inverted = angle_inverted
        self.bbox_aspect_ratio = bbox_aspect_ratio
        self.span_bbox_aspect_ratio = span_bbox_aspect_ratio
        self.span_ratio = span_ratio
        self.score_threshold = score_threshold
        self.enable_folded_pose = enable_folded_pose
        self.folded_pose_max_span_ratio = folded_pose_max_span_ratio
        self.suppress_sitting_like_pose = suppress_sitting_like_pose
        self.sitting_like_aspect_ratio = sitting_like_aspect_ratio
        self.min_keypoint_confidence = min_keypoint_confidence
        self.min_hip_confidence = min_hip_confidence
        self.min_leg_confidence = min_leg_confidence

    # ── 공개 API ──────────────────────────────────────────────────────

    def detect(
        self,
        keypoints,
        idx: int,
        bbox_width: int,
        bbox_height: int,
        bbox_y: int = 0,
    ) -> bool:
        """낙상 여부를 반환한다 (True = 낙상)."""
        kpts = extract_keypoints(keypoints, idx)
        if kpts is None:
            return False
        try:
            return self._check_fall(kpts, bbox_width, bbox_height, bbox_y=bbox_y)
        except Exception as exc:
            logger.debug(
                "낙상 감지 키포인트 처리 실패(idx=%s): %s", idx, exc, exc_info=True
            )
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

    def _check_fall(
        self, kpts: np.ndarray, bbox_w: int, bbox_h: int, *, bbox_y: int = 0
    ) -> bool:
        """낙상 점수 기반 판정."""
        result = self._score_fall(kpts, bbox_w, bbox_h, bbox_y=bbox_y)
        is_fall = result.score >= self.score_threshold
        if is_fall:
            logger.debug(
                "낙상 후보 승인: score=%.2f threshold=%.2f reasons=%s",
                result.score,
                self.score_threshold,
                ",".join(result.reasons),
            )
        else:
            logger.debug(
                "낙상 후보 미충족: score=%.2f threshold=%.2f reasons=%s",
                result.score,
                self.score_threshold,
                ",".join(result.reasons),
            )
        return is_fall

    def _score_fall(
        self, kpts: np.ndarray, bbox_w: int, bbox_h: int, *, bbox_y: int = 0
    ) -> FallScore:
        """단일 프레임 포즈에서 낙상 가능성 점수와 근거를 계산한다."""
        # COCO: 0-코, 5-왼쪽어깨, 6-오른쪽어깨
        #        11-왼쪽엉덩이, 12-오른쪽엉덩이
        #        13-왼쪽무릎, 14-오른쪽무릎, 15-왼쪽발목, 16-오른쪽발목
        nose = kpts[0][:2]
        left_shoulder = kpts[5][:2]
        right_shoulder = kpts[6][:2]
        left_hip = kpts[11][:2]
        right_hip = kpts[12][:2]
        nose_valid = kpts[0][2] >= self.min_keypoint_confidence
        left_shoulder_v = kpts[5][2] >= self.min_keypoint_confidence
        right_shoulder_v = kpts[6][2] >= self.min_keypoint_confidence
        left_hip_v = kpts[11][2] >= self.min_hip_confidence
        right_hip_v = kpts[12][2] >= self.min_hip_confidence

        # 어깨 키포인트가 최소 하나 있어야 함
        if not left_shoulder_v and not right_shoulder_v:
            return FallScore(0.0, ("missing_shoulder",))

        # 의자에 기대거나 상체만 기울어진 자세 오탐을 줄이기 위해
        # 무릎/발목 중 최소 하나가 확인될 때만 낙상 판정을 시작한다.
        if not self._has_visible_leg(kpts):
            logger.debug("낙상 후보 거부: 무릎/발목 키포인트 신뢰도 부족")
            return FallScore(0.0, ("missing_leg",))

        score = 0.0
        reasons: list[str] = []

        # 방법 1: 어깨-엉덩이 벡터 각도
        if left_hip_v or right_hip_v:
            shoulder_xs, shoulder_ys = [], []
            if left_shoulder_v:
                shoulder_xs.append(left_shoulder[0])
                shoulder_ys.append(left_shoulder[1])
            if right_shoulder_v:
                shoulder_xs.append(right_shoulder[0])
                shoulder_ys.append(right_shoulder[1])
            sc = np.array(
                [
                    sum(shoulder_xs) / len(shoulder_xs),
                    sum(shoulder_ys) / len(shoulder_ys),
                ]
            )

            hip_xs, hip_ys = [], []
            if left_hip_v:
                hip_xs.append(left_hip[0])
                hip_ys.append(left_hip[1])
            if right_hip_v:
                hip_xs.append(right_hip[0])
                hip_ys.append(right_hip[1])
            hc = np.array([sum(hip_xs) / len(hip_xs), sum(hip_ys) / len(hip_ys)])

            body_vec = hc - sc
            angle = np.abs(np.arctan2(body_vec[1], body_vec[0]) * 180 / np.pi)
            if angle < self.angle_horizontal or angle > self.angle_inverted:
                score += 2.0
                reasons.append(f"torso_horizontal:{angle:.1f}")

            vertical_gap = abs(float(hc[1] - sc[1]))
            horizontal_gap = abs(float(hc[0] - sc[0]))
            if bbox_h > 0 and vertical_gap / bbox_h < 0.35 and horizontal_gap > vertical_gap:
                score += 0.5
                reasons.append("torso_flattened")

        # 방법 2: 무릎/발목이 코보다 높은 경우
        if nose_valid:
            _inf = float("inf")
            knee_y_min = min(
                kpts[13][1] if kpts[13][2] >= self.min_leg_confidence else _inf,
                kpts[14][1] if kpts[14][2] >= self.min_leg_confidence else _inf,
            )
            ankle_y_min = min(
                kpts[15][1] if kpts[15][2] >= self.min_leg_confidence else _inf,
                kpts[16][1] if kpts[16][2] >= self.min_leg_confidence else _inf,
            )
            head_y = nose[1]
            if (knee_y_min != _inf and knee_y_min < head_y) or (
                ankle_y_min != _inf and ankle_y_min < head_y
            ):
                score += 2.5
                reasons.append("leg_above_head")

        if self.enable_folded_pose:
            folded_floor_pose = self._is_folded_floor_pose(kpts, bbox_h)
            if folded_floor_pose is not None:
                score += 3.0
                reasons.append(f"folded_floor_pose:{folded_floor_pose:.2f}")

        # 방법 3: bbox 가로 비율 + 코 위치
        aspect_ratio = bbox_w / max(bbox_h, 1)
        if (
            nose_valid
            and bbox_w > bbox_h * self.bbox_aspect_ratio
            and nose[1] - bbox_y > bbox_h * self.fall_height_ratio
        ):
            score += 2.0
            reasons.append(f"wide_bbox_low_head:{aspect_ratio:.2f}")
        elif bbox_w > bbox_h * self.span_bbox_aspect_ratio:
            score += 0.5
            reasons.append(f"wide_bbox_candidate:{aspect_ratio:.2f}")

        # 방법 4: 키포인트 수직 분산 비율
        if bbox_h > 0 and bbox_w > bbox_h * self.span_bbox_aspect_ratio:
            ys_valid = [
                kpts[ki][1]
                for ki in range(min(len(kpts), 17))
                if kpts[ki][2] >= self.min_keypoint_confidence
            ]
            if len(ys_valid) >= 3:
                span_ratio = (max(ys_valid) - min(ys_valid)) / bbox_h
                if span_ratio < self.span_ratio:
                    score += 1.5
                    reasons.append(f"low_vertical_span:{span_ratio:.2f}")

        if (
            self.suppress_sitting_like_pose
            and aspect_ratio >= self.sitting_like_aspect_ratio
            and any(reason == "torso_flattened" for reason in reasons)
            and any(reason.startswith("wide_bbox_low_head:") for reason in reasons)
            and not any(reason == "leg_above_head" for reason in reasons)
            and not any(reason.startswith("low_vertical_span:") for reason in reasons)
            and not any(reason.startswith("folded_floor_pose:") for reason in reasons)
        ):
            score = max(0.0, score - 2.0)
            reasons.append(f"sitting_like_wide_pose:-2.0:{aspect_ratio:.2f}")

        return FallScore(score, tuple(reasons))

    def folded_floor_pose_score(self, keypoints, bbox_height: int) -> float | None:
        """운영 판정과 분리해 후면/측면 바닥 착좌형 후보 신호를 계산한다."""
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            return self._is_folded_floor_pose(kpts, bbox_height)
        except Exception as exc:
            logger.debug("접힌 바닥 자세 후보 계산 실패: %s", exc)
            return None

    def _is_folded_floor_pose(self, kpts: np.ndarray, bbox_h: int) -> float | None:
        """후면/측면에서 주저앉거나 무릎을 접은 낙상 자세를 감지한다.

        얼굴/코가 보이지 않는 후면 낙상은 몸통이 수평으로 눕기보다
        바닥에 앉은 형태로 끝나는 경우가 있어, 하체 키포인트가 엉덩이
        주변에 압축되어 있는지를 별도 신호로 본다.
        """
        if bbox_h <= 0:
            return None

        shoulder_points = self._visible_points(kpts, (5, 6), self.min_keypoint_confidence)
        hip_points = self._visible_points(kpts, (11, 12), self.min_hip_confidence)
        knee_points = self._visible_points(kpts, (13, 14), self.min_leg_confidence)
        ankle_points = self._visible_points(kpts, (15, 16), self.min_leg_confidence)
        if not shoulder_points or not hip_points or not knee_points:
            return None

        shoulder_center = np.mean(shoulder_points, axis=0)
        hip_center = np.mean(hip_points, axis=0)
        lower_points = hip_points + knee_points + ankle_points
        lower_ys = [float(point[1]) for point in lower_points]
        lower_span_ratio = (max(lower_ys) - min(lower_ys)) / bbox_h
        torso_gap_ratio = abs(float(hip_center[1] - shoulder_center[1])) / bbox_h

        # 바닥에 접힌 자세는 하체가 엉덩이 주변에 몰리고, 어깨-엉덩이 간격은
        # 어느 정도 남아 있어 단순 키포인트 노이즈와 구분된다.
        if lower_span_ratio > self.folded_pose_max_span_ratio:
            return None
        if torso_gap_ratio < 0.22 or torso_gap_ratio > 0.78:
            return None
        if hip_center[1] <= shoulder_center[1]:
            return None
        return lower_span_ratio

    def _has_visible_leg(self, kpts: np.ndarray) -> bool:
        """무릎 또는 발목 키포인트가 충분히 보이는지 확인한다."""
        return any(
            len(kpts) > ki and kpts[ki][2] >= self.min_leg_confidence
            for ki in (13, 14, 15, 16)
        )

    @staticmethod
    def _visible_points(
        kpts: np.ndarray,
        indices: tuple[int, ...],
        min_confidence: float,
    ) -> list[np.ndarray]:
        return [
            kpts[ki][:2]
            for ki in indices
            if len(kpts) > ki and kpts[ki][2] >= min_confidence
        ]

    # ── 사람 검증 로직 ────────────────────────────────────────────────

    def _check_person(
        self,
        kpts: np.ndarray,
        *,
        enforce_vertical_order: bool = True,
    ) -> bool:
        """키포인트 신뢰도 및 해부학적 수직 순서 검증."""
        # COCO: 0-코, 5-왼어깨, 6-오른어깨, 11-왼엉덩이, 12-오른엉덩이
        nose_conf = kpts[0][2] if len(kpts) > 0 else 0.0
        ls_conf = kpts[5][2] if len(kpts) > 5 else 0.0
        rs_conf = kpts[6][2] if len(kpts) > 6 else 0.0
        lh_conf = kpts[11][2] if len(kpts) > 11 else 0.0
        rh_conf = kpts[12][2] if len(kpts) > 12 else 0.0

        has_nose = nose_conf > MIN_KEYPOINT_CONFIDENCE
        has_shoulder = (
            ls_conf > MIN_KEYPOINT_CONFIDENCE or rs_conf > MIN_KEYPOINT_CONFIDENCE
        )
        has_hip = lh_conf > MIN_KEYPOINT_CONFIDENCE or rh_conf > MIN_KEYPOINT_CONFIDENCE

        # 검사 1: 주요 키포인트 2개 이상 필요
        if sum([has_nose, has_shoulder, has_hip]) < 2:
            logger.debug(
                "키포인트 부족: nose=%s, shoulder=%s, hip=%s",
                has_nose,
                has_shoulder,
                has_hip,
            )
            return False

        # 검사 2: 수직 순서 (y 좌표계: 위로 갈수록 값이 작음)
        if enforce_vertical_order and has_nose and has_shoulder:
            nose_y = kpts[0][1]
            sh_ys = [kpts[5][1] for _ in [()] if ls_conf > MIN_KEYPOINT_CONFIDENCE] + [
                kpts[6][1] for _ in [()] if rs_conf > MIN_KEYPOINT_CONFIDENCE
            ]
            sh_ys = ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) + (
                [kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else []
            )
            if sh_ys and nose_y >= min(sh_ys):
                logger.debug(
                    "수직 순서 위반(코>=어깨): nose_y=%.1f, shoulder_y=%.1f",
                    nose_y,
                    min(sh_ys),
                )
                return False

        if enforce_vertical_order and has_shoulder and has_hip:
            sh_ys = ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) + (
                [kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else []
            )
            hip_ys = ([kpts[11][1]] if lh_conf > MIN_KEYPOINT_CONFIDENCE else []) + (
                [kpts[12][1]] if rh_conf > MIN_KEYPOINT_CONFIDENCE else []
            )
            if sh_ys and hip_ys:
                avg_sh = sum(sh_ys) / len(sh_ys)
                avg_hip = sum(hip_ys) / len(hip_ys)
                if avg_sh >= avg_hip:
                    logger.debug(
                        "수직 순서 위반(어깨>=엉덩이): shoulder_y=%.1f, hip_y=%.1f",
                        avg_sh,
                        avg_hip,
                    )
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
            sh_ys = ([kpts[5][1]] if ls_conf > MIN_KEYPOINT_CONFIDENCE else []) + (
                [kpts[6][1]] if rs_conf > MIN_KEYPOINT_CONFIDENCE else []
            )
            if not sh_ys:
                return True
            avg_sh_y = sum(sh_ys) / len(sh_ys)
            ratio = (avg_sh_y - bbox_y1) / max(bbox_height, 1)
            if ratio < SHOULDER_TOP_MIN_RATIO:
                logger.debug(
                    "어깨 bbox 상단 치우침 거부(옷걸이 오탐): ratio=%.2f", ratio
                )
                return False
            return True
        except Exception as exc:
            logger.debug("어깨 위치 검증 실패: %s", exc)
            return True
