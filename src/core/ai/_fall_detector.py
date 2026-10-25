"""포즈 키포인트 기반 낙상 감지 및 사람 검증 — FallDetector.

AIAnalyzer에서 낙상/검증 로직만 분리하여 단독 테스트 및 재사용이 가능하다.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

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


@dataclass
class _StandingReference:
    """같은 영상·사람에서 실제 관측한 직립 기준만 짧게 보관한다."""

    last_pts: float
    first_standing_pts: float
    standing_pts: float
    standing_frames: int
    height: int
    ankle_y: float
    last_simultaneous_drop_pts: float | None = None


@dataclass(frozen=True)
class _BodyObservation:
    """같은 인물의 고신뢰 관절과 영상 시각만 하강 비교에 사용한다."""

    pts: float
    height: int
    hip_y: float
    shoulder_y: float
    ankles: tuple[float | None, float | None]


@dataclass
class _DescentHistory:
    """짧은 움직임과 아직 회복하지 않은 하강 시작 자세를 보관한다."""

    samples: deque[_BodyObservation] = field(default_factory=lambda: deque(maxlen=16))
    origin: _BodyObservation | None = None


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
        self._standing_references: dict[tuple[str, int], _StandingReference] = {}
        self._transition_clocks: dict[str, tuple[int, float]] = {}
        self._descent_histories: dict[tuple[str, int], _DescentHistory] = {}

    def clear_camera_history(self, camera_id: str) -> None:
        """카메라 제거·세대 전환 때 이전 사람의 자세 기준을 폐기한다."""
        self._transition_clocks.pop(camera_id, None)
        for key in list(self._standing_references):
            if key[0] == camera_id:
                self._standing_references.pop(key)
        for key in list(self._descent_histories):
            if key[0] == camera_id:
                self._descent_histories.pop(key)

    def observe_floor_transition(self, camera_id: str, event) -> bool:
        """같은 인물의 직립·하강 이력과 현재 자세로 바닥 전환을 판정한다."""
        if not self.enable_folded_pose:
            return False
        metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
        identity = metadata.get("frame_identity")
        if not isinstance(identity, Mapping):
            self.clear_camera_history(camera_id)
            return False
        generation, pts_ns = identity.get("generation"), identity.get("source_pts_ns")
        if (
            identity.get("camera_id") != camera_id
            or type(generation) is not int or generation < 0
            or type(pts_ns) is not int or not 0 <= pts_ns < 2**64 - 1
        ):
            self.clear_camera_history(camera_id)
            return False
        pts = pts_ns / 1_000_000_000
        previous = self._transition_clocks.get(camera_id)
        if previous is not None:
            if generation < previous[0]:
                return False
            if generation != previous[0] or pts < previous[1]:
                self.clear_camera_history(camera_id)
        self._transition_clocks[camera_id] = (generation, pts)
        # 검출 공백을 과거 자세로 채우지 않는다. 실제 고신뢰 관측이 재개돼야 판단한다.
        for key, reference in list(self._standing_references.items()):
            if key[0] == camera_id and pts - reference.last_pts > 1.0:
                self._standing_references.pop(key)
        for key, history in list(self._descent_histories.items()):
            if key[0] == camera_id and pts - history.samples[-1].pts > 1.0:
                self._descent_histories.pop(key)
        object_id = event.object_id
        if type(object_id) is not int or not 0 <= object_id < 2**64 - 1:
            return False
        key = (camera_id, object_id)
        try:
            kpts = np.asarray(event.keypoints, dtype=np.float32)
            valid = (
                kpts.shape == (17, 3) and np.isfinite(kpts).all()
                and np.all((kpts[:, 2] >= 0) & (kpts[:, 2] <= 1))
                and np.isfinite(event.confidence) and 0.5 <= event.confidence <= 1.0
                and event.width > 0 and event.height > 0
            )
        except (ValueError, TypeError):
            valid = False
        if not valid:
            self._standing_references.pop(key, None)
            self._descent_histories.pop(key, None)
            return False
        reference = self._standing_references.get(key)
        if reference is not None:
            if pts <= reference.last_pts:
                return False
            reference.last_pts = pts
        if min(kpts[j, 2] for j in (5, 6, 11, 12)) < 0.5:
            return False
        height = event.height
        shoulder_y = float(kpts[[5, 6], 1].mean())
        hip_y = float(kpts[[11, 12], 1].mean())
        ankles = [j for j in (15, 16) if kpts[j, 2] >= 0.5]
        if not ankles:
            return False
        ankle_y = float(kpts[ankles, 1].mean())
        upright = (
            min(kpts[j, 2] for j in (13, 14, 15, 16)) >= 0.5
            and event.width / height <= 0.65
            and hip_y - shoulder_y >= height * 0.25
            and ankle_y - hip_y >= height * 0.35
        )
        descending = self._observe_body_descent(key, pts, kpts, height, shoulder_y, hip_y)
        if upright:
            history = self._descent_histories.get(key)
            if history is not None:
                history.origin = None
            if reference is not None:
                reference.last_simultaneous_drop_pts = None
            if reference is None:
                camera_keys = [k for k in self._standing_references if k[0] == camera_id]
                # 장시간 운영에서 잘못 연결된 ID가 무한히 쌓이지 않도록 상한을 둔다.
                if len(camera_keys) >= 256:
                    oldest = min(camera_keys, key=lambda k: self._standing_references[k].last_pts)
                    self._standing_references.pop(oldest)
                self._standing_references[key] = _StandingReference(pts, pts, pts, 1, height, ankle_y)
            else:
                reference.standing_frames += 1
                reference.standing_pts = pts
                reference.height, reference.ankle_y = height, ankle_y
            return False
        if descending:
            return True
        if (
            reference is None or reference.standing_frames < 3
            or reference.standing_pts - reference.first_standing_pts < 0.2
            or pts - reference.standing_pts > 8.0
            or abs(ankle_y - reference.ankle_y) > reference.height * 0.25
            or shoulder_y >= hip_y
        ):
            return False
        # 착석과 구분할 독립 근거로 실제 하강 또는 수평 몸통을 요구한다.
        torso_angle = abs(float(np.degrees(np.arctan2(
            hip_y - shoulder_y, kpts[[11, 12], 0].mean() - kpts[[5, 6], 0].mean(),
        ))))
        observed_drop = (reference.last_simultaneous_drop_pts is not None
                         and 0 <= pts - reference.last_simultaneous_drop_pts <= 8.0)
        history = self._descent_histories.get(key)
        # 높이가 남은 느린 하강은 직립 직후에만 인정해 착석 뒤의 작은 움직임과 구분한다.
        still_descending = (
            pts - reference.standing_pts <= 1.0
            and height > reference.height * 0.65 and history is not None
            and history.samples[-1].pts == pts
            and any(0.25 <= pts - before.pts <= 0.45
                    and (drop := self._relative_body_drop(before, history.samples[-1])) is not None
                    and min(drop) >= 0.05 for before in history.samples)
        )
        # 하강 이력만 남은 자세는 충분히 낮아지거나 옆으로 펼쳐진 경우에만 유지한다.
        if (
            ((observed_drop and height <= reference.height * 0.65
              and (height <= reference.height * 0.55 or event.width >= height)) or still_descending
             or torso_angle < self.angle_horizontal or torso_angle > self.angle_inverted)
            and height < reference.height
            and any(min(kpts[j, 2] for j in leg) >= 0.5 for leg in ((11, 13, 15), (12, 14, 16)))
            and self._is_folded_floor_pose(kpts, height) is not None
        ):
            return True
        # 다리 투영만으로 착석을 승격하지 않도록 두 투영 경로의 높이 감소 기준을 맞춘다.
        if height > reference.height * 0.55 or min(kpts[j, 2] for j in (13, 14, 15, 16)) < 0.5:
            return False
        # 전방으로 누우면 다리가 짧게 투영돼 착석 억제에 걸릴 수 있다.
        # 실제 직립 대비 높이 감소·수평 몸통·양다리 높이 감소를 함께 요구한다.
        torso_x = abs(float(kpts[[11, 12], 0].mean() - kpts[[5, 6], 0].mean()))
        if (
            event.width >= height * 1.45
            and torso_x >= max(abs(hip_y - shoulder_y) * 2.0, height * 0.35)
            and all(abs(float(kpts[ankle, 1] - kpts[hip, 1])) <= reference.height * 0.25
                    for hip, ankle in ((11, 15), (12, 16)))
        ):
            return True
        # 무릎·발목의 수직 간격만 넓히면 정상 착석을 오인하므로 양쪽 다리의
        # 옆으로 뻗은 투영과 직립 대비 높이 감소를 함께 요구한다.
        return all(
            abs(float(kpts[hip, 1] - kpts[knee, 1])) <= height * 0.15
            and abs(float(kpts[ankle, 0] - kpts[hip, 0])) >= height * 0.25
            and abs(float(kpts[ankle, 0] - kpts[hip, 0])) >= abs(float(kpts[ankle, 1] - kpts[hip, 1]))
            and float(kpts[ankle, 1] - kpts[knee, 1]) >= height * 0.15
            for hip, knee, ankle in ((11, 13, 15), (12, 14, 16))
        )

    @staticmethod
    def _relative_body_drop(
        before: _BodyObservation, after: _BodyObservation,
    ) -> tuple[float, float] | None:
        # 가림으로 보이는 발이 바뀌어도 서로 다른 발목 위치를 하강으로 오인하지 않는다.
        feet = [(a, b) for a, b in zip(before.ankles, after.ankles) if a is not None and b is not None]
        if not feet:
            return None
        foot_shift = sum(b - a for a, b in feet) / len(feet)
        if abs(foot_shift) > before.height * 0.25:
            return None
        # 발목만 들어 올린 움직임이 몸 전체의 하강으로 증폭되지 않게 높이 감소를 교차 확인한다.
        if after.height > before.height * 0.75:
            foot_shift = max(0.0, foot_shift)
        return ((after.hip_y - before.hip_y - foot_shift) / before.height,
                (after.shoulder_y - before.shoulder_y - foot_shift) / before.height)

    def _observe_body_descent(
        self, key: tuple[str, int], pts: float, kpts: np.ndarray,
        height: int, shoulder_y: float, hip_y: float,
    ) -> bool:
        """동시 하강 뒤 실제 관측되는 낮은 골반을 기존 경보 창에 연결한다."""
        ankles = tuple(float(kpts[ankle, 1]) if min(kpts[j, 2] for j in leg) >= 0.5 else None
                       for leg, ankle in (((11, 13, 15), 15), ((12, 14, 16), 16)))
        if all(value is None for value in ankles):
            return False
        # 한쪽 다리 가림 중에도 같은 고신뢰 발목의 움직임은 기록하되 자세 확인 전에는 승격하지 않는다.
        observed_floor_pose = (all(value is not None for value in ankles)
                               or self._is_folded_floor_pose(kpts, height) is not None)
        current = _BodyObservation(pts, height, hip_y, shoulder_y, ankles)
        history = self._descent_histories.get(key)
        if history is None:
            camera_keys = [k for k in self._descent_histories if k[0] == key[0]]
            if len(camera_keys) >= 256:
                oldest = min(camera_keys, key=lambda k: self._descent_histories[k].samples[-1].pts)
                self._descent_histories.pop(oldest)
            history = self._descent_histories[key] = _DescentHistory()
        elif pts <= history.samples[-1].pts:
            return False
        while history.samples and pts - history.samples[0].pts > 0.45:
            history.samples.popleft()
        history.samples.append(current)
        if history.origin is not None:
            drop = self._relative_body_drop(history.origin, current)
            reference = self._standing_references.get(key)
            torso_angle = abs(float(np.degrees(np.arctan2(
                hip_y - shoulder_y, kpts[[11, 12], 0].mean() - kpts[[5, 6], 0].mean(),
            ))))
            # 골반만 낮은 착석은 해제하되 실제 높이 감소·수평 몸통은 낙상 유지 근거로 보존한다.
            if (pts - history.origin.pts <= 8.0 and drop is not None
                    and (reference is None or height <= reference.height * 0.65
                         or height <= history.origin.height * 0.75
                         or torso_angle < self.angle_horizontal or torso_angle > self.angle_inverted)
                    and (drop[0] >= 0.10 or height <= history.origin.height * 0.65)):
                return observed_floor_pose
            history.origin = None
        simultaneous_samples = 0
        for before in history.samples:
            reference = self._standing_references.get(key)
            recent_standing = (reference is not None and reference.standing_frames >= 3
                               and reference.standing_pts - reference.first_standing_pts >= 0.2
                               and 0 <= before.pts - reference.standing_pts <= 1.0)
            if not 0.25 <= pts - before.pts <= 0.45:
                continue
            drop = self._relative_body_drop(before, current)
            # 단일 관절 흔들림은 여러 과거 관측 또는 bbox 높이 감소로 교차 확인한다.
            if reference is not None and drop is not None and min(drop) >= 0.15:
                simultaneous_samples += 1
                if simultaneous_samples >= 2 or height <= before.height * 0.85:
                    reference.last_simultaneous_drop_pts = pts
            if not observed_floor_pose:
                continue
            # 숙인 시작은 양다리 지지 상실과 큰 하강 또는 확인된 직립 대비 높이 감소를 함께 요구한다.
            known_standing = (reference is not None and reference.standing_frames >= 3
                              and reference.standing_pts - reference.first_standing_pts >= 0.2
                              and 0 <= pts - reference.standing_pts <= 8.0)
            lost_leg_support = (drop is not None and (drop[0] >= 0.30 or known_standing)
                                and all(ankle is not None and abs(ankle - kpts[hip, 1]) <= height * 0.35
                                        for ankle, hip in zip(ankles, (11, 12))))
            if (before.hip_y - before.shoulder_y < before.height * 0.25
                    and not recent_standing and not lost_leg_support):
                continue
            # 쪼그려 앉기와 구분하되, 직전에 충분히 서 있다가 숙이며 넘어지는 흐름도 보존한다.
            if (drop is not None and drop[0] >= 0.15 and drop[1] >= 0.20
                    and (height <= before.height * 0.75 or drop[0] >= 0.30
                         or (lost_leg_support and known_standing and height <= reference.height * 0.65))):
                history.origin = before
                return True
        return False

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

    def detect_keypoints(
        self,
        keypoints: np.ndarray | list,
        bbox_width: int,
        bbox_height: int,
        bbox_y: int = 0,
    ) -> bool:
        """사람 단위 ``(N, 3)`` keypoint 배열로 낙상을 판정한다.

        crop-pose처럼 YOLO 결과 객체와 인덱스가 없는 경로에서 사용한다.
        잘못된 배열은 운영 루프를 중단하지 않고 미검출로 처리한다.
        """
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[1] < 3 or kpts.shape[0] == 0:
                return False
            return self._check_fall(kpts[:, :3], bbox_width, bbox_height, bbox_y=bbox_y)
        except Exception as exc:
            logger.debug("사람 단위 키포인트 낙상 판정 실패: %s", exc)
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

    def validate_person_keypoints(self, keypoints: np.ndarray | list) -> bool:
        """사람 단위 ``(N, 3)`` keypoint 배열을 검증한다."""
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[1] < 3 or kpts.shape[0] == 0:
                return True
            return self._check_person(kpts[:, :3])
        except Exception as exc:
            logger.debug("사람 단위 키포인트 검증 실패: %s", exc)
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
            elif (
                bbox_h > 0
                and float(sc[1] - hc[1]) >= bbox_h * 0.20
                and self._visible_points(kpts, (15, 16), self.min_leg_confidence)
            ):
                # 카메라 쪽 전방 낙상은 세로로 투영되어 절댓값 각도만으로 놓친다.
                # 발목까지 관측될 때만 보완하며 작은 흔들림·기존 몸통 중복 가산은 제외한다.
                score += 2.0
                reasons.append("torso_inverted")

            vertical_gap = abs(float(hc[1] - sc[1]))
            horizontal_gap = abs(float(hc[0] - sc[0]))
            if bbox_h > 0 and vertical_gap / bbox_h < 0.35 and horizontal_gap > vertical_gap:
                score += 0.5
                reasons.append("torso_flattened")

        # 다리로 지지한 숙임 형태에서는 머리·넓이 신호를 독립 근거로 중복 가산하지 않는다.
        # shortcut: 손 짚은 낙상과 겹치는 2D 자세는 지연될 수 있어, 독립 대조 후 시계열 근거로 보완한다.
        if (bbox_h > 0 and nose_valid and nose[1] > max(left_hip[1], right_hip[1])
                and min(kpts[j, 2] for j in (11, 12, 13, 14, 15, 16)) >= 0.5
                and all(bbox_y - bbox_h * 0.10 <= kpts[j, 1] <= bbox_y + bbox_h * 1.10
                        for j in (11, 12, 13, 14, 15, 16))
                and float(kpts[[15, 16], 1].mean() - kpts[[11, 12], 1].mean()) >= bbox_h * 0.35
                and min(kpts[15, 1], kpts[16, 1]) >= nose[1] - bbox_h * 0.05
                and (min(kpts[13, 1], kpts[14, 1]) < nose[1]
                     or max(kpts[15, 1] - kpts[13, 1], kpts[16, 1] - kpts[14, 1]) >= bbox_h * 0.10)):
            return FallScore(score, tuple(reasons + ["supported_bend"]))

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
            elif (
                bbox_h > 0
                and left_hip_v and right_hip_v
                and min(kpts[0][2], kpts[11][2], kpts[12][2]) >= 0.5
                and kpts[15][2] >= self.min_leg_confidence
                and kpts[16][2] >= self.min_leg_confidence
                and head_y - max(left_hip[1], right_hip[1]) >= bbox_h * 0.20
            ):
                # 전방 투영에서 머리가 무릎 위에 남아도 골반 아래로 내려간 근거를 보완한다.
                # 머리·양 골반 신뢰도0.5와 양 발목 관측을 요구하며 기존 다리 신호와 중복하지 않는다.
                score += 1.0
                reasons.append("head_below_hips")

        if self.enable_folded_pose:
            folded_floor_pose = self._is_folded_floor_pose(kpts, bbox_h)
            if folded_floor_pose is not None:
                # 접힌 자세만으로는 정상 착석과 구분할 수 없어 전환 이력에서 승격한다.
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
        # 발목이 가려진 착석을 짧은 하체로 오인하지 않도록 관측 근거를 요구한다.
        if not shoulder_points or not hip_points or not knee_points or not ankle_points:
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

    def validate_shoulder_keypoints(
        self,
        keypoints: np.ndarray | list,
        bbox_y1: int,
        bbox_height: int,
    ) -> bool:
        """사람 단위 ``(N, 3)`` keypoint 배열의 어깨 위치를 검증한다."""
        try:
            kpts = np.asarray(keypoints, dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[1] < 3:
                return True
            if len(kpts) <= 6:
                return True
            sh_ys = [
                kpts[index][1]
                for index in (5, 6)
                if kpts[index][2] > MIN_KEYPOINT_CONFIDENCE
            ]
            if not sh_ys:
                return True
            ratio = (float(sum(sh_ys) / len(sh_ys)) - bbox_y1) / max(bbox_height, 1)
            if ratio < SHOULDER_TOP_MIN_RATIO:
                logger.debug("어깨 bbox 상단 치우침 거부(옷걸이 오탐): ratio=%.2f", ratio)
                return False
            return True
        except Exception as exc:
            logger.debug("사람 단위 어깨 위치 검증 실패: %s", exc)
            return True
