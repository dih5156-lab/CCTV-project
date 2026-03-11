"""
test_ai_analysis.py — AIAnalyzer 단위 테스트

커버리지 대상:
  - _map_class_to_event_type
  - _generate_temp_id
  - _filter_helmet_boxes
  - _remove_duplicates
  - split_events
  - check_helmet_compliance
  - update_threshold
  - _detect_fall_from_keypoints (포즈 기반 낙상 감지 로직)
  - _validate_person_keypoints

전략: YOLO 모델 파일 없이 테스트하기 위해 load_models 를 패치한다.
      순수 계산 로직만 검증하므로 실제 모델 추론은 수행하지 않는다.
"""

import time

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.core.events import DetectionEvent, EventType


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _det(
    etype: str = "person",
    x: int = 10, y: int = 10,
    w: int = 50, h: int = 100,
    conf: float = 0.9,
    oid: int = 1,
) -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.from_string(etype),
        x=x, y=y, width=w, height=h,
        confidence=conf,
        timestamp=time.time(),
        object_id=oid,
    )


class MockKeypoints:
    """YOLO keypoints 객체 시뮬레이터.

    _extract_keypoints 의 ``hasattr(keypoints, "data")`` 분기를 이용한다.
    data[idx] 는 (17, 3) numpy 배열 (각 행: [x, y, confidence]) 로 구성된다.
    """

    def __init__(self, kpts_array: np.ndarray):
        self.data = [kpts_array]


def _make_kpts(overrides: dict | None = None) -> np.ndarray:
    """직립 자세 COCO 17 keypoints (17, 3) 배열 생성.

    직립 자세 기준:
      nose(0)   y=50,  어깨(5,6) y=100,  엉덩이(11,12) y=200,  발목(15,16) y=350
    """
    DEFAULT_CONF = 0.9
    kpts = np.zeros((17, 3), dtype=float)

    # 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
    kpts[0]  = [100, 50,  DEFAULT_CONF]
    kpts[1]  = [95,  45,  DEFAULT_CONF]
    kpts[2]  = [105, 45,  DEFAULT_CONF]
    kpts[3]  = [90,  48,  DEFAULT_CONF]
    kpts[4]  = [110, 48,  DEFAULT_CONF]
    # 5:left_shoulder  6:right_shoulder
    kpts[5]  = [90,  100, DEFAULT_CONF]
    kpts[6]  = [110, 100, DEFAULT_CONF]
    # 7:left_elbow  8:right_elbow
    kpts[7]  = [85,  150, 0.8]
    kpts[8]  = [115, 150, 0.8]
    # 9:left_wrist  10:right_wrist
    kpts[9]  = [80,  180, 0.7]
    kpts[10] = [120, 180, 0.7]
    # 11:left_hip  12:right_hip
    kpts[11] = [90,  200, DEFAULT_CONF]
    kpts[12] = [110, 200, DEFAULT_CONF]
    # 13:left_knee  14:right_knee
    kpts[13] = [90,  280, 0.8]
    kpts[14] = [110, 280, 0.8]
    # 15:left_ankle  16:right_ankle
    kpts[15] = [90,  350, 0.8]
    kpts[16] = [110, 350, 0.8]

    if overrides:
        for idx, vals in overrides.items():
            kpts[idx] = vals

    return kpts


# ---------------------------------------------------------------------------
# fixture: 모델 없는 AIAnalyzer
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    """YOLO 모델 로딩 없이 순수 로직 테스트용 AIAnalyzer 인스턴스."""
    from src.core.ai_analysis import AIAnalyzer
    with patch("src.core.ai_analysis.YOLO", MagicMock()):
        with patch.object(AIAnalyzer, "load_models"):
            inst = AIAnalyzer(confidence_threshold=0.5)
    return inst


# ===========================================================================
# _map_class_to_event_type
# ===========================================================================


class TestMapClassToEventType:
    def test_helmet_class_helmet_model(self, analyzer):
        assert analyzer._map_class_to_event_type("helmet", "helmet") == EventType.HELMET

    def test_helmet_wearing_class_helmet_model(self, analyzer):
        assert analyzer._map_class_to_event_type("helmet_wearing", "helmet") == EventType.HELMET

    def test_head_class_helmet_model(self, analyzer):
        assert analyzer._map_class_to_event_type("head", "helmet") == EventType.HEAD

    def test_no_helmet_class_maps_to_head(self, analyzer):
        assert analyzer._map_class_to_event_type("no_helmet", "helmet") == EventType.HEAD

    def test_helmet_missing_maps_to_head(self, analyzer):
        assert analyzer._map_class_to_event_type("helmet_missing", "helmet") == EventType.HEAD

    def test_person_model_person_class(self, analyzer):
        assert analyzer._map_class_to_event_type("person", "person") == EventType.PERSON

    def test_person_model_non_person_returns_other(self, analyzer):
        assert analyzer._map_class_to_event_type("dog", "person") == EventType.OTHER

    def test_empty_class_returns_other(self, analyzer):
        assert analyzer._map_class_to_event_type("", "helmet") == EventType.OTHER

    def test_unknown_class_returns_other(self, analyzer):
        assert analyzer._map_class_to_event_type("xyz_unknown", "helmet") == EventType.OTHER

    def test_case_insensitive_normalization(self, analyzer):
        assert analyzer._map_class_to_event_type("HELMET", "helmet") == EventType.HELMET

    def test_spaces_normalized_to_underscore(self, analyzer):
        """공백은 _로 정규화되어 'no helmet' → 'no_helmet' → HEAD 매핑."""
        assert analyzer._map_class_to_event_type("no helmet", "helmet") == EventType.HEAD


# ===========================================================================
# _generate_temp_id
# ===========================================================================


class TestGenerateTempId:
    def test_result_within_expected_range(self, analyzer):
        tid = analyzer._generate_temp_id(10, 20, 50, 60)
        assert 1_500_000_000 <= tid < 2_000_000_000

    def test_deterministic_same_inputs(self, analyzer):
        assert analyzer._generate_temp_id(10, 20, 50, 60) == analyzer._generate_temp_id(10, 20, 50, 60)

    def test_different_positions_produce_different_ids(self, analyzer):
        t1 = analyzer._generate_temp_id(0, 0, 50, 50)
        t2 = analyzer._generate_temp_id(500, 500, 50, 50)
        assert t1 != t2

    def test_zero_size_does_not_raise(self, analyzer):
        tid = analyzer._generate_temp_id(0, 0, 0, 0)
        assert isinstance(tid, int)

    def test_negative_size_clamped(self, analyzer):
        """음수 크기는 max(w, 0) 처리 — 예외 없이 정수 반환."""
        tid = analyzer._generate_temp_id(0, 0, -10, -20)
        assert isinstance(tid, int)


# ===========================================================================
# _filter_helmet_boxes
# ===========================================================================


class TestFilterHelmetBoxes:
    def test_normal_size_passes(self, analyzer):
        ev = _det("helmet", w=30, h=30)
        assert len(analyzer._filter_helmet_boxes([ev])) == 1

    def test_too_small_filtered_out(self, analyzer):
        ev = _det("helmet", w=5, h=5)  # MIN_HELMET_SIZE = 15
        assert len(analyzer._filter_helmet_boxes([ev])) == 0

    def test_too_large_filtered_out(self, analyzer):
        ev = _det("helmet", w=350, h=350)  # MAX_HELMET_WIDTH = 300
        assert len(analyzer._filter_helmet_boxes([ev])) == 0

    def test_bad_aspect_ratio_filtered_out(self, analyzer):
        ev = _det("helmet", w=200, h=50)  # ratio 4 > MAX_HELMET_ASPECT_RATIO 2
        assert len(analyzer._filter_helmet_boxes([ev])) == 0

    def test_head_event_also_filtered(self, analyzer):
        ev = _det("head", w=5, h=5)
        assert len(analyzer._filter_helmet_boxes([ev])) == 0

    def test_non_helmet_passthrough_regardless_of_size(self, analyzer):
        """헬멧/HEAD 아닌 클래스(danger_zone 등)는 크기 체크 없이 통과."""
        ev = _det("danger_zone", w=5, h=5)
        assert len(analyzer._filter_helmet_boxes([ev])) == 1

    def test_empty_list(self, analyzer):
        assert analyzer._filter_helmet_boxes([]) == []

    def test_mixed_list(self, analyzer):
        good = _det("helmet", w=40, h=40, oid=1)
        small = _det("helmet", w=3, h=3, oid=2)
        other = _det("danger_zone", w=5, h=5, oid=3)
        result = analyzer._filter_helmet_boxes([good, small, other])
        # good + other 남아야 함
        assert len(result) == 2


# ===========================================================================
# _remove_duplicates
# ===========================================================================


class TestRemoveDuplicates:
    def test_empty_list_returns_empty(self, analyzer):
        assert analyzer._remove_duplicates([]) == []

    def test_single_event_kept(self, analyzer):
        ev = _det("helmet")
        assert analyzer._remove_duplicates([ev]) == [ev]

    def test_overlapping_keeps_higher_confidence(self, analyzer):
        high = _det("helmet", x=0, y=0, w=100, h=100, conf=0.9, oid=1)
        low  = _det("helmet", x=0, y=0, w=100, h=100, conf=0.5, oid=2)
        result = analyzer._remove_duplicates([high, low])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_non_overlapping_both_kept(self, analyzer):
        ev1 = _det("helmet", x=0,   y=0,   w=50, h=50, oid=1)
        ev2 = _det("helmet", x=500, y=500, w=50, h=50, oid=2)
        assert len(analyzer._remove_duplicates([ev1, ev2])) == 2

    def test_three_events_two_overlapping_one_separate(self, analyzer):
        e1 = _det("helmet", x=0, y=0, w=100, h=100, conf=0.9, oid=1)
        e2 = _det("helmet", x=0, y=0, w=100, h=100, conf=0.7, oid=2)  # 중복
        e3 = _det("helmet", x=500, y=500, w=50, h=50, conf=0.8, oid=3)  # 별개
        result = analyzer._remove_duplicates([e1, e2, e3])
        assert len(result) == 2


# ===========================================================================
# split_events
# ===========================================================================


class TestSplitEvents:
    def test_empty_input(self, analyzer):
        p, h, o = analyzer.split_events([])
        assert p == h == o == []

    def test_person_classified_as_person(self, analyzer):
        ev = _det("person")
        p, h, o = analyzer.split_events([ev])
        assert ev in p
        assert ev not in h
        assert ev not in o

    def test_helmet_classified_as_helmet(self, analyzer):
        ev = _det("helmet")
        p, h, o = analyzer.split_events([ev])
        assert ev in h

    def test_head_classified_as_helmet(self, analyzer):
        ev = _det("head")
        p, h, o = analyzer.split_events([ev])
        assert ev in h

    def test_fall_classified_as_other(self, analyzer):
        ev = _det("fall_detected")
        p, h, o = analyzer.split_events([ev])
        assert ev in o

    def test_danger_zone_classified_as_other(self, analyzer):
        ev = _det("danger_zone")
        p, h, o = analyzer.split_events([ev])
        assert ev in o

    def test_mixed_events_split_correctly(self, analyzer):
        person = _det("person",       oid=1)
        helmet = _det("helmet",       oid=2)
        head   = _det("head",         oid=3)
        fall   = _det("fall_detected", oid=4)
        p, h, o = analyzer.split_events([person, helmet, head, fall])
        assert person in p
        assert helmet in h
        assert head in h
        assert fall in o


# ===========================================================================
# check_helmet_compliance
# ===========================================================================


class TestCheckHelmetCompliance:
    def test_person_with_overlapping_helmet_is_wearing(self, analyzer):
        person = _det("person", x=0,  y=0,  w=100, h=200, oid=1)
        helmet = _det("helmet", x=30, y=5,  w=40,  h=40,  oid=2)
        results = analyzer.check_helmet_compliance([], persons=[person], helmets=[helmet])
        assert results[0]["is_wearing"] is True

    def test_person_without_helmet_not_wearing(self, analyzer):
        person = _det("person", x=0, y=0, w=100, h=200, oid=1)
        results = analyzer.check_helmet_compliance([], persons=[person], helmets=[])
        assert results[0]["is_wearing"] is False

    def test_helmet_far_away_not_wearing(self, analyzer):
        person = _det("person", x=0,   y=0,   w=100, h=100, oid=1)
        helmet = _det("helmet", x=900, y=900, w=30,  h=30,  oid=2)
        results = analyzer.check_helmet_compliance([], persons=[person], helmets=[helmet])
        assert results[0]["is_wearing"] is False

    def test_empty_persons_returns_empty(self, analyzer):
        assert analyzer.check_helmet_compliance([], persons=[], helmets=[]) == []

    def test_multiple_persons_all_returned(self, analyzer):
        p1 = _det("person", x=0,   y=0, w=100, h=200, oid=1)
        p2 = _det("person", x=500, y=0, w=100, h=200, oid=2)
        results = analyzer.check_helmet_compliance([], persons=[p1, p2], helmets=[])
        assert len(results) == 2

    def test_result_structure_has_required_keys(self, analyzer):
        person = _det("person", oid=1)
        result = analyzer.check_helmet_compliance([], persons=[person], helmets=[])
        assert "person" in result[0]
        assert "is_wearing" in result[0]


# ===========================================================================
# update_threshold
# ===========================================================================


class TestUpdateThreshold:
    def test_valid_threshold_updated(self, analyzer):
        analyzer.update_threshold(0.7)
        assert analyzer.confidence_threshold == pytest.approx(0.7)

    def test_boundary_zero_accepted(self, analyzer):
        analyzer.update_threshold(0.0)
        assert analyzer.confidence_threshold == pytest.approx(0.0)

    def test_boundary_one_accepted(self, analyzer):
        analyzer.update_threshold(1.0)
        assert analyzer.confidence_threshold == pytest.approx(1.0)

    def test_below_zero_raises_value_error(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.update_threshold(-0.1)

    def test_above_one_raises_value_error(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.update_threshold(1.1)


# ===========================================================================
# _detect_fall_from_keypoints
# ===========================================================================


class TestDetectFallFromKeypoints:
    def test_upright_person_not_fall(self, analyzer):
        """직립 자세 (head 위 / 발목 아래) → 낙상 아님."""
        kpts = _make_kpts()  # 직립 기본값
        keypoints = MockKeypoints(kpts)
        result = analyzer._detect_fall_from_keypoints(
            keypoints, idx=0, bbox_width=30, bbox_height=300
        )
        assert result is False

    def test_horizontal_body_is_fall_method1(self, analyzer):
        """어깨-엉덩이 벡터가 수평(~0°) → 낙상 (방법 1)."""
        # 사람이 옆으로 누운 상태: 어깨 센터(50,150), 엉덩이 센터(150,150) → 수평
        kpts = _make_kpts({
            0:  [100, 100, 0.9],   # nose (신뢰도 충족)
            5:  [50,  130, 0.9],   # left shoulder
            6:  [50,  170, 0.9],   # right shoulder → shoulder_center = (50, 150)
            11: [150, 130, 0.9],   # left hip
            12: [150, 170, 0.9],   # right hip → hip_center = (150, 150)
        })
        keypoints = MockKeypoints(kpts)
        result = analyzer._detect_fall_from_keypoints(
            keypoints, idx=0, bbox_width=200, bbox_height=50
        )
        assert result is True

    def test_knee_above_head_is_fall_method2(self, analyzer):
        """무릎의 y좌표가 코보다 작으면 (화면상 더 위) → 낙상 (방법 2)."""
        kpts = _make_kpts({
            0:  [100, 300, 0.9],   # nose: 아래에 있음 (y=300 큰 값)
            5:  [90,  280, 0.9],
            6:  [110, 280, 0.9],
            11: [90,  250, 0.9],
            12: [110, 250, 0.9],
            13: [90,   20, 0.9],   # left knee: y=20 < nose y=300 → 머리 위
            14: [110,  20, 0.9],
        })
        keypoints = MockKeypoints(kpts)
        result = analyzer._detect_fall_from_keypoints(
            keypoints, idx=0, bbox_width=50, bbox_height=50
        )
        assert result is True

    def test_wide_bbox_with_low_nose_is_fall_method3(self, analyzer):
        """가로 > 세로×2 이고 코가 충분히 낮은 위치 → 낙상 (방법 3)."""
        kpts = _make_kpts({
            0:  [100, 80, 0.9],    # nose y=80 > bbox_height*0.3=30
            5:  [50,  50, 0.9],
            6:  [150, 50, 0.9],
            11: [50,  60, 0.2],    # 엉덩이 신뢰도 낮음 → 방법1 스킵
            12: [150, 60, 0.2],
        })
        keypoints = MockKeypoints(kpts)
        # bbox_width(300) > bbox_height(100) * 2 = 200, nose y(80) > 100*0.3=30
        result = analyzer._detect_fall_from_keypoints(
            keypoints, idx=0, bbox_width=300, bbox_height=100
        )
        assert result is True

    def test_none_keypoints_returns_false(self, analyzer):
        """keypoints가 None이면 _extract_keypoints에서 None 반환 → False."""
        result = analyzer._detect_fall_from_keypoints(
            None, idx=0, bbox_width=50, bbox_height=100
        )
        assert result is False

    def test_low_confidence_keypoints_return_false(self, analyzer):
        """핵심 키포인트(코, 어깨) 신뢰도 < MIN_KEYPOINT_CONFIDENCE(0.2) → False."""
        kpts = _make_kpts({
            0: [100, 50, 0.05],    # nose low conf
            5: [90,  100, 0.05],   # left shoulder low conf
            6: [110, 100, 0.05],   # right shoulder low conf
        })
        keypoints = MockKeypoints(kpts)
        result = analyzer._detect_fall_from_keypoints(
            keypoints, idx=0, bbox_width=50, bbox_height=200
        )
        assert result is False


# ===========================================================================
# _validate_person_keypoints
# ===========================================================================


class TestValidatePersonKeypoints:
    def test_valid_upright_person(self, analyzer):
        keypoints = MockKeypoints(_make_kpts())
        assert analyzer._validate_person_keypoints(keypoints, 0) is True

    def test_all_zero_confidence_returns_false(self, analyzer):
        kpts = np.zeros((17, 3), dtype=float)  # 모두 신뢰도 0
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is False

    def test_nose_only_not_enough(self, analyzer):
        """코만 신뢰도 있으면 2/3 기준 미달 → False (옷 오탐 방지)."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[0] = [100, 50, 0.9]  # nose만 신뢰도 있음
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is False

    def test_nose_and_shoulder_valid(self, analyzer):
        """코 + 어깨 2개 체크 → 유효한 사람."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[0] = [100, 50, 0.9]   # nose
        kpts[5] = [90,  100, 0.9]  # left shoulder
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is True

    def test_shoulder_hip_knee_valid(self, analyzer):
        """어깨 + 엉덩이 + 무릎 조합 → 유효한 사람 (뒤돌아선 자세)."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[5]  = [90,  100, 0.9]  # left shoulder
        kpts[11] = [90,  200, 0.9]  # left hip
        kpts[13] = [90,  280, 0.9]  # left knee
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is True

    def test_shoulder_hip_only_no_face_leg_returns_false(self, analyzer):
        """어깨 + 엉덩이만 있고 코·무릎·발목 없음 → 옷걸이/의류 오탐 → False."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[5]  = [90,  100, 0.9]  # left shoulder
        kpts[11] = [90,  200, 0.9]  # left hip
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is False

    def test_nose_below_shoulder_returns_false(self, analyzer):
        """코가 어깨보다 아래(큰 y)이면 옷걸이/의류 오탐 → False."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[0] = [100, 150, 0.9]   # nose y=150 (아래)
        kpts[5] = [90,  100, 0.9]   # left shoulder y=100 (위)
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is False

    def test_shoulder_below_hip_returns_false(self, analyzer):
        """어깨가 엉덩이보다 아래(큰 y)이면 비정상 자세 → False."""
        kpts = np.zeros((17, 3), dtype=float)
        kpts[5]  = [90, 250, 0.9]   # shoulder y=250 (아래)
        kpts[11] = [90, 100, 0.9]   # hip y=100 (위)
        keypoints = MockKeypoints(kpts)
        assert analyzer._validate_person_keypoints(keypoints, 0) is False

    def test_none_keypoints_returns_true_default(self, analyzer):
        """_extract_keypoints가 None 반환 시 기본값 True 반환."""
        # MockKeypoints를 만들되 data 접근 시 예외 발생하도록
        class BadKeypoints:
            @property
            def data(self):
                raise RuntimeError("oops")

        result = analyzer._validate_person_keypoints(BadKeypoints(), 0)
        # 예외 발생 → except 블록에서 True 반환
        assert result is True
