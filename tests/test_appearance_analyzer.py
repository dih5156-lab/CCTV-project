"""test_appearance_analyzer.py — AppearanceAnalyzer 단위 테스트.

커버리지 대상:
  - _dominant_color: HSV 색상 분류
  - extract_attributes: 상/하체 분리 + 색상 추출
  - match_conditions: 조건 매칭 점수 계산
  - find_matches: 통합 매칭 파이프라인
  - 조건 CRUD: add / remove / set / get_enabled
"""

import numpy as np
import pytest

from src.core.ai._appearance_analyzer import AppearanceAnalyzer


# ── 헬퍼 ─────────────────────────────────────────────────────────────


def _solid_frame(bgr: tuple, h: int = 200, w: int = 100) -> np.ndarray:
    """단색 BGR 프레임을 생성한다."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def _two_tone_frame(
    upper_bgr: tuple,
    lower_bgr: tuple,
    h: int = 200,
    w: int = 100,
) -> np.ndarray:
    """상/하체가 다른 색상인 프레임 (person crop 시뮬레이션)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    mid = int(h * 0.45)  # _UPPER_BODY_RATIO = 0.45
    frame[:mid, :] = upper_bgr
    frame[mid:, :] = lower_bgr
    return frame


# ── 색상 분류 테스트 ─────────────────────────────────────────────────


class TestDominantColor:
    """_dominant_color 메서드 테스트."""

    @pytest.fixture()
    def analyzer(self):
        return AppearanceAnalyzer()

    @pytest.mark.parametrize(
        "bgr, expected",
        [
            ((0, 0, 255), "red"),       # 순수 빨강
            ((0, 0, 0), "black"),       # 순수 검정
            ((255, 255, 255), "white"), # 순수 흰색
            ((255, 0, 0), "blue"),      # 순수 파랑
            ((0, 255, 0), "green"),     # 순수 초록
        ],
    )
    def test_solid_colors(self, analyzer, bgr, expected):
        region = _solid_frame(bgr, h=50, w=50)
        result = analyzer._dominant_color(region)
        assert result == expected

    def test_empty_region(self, analyzer):
        region = np.zeros((0, 0, 3), dtype=np.uint8)
        assert analyzer._dominant_color(region) == "unknown"

    def test_tiny_region(self, analyzer):
        region = np.zeros((3, 3, 3), dtype=np.uint8)
        assert analyzer._dominant_color(region) == "unknown"


# ── 속성 추출 테스트 ─────────────────────────────────────────────────


class TestExtractAttributes:
    """extract_attributes 메서드 테스트."""

    @pytest.fixture()
    def analyzer(self):
        return AppearanceAnalyzer()

    def test_red_top_black_bottom(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(0, 0, 255),   # red
            lower_bgr=(0, 0, 0),     # black
        )
        attrs = analyzer.extract_attributes(frame, 0, 0, 100, 200)
        assert attrs["upper_color"] == "red"
        assert attrs["lower_color"] == "black"

    def test_white_top_blue_bottom(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(255, 255, 255),  # white
            lower_bgr=(255, 0, 0),      # blue
        )
        attrs = analyzer.extract_attributes(frame, 0, 0, 100, 200)
        assert attrs["upper_color"] == "white"
        assert attrs["lower_color"] == "blue"

    def test_tiny_bbox_returns_unknown(self, analyzer):
        frame = _solid_frame((0, 0, 255), h=200, w=100)
        attrs = analyzer.extract_attributes(frame, 0, 0, 10, 10)  # 너무 작음
        assert attrs["upper_color"] == "unknown"
        assert attrs["lower_color"] == "unknown"

    def test_bbox_clipping(self, analyzer):
        """bbox가 프레임을 벗어나도 에러 없이 처리."""
        frame = _solid_frame((0, 0, 255), h=100, w=100)
        attrs = analyzer.extract_attributes(frame, 50, 50, 200, 200)
        # 클리핑 후 50x50 crop → 충분한 크기
        assert attrs["upper_color"] in (
            "red", "orange", "unknown",
        )


# ── 조건 매칭 테스트 ─────────────────────────────────────────────────


class TestMatchConditions:
    """match_conditions 메서드 테스트."""

    @pytest.fixture()
    def analyzer(self):
        return AppearanceAnalyzer()

    def test_perfect_match(self, analyzer):
        attrs = {"upper_color": "red", "lower_color": "black"}
        cond = {"upper_color": "red", "lower_color": "black"}
        assert analyzer.match_conditions(attrs, cond) == 1.0

    def test_partial_match(self, analyzer):
        attrs = {"upper_color": "red", "lower_color": "blue"}
        cond = {"upper_color": "red", "lower_color": "black"}
        assert analyzer.match_conditions(attrs, cond) == 0.5

    def test_no_match(self, analyzer):
        attrs = {"upper_color": "green", "lower_color": "blue"}
        cond = {"upper_color": "red", "lower_color": "black"}
        assert analyzer.match_conditions(attrs, cond) == 0.0

    def test_single_condition_match(self, analyzer):
        attrs = {"upper_color": "red", "lower_color": "blue"}
        cond = {"upper_color": "red"}  # 상의만 조건
        assert analyzer.match_conditions(attrs, cond) == 1.0

    def test_no_conditions(self, analyzer):
        attrs = {"upper_color": "red", "lower_color": "blue"}
        cond = {}
        assert analyzer.match_conditions(attrs, cond) == 0.0


# ── 조건 CRUD 테스트 ─────────────────────────────────────────────────


class TestConditionCRUD:
    """조건 등록/삭제/조회 테스트."""

    @pytest.fixture()
    def analyzer(self):
        return AppearanceAnalyzer()

    def test_add_condition(self, analyzer):
        cond = analyzer.add_condition({
            "name": "의심인물_A",
            "upper_color": "red",
            "lower_color": "black",
        })
        assert cond["id"].startswith("cond_")
        assert cond["name"] == "의심인물_A"
        assert len(analyzer.conditions) == 1

    def test_remove_condition(self, analyzer):
        cond = analyzer.add_condition({"name": "test", "upper_color": "red"})
        result = analyzer.remove_condition(cond["id"])
        assert result is True
        assert len(analyzer.conditions) == 0

    def test_remove_nonexistent(self, analyzer):
        result = analyzer.remove_condition("nonexistent")
        assert result is False

    def test_set_conditions(self, analyzer):
        analyzer.set_conditions([
            {"id": "c1", "name": "A", "upper_color": "red"},
            {"id": "c2", "name": "B", "lower_color": "black"},
        ])
        assert len(analyzer.conditions) == 2

    def test_get_enabled_conditions(self, analyzer):
        analyzer.set_conditions([
            {"id": "c1", "name": "A", "upper_color": "red", "enabled": True},
            {"id": "c2", "name": "B", "lower_color": "black", "enabled": False},
        ])
        enabled = analyzer.get_enabled_conditions()
        assert len(enabled) == 1
        assert enabled[0]["id"] == "c1"

    def test_camera_filter(self, analyzer):
        analyzer.set_conditions([
            {"id": "c1", "name": "A", "upper_color": "red", "cameras": ["cam_01"]},
            {"id": "c2", "name": "B", "lower_color": "black", "cameras": ["cam_02"]},
        ])
        result = analyzer.get_enabled_conditions(camera_id="cam_01")
        assert len(result) == 1
        assert result[0]["id"] == "c1"


# ── 통합 매칭 테스트 ─────────────────────────────────────────────────


class TestFindMatches:
    """find_matches 엔드투엔드 테스트."""

    @pytest.fixture()
    def analyzer(self):
        a = AppearanceAnalyzer()
        a.set_conditions([
            {
                "id": "c1",
                "name": "빨간상의_검은하의",
                "upper_color": "red",
                "lower_color": "black",
                "threshold": 0.8,
            },
        ])
        return a

    def test_matching_person(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(0, 0, 255),   # red
            lower_bgr=(0, 0, 0),     # black
        )
        matches = analyzer.find_matches(frame, 0, 0, 100, 200)
        assert len(matches) == 1
        assert matches[0]["condition_id"] == "c1"
        assert matches[0]["score"] == 1.0

    def test_non_matching_person(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(255, 0, 0),   # blue
            lower_bgr=(255, 255, 255),  # white
        )
        matches = analyzer.find_matches(frame, 0, 0, 100, 200)
        assert len(matches) == 0

    def test_no_conditions(self):
        analyzer = AppearanceAnalyzer()
        frame = _solid_frame((0, 0, 255))
        matches = analyzer.find_matches(frame, 0, 0, 100, 200)
        assert matches == []


# ── 모자(hat_color) 테스트 ───────────────────────────────────────────


class TestHatColor:
    """hat_color 추출 및 조건 매칭 테스트."""

    @pytest.fixture()
    def analyzer(self):
        return AppearanceAnalyzer()

    def test_hat_color_extracted(self, analyzer):
        """머리 영역(상단 15%)의 색상이 hat_color로 추출된다."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        head_h = int(200 * 0.15)  # 30px
        frame[:head_h, :] = (0, 0, 255)    # red hat
        frame[head_h:90, :] = (255, 255, 255)  # white upper
        frame[90:, :] = (0, 0, 0)           # black lower
        attrs = analyzer.extract_attributes(frame, 0, 0, 100, 200)
        assert attrs["hat_color"] == "red"

    def test_hat_color_condition_match(self, analyzer):
        attrs = {"upper_color": "white", "lower_color": "black", "hat_color": "red"}
        cond = {"hat_color": "red"}
        assert analyzer.match_conditions(attrs, cond) == 1.0

    def test_hat_color_condition_no_match(self, analyzer):
        attrs = {"upper_color": "white", "lower_color": "black", "hat_color": "blue"}
        cond = {"hat_color": "red"}
        assert analyzer.match_conditions(attrs, cond) == 0.0

    def test_hat_color_partial_match(self, analyzer):
        """hat_color + upper_color 중 하나만 매칭."""
        attrs = {"upper_color": "red", "lower_color": "black", "hat_color": "blue"}
        cond = {"upper_color": "red", "hat_color": "red"}
        assert analyzer.match_conditions(attrs, cond) == 0.5


# ── 가방 감지 테스트 ─────────────────────────────────────────────────


class TestBagDetection:
    """_detect_bags 및 조건 매칭 테스트."""

    def test_backpack_nearby(self):
        """사람 근처 backpack 감지."""
        nearby = [{"class_name": "backpack", "x": 50, "y": 50, "width": 30, "height": 30}]
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, nearby)
        assert result["has_backpack"] is True
        assert result["has_handbag"] is False

    def test_handbag_nearby(self):
        nearby = [{"class_name": "handbag", "x": 55, "y": 100, "width": 20, "height": 20}]
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, nearby)
        assert result["has_handbag"] is True

    def test_suitcase_nearby(self):
        nearby = [{"class_name": "suitcase", "x": 60, "y": 160, "width": 40, "height": 40}]
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, nearby)
        assert result["has_suitcase"] is True

    def test_bag_too_far(self):
        """사람과 너무 멀리 있는 가방은 감지하지 않음."""
        nearby = [{"class_name": "backpack", "x": 500, "y": 500, "width": 30, "height": 30}]
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, nearby)
        assert result["has_backpack"] is False

    def test_no_nearby_objects(self):
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, None)
        assert result == {"has_backpack": False, "has_handbag": False, "has_suitcase": False}

    def test_multiple_bags(self):
        nearby = [
            {"class_name": "backpack", "x": 55, "y": 55, "width": 20, "height": 20},
            {"class_name": "handbag", "x": 60, "y": 100, "width": 15, "height": 15},
        ]
        result = AppearanceAnalyzer._detect_bags(50, 50, 60, 150, nearby)
        assert result["has_backpack"] is True
        assert result["has_handbag"] is True
        assert result["has_suitcase"] is False

    def test_bag_condition_match(self):
        analyzer = AppearanceAnalyzer()
        attrs = {
            "upper_color": "red", "lower_color": "black",
            "has_backpack": True, "has_handbag": False, "has_suitcase": False,
        }
        cond = {"has_backpack": True}
        assert analyzer.match_conditions(attrs, cond) == 1.0

    def test_bag_condition_false_positive(self):
        analyzer = AppearanceAnalyzer()
        attrs = {
            "upper_color": "red", "lower_color": "black",
            "has_backpack": False, "has_handbag": False, "has_suitcase": False,
        }
        cond = {"has_backpack": True}
        assert analyzer.match_conditions(attrs, cond) == 0.0


# ── 가방 + 색상 통합 매칭 테스트 ─────────────────────────────────────


class TestCombinedConditions:
    """색상 + 소지품 복합 조건 통합 테스트."""

    @pytest.fixture()
    def analyzer(self):
        a = AppearanceAnalyzer()
        a.set_conditions([{
            "id": "c_combo",
            "name": "빨간상의_백팩",
            "upper_color": "red",
            "has_backpack": True,
            "threshold": 0.8,
        }])
        return a

    def test_color_and_bag_match(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(0, 0, 255),   # red
            lower_bgr=(0, 0, 0),     # black
        )
        nearby = [{"class_name": "backpack", "x": 5, "y": 5, "width": 20, "height": 20}]
        matches = analyzer.find_matches(frame, 0, 0, 100, 200, nearby_objects=nearby)
        assert len(matches) == 1
        assert matches[0]["condition_id"] == "c_combo"

    def test_color_match_but_no_bag(self, analyzer):
        frame = _two_tone_frame(
            upper_bgr=(0, 0, 255),   # red
            lower_bgr=(0, 0, 0),     # black
        )
        matches = analyzer.find_matches(frame, 0, 0, 100, 200)
        assert len(matches) == 0  # has_backpack=False → 0.5 < 0.8 threshold

    def test_add_condition_with_bag_fields(self):
        analyzer = AppearanceAnalyzer()
        cond = analyzer.add_condition({
            "name": "백팩소지자",
            "has_backpack": True,
            "upper_color": "red",
        })
        assert cond["has_backpack"] is True
        assert cond["has_handbag"] is None
        assert cond["has_suitcase"] is None
        assert cond["hat_color"] is None
