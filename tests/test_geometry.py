"""
test_geometry.py — geometry 유틸리티 단위 테스트
"""
import pytest
from src.core.events import DetectionEvent, EventType
from src.utils.geometry import (
    calculate_iou,
    calculate_bbox_iou,
    boxes_overlap,
    get_center,
    point_in_bbox,
    get_head_bbox,
    calculate_overlap_ratio,
    is_helmet_worn,
)
import time


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _evt(x, y, w, h, eid=1, etype="helmet") -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.from_string(etype),
        x=x, y=y, width=w, height=h,
        confidence=0.9,
        timestamp=time.time(),
        object_id=eid,
    )


def _bbox(x, y, w, h) -> dict:
    return {"x": x, "y": y, "width": w, "height": h}


# ---------------------------------------------------------------------------
# calculate_bbox_iou
# ---------------------------------------------------------------------------


class TestCalculateBboxIou:
    def test_identical_boxes(self):
        b = _bbox(0, 0, 100, 100)
        assert calculate_bbox_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert calculate_bbox_iou(_bbox(0, 0, 10, 10), _bbox(20, 20, 10, 10)) == 0.0

    def test_partial_overlap(self):
        # 두 박스가 절반씩 겹침 → IoU = 50*100 / (100*100 + 50*100 - 50*100)
        iou = calculate_bbox_iou(_bbox(0, 0, 100, 100), _bbox(50, 0, 100, 100))
        assert 0.0 < iou < 1.0

    def test_one_inside_other(self):
        big = _bbox(0, 0, 100, 100)
        small = _bbox(25, 25, 50, 50)
        iou = calculate_bbox_iou(big, small)
        # intersection == small 넓이(2500), union == 큰박스(10000)
        assert iou == pytest.approx(2500 / 10000)

    def test_type_error_non_dict(self):
        with pytest.raises(TypeError):
            calculate_bbox_iou("not_a_dict", _bbox(0, 0, 10, 10))  # type: ignore

    def test_zero_area_box(self):
        assert calculate_bbox_iou(_bbox(0, 0, 0, 0), _bbox(0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# calculate_iou  (DetectionEvent 버전)
# ---------------------------------------------------------------------------


class TestCalculateIou:
    def test_same_event_iou_one(self):
        e = _evt(0, 0, 50, 50)
        assert calculate_iou(e, e) == pytest.approx(1.0)

    def test_no_overlap_events(self):
        assert calculate_iou(_evt(0, 0, 10, 10), _evt(100, 100, 10, 10)) == 0.0

    def test_type_error_non_event(self):
        with pytest.raises(TypeError):
            calculate_iou(_evt(0, 0, 10, 10), {"x": 0})  # type: ignore


# ---------------------------------------------------------------------------
# boxes_overlap
# ---------------------------------------------------------------------------


class TestBoxesOverlap:
    def test_overlapping(self):
        e1 = _evt(0, 0, 100, 100)
        e2 = _evt(10, 10, 100, 100)
        assert boxes_overlap(e1, e2, threshold=0.05) is True

    def test_not_overlapping(self):
        e1 = _evt(0, 0, 10, 10)
        e2 = _evt(100, 100, 10, 10)
        assert boxes_overlap(e1, e2) is False

    def test_invalid_threshold_raises(self):
        e = _evt(0, 0, 10, 10)
        with pytest.raises(ValueError):
            boxes_overlap(e, e, threshold=1.5)


# ---------------------------------------------------------------------------
# get_center
# ---------------------------------------------------------------------------


class TestGetCenter:
    def test_basic(self):
        cx, cy = get_center(_bbox(0, 0, 100, 100))
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(50.0)

    def test_offset(self):
        cx, cy = get_center(_bbox(10, 20, 40, 60))
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_type_error(self):
        with pytest.raises(TypeError):
            get_center("not_a_dict")  # type: ignore


# ---------------------------------------------------------------------------
# point_in_bbox
# ---------------------------------------------------------------------------


class TestPointInBbox:
    def test_inside(self):
        assert point_in_bbox(50, 50, _bbox(0, 0, 100, 100)) is True

    def test_on_edge(self):
        assert point_in_bbox(0, 0, _bbox(0, 0, 100, 100)) is True

    def test_outside(self):
        assert point_in_bbox(200, 200, _bbox(0, 0, 100, 100)) is False


# ---------------------------------------------------------------------------
# get_head_bbox
# ---------------------------------------------------------------------------


class TestGetHeadBbox:
    def test_height_reduced(self):
        hb = get_head_bbox(_bbox(0, 0, 100, 100), head_ratio=0.5)
        assert hb["height"] == pytest.approx(50.0)
        assert hb["x"] == 0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            get_head_bbox(_bbox(0, 0, 100, 100), head_ratio=0.0)


# ---------------------------------------------------------------------------
# is_helmet_worn
# ---------------------------------------------------------------------------


class TestIsHelmetWorn:
    def test_helmet_over_head(self):
        person = _bbox(0, 0, 100, 200)
        helmet = _bbox(5, 5, 40, 40)   # 머리 영역 안에 위치
        assert is_helmet_worn(person, [helmet]) is True

    def test_no_helmet(self):
        person = _bbox(0, 0, 100, 200)
        helmet_far = _bbox(500, 500, 50, 50)
        assert is_helmet_worn(person, [helmet_far]) is False

    def test_empty_helmets(self):
        assert is_helmet_worn(_bbox(0, 0, 100, 200), []) is False
