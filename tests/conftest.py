"""
conftest.py — 공통 pytest 픽스처
"""
import time
import pytest
from src.core.events import DetectionEvent, EventType


# ---------------------------------------------------------------------------
# DetectionEvent 팩토리 헬퍼
# ---------------------------------------------------------------------------


def make_event(
    event_type: str = "helmet",
    x: int = 10,
    y: int = 20,
    width: int = 50,
    height: int = 60,
    confidence: float = 0.9,
    object_id: int = 1,
    timestamp: float | None = None,
) -> DetectionEvent:
    """테스트용 DetectionEvent 팩토리."""
    return DetectionEvent(
        event_type=EventType.from_string(event_type),
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
        timestamp=timestamp if timestamp is not None else time.time(),
        object_id=object_id,
    )


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def helmet_event() -> DetectionEvent:
    return make_event("helmet", object_id=1)


@pytest.fixture
def head_event() -> DetectionEvent:
    return make_event("head", object_id=2)


@pytest.fixture
def fall_event() -> DetectionEvent:
    return make_event("fall_detected", x=100, y=100, object_id=3)


@pytest.fixture
def sample_bbox() -> dict:
    return {"x": 10, "y": 20, "width": 50, "height": 60}


@pytest.fixture
def overlapping_bbox() -> dict:
    """sample_bbox 와 크게 겹치는 박스."""
    return {"x": 20, "y": 30, "width": 50, "height": 60}


@pytest.fixture
def non_overlapping_bbox() -> dict:
    """sample_bbox 와 전혀 겹치지 않는 박스."""
    return {"x": 200, "y": 200, "width": 50, "height": 60}
