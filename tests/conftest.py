"""
conftest.py — 공통 pytest 픽스처
"""
import json
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

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


def http_request(method: str, url: str, body: dict | None = None):
    """urllib 래퍼 — (status_code, dict) 반환."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


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


@pytest.fixture
def tmp_path() -> Path:
    """Windows/sandbox 환경에서 직접 생성한 워크스페이스 임시 디렉터리를 제공한다."""
    base = Path("tmp_test_dirs").resolve()
    base.mkdir(exist_ok=True)
    temp_dir = base / f"cctv_test_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
