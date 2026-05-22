"""
test_events.py — DetectionEvent / EventType 단위 테스트
"""
import time
import pytest
from src.core.events import DetectionEvent, EventType, severity_for_event_type


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class TestEventType:
    def test_from_string_known(self):
        assert EventType.from_string("helmet") == EventType.HELMET
        assert EventType.from_string("fall_detected") == EventType.FALL_DETECTED
        assert EventType.from_string("person") == EventType.PERSON

    def test_from_string_uppercase(self):
        assert EventType.from_string("HELMET") == EventType.HELMET
        assert EventType.from_string("HEAD") == EventType.HEAD

    def test_from_string_unknown_returns_other(self):
        assert EventType.from_string("xxxx_unknown") == EventType.OTHER

    def test_from_string_empty_returns_other(self):
        assert EventType.from_string("") == EventType.OTHER

    def test_all_values_roundtrip(self):
        """모든 enum 값이 from_string(value.value) 로 복원되는지 확인."""
        for evt in EventType:
            assert EventType.from_string(evt.value) == evt


class TestEventSeverity:
    def test_critical_event_types(self):
        assert severity_for_event_type(EventType.FALL_DETECTED) == "critical"
        assert severity_for_event_type(EventType.DANGER_ZONE) == "critical"
        assert severity_for_event_type(EventType.UNSAFE_BEHAVIOR) == "critical"

    def test_normal_event_type(self):
        assert severity_for_event_type(EventType.HELMET) == "normal"


# ---------------------------------------------------------------------------
# DetectionEvent
# ---------------------------------------------------------------------------


class TestDetectionEvent:
    def _make(self, **kwargs) -> DetectionEvent:
        defaults = dict(
            event_type=EventType.HELMET,
            x=10, y=20, width=50, height=60,
            confidence=0.85,
            timestamp=time.time(),
            object_id=42,
        )
        defaults.update(kwargs)
        return DetectionEvent(**defaults)

    def test_to_dict_structure(self):
        evt = self._make()
        d = evt.to_dict()
        assert d["type"] == EventType.HELMET.value
        assert d["bbox"] == {"x": 10, "y": 20, "width": 50, "height": 60}
        assert d["confidence"] == pytest.approx(0.85)
        assert d["object_id"] == 42

    def test_to_dict_keeps_legacy_shape_without_canonical_fields(self):
        evt = self._make()
        d = evt.to_dict()
        assert "schema_version" not in d
        assert "event" not in d

    def test_to_dict_optional_none(self):
        evt = self._make(object_id=None, class_idx=None, keypoints=None)
        d = evt.to_dict()
        assert d["object_id"] is None
        assert d["keypoints"] is None

    def test_to_dict_keypoints_preserved(self):
        kp = [[1.0, 2.0, 0.9]]
        evt = self._make(keypoints=kp)
        assert evt.to_dict()["keypoints"] == kp

    def test_repr_contains_type_and_conf(self):
        evt = self._make()
        r = repr(evt)
        assert "helmet" in r
        assert "0.85" in r

    def test_confidence_zero(self):
        evt = self._make(confidence=0.0)
        assert evt.to_dict()["confidence"] == 0.0

    def test_large_bbox_values(self):
        """음수 좌표·큰 값도 저장 가능해야 함."""
        evt = self._make(x=-5, y=-10, width=10000, height=9000)
        d = evt.to_dict()["bbox"]
        assert d["x"] == -5
        assert d["width"] == 10000
