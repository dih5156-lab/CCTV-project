from src.core.events import EventType
from src.utils.visualizer import _parse_event_data


def test_parse_event_data_maps_zone_object_alias() -> None:
    parsed = _parse_event_data(
        {
            "type": "zone_object_detected",
            "confidence": 0.9,
            "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
        }
    )

    assert parsed is not None
    assert parsed["type_str"] == EventType.ZONE_OBJECT.value


def test_parse_event_data_skips_unknown_event_type() -> None:
    parsed = _parse_event_data(
        {
            "type": "totally_unknown_type",
            "confidence": 0.4,
            "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
        }
    )

    assert parsed is None


def test_parse_event_data_accepts_mixed_case_event_type() -> None:
    parsed = _parse_event_data(
        {
            "type": "HeLmEt",
            "confidence": 0.8,
            "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
        }
    )

    assert parsed is not None
    assert parsed["type_str"] == EventType.HELMET.value
