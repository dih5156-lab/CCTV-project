from dataclasses import dataclass

from src.services.event_intake_processor import EventIntakeProcessor


@dataclass
class FakeSiteRegistry:
    mode: object = "auto"
    settings: dict | None = None

    def resolve_mode(self, camera_id):
        return self.mode, "site-1"

    def resolve_action_settings(self, camera_id):
        return self.settings or {}


def test_processor_normalizes_event_and_applies_site_messages():
    processor = EventIntakeProcessor(
        site_registry=FakeSiteRegistry(
            settings={"display_message": "현장주의", "tts_message": "주의하세요"}
        ),
        manual_mode="manual",
    )

    result = processor.prepare({"camera_id": "cam-1", "type": "Fall"})

    assert result.camera_id == "cam-1"
    assert result.event_type == "fall"
    assert result.payload["event"]["display_message"] == "현장주의"
    assert result.payload["event"]["tts_message"] == "주의하세요"
    assert result.is_filtered is False


def test_processor_marks_low_confidence_event_as_filtered():
    processor = EventIntakeProcessor(
        site_registry=FakeSiteRegistry(settings={"confidence_threshold": 0.8}),
        manual_mode="manual",
    )

    result = processor.prepare(
        {"camera_id": "cam-1", "type": "fall", "confidence": 0.5}
    )

    assert result.is_filtered is True
