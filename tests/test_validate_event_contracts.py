from scripts.validate_event_contracts import _sample_payloads, validate_payload


def test_supported_event_samples_pass_contract_validation():
    results = [validate_payload(payload, index=index) for index, payload in enumerate(_sample_payloads())]

    assert all(result["valid"] for result in results)
    assert {result["event_type"] for result in results} >= {
        "person",
        "helmet",
        "fall_detected",
        "face_recognized",
        "danger_zone",
        "appearance_match",
        "tilt_alert",
    }


def test_invalid_confidence_is_error():
    result = validate_payload(
        {"camera_id": "cam-1", "type": "person", "confidence": 1.2, "timestamp": 1.0}
    )

    assert result["valid"] is False
    assert "confidence must be between 0 and 1" in result["errors"]


def test_optional_detail_missing_is_warning_not_error():
    result = validate_payload(
        {"camera_id": "cam-1", "type": "fall_detected", "confidence": 0.8, "timestamp": 1.0}
    )

    assert result["valid"] is True
    assert any("fall_direction" in warning for warning in result["warnings"])


def test_canonical_fields_and_legacy_fields_are_both_present():
    result = validate_payload(
        {"camera_id": "cam-1", "type": "helmet", "confidence": 0.9, "timestamp": 1.0}
    )

    canonical = result["canonical"]
    assert canonical["type"] == "helmet"
    assert canonical["camera_id"] == "cam-1"
    assert canonical["event"]["event_type"] == "helmet"
    assert canonical["schema_version"] == "1.0"
