#!/usr/bin/env python3
"""AI·센서 이벤트 JSON 계약 검증 도구.

기본 필드, confidence 범위, Canonical Event 보강 결과를 검사한다.
이벤트별 상세 필드가 없으면 호환성을 위해 warning으로 보고한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# `python scripts/validate_event_contracts.py` 실행 시에도 프로젝트 패키지를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical_event import canonicalize_event_payload, get_payload_event_type  # noqa: E402


EVENT_TYPES = {
    "person",
    "helmet",
    "head",
    "fall_detected",
    "not_fall",
    "face_recognized",
    "face_unknown",
    "danger_zone",
    "intrusion",
    "zone_object",
    "crowd_warning",
    "appearance_match",
    "unsafe_behavior",
    "tilt_alert",
    "temperature_alert",
    "vibration_alert",
    "sensor_fault",
    "sensor_temperature",
    "smoke_test_alert",
}

DETAIL_FIELDS = {
    "fall_detected": ("fall_score", "fall_direction", "fall_type"),
    "face_recognized": ("face_name", "face_score"),
    "face_unknown": ("face_score", "recognizer"),
    "danger_zone": ("zone_id", "zone_event"),
    "intrusion": ("zone_id", "zone_event"),
    "zone_object": ("zone_id", "mode"),
    "crowd_warning": ("person_count", "threshold"),
    "appearance_match": ("upper_color", "lower_color", "has_helmet"),
    "unsafe_behavior": ("reason",),
}


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("metadata")
    if isinstance(value, Mapping):
        return value
    raw = payload.get("raw")
    if isinstance(raw, Mapping) and isinstance(raw.get("metadata"), Mapping):
        return raw["metadata"]
    return {}


def _event_value(payload: Mapping[str, Any], key: str) -> Any:
    event = payload.get("event")
    if isinstance(event, Mapping) and event.get(key) is not None:
        return event[key]
    return payload.get(key)


def validate_payload(payload: Mapping[str, Any], *, index: int = 0) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, Mapping):
        return {"index": index, "valid": False, "errors": ["payload must be an object"], "warnings": []}

    event_type = get_payload_event_type(payload).strip().lower()
    device = payload.get("device") if isinstance(payload.get("device"), Mapping) else {}
    camera_id = (
        payload.get("camera_id")
        or payload.get("device_id")
        or payload.get("dev_eui")
        or device.get("camera_id")
        or device.get("device_id")
    )
    if not camera_id:
        errors.append("camera_id is required")
    if event_type == "unknown":
        errors.append("event type is required (type or event.event_type)")
    elif event_type not in EVENT_TYPES:
        warnings.append(f"unknown event type: {event_type}")

    confidence = _event_value(payload, "confidence")
    if confidence is not None:
        try:
            numeric_confidence = float(confidence)
            if not 0.0 <= numeric_confidence <= 1.0:
                errors.append("confidence must be between 0 and 1")
        except (TypeError, ValueError):
            errors.append("confidence must be numeric")

    if not any(payload.get(key) is not None for key in ("timestamp", "occurred_at", "received_at", "queued_at")):
        warnings.append("timestamp/occurred_at is missing; canonicalization will use current time")

    if "metadata" in payload and not isinstance(payload["metadata"], Mapping):
        errors.append("metadata must be an object when provided")

    metadata = _metadata(payload)
    for field in DETAIL_FIELDS.get(event_type, ()):
        if metadata.get(field) is None and payload.get(field) is None:
            warnings.append(f"{event_type} detail field is missing: {field}")

    canonical: dict[str, Any] | None = None
    try:
        canonical = canonicalize_event_payload(payload)
        for key in ("schema_version", "message_type", "occurred_at", "device", "event", "raw", "event_id"):
            if key not in canonical:
                errors.append(f"canonical field is missing: {key}")
        if canonical.get("type") != event_type:
            errors.append("legacy type was not preserved during canonicalization")
        if not isinstance(canonical.get("event"), Mapping):
            errors.append("canonical event must be an object")
    except Exception as exc:  # pragma: no cover - defensive boundary for CLI input
        errors.append(f"canonicalization failed: {exc}")

    return {
        "index": index,
        "event_type": event_type,
        "camera_id": str(camera_id) if camera_id else None,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "canonical": canonical,
    }


def _sample_payloads() -> list[dict[str, Any]]:
    base = {"camera_id": "contract-test-camera", "confidence": 0.95, "timestamp": 1770000000.0}
    return [
        {**base, "type": "person", "object_id": 1},
        {**base, "type": "helmet", "object_id": 1},
        {**base, "type": "head", "object_id": 1},
        {**base, "type": "fall_detected", "metadata": {"fall_score": 4.2, "fall_direction": "back", "fall_type": "뒤로 넘어짐"}},
        {**base, "type": "face_recognized", "metadata": {"face_name": "sample", "face_score": 0.95}},
        {**base, "type": "face_unknown", "metadata": {"face_score": 0.72, "recognizer": "test"}},
        {**base, "type": "danger_zone", "metadata": {"zone_id": "zone-a", "zone_event": "zone_entered"}},
        {**base, "type": "intrusion", "metadata": {"zone_id": "zone-a", "zone_event": "zone_entered"}},
        {**base, "type": "zone_object", "metadata": {"zone_id": "zone-a", "mode": "object_watch"}},
        {**base, "type": "crowd_warning", "metadata": {"person_count": 10, "threshold": 8}},
        {**base, "type": "appearance_match", "metadata": {"upper_color": "blue", "lower_color": "black", "has_helmet": True}},
        {**base, "type": "unsafe_behavior", "metadata": {"reason": "restricted_motion"}},
        {"device_id": "sensor-01", "type": "tilt_alert", "timestamp": 1770000000.0, "metadata": {"telemetry": {"angle_x_deg": 52.0}}},
    ]


def _load_payloads(path: Path) -> list[Mapping[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        return [data]
    if isinstance(data, list) and all(isinstance(item, Mapping) for item in data):
        return data
    raise ValueError("input JSON must be an object or an array of objects")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CCTV 이벤트 JSON 계약 검증")
    parser.add_argument("--file", type=Path, help="검증할 JSON 파일(객체 또는 객체 배열)")
    parser.add_argument("--samples", action="store_true", help="지원 이벤트 샘플 전체 검증")
    args = parser.parse_args(argv)

    if bool(args.file) == bool(args.samples):
        parser.error("--file 또는 --samples 중 하나만 지정해야 합니다")

    try:
        payloads = _load_payloads(args.file) if args.file else _sample_payloads()
        results = [validate_payload(payload, index=index) for index, payload in enumerate(payloads)]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 2

    valid = all(result["valid"] for result in results)
    output = {"valid": valid, "total": len(results), "failed": sum(not item["valid"] for item in results), "results": results}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
