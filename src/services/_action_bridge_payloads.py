"""ActionBridge 입력 페이로드 정규화 헬퍼."""

from __future__ import annotations

from typing import Dict


def normalize_sensor_payload(topic: str, payload: Dict) -> Dict:
    """센서 경보 페이로드를 Action Bridge 공통 형식으로 변환한다.

    Kuiper 출력: {"dev_eui":..., "device_id":"factory-24", "type":"tilt_alert", ...}
    공통 형식 : {"camera_id":"factory-24", "type":"tilt_alert", "source":"sensor", ...}
    """
    if not isinstance(payload, dict):
        return {
            "type": f"{topic.split('/')[-1]}_alert",
            "source": "sensor",
            "camera_id": "unknown",
        }
    normalized = dict(payload)
    if "camera_id" not in normalized and "device_id" in normalized:
        normalized["camera_id"] = normalized["device_id"]
    if "type" not in normalized:
        sensor_kind = topic.split("/")[-1]
        normalized["type"] = f"{sensor_kind}_alert"
    normalized.setdefault("source", "sensor")
    return normalized
