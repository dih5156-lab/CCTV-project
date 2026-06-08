"""센서 디바이스 도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from src.time_utils import coerce_timestamp_seconds


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class SensorReading:
    """TLV 디코드 결과를 Python 도메인 모델로 정규화한 센서 측정값."""

    device_id: str
    app_eui: str
    dev_eui: str
    table_name: str
    telemetry: Dict[str, Any]
    received_at: float
    source: str = "lora_tlv"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_decoded(
        cls,
        uplink_message: Mapping[str, Any],
        decoded_payload: Mapping[str, Any],
    ) -> "SensorReading":
        rx_metadata = uplink_message.get("rx_metadata") or []
        rx0 = rx_metadata[0] if isinstance(rx_metadata, list) and rx_metadata else {}

        if "data" in decoded_payload:
            raw_data = _as_mapping(decoded_payload.get("data"))
        else:
            raw_data = _as_mapping(decoded_payload)
        telemetry = {
            key: value
            for key, value in raw_data.items()
            if key != "tableName"
        }
        table_name = str(
            decoded_payload.get("tableName")
            or raw_data.get("tableName")
            or "unknown"
        )
        dev_eui = str(uplink_message.get("dev_eui") or "").lower()
        device_id = str(uplink_message.get("device_id") or dev_eui or "unknown")
        received_at = coerce_timestamp_seconds(
            rx0.get("time") or uplink_message.get("received_at") or uplink_message.get("timestamp")
        )

        return cls(
            device_id=device_id,
            app_eui=str(uplink_message.get("app_eui") or "").lower(),
            dev_eui=dev_eui,
            table_name=table_name,
            telemetry=telemetry,
            received_at=received_at,
            metadata={
                "f_port": uplink_message.get("f_port"),
                "f_cnt_up": uplink_message.get("f_cnt_up"),
                "channel": rx0.get("channel"),
                "frequency": rx0.get("frequency"),
                "rssi": rx0.get("rssi"),
                "snr": rx0.get("snr"),
            },
        )
