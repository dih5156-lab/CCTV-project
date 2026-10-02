"""Decode retained MQTT uplinks without device or database side effects."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UplinkDecoder:
    def decode(self, topic: str, payload: bytes) -> dict:
        if topic.split("/")[-1:] != ["up"]:
            raise ValueError("not an uplink topic")
        message_data = json.loads(payload)
        if not isinstance(message_data, dict):
            raise ValueError("JSON payload must be an object")
        # JSON parsers accept NaN/Infinity by default, including nested values.
        json.dumps(message_data, allow_nan=False)
        for key in ("uplink_message", "end_device_ids"):
            if key in message_data and not isinstance(message_data[key], dict):
                raise ValueError(f"{key} must be an object")
        end_ids = message_data.get("end_device_ids", {})
        if "application_ids" in end_ids and not isinstance(
            end_ids["application_ids"], dict
        ):
            raise ValueError("application_ids must be an object")
        for identity in (message_data, end_ids, end_ids.get("application_ids", {})):
            for key in ("app_eui", "application_id", "dev_eui", "device_id"):
                if identity.get(key) is not None and not isinstance(identity[key], str):
                    raise ValueError(f"{key} must be a string")
        app_id, dev_eui = self._extract_app_and_device(topic, message_data)
        sensor_payload = self._extract_payload(message_data)
        if not isinstance(app_id, str) or not app_id or not dev_eui:
            raise ValueError("missing appID or devEUI")
        if not isinstance(sensor_payload, str) or not sensor_payload:
            raise ValueError("missing or invalid sensor payload")
        uplink = message_data.get("uplink_message", {})
        metadata = message_data.get("rx_metadata") or uplink.get("rx_metadata", [])
        if not isinstance(metadata, list) or any(
            not isinstance(item, dict) for item in metadata
        ):
            raise ValueError("rx_metadata must be an array of objects")
        primary = metadata[0] if metadata else {}
        if "gateway_info" in primary and not isinstance(primary["gateway_info"], dict):
            raise ValueError("gateway_info must be an object")
        return dict(
            app_id=app_id,
            dev_eui=dev_eui,
            payload=sensor_payload,
            channel=self._parse_integer(primary.get("channel", 0)),
            frequency=self._parse_integer(primary.get("frequency", 0)),
            received_at=self._parse_received_at(primary.get("time", 0)),
            uplink_metadata={
                "message_id": message_data.get("message_id", ""),
                "f_port": message_data.get("f_port", 0),
                "f_cnt_up": message_data.get("f_cnt_up", 0),
                "is_confirmed": bool(message_data.get("is_confirmed", False)),
                "is_ack": bool(message_data.get("is_ack", False)),
                "radio": self._extract_radio_metadata(primary),
            },
        )

    def _extract_app_and_device(
        self, topic: str, message_data: dict
    ) -> tuple[str, str]:
        app_id = message_data.get("app_eui") or message_data.get("application_id") or ""
        dev_eui = message_data.get("dev_eui") or message_data.get("device_id") or ""

        end_device_ids = message_data.get("end_device_ids") or {}
        app_ids = end_device_ids.get("application_ids") or {}
        app_id = app_id or app_ids.get("application_id", "")
        dev_eui = (
            dev_eui
            or end_device_ids.get("dev_eui")
            or end_device_ids.get("device_id", "")
        )

        if app_id and dev_eui:
            return app_id, self._normalize_dev_eui(dev_eui)

        try:
            parsed_app_id, parsed_dev_eui = self._parse_topic(topic)
        except ValueError:
            return app_id, self._normalize_dev_eui(dev_eui)
        return app_id or parsed_app_id, self._normalize_dev_eui(
            dev_eui or parsed_dev_eui
        )

    def _extract_payload(self, message_data: dict) -> str:
        uplink = message_data.get("uplink_message") or {}
        return (
            message_data.get("payload")
            or message_data.get("frm_payload")
            or uplink.get("frm_payload")
            or ""
        )

    def _parse_received_at(self, value) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if not value:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Failed to parse received_at time: %s", value)
            return 0
        return int(dt.timestamp() * 1000)

    def _parse_integer(self, value) -> int:
        """구현사별 숫자/문자열 metadata를 DB 정수 컬럼에 안전하게 맞춘다."""
        try:
            return int(float(value or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _extract_radio_metadata(self, metadata: dict) -> dict:
        """내부 이벤트에 필요한 첫 게이트웨이 무선 품질만 가볍게 보존한다."""
        if not metadata:
            return {}
        gateway_info = metadata.get("gateway_info") or {}
        return {
            "gateway_id": gateway_info.get("gw_id", ""),
            "data_rate": metadata.get("data_rate", ""),
            "channel": metadata.get("channel", 0),
            "frequency": metadata.get("frequency", 0),
            "rssi": metadata.get("rssi", 0),
            "snr": metadata.get("snr", 0),
        }

    def _normalize_dev_eui(self, dev_eui: str) -> str:
        value = str(dev_eui or "").strip()
        if value.lower().startswith("eui-"):
            value = value[4:]
        return value

    def _parse_topic(self, topic: str):
        """
        토픽에서 appID, devEUI 추출
        Go: func (m *Manager) parseTopic(topic string) (appID, devEUI string, err error)

        토픽 형식:
          - {appID}/{devEUI}/up
          - v3/{appID}/devices/eui-{devEUI}/up
        """
        parts = topic.split("/")
        if len(parts) >= 5 and parts[0] == "v3" and parts[2] == "devices":
            return parts[1], parts[3]
        if len(parts) < 3:
            raise ValueError(f"invalid topic format: {topic}")
        return parts[0], parts[1]
