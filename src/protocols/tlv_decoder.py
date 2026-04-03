"""Go TLV decoder HTTP client."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import requests

logger = logging.getLogger(__name__)


class GoTLVDecoderClient:
    """Go 기반 TLV 디코더 서비스와 통신하는 얇은 HTTP 클라이언트."""

    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = max(0.1, float(timeout))

    @property
    def decode_url(self) -> str:
        if self.base_url.endswith("/decode-tlv"):
            return self.base_url
        return f"{self.base_url}/decode-tlv"

    def decode_uplink(self, uplink_message: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        rx_metadata = uplink_message.get("rx_metadata") or []
        rx0 = rx_metadata[0] if isinstance(rx_metadata, list) and rx_metadata else {}
        request_body = {
            "payload": uplink_message.get("payload"),
            "app_eui": uplink_message.get("app_eui"),
            "dev_eui": uplink_message.get("dev_eui"),
            "f_port": uplink_message.get("f_port"),
            "f_cnt_up": uplink_message.get("f_cnt_up"),
            "channel": rx0.get("channel"),
            "frequency": rx0.get("frequency"),
            "received_at": rx0.get("time"),
        }

        try:
            response = requests.post(
                self.decode_url,
                json=request_body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                logger.warning("TLV decoder 응답 형식이 예상과 다릅니다: %r", payload)
                return None
            return payload
        except requests.RequestException as exc:
            logger.error("TLV decoder 요청 실패: %s", exc)
            return None
        except ValueError as exc:
            logger.error("TLV decoder JSON 파싱 실패: %s", exc)
            return None
