"""경광등/사이렌 디바이스 컨트롤러.

신규 표준 모듈 경로다. 기존 ``src.devices.sensor`` 는
하위 호환용 shim 으로 유지한다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """센서/경광등 연결 설정."""

    host: str = ""
    port: int = 80
    username: str = ""
    password: str = ""
    auto_stop_seconds: float = 10.0
    connect_timeout: int = 3
    read_timeout: int = 7

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.username and self.password)


class SirenNetworkError(Exception):
    """경광등 디바이스 네트워크 오류."""


class _SirenClient:
    """InterM 경광등 디바이스 HTTP 클라이언트."""

    def __init__(self, cfg: SensorConfig):
        self._cfg = cfg
        self._base = f"http://{cfg.host}/interm-api"
        self._auth = HTTPDigestAuth(cfg.username, cfg.password)
        self._timeout = (cfg.connect_timeout, cfg.read_timeout)

    def _post(self, path: str, body: Dict) -> Dict:
        url = f"{self._base}{path}"
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=body,
                auth=self._auth,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            logger.warning("[Siren] %s %s 네트워크 오류 (경광등 오프라인?): %s", "POST", path, exc)
            raise SirenNetworkError(str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            logger.error("[Siren] POST %s 오류: %s", path, exc)
            raise

    def trigger(self) -> Dict:
        """경광등을 켠다."""
        body = {"Control": True, "Run": True}
        return self._post("/Warnbell/Control", body)

    def stop(self) -> Dict:
        """경광등을 끈다."""
        body = {"Control": True, "Run": False}
        return self._post("/Warnbell/Control", body)


class SirenDevice:
    """경광등/사이렌 디바이스 컨트롤러."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self._client: Optional[_SirenClient] = None
        self._stop_timer: Optional[threading.Timer] = None

    def _get_client(self) -> Optional[_SirenClient]:
        if not self.config.is_configured:
            logger.warning("[Siren] host/username/password 미설정 - 경광등 비활성화")
            return None
        if self._client is None:
            self._client = _SirenClient(self.config)
        return self._client

    def trigger(self, event_type: str = "", camera_id: str = "") -> bool:
        """경광등을 켠다. auto_stop_seconds 설정 시 자동 정지 타이머 등록."""
        client = self._get_client()
        if client is None:
            return False

        if self._stop_timer and self._stop_timer.is_alive():
            self._stop_timer.cancel()

        try:
            client.trigger()
            logger.info("[Siren] 경광등 ON (camera=%s, type=%s)", camera_id, event_type)

            if self.config.auto_stop_seconds > 0:
                self._stop_timer = threading.Timer(
                    self.config.auto_stop_seconds, self.stop
                )
                self._stop_timer.daemon = True
                self._stop_timer.start()
                logger.debug(
                    "[Siren] 자동 정지 예약: %.1f초 후", self.config.auto_stop_seconds
                )
            return True

        except SirenNetworkError:
            logger.warning(
                "[Siren] 경광등 오프라인 (%s:%s) - trigger 건너뜀",
                self.config.host, self.config.port,
            )
            return False
        except Exception as exc:
            logger.error("[Siren] trigger() 오류: %s", exc)
            return False

    def stop(self) -> bool:
        """경광등을 끈다."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.stop()
            logger.info("[Siren] 경광등 OFF")
            return True
        except Exception as exc:
            logger.error("[Siren] stop() 오류: %s", exc)
            return False


__all__ = ["SensorConfig", "SirenDevice", "SirenNetworkError"]
