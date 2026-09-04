"""EdgeX 명령을 InterM 경광등 제어로 변환하는 서비스 경계."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..devices.siren import SensorConfig, SirenDevice


@dataclass(frozen=True)
class SirenCommandResult:
    """경광등 명령 처리 결과를 표현한다."""

    request_id: str
    event_id: str
    device_id: str
    status: str
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """MQTT 결과 발행에 사용할 딕셔너리로 변환한다."""
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "device_id": self.device_id,
            "status": self.status,
            "error_code": self.error_code,
        }


class SirenDeviceService:
    """EdgeX 경광등 명령을 InterM API 호출로 변환한다."""

    COMMANDS = frozenset({"trigger", "stop"})

    def __init__(
        self,
        *,
        device_id: str,
        config: SensorConfig,
        device: Optional[Any] = None,
        devices: Optional[Mapping[str, Any]] = None,
        dry_run: bool = False,
    ) -> None:
        """경광등 식별자와 InterM 장치 클라이언트를 초기화한다."""
        self.device_id = device_id
        self._device = device or SirenDevice(config)
        self._devices = dict(devices or {device_id: self._device})
        self._dry_run = bool(dry_run)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """현재 서비스가 처리할 수 있는 장치 식별자 목록을 반환한다."""
        return tuple(self._devices)

    def execute_request(self, request: Mapping[str, Any]) -> SirenCommandResult:
        """공통 Command 요청을 검증하고 경광등 동작으로 실행한다."""
        request_id = str(request.get("request_id") or "")
        event_id = str(request.get("event_id") or "")
        if not request_id or not event_id or request.get("device") != "siren":
            return self._result(request_id, event_id, "invalid_request")
        target_device_id = str(request.get("device_id") or self.device_id)
        target_device = self._devices.get(target_device_id)
        if target_device is None:
            return SirenCommandResult(
                request_id, event_id, target_device_id, "failed", "device_not_found"
            )

        action = str(request.get("action") or "")
        if action not in self.COMMANDS:
            return self._result(request_id, event_id, "unsupported_command")

        payload = request.get("payload") or {}
        if not isinstance(payload, Mapping):
            return self._result(request_id, event_id, "invalid_payload")
        if self._dry_run:
            return SirenCommandResult(request_id, event_id, self.device_id, "simulated")

        try:
            success = self._execute_action(action, payload, target_device)
        except Exception:
            return self._result(request_id, event_id, "device_error")
        return SirenCommandResult(
            request_id,
            event_id,
            target_device_id,
            "acknowledged" if success else "failed",
            None if success else "device_unreachable",
        )

    def _execute_action(self, action: str, payload: Mapping[str, Any], device: Any) -> bool:
        """명령 이름에 맞는 InterM 경광등 메서드를 호출한다."""
        if action == "trigger":
            return bool(
                device.trigger(
                    str(payload.get("event_type") or "unknown"),
                    str(payload.get("camera_id") or "unknown"),
                )
            )
        return bool(device.stop())

    def _result(self, request_id: str, event_id: str, error_code: str) -> SirenCommandResult:
        """공통 실패 결과를 생성한다."""
        return SirenCommandResult(
            request_id,
            event_id,
            self.device_id,
            "failed",
            error_code,
        )

    def close(self) -> None:
        """경광등 서비스 종료 시 필요한 자원을 정리한다."""
        for device in self._devices.values():
            stop_timer = getattr(device, "stop_timer", None)
            if callable(stop_timer):
                stop_timer()
