"""EdgeX 명령을 InterM 스피커 제어로 변환하는 서비스 경계."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..devices.speaker import SpeakerConfig, SpeakerDevice


@dataclass(frozen=True)
class SpeakerCommandResult:
    """스피커 명령 처리 결과를 표현한다."""

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


class SpeakerDeviceService:
    """EdgeX 스피커 명령을 InterM API 호출로 변환한다."""

    COMMANDS = frozenset({"play", "stop", "power_on", "power_off"})

    def __init__(
        self,
        *,
        device_id: str,
        config: SpeakerConfig,
        device: Optional[Any] = None,
        devices: Optional[Mapping[str, Any]] = None,
        dry_run: bool = False,
    ) -> None:
        """스피커 식별자와 InterM 장치 클라이언트를 초기화한다."""
        self.device_id = device_id
        self._device = device or SpeakerDevice(config)
        self._devices = dict(devices or {device_id: self._device})
        self._dry_run = bool(dry_run)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """현재 서비스가 처리할 수 있는 장치 식별자 목록을 반환한다."""
        return tuple(self._devices)

    def execute_request(self, request: Mapping[str, Any]) -> SpeakerCommandResult:
        """공통 Command 요청을 검증하고 스피커 동작으로 실행한다."""
        request_id = str(request.get("request_id") or "")
        event_id = str(request.get("event_id") or "")
        if not request_id or not event_id or request.get("device") != "speaker":
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=self.device_id,
                status="failed",
                error_code="invalid_request",
            )
        target_device_id = str(request.get("device_id") or self.device_id)
        target_device = self._devices.get(target_device_id)
        if target_device is None:
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=target_device_id,
                status="failed",
                error_code="device_not_found",
            )

        action = str(request.get("action") or "")
        if action not in self.COMMANDS:
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=target_device_id,
                status="failed",
                error_code="unsupported_command",
            )

        payload = request.get("payload") or {}
        if not isinstance(payload, Mapping):
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=target_device_id,
                status="failed",
                error_code="invalid_payload",
            )

        if self._dry_run:
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=target_device_id,
                status="simulated",
            )

        try:
            success = self._execute_action(action, payload, target_device)
        except Exception:
            return SpeakerCommandResult(
                request_id=request_id,
                event_id=event_id,
                device_id=target_device_id,
                status="failed",
                error_code="device_error",
            )

        return SpeakerCommandResult(
            request_id=request_id,
            event_id=event_id,
            device_id=target_device_id,
            status="acknowledged" if success else "failed",
            error_code=None if success else "device_unreachable",
        )

    def _execute_action(self, action: str, payload: Mapping[str, Any], device: Any) -> bool:
        """명령 이름에 맞는 InterM 스피커 메서드를 호출한다."""
        if action == "play":
            return bool(
                    device.play(
                    str(payload.get("event_type") or "unknown"),
                    str(payload.get("severity") or "warning"),
                    str(payload.get("camera_id") or "unknown"),
                    text=str(payload.get("text") or ""),
                )
            )
        if action == "stop":
            return bool(device.stop())
        if action == "power_on":
            return bool(device.power_on())
        return bool(device.power_off())

    def close(self) -> None:
        """스피커 장치가 사용하는 백그라운드 자원을 정리한다."""
        for device in self._devices.values():
            stop_method = getattr(device, "stop_background_tasks", None)
            if callable(stop_method):
                stop_method()
