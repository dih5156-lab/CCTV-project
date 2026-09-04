"""Dabit 전광판용 EdgeX Command 변환 경계.

EdgeX Device Service 프로세스가 사용할 장치별 명령 계약을 고정한다.
현재 운영 경로(Action Layer 직접 TCP)는 유지하며, 이 모듈은 전환 전 검증용이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..devices.signboard import SignboardConfig, SignboardDevice


@dataclass(frozen=True)
class DabitCommandResult:
    command_id: str
    device_id: str
    status: str
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """명령 결과를 API와 MQTT에서 사용하는 딕셔너리로 변환한다."""
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "status": self.status,
            "error_code": self.error_code,
        }


class DabitDeviceService:
    """EdgeX Command를 Dabit TCP 장치 명령으로 변환한다."""

    COMMANDS = frozenset({"display", "clear", "power"})

    def __init__(
        self,
        *,
        device_id: str,
        config: SignboardConfig,
        device: Optional[Any] = None,
        devices: Optional[Mapping[str, Any]] = None,
        dry_run: bool = False,
    ) -> None:
        """전광판 식별자와 Dabit 장치 클라이언트를 초기화한다."""
        self.device_id = device_id
        self._device = device or SignboardDevice(config)
        self._devices = dict(devices or {device_id: self._device})
        self._dry_run = bool(dry_run)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """현재 서비스가 처리할 수 있는 장치 식별자 목록을 반환한다."""
        return tuple(self._devices)

    def execute(
        self,
        command_id: str,
        command: str,
        parameters: Mapping[str, Any] | None = None,
        device: Any | None = None,
        device_id: str | None = None,
    ) -> DabitCommandResult:
        if command not in self.COMMANDS:
            return DabitCommandResult(command_id, self.device_id, "failed", "unsupported_command")
        params = parameters or {}
        # 기존 테스트와 단일 장치 사용자가 _device를 교체하는 호환 동작을 유지한다.
        target_device = device or (
            self._device if device_id is None else self._devices.get(device_id)
        )
        target_device_id = device_id or self.device_id
        if target_device is None:
            return DabitCommandResult(command_id, target_device_id, "failed", "device_not_found")
        if self._dry_run:
            return DabitCommandResult(command_id, target_device_id, "simulated")
        try:
            if command == "display":
                ok = target_device.display(
                    text=str(params.get("display_text") or ""),
                    title=str(params.get("title") or "CCTV 알림"),
                    text_color=_optional_int(params.get("display_color")),
                    back_color=_optional_int(params.get("back_color")),
                    text_size=_optional_int(params.get("text_size")),
                    text_speed=_optional_int(params.get("text_speed")),
                )
            elif command == "clear":
                ok = target_device.clear()
            else:
                ok = target_device.power_on() if bool(params.get("power", True)) else target_device.power_off()
        except Exception:
            return DabitCommandResult(command_id, target_device_id, "failed", "device_error")
        return DabitCommandResult(command_id, target_device_id, "acknowledged" if ok else "failed", None if ok else "device_unreachable")

    def close(self) -> None:
        """전광판 장치의 유휴 갱신 스레드를 종료한다."""
        for device in self._devices.values():
            device.stop_idle()

    def execute_request(self, request: Mapping[str, Any]) -> DabitCommandResult:
        """공통 Command 요청을 Dabit 명령 형식으로 변환해 실행한다."""
        request_id = str(request.get("request_id") or "")
        if not request_id or not request.get("event_id") or request.get("device") != "signboard":
            return DabitCommandResult(request_id, self.device_id, "failed", "invalid_request")
        payload = request.get("payload") or {}
        if not isinstance(payload, Mapping):
            return DabitCommandResult(request_id, self.device_id, "failed", "invalid_payload")
        target_device_id = str(request.get("device_id") or self.device_id)
        if target_device_id not in self._devices:
            return DabitCommandResult(request_id, target_device_id, "failed", "device_not_found")
        action = str(request.get("action") or "")
        parameters = {
            "display_text": payload.get("text") or payload.get("display_text"),
            "title": payload.get("title"),
            "power": payload.get("power", True),
        }
        command = "power" if action in {"power_on", "power_off"} else action
        if action == "power_off":
            parameters["power"] = False
        result = self.execute(
            request_id,
            command,
            parameters,
            device=self._devices[target_device_id],
            device_id=target_device_id,
        )
        return DabitCommandResult(
            result.command_id,
            result.device_id,
            result.status,
            result.error_code,
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
