"""Dabit 전광판용 EdgeX Command 변환 경계.

EdgeX Device Service 프로세스가 사용할 장치별 명령 계약을 고정한다.
현재 운영 경로(Action Layer 직접 TCP)는 유지하며, 이 모듈은 전환 전 검증용이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..devices.signboard import SignboardConfig, SignboardDevice


@dataclass(frozen=True)
class DabitCommandResult:
    command_id: str
    device_id: str
    status: str
    error_code: str | None = None


class DabitDeviceService:
    """EdgeX Command를 Dabit TCP 장치 명령으로 변환한다."""

    COMMANDS = frozenset({"display", "clear", "power"})

    def __init__(self, *, device_id: str, config: SignboardConfig) -> None:
        self.device_id = device_id
        self._device = SignboardDevice(config)

    def execute(self, command_id: str, command: str, parameters: Mapping[str, Any] | None = None) -> DabitCommandResult:
        if command not in self.COMMANDS:
            return DabitCommandResult(command_id, self.device_id, "failed", "unsupported_command")
        params = parameters or {}
        try:
            if command == "display":
                ok = self._device.display(
                    text=str(params.get("display_text") or ""),
                    title=str(params.get("title") or "CCTV 알림"),
                    text_color=_optional_int(params.get("display_color")),
                    back_color=_optional_int(params.get("back_color")),
                    text_size=_optional_int(params.get("text_size")),
                    text_speed=_optional_int(params.get("text_speed")),
                )
            elif command == "clear":
                ok = self._device.clear()
            else:
                ok = self._device.power_on() if bool(params.get("power", True)) else self._device.power_off()
        except Exception:
            return DabitCommandResult(command_id, self.device_id, "failed", "device_error")
        return DabitCommandResult(command_id, self.device_id, "acknowledged" if ok else "failed", None if ok else "device_unreachable")

    def close(self) -> None:
        self._device.stop_idle()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
