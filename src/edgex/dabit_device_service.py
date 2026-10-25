"""Dabit 전광판용 EdgeX Command 변환 경계.

EdgeX Device Service 프로세스가 사용할 장치별 명령 계약을 고정한다.
MQTT·HTTP 요청의 장치 실행과 결과를 같은 경계에서 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..devices.signboard import SignboardConfig, SignboardDevice
from .request_idempotency import RequestIdempotencyCache, RequestIdempotencyError


@dataclass(frozen=True)
class DabitCommandResult:
    command_id: str
    device_id: str
    status: str
    error_code: str | None = None
    event_id: str = ""
    request_id: str = ""
    execution_completed: bool = False
    resource_idle: bool = False
    completion_evidence: str | None = None

    def __post_init__(self) -> None:
        """기존 command_id를 공통 결과 계약의 request_id로 보완한다."""
        if not self.request_id:
            object.__setattr__(self, "request_id", self.command_id)

    def to_dict(self) -> dict[str, Any]:
        """명령 결과를 API와 MQTT에서 사용하는 딕셔너리로 변환한다."""
        return {
            "command_id": self.command_id,
            "request_id": self.request_id,
            "device_id": self.device_id,
            "event_id": self.event_id,
            "status": self.status,
            "error_code": self.error_code,
            "execution_completed": self.execution_completed,
            "resource_idle": self.resource_idle,
            "completion_evidence": self.completion_evidence,
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
        request_db_path: str | None = None,
    ) -> None:
        """전광판 식별자와 Dabit 장치 클라이언트를 초기화한다."""
        self.device_id = device_id
        self._device = device or SignboardDevice(config)
        self._devices = dict(devices or {device_id: self._device})
        self._dry_run = bool(dry_run)
        self._request_results = RequestIdempotencyCache[DabitCommandResult](database_path=request_db_path)
        self.on_completion = None

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
        event_id: str = "",
        on_idle=None,
    ) -> DabitCommandResult:
        if command not in self.COMMANDS:
            return DabitCommandResult(
                command_id, self.device_id, "failed", "unsupported_command", event_id
            )
        params = parameters or {}
        # 기존 테스트와 단일 장치 사용자가 _device를 교체하는 호환 동작을 유지한다.
        target_device = device or (
            self._device if device_id is None else self._devices.get(device_id)
        )
        target_device_id = device_id or self.device_id
        if target_device is None:
            return DabitCommandResult(
                command_id, target_device_id, "failed", "device_not_found", event_id
            )
        if self._dry_run:
            return DabitCommandResult(
                command_id, target_device_id, "simulated", event_id=event_id
            )
        try:
            if command == "display":
                ok = target_device.display(
                    text=str(params.get("display_text") or ""),
                    title=str(params.get("title") or "CCTV 알림"),
                    text_color=_optional_int(params.get("display_color")),
                    back_color=_optional_int(params.get("back_color")),
                    text_size=_optional_int(params.get("text_size")),
                    text_speed=_optional_int(params.get("text_speed")),
                    brightness=_optional_int(params.get("brightness")),
                    **({"on_idle": on_idle} if on_idle is not None else {}),
                )
            elif command == "clear":
                ok = target_device.clear()
            else:
                ok = target_device.power_on() if bool(params.get("power", True)) else target_device.power_off()
        except Exception:
            return DabitCommandResult(
                command_id, target_device_id, "failed", "device_error", event_id
            )
        return DabitCommandResult(
            command_id,
            target_device_id,
            "acknowledged" if ok else "failed",
            None if ok else "device_unreachable",
            event_id,
        )

    def close(self) -> None:
        """전광판 장치의 유휴 갱신 스레드를 종료한다."""
        for device in self._devices.values():
            device.stop_idle()

    def get_request_status(self, request_id: str) -> dict[str, Any]:
        """출력 명령을 재전송하지 않고 보존된 실행 상태를 조회한다."""
        return self._request_results.lookup(request_id)

    def _complete_display(self, request_id: str, event_id: str, device_id: str) -> bool:
        """유휴 복귀 ACK를 먼저 보존해 MQTT 결과 유실 후에도 조회할 수 있게 한다."""
        result = self._request_results.complete(
            request_id, event_id=event_id, device_id=device_id,
            result_type=DabitCommandResult, evidence="dabit_idle_frame_ack",
        )
        if result is None:
            return False
        return self.on_completion(result.to_dict()) if self.on_completion is not None else True

    def execute_request(self, request: Mapping[str, Any]) -> DabitCommandResult:
        """공통 Command 요청을 Dabit 명령 형식으로 변환해 실행한다."""
        request_id = str(request.get("request_id") or "")
        event_id = str(request.get("event_id") or "")
        if not request_id or not request.get("event_id") or request.get("device") != "signboard":
            return DabitCommandResult(request_id, self.device_id, "failed", "invalid_request")
        payload = request.get("payload") or {}
        if not isinstance(payload, Mapping):
            return DabitCommandResult(request_id, self.device_id, "failed", "invalid_payload")
        target_device_id = str(request.get("device_id") or self.device_id)
        if target_device_id not in self._devices:
            return DabitCommandResult(request_id, target_device_id, "failed", "device_not_found")
        if str(request.get("action") or "") not in self.COMMANDS | {"power_on", "power_off"}:
            return DabitCommandResult(request_id, target_device_id, "failed", "unsupported_command", event_id)
        # 기본 대상·모의 실행 여부도 고정 내용에 포함해 다른 작업의 이력을 재사용하지 않는다.
        def execute():
            action = str(request.get("action") or "")
            parameters = {
                "display_text": payload.get("text") or payload.get("display_text"),
                "title": payload.get("title"),
                "display_color": payload.get("display_color"),
                "back_color": payload.get("back_color"),
                "text_size": payload.get("text_size"),
                "text_speed": payload.get("text_speed"),
                "brightness": payload.get("brightness"),
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
                event_id=event_id,
                on_idle=(lambda: self._complete_display(request_id, event_id, target_device_id))
                if action == "display" and isinstance(self._devices[target_device_id], SignboardDevice) else None,
            )
            return result

        try:
            return self._request_results.run(
                request_id, execute, request={**request, "device_id": target_device_id,
                    "payload": dict(payload), "dry_run": self._dry_run}, result_type=DabitCommandResult,
            )
        except RequestIdempotencyError as exc:
            return DabitCommandResult(
                command_id=request_id, request_id=request_id, event_id=event_id,
                device_id=target_device_id, status=exc.status, error_code=exc.error_code,
            )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
