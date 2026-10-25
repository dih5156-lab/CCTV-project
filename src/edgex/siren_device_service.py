"""EdgeX 명령을 InterM 경광등 제어로 변환하는 서비스 경계."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..devices.siren import SensorConfig, SirenDevice
from .request_idempotency import RequestIdempotencyCache, RequestIdempotencyError


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
        request_db_path: str | None = None,
    ) -> None:
        """경광등 식별자와 InterM 장치 클라이언트를 초기화한다."""
        self.device_id = device_id
        self._device = device or SirenDevice(config)
        self._devices = dict(devices or {device_id: self._device})
        self._dry_run = bool(dry_run)
        self._request_results = RequestIdempotencyCache[SirenCommandResult](database_path=request_db_path)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """현재 서비스가 처리할 수 있는 장치 식별자 목록을 반환한다."""
        return tuple(self._devices)

    def get_request_status(self, request_id: str) -> dict[str, Any]:
        """출력 명령을 재전송하지 않고 보존된 실행 상태를 조회한다."""
        return self._request_results.lookup(request_id)

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
        # 기본 대상·모의 실행 여부도 고정 내용에 포함해 다른 작업의 이력을 재사용하지 않는다.
        def execute():
            if self._dry_run:
                result = SirenCommandResult(request_id, event_id, target_device_id, "simulated")
                return result

            try:
                success = self._execute_action(action, payload, target_device)
            except Exception:
                # 다른 물리 장치의 실패로 기록되지 않도록 실제 대상 ID를 보존한다.
                return SirenCommandResult(request_id, event_id, target_device_id, "failed", "device_error")
            result = SirenCommandResult(
                request_id,
                event_id,
                target_device_id,
                "acknowledged" if success else "failed",
                None if success else "device_unreachable",
            )
            return result

        try:
            return self._request_results.run(
                request_id, execute, request={**request, "device_id": target_device_id,
                    "payload": dict(payload), "dry_run": self._dry_run}, result_type=SirenCommandResult,
            )
        except RequestIdempotencyError as exc:
            return SirenCommandResult(
                request_id=request_id, event_id=event_id,
                device_id=target_device_id, status=exc.status, error_code=exc.error_code,
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
