"""장치 명령 결과의 저장과 메트릭 기록을 담당한다."""

from __future__ import annotations

from typing import Dict

from .device_command_transport import DeviceCommand, DeviceCommandResult


class CommandResultCollector:
    """장치 전송 결과를 공통 응답으로 변환하고 저장소·메트릭에 기록한다."""

    def __init__(self, *, repository, device_commands_metric, device_results_metric) -> None:
        """결과 저장소와 장치 명령 메트릭을 보관한다."""
        self._repository = repository
        self._device_commands_metric = device_commands_metric
        self._device_results_metric = device_results_metric

    def collect(self, command: DeviceCommand, result: DeviceCommandResult) -> Dict:
        """전송 결과를 기록하고 Action Layer가 사용할 응답 형태로 반환한다."""
        self._device_commands_metric.labels(
            device=command.device,
            status="ok" if result.ok else "skip",
        ).inc()
        self._device_results_metric.labels(
            device=command.device,
            status=result.status,
        ).inc()
        # 발행 전에 보류된 명령도 만료/정책 사유를 남겨 장치 실패와 구분한다.
        payload = command.payload
        if result.target_results:
            payload = {**payload, "event_id": command.event_id,
                       "device_results": list(result.target_results)}
        elif result.status in {"expired", "held"}:
            payload = {**payload, "action_decision": {"status": result.status, "reason": result.error}}
        stored = self._repository.record_command(
            command.command_id,
            f"device/{command.device}",
            result.status,
            payload,
        )
        # 비동기 장치 결과가 먼저 왔다면 저장소에서 병합한 상태를 응답에도 사용한다.
        status = stored["status"] if result.target_results and isinstance(stored, dict) else result.status
        return {
            "device": command.device,
            "command_id": command.command_id,
            "status": status,
            "event_id": command.event_id,
            "device_ids": list(result.device_ids),
            "request_ids": list(result.request_ids),
            **({"device_results": stored["payload"]["device_results"]
                if isinstance(stored, dict) else list(result.target_results)} if result.target_results else {}),
            **({"error": result.error} if result.error else {}),
        }

    def register_targets(self, command: DeviceCommand, targets: list[Dict]) -> bool:
        """전체 대상을 영속 기록한 뒤에만 발행해 응답 유실을 성공으로 오인하지 않는다."""
        return self._repository.record_command(
            command.command_id, f"device/{command.device}", "pending",
            {**command.payload, "event_id": command.event_id,
             "target_request_ids": [row["request_id"] for row in targets], "device_results": targets},
        ) is not None
