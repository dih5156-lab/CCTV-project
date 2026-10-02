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
        self._repository.record_command(
            command.command_id,
            f"device/{command.device}",
            result.status,
            command.payload,
        )
        return {
            "device": command.device,
            "command_id": command.command_id,
            "status": result.status,
            "event_id": command.event_id,
            "device_ids": list(result.device_ids),
            "request_ids": list(result.request_ids),
            **({"error": result.error} if result.error else {}),
        }
