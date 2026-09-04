"""Action Layer의 장치별 직접 호출을 공통 명령 경계로 감싼다."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..edgex.command_contract import build_command_request, build_command_topic


@dataclass(frozen=True)
class DeviceCommand:
    """장치 전송 방식과 무관하게 사용하는 공통 명령이다."""

    device: str
    action: str
    payload: Mapping[str, Any]
    command_id: str
    event_id: str
    camera_id: str


@dataclass(frozen=True)
class DeviceCommandResult:
    """장치 명령 실행 결과를 공통 상태로 표현한다."""

    device: str
    command_id: str
    ok: bool
    status: str
    error: str | None = None


class DeviceCommandTransport(Protocol):
    """장치 명령을 실행하는 전송 계층의 최소 인터페이스다."""

    def send(self, command: DeviceCommand) -> DeviceCommandResult:
        """공통 장치 명령을 실행하고 결과를 반환한다."""


class DirectDeviceCommandTransport:
    """기존 장치 클라이언트를 공통 명령 인터페이스로 감싼다."""

    def __init__(self, *, speaker, signboard, siren) -> None:
        """기존 스피커·전광판·사이렌 클라이언트를 보관한다."""
        self._speaker = speaker
        self._signboard = signboard
        self._siren = siren

    def send(self, command: DeviceCommand) -> DeviceCommandResult:
        """장치 종류에 맞는 기존 클라이언트를 호출한다."""
        try:
            ok = self._send_to_device(command)
        except Exception as exc:
            return DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=False,
                status="failed",
                error=str(exc),
            )
        return DeviceCommandResult(
            device=command.device,
            command_id=command.command_id,
            ok=bool(ok),
            status="acknowledged" if ok else "failed",
        )

    def _send_to_device(self, command: DeviceCommand) -> bool:
        """장치별 기존 호출 규격으로 공통 명령을 변환한다."""
        if command.device == "speaker" and command.action == "play":
            return self._speaker.play(
                command.payload.get("event_type", ""),
                command.payload.get("severity", ""),
                command.camera_id,
                text=command.payload.get("text", ""),
            )
        if command.device == "signboard" and command.action == "display":
            return self._signboard.display(
                text=command.payload.get("text", ""),
                title=command.payload.get("title", "경고!"),
                class_name=command.payload.get("class_name", command.payload.get("event_type", "")),
            )
        if command.device == "siren" and command.action == "trigger":
            return self._siren.trigger(
                command.payload.get("event_type", ""),
                command.camera_id,
            )
        raise ValueError(f"지원하지 않는 장치: {command.device}")


class EdgeXCommandTransport:
    """공통 명령을 EdgeX MQTT Command 계약으로 변환해 발행한다."""

    def __init__(
        self,
        *,
        publish,
        resolve_device_ids,
        jetson_id: str,
        topic_prefix: str,
    ) -> None:
        """EdgeX 발행기와 물리 장치 ID 조회기를 보관한다."""
        self._publish = publish
        self._resolve_device_ids = resolve_device_ids
        self._jetson_id = jetson_id
        self._topic_prefix = topic_prefix

    def send(self, command: DeviceCommand) -> DeviceCommandResult:
        """등록된 물리 장치별로 EdgeX Command를 발행한다."""
        device_ids = self._resolve_device_ids(command.device, command.camera_id)
        failed_devices = []
        for device_id in device_ids or [""]:
            target_command_id = (
                f"{command.command_id}:{device_id}" if device_id else command.command_id
            )
            payload = build_command_request(
                event_id=command.event_id,
                request_id=target_command_id,
                device=command.device,
                device_id=device_id or None,
                action=command.action,
                payload=dict(command.payload),
            )
            topic = build_command_topic(
                self._topic_prefix,
                self._jetson_id,
                command.device,
                device_id=device_id or None,
            )
            if not self._publish(topic, payload):
                failed_devices.append(device_id or command.device)

        if failed_devices:
            return DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=False,
                status="failed",
                error=f"EdgeX 명령 발행 실패: {', '.join(failed_devices)}",
            )
        return DeviceCommandResult(
            device=command.device,
            command_id=command.command_id,
            ok=True,
            status="acknowledged",
        )
