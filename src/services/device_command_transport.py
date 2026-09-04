"""Action Layer의 장치별 직접 호출을 공통 명령 경계로 감싼다."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


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
