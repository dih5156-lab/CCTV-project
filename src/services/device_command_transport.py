"""Action Layer의 장치별 직접 호출을 공통 명령 경계로 감싼다."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from ..edgex.command_contract import build_command_request, build_command_topic
from .command_identifiers import build_transport_request_id


class DeviceCommandMode(str, Enum):
    """장치 명령을 실행하는 운영 모드다."""

    DIRECT = "direct"
    SHADOW = "shadow"
    EDGEX = "edgex"


def normalize_device_command_mode(value: str | DeviceCommandMode | None) -> DeviceCommandMode:
    """환경변수나 명령행 값을 안전한 장치 명령 모드로 변환한다."""
    if isinstance(value, DeviceCommandMode):
        return value
    normalized = str(value or "direct").strip().lower()
    try:
        return DeviceCommandMode(normalized)
    except ValueError as exc:
        raise ValueError(f"지원하지 않는 장치 명령 모드: {normalized}") from exc


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
    device_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    target_results: tuple[dict[str, Any], ...] = ()


class DeviceCommandTransport(Protocol):
    """장치 명령을 실행하는 전송 계층의 최소 인터페이스다."""

    def send(self, command: DeviceCommand) -> DeviceCommandResult:
        """공통 장치 명령을 실행하고 결과를 반환한다."""


class DirectDeviceCommandTransport:
    """기존 장치 클라이언트를 공통 명령 인터페이스로 감싼다."""

    def __init__(self, *, speaker, signboard, siren, dry_run: bool = False) -> None:
        """기존 장치 클라이언트와 물리 출력 차단 여부를 보관한다."""
        self._speaker = speaker
        self._signboard = signboard
        self._siren = siren
        self._dry_run = dry_run

    def send(self, command: DeviceCommand) -> DeviceCommandResult:
        """장치 종류에 맞는 기존 클라이언트를 호출한다."""
        if self._dry_run:
            return DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=True,
                status="simulated",
            )
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
        register_targets=None,
    ) -> None:
        """EdgeX 발행기와 물리 장치 ID 조회기를 보관한다."""
        self._publish = publish
        self._resolve_device_ids = resolve_device_ids
        self._jetson_id = jetson_id
        self._topic_prefix = topic_prefix
        self._register_targets = register_targets

    def send(self, command: DeviceCommand, *, before_publish=None) -> DeviceCommandResult:
        """등록된 물리 장치별로 EdgeX Command를 발행한다."""
        device_ids = self._resolve_device_ids(command.device, command.camera_id)
        if not device_ids:
            return DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=False,
                status="failed",
                error=f"등록된 EdgeX 대상 장치 없음: {command.device}",
            )
        # 응답이 발행 함수 반환보다 빨라도 전체 대상과 미응답 상태를 먼저 보존한다.
        target_results = [{
            "request_id": build_transport_request_id(command.command_id, device_id),
            "event_id": command.event_id, "device_id": device_id,
            "status": "pending", "error_code": None,
        } for device_id in device_ids]
        if self._register_targets is not None and not self._register_targets(command, target_results):
            return DeviceCommandResult(command.device, command.command_id, False, "failed",
                                       "장치 대상 저장 실패")
        failed_devices = []
        target_device_ids = []
        target_request_ids = []
        for device_id, target_result in zip(device_ids, target_results):
            target_command_id = build_transport_request_id(
                command.command_id, device_id
            )
            target_device_ids.append(device_id)
            target_request_ids.append(target_command_id)
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
            try:
                reason = before_publish(device_id) if before_publish is not None else None
                if reason:
                    target_result["status"] = "expired" if reason == "expired" else "held"
                    target_result["error_code"] = reason
                else:
                    accepted = self._publish(topic, payload)
                    target_result["status"] = "accepted" if accepted else "failed"
                    target_result["error_code"] = None if accepted else "publish_failed"
            except Exception:
                # 발행 중 예외는 전달 여부를 단정할 수 없으며 다른 대상의 전송은 계속한다.
                target_result["status"] = "uncertain"
                target_result["error_code"] = "publish_error"
            if target_result["status"] != "accepted":
                failed_devices.append(device_id or command.device)

        if failed_devices:
            statuses = {row["status"] for row in target_results}
            return DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=False,
                status=(next(iter(statuses)) if statuses in ({"expired"}, {"held"})
                        else "uncertain" if any(row["status"] == "uncertain" for row in target_results)
                        else "failed" if len(failed_devices) == len(device_ids) else "partial_failure"),
                error=f"EdgeX 명령 발행 실패: {', '.join(failed_devices)}",
                device_ids=tuple(target_device_ids),
                request_ids=tuple(target_request_ids),
                target_results=tuple(target_results),
            )
        return DeviceCommandResult(
            device=command.device,
            command_id=command.command_id,
            ok=True,
            # MQTT 발행 접수는 장치의 성공 응답이 아니다.
            status="accepted",
            device_ids=tuple(target_device_ids),
            request_ids=tuple(target_request_ids),
            target_results=tuple(target_results),
        )
