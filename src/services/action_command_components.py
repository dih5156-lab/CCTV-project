"""Action Layer의 장치 선택과 공통 명령 생성을 담당한다."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import TypeVar

from .device_command_transport import DeviceCommand, DeviceCommandMode

DeviceType = TypeVar("DeviceType")


class DeviceResolver:
    """이벤트에 사용할 출력 장치를 정책과 가용성 기준으로 선택한다."""

    def __init__(
        self,
        *,
        resolve_candidates: Callable[[str], Sequence[DeviceType]],
        is_available: Callable[..., bool],
        command_mode: DeviceCommandMode,
        allowed_devices: Collection[DeviceType] | None = None,
    ) -> None:
        """장치 후보 조회기와 가용성 확인기를 보관한다."""
        self._resolve_candidates = resolve_candidates
        self._is_available = is_available
        self._command_mode = DeviceCommandMode(command_mode)
        self._allowed_devices = set(allowed_devices or ())

    def resolve(
        self,
        camera_id: str,
        *,
        force_refresh: bool = False,
    ) -> list[DeviceType]:
        """카메라에 연결된 장치 중 실행 가능한 장치만 반환한다."""
        candidates = list(self._resolve_candidates(camera_id))
        if self._command_mode is not DeviceCommandMode.DIRECT:
            candidates = [
                device
                for device in candidates
                if not self._allowed_devices or device in self._allowed_devices
            ]
        return [
            device
            for device in candidates
            if self._is_available(device, force_refresh=force_refresh)
        ]


class CommandBuilder:
    """장치별 입력을 공통 DeviceCommand 객체로 변환한다."""

    def build(
        self,
        *,
        device: str,
        action: str,
        payload: dict,
        command_id: str,
        event_id: str,
        camera_id: str,
    ) -> DeviceCommand:
        """명령 payload를 복사해 공통 장치 명령을 생성한다."""
        return DeviceCommand(
            device=device,
            action=action,
            payload=dict(payload),
            command_id=command_id,
            event_id=event_id,
            camera_id=camera_id,
        )

    def build_alarm_command(
        self,
        *,
        device: str,
        action: str,
        event_type: str,
        severity: str,
        camera_id: str,
        event_id: str,
        command_id: str,
        text: str = "",
        title: str = "경고!",
        class_name: str | None = None,
    ) -> DeviceCommand:
        """알람 장치 종류에 맞는 표준 payload와 명령을 생성한다."""
        if device == "speaker":
            payload = {
                "event_type": event_type,
                "severity": severity,
                "camera_id": camera_id,
                "text": text,
            }
        elif device == "signboard":
            payload = {
                "event_type": event_type,
                "severity": severity,
                "camera_id": camera_id,
                "text": text,
                "title": title,
                "class_name": class_name or event_type,
            }
        elif device == "siren":
            payload = {
                "event_type": event_type,
                "camera_id": camera_id,
            }
        else:
            raise ValueError(f"지원하지 않는 알람 장치: {device}")
        return self.build(
            device=device,
            action=action,
            payload=payload,
            command_id=command_id,
            event_id=event_id,
            camera_id=camera_id,
        )
