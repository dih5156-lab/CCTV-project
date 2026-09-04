"""ActionBridge 이벤트 실행기."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_display_message,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_severity,
    get_payload_tts_message,
)
from ..edgex.command_contract import build_command_request, build_command_topic
from ..event_priority import event_priority
from ..event_routing import decide_alert_forward
from .cctv_metrics import (
    device_command_results as _device_command_results,
)
from .cctv_metrics import (
    device_commands as _device_commands,
)

logger = logging.getLogger(__name__)


class _ActionExecutor:
    """디바이스 알람 실행, HTTP 전송, 저장소 기록을 묶어 처리한다."""

    def __init__(
        self,
        *,
        repo,
        alarm,
        forwarder,
        speaker,
        signboard,
        siren,
        resolve_devices,
        publish_status,
        publish_edgex_command: Optional[Callable[[str, Dict], bool]],
        resolve_edgex_device_ids: Optional[Callable[[str, str], List[str]]] = None,
        edgex_jetson_id: str,
        edgex_command_topic_prefix: str,
        build_display_text,
        alarm_device_enum,
    ) -> None:
        """장치 실행에 필요한 저장소·클라이언트·상태 발행기를 보관한다."""
        self._repo = repo
        self._alarm = alarm
        self._forwarder = forwarder
        self._speaker = speaker
        self._signboard = signboard
        self._siren = siren
        self._resolve_devices = resolve_devices
        self._publish_status = publish_status
        self._publish_edgex_command = publish_edgex_command
        self._resolve_edgex_device_ids = resolve_edgex_device_ids
        self._edgex_jetson_id = edgex_jetson_id
        self._edgex_command_topic_prefix = edgex_command_topic_prefix
        self._build_display_text = build_display_text
        self._alarm_device_enum = alarm_device_enum

    def _publish_shadow_command(
        self,
        *,
        event_id: str,
        device: str,
        action: str,
        payload: Dict,
        command_id: str,
        camera_id: str,
    ) -> None:
        """기존 직접 제어와 비교할 EdgeX Command를 생성해 발행한다."""
        if self._publish_edgex_command is None:
            return
        try:
            device_ids = (
                self._resolve_edgex_device_ids(device, camera_id)
                if self._resolve_edgex_device_ids
                else [""]
            )
            for device_id in device_ids:
                target_command_id = f"{command_id}:{device_id}" if device_id else command_id
                command = build_command_request(
                    event_id=event_id,
                    request_id=target_command_id,
                    device=device,
                    device_id=device_id or None,
                    action=action,
                    payload=payload,
                )
                topic = build_command_topic(
                    self._edgex_command_topic_prefix,
                    self._edgex_jetson_id,
                    device,
                    device_id=device_id or None,
                )
                if not self._publish_edgex_command(topic, command):
                    logger.warning("EdgeX shadow Command 발행 결과 실패: command_id=%s", target_command_id)
        except Exception as exc:
            logger.warning("EdgeX shadow Command 구성 실패: command_id=%s error=%s", command_id, exc)

    def execute(self, topic: str, payload: Dict) -> None:
        """단일 이벤트에 대한 장치 조치와 외부 전송을 수행한다."""
        camera_id = get_payload_camera_id(payload)
        event_type = get_payload_event_type(payload).lower()
        severity = get_payload_severity(payload).lower()
        display_message = get_payload_display_message(payload)
        tts_message = get_payload_tts_message(payload)
        force_reachability_refresh = self._alarm.is_demo_event(payload)

        alarm_played = False
        device_results = []
        if self._alarm.should_alarm(topic, payload) and self._alarm.try_acquire_slot(
            camera_id,
            event_type,
            priority=event_priority(payload),
            object_id=payload.get("object_id"),
            force=self._alarm.is_demo_event(payload),
        ):
            devices = self._resolve_devices(
                camera_id,
                force_refresh=force_reachability_refresh,
            )

            if self._alarm_device_enum.SPEAKER in devices:
                command_id = f"{get_payload_event_id(payload)}:speaker"
                self._repo.record_command(command_id, "device/speaker", "sent", payload)
                speaker_ok = self._speaker.play(
                    event_type,
                    severity,
                    camera_id,
                    text=tts_message,
                )
                if speaker_ok:
                    alarm_played = True
                _device_commands.labels(
                    device="speaker", status="ok" if speaker_ok else "skip"
                ).inc()
                status = "acknowledged" if speaker_ok else "failed"
                _device_command_results.labels(device="speaker", status=status).inc()
                self._repo.record_command(command_id, "device/speaker", status, payload)
                device_results.append({"device": "speaker", "command_id": command_id, "status": status})
                self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="speaker",
                    action="play",
                    payload={
                        "event_type": event_type,
                        "severity": severity,
                        "camera_id": camera_id,
                        "text": tts_message,
                    },
                )

            if self._alarm_device_enum.SIGNBOARD in devices:
                command_id = f"{get_payload_event_id(payload)}:signboard"
                self._repo.record_command(command_id, "device/signboard", "sent", payload)
                signboard_ok = self._signboard.display(
                    text=display_message
                    or self._build_display_text(event_type, severity, camera_id),
                    title="경고!",
                    class_name=event_type,
                )
                if signboard_ok:
                    alarm_played = True
                _device_commands.labels(
                    device="signboard", status="ok" if signboard_ok else "skip"
                ).inc()
                status = "acknowledged" if signboard_ok else "failed"
                _device_command_results.labels(device="signboard", status=status).inc()
                self._repo.record_command(command_id, "device/signboard", status, payload)
                device_results.append({"device": "signboard", "command_id": command_id, "status": status})
                self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="signboard",
                    action="display",
                    payload={
                        "event_type": event_type,
                        "severity": severity,
                        "camera_id": camera_id,
                        "text": display_message
                        or self._build_display_text(event_type, severity, camera_id),
                        "title": "경고!",
                    },
                )

            if self._alarm_device_enum.SIREN in devices:
                command_id = f"{get_payload_event_id(payload)}:siren"
                self._repo.record_command(command_id, "device/siren", "sent", payload)
                siren_ok = self._siren.trigger(event_type, camera_id)
                if siren_ok:
                    alarm_played = True
                _device_commands.labels(
                    device="siren", status="ok" if siren_ok else "skip"
                ).inc()
                status = "acknowledged" if siren_ok else "failed"
                _device_command_results.labels(device="siren", status=status).inc()
                self._repo.record_command(command_id, "device/siren", status, payload)
                device_results.append({"device": "siren", "command_id": command_id, "status": status})
                self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="siren",
                    action="trigger",
                    payload={
                        "event_type": event_type,
                        "camera_id": camera_id,
                    },
                )

        forward_decision = decide_alert_forward(
            payload,
            has_targets=self._forwarder.has_targets,
        )
        if forward_decision.reason == "already_stored":
            logger.debug(
                "중복 Alert 저장 방지를 위해 HTTP forward 생략: camera=%s type=%s",
                camera_id,
                event_type,
            )
        elif forward_decision.should_forward:
            self._forwarder.forward(topic, payload)
        http_sent = forward_decision.http_sent

        self._repo.save(topic, payload, alarm_played=alarm_played, http_sent=http_sent)
        self._publish_status(
            "events/executed",
            {
                "camera_id": camera_id,
                "event_type": event_type,
                "severity": severity,
                "status": "executed",
                "alarm_played": alarm_played,
                "http_sent": http_sent,
                "devices": [
                    device.value
                    for device in self._resolve_devices(
                        camera_id,
                        force_refresh=force_reachability_refresh,
                    )
                ],
                "device_results": device_results,
            },
        )
