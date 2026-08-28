"""ActionBridge 이벤트 실행기."""

from __future__ import annotations

import logging
from typing import Dict

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_display_message,
    get_payload_event_id,
    get_payload_event_type,
    get_payload_severity,
    get_payload_tts_message,
)
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
        build_display_text,
        alarm_device_enum,
    ) -> None:
        self._repo = repo
        self._alarm = alarm
        self._forwarder = forwarder
        self._speaker = speaker
        self._signboard = signboard
        self._siren = siren
        self._resolve_devices = resolve_devices
        self._publish_status = publish_status
        self._build_display_text = build_display_text
        self._alarm_device_enum = alarm_device_enum

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
            camera_id, event_type, force=self._alarm.is_demo_event(payload)
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
