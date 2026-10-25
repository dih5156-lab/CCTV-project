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
from ..event_routing import decide_alert_forward
from .action_command_components import CommandBuilder
from .action_policy import ActionPolicy
from .cctv_metrics import (
    device_command_results as _device_command_results,
)
from .cctv_metrics import (
    device_commands as _device_commands,
)
from .command_identifiers import build_device_command_id
from .command_result_collector import CommandResultCollector
from .device_command_transport import (
    DeviceCommand,
    DeviceCommandMode,
    DeviceCommandResult,
    DirectDeviceCommandTransport,
    EdgeXCommandTransport,
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
        device_transport=None,
        edgex_transport=None,
        device_command_mode=DeviceCommandMode.DIRECT,
        device_dry_run: bool = False,
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
        self._device_transport = device_transport
        self._edgex_transport = edgex_transport
        self._device_command_mode = DeviceCommandMode(device_command_mode)
        self._device_dry_run = device_dry_run
        self._command_builder = CommandBuilder()
        self._command_result_collector = CommandResultCollector(
            repository=repo,
            device_commands_metric=_device_commands,
            device_results_metric=_device_command_results,
        )
        self._action_policy = ActionPolicy(alarm)
        self._expiry_policy = None
        self._physical_queue = None

    def _execute_device_command(
        self,
        *,
        command: DeviceCommand,
        execution_context=None,
    ) -> Dict:
        """공통 명령을 전송하고 저장소·메트릭용 결과를 만든다."""
        if self._expiry_policy is not None:
            reason = self._expiry_policy.check(execution_context, command.device)
            if reason:
                result = DeviceCommandResult(command.device, command.command_id, False,
                                             "expired" if reason == "expired" else "held", reason)
                return self._command_result_collector.collect(command, result)
        if self._device_dry_run:
            result = DeviceCommandResult(
                device=command.device,
                command_id=command.command_id,
                ok=True,
                status="simulated",
            )
            return self._command_result_collector.collect(command, result)
        if self._physical_queue is not None:
            result = self._physical_queue.enqueue(command, execution_context)
            return self._command_result_collector.collect(command, result)
        if self._device_command_mode is DeviceCommandMode.EDGEX:
            if self._publish_edgex_command is None:
                result = DeviceCommandResult(
                    device=command.device,
                    command_id=command.command_id,
                    ok=False,
                    status="failed",
                    error="EdgeX 전송기가 비활성화됨",
                )
            else:
                transport = self._edgex_transport or EdgeXCommandTransport(
                    publish=self._publish_edgex_command,
                    resolve_device_ids=self._resolve_edgex_device_ids or (lambda *_: []),
                    jetson_id=self._edgex_jetson_id,
                    topic_prefix=self._edgex_command_topic_prefix,
                    register_targets=self._command_result_collector.register_targets,
                )
                if self._expiry_policy is None:
                    result = transport.send(command)
                else:
                    # 앞 장치/대상 발행이 지연된 경우에도 각 실제 발행 직전에 같은 기한을 검사한다.
                    result = transport.send(command, before_publish=lambda device_id: self._expiry_policy.check(
                        execution_context, command.device, device_id))
        else:
            transport = self._device_transport or DirectDeviceCommandTransport(
                speaker=self._speaker,
                signboard=self._signboard,
                siren=self._siren,
                dry_run=self._device_dry_run,
            )
            result = transport.send(command)
        return self._command_result_collector.collect(command, result)

    def _publish_shadow_command(
        self,
        *,
        event_id: str,
        device: str,
        action: str,
        payload: Dict,
        command_id: str,
        camera_id: str,
        direct_status: str = "unknown",
    ) -> Optional[Dict]:
        """기존 직접 제어와 비교할 EdgeX Command를 생성해 발행한다."""
        if (
            self._device_command_mode is not DeviceCommandMode.SHADOW
            or self._publish_edgex_command is None
        ):
            return None
        try:
            transport = self._edgex_transport or EdgeXCommandTransport(
                publish=self._publish_edgex_command,
                resolve_device_ids=self._resolve_edgex_device_ids or (lambda *_: []),
                jetson_id=self._edgex_jetson_id,
                topic_prefix=self._edgex_command_topic_prefix,
            )
            result = transport.send(
                self._command_builder.build(
                    device=device,
                    action=action,
                    payload=payload,
                    command_id=command_id,
                    event_id=event_id,
                    camera_id=camera_id,
                )
            )
            if not result.ok:
                logger.warning(
                    "EdgeX shadow Command 발행 결과 실패: command_id=%s error=%s",
                    command_id,
                    result.error,
                )
            return {
                "mode": "shadow",
                "direct_status": direct_status,
                "edgex_publish_status": result.status,
                "comparison": (
                    # 전송 접수만으로 직접 실행과 장치 결과가 일치한다고 판단하지 않는다.
                    "awaiting_device_result" if result.status == "accepted" else "mismatch"
                ),
                **({"error": result.error} if result.error else {}),
            }
        except Exception as exc:
            logger.warning("EdgeX shadow Command 구성 실패: command_id=%s error=%s", command_id, exc)
            return {
                "mode": "shadow",
                "direct_status": direct_status,
                "edgex_publish_status": "failed",
                "comparison": "mismatch",
                "error": str(exc),
            }

    def _record_shadow_comparison(
        self,
        *,
        command_id: str,
        device: str,
        payload: Dict,
        comparison: Optional[Dict],
        direct_status: str,
    ) -> Optional[Dict]:
        """Shadow 발행 결과를 장치 결과와 명령 저장소에 함께 기록한다."""
        if comparison is None:
            return None
        payload_with_comparison = {
            **payload,
            "shadow_comparison": comparison,
        }
        self._repo.record_command(
            command_id,
            f"device/{device}",
            direct_status,
            payload_with_comparison,
        )
        return comparison

    def execute(self, topic: str, payload: Dict, *, execution_context=None) -> None:
        """단일 이벤트에 대한 장치 조치와 외부 전송을 수행한다."""
        if self._expiry_policy is not None:
            reason = self._expiry_policy.check(execution_context)
            if not reason and (execution_context["event_id"] != get_payload_event_id(payload)
                               or execution_context["camera_id"] != get_payload_camera_id(payload)):
                reason = "event_context_mismatch"
            if reason:
                decision = {"status": "expired" if reason == "expired" else "held", "reason": reason}
                self._repo.save(topic, {**payload, "action_decision": decision}, alarm_played=False, http_sent=False)
                self._publish_status("events/held", {"event_id": get_payload_event_id(payload), **decision})
                return
        camera_id = get_payload_camera_id(payload)
        event_type = get_payload_event_type(payload).lower()
        severity = get_payload_severity(payload).lower()
        display_message = get_payload_display_message(payload)
        tts_message = get_payload_tts_message(payload)
        force_reachability_refresh = self._alarm.is_demo_event(payload)

        alarm_played = False
        device_results = []
        # 영속 큐에서는 카메라 메모리 슬롯이 유효한 대기 경보를 버리지 않게 한다.
        should_execute = (self._alarm.should_alarm(topic, payload) if self._physical_queue is not None
                          else self._action_policy.should_execute(topic, payload))
        if should_execute:
            devices = self._resolve_devices(
                camera_id,
                force_refresh=force_reachability_refresh,
            )

            if self._alarm_device_enum.SPEAKER in devices:
                command_id = build_device_command_id(
                    get_payload_event_id(payload), "speaker"
                )
                speaker_command = self._command_builder.build_alarm_command(
                    device="speaker",
                    action="play",
                    event_type=event_type,
                    severity=severity,
                    camera_id=camera_id,
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    text=tts_message,
                )
                self._repo.record_command(command_id, "device/speaker", "sent", payload)
                speaker_result = self._execute_device_command(
                    command=speaker_command,
                    execution_context=execution_context,
                )
                if speaker_result["status"] == "acknowledged":
                    alarm_played = True
                device_results.append(speaker_result)
                shadow_comparison = self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="speaker",
                    action="play",
                    payload=speaker_command.payload,
                    direct_status=speaker_result["status"],
                )
                speaker_result["shadow_comparison"] = self._record_shadow_comparison(
                    command_id=command_id,
                    device="speaker",
                    payload=speaker_command.payload,
                    comparison=shadow_comparison,
                    direct_status=speaker_result["status"],
                )

            if self._alarm_device_enum.SIGNBOARD in devices:
                command_id = build_device_command_id(
                    get_payload_event_id(payload), "signboard"
                )
                signboard_text = display_message or self._build_display_text(
                    event_type,
                    severity,
                    camera_id,
                )
                signboard_command = self._command_builder.build_alarm_command(
                    device="signboard",
                    action="display",
                    event_type=event_type,
                    severity=severity,
                    camera_id=camera_id,
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    text=signboard_text,
                )
                self._repo.record_command(command_id, "device/signboard", "sent", payload)
                signboard_result = self._execute_device_command(
                    command=signboard_command,
                    execution_context=execution_context,
                )
                if signboard_result["status"] == "acknowledged":
                    alarm_played = True
                device_results.append(signboard_result)
                shadow_comparison = self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="signboard",
                    action="display",
                    payload=signboard_command.payload,
                    direct_status=signboard_result["status"],
                )
                signboard_result["shadow_comparison"] = self._record_shadow_comparison(
                    command_id=command_id,
                    device="signboard",
                    payload=signboard_command.payload,
                    comparison=shadow_comparison,
                    direct_status=signboard_result["status"],
                )

            if self._alarm_device_enum.SIREN in devices:
                command_id = build_device_command_id(
                    get_payload_event_id(payload), "siren"
                )
                siren_command = self._command_builder.build_alarm_command(
                    device="siren",
                    action="trigger",
                    event_type=event_type,
                    severity=severity,
                    camera_id=camera_id,
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                )
                self._repo.record_command(command_id, "device/siren", "sent", payload)
                siren_result = self._execute_device_command(
                    command=siren_command,
                    execution_context=execution_context,
                )
                if siren_result["status"] == "acknowledged":
                    alarm_played = True
                device_results.append(siren_result)
                shadow_comparison = self._publish_shadow_command(
                    event_id=get_payload_event_id(payload),
                    command_id=command_id,
                    camera_id=camera_id,
                    device="siren",
                    action="trigger",
                    payload=siren_command.payload,
                    direct_status=siren_result["status"],
                )
                siren_result["shadow_comparison"] = self._record_shadow_comparison(
                    command_id=command_id,
                    device="siren",
                    payload=siren_command.payload,
                    comparison=shadow_comparison,
                    direct_status=siren_result["status"],
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
                "event_id": get_payload_event_id(payload),
                **(
                    {"request_id": str(payload["request_id"])}
                    if payload.get("request_id")
                    else {}
                ),
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
