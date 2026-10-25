"""액션 레이어 (action-bridge)

역할:
  - Kuiper 룰 결과 MQTT 구독
  - 외부 이벤트 REST 수신
  - 사이트별 디바이스(스피커 / 전광판 / 경광등) 조치 실행
  - 다중 외부 플랫폼 HTTP 전송 (S-PARK_SP / D_HUB / CITY_SP)
  - 자동(AUTO) / 수동(MANUAL) 조치 모드 관리
  - SQLite 이벤트 저장

클래스 구성:
  ControlMode, AlarmDevice, SiteConfig — 도메인 모델
  _EventRepo       — SQLite CRUD          (ActionBridge 내부용)
  _SiteRegistry    — 사이트·수동큐 관리  (ActionBridge 내부용)
  _AlarmCoordinator — 알람 타이밍 제어   (ActionBridge 내부용)
  ActionBridge     — 오케스트레이터 (공개 API)

디바이스별 상세 로직은 devices/ 이하 파일에 위치한다.
HTTP 포워더 로직은 protocols/http.py 에 위치한다.
REST 수신 서버는 protocols/rest.py 의 RestEventReceiver 를 사용한다.
"""

import json
import logging
import signal
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_event_id,
    get_payload_event_type,
)
from ..config import ActionBridgeConfig
from ..devices.signboard import SignboardConfig, SignboardDevice, build_display_text
from ..devices.siren import SensorConfig, SirenDevice
from ..devices.speaker import SpeakerConfig, SpeakerDevice
from ..edgex.device_registry import DeviceRegistry
from ..protocols._mqtt_factory import create_mqtt_client
from ..protocols.http import HttpEventForwarder, HttpEventTarget
from ..protocols.mqtt_subscriber import MqttTopicSubscriber
from ..protocols.rest import RestEventReceiver
from ..time_utils import now_kst_iso
from . import _action_bridge_rest_queue as _rest_queue
from . import _device_reachability
from ._action_bridge_payloads import normalize_sensor_payload
from ._action_bridge_support import (
    AlarmDevice,
    ControlMode,
    SiteConfig,
    _ActionExecutor,
    _AlarmCoordinator,
    _EventRepo,
    _SiteRegistry,
)
from ._action_bridge_topics import (
    CMD_TOPIC_APPROVE,
    CMD_TOPIC_MODE,
    CMD_TOPIC_REJECT,
    EDGEX_RESULT_TOPIC,
    STATUS_TOPIC_PREFIX,
    default_alarm_topics,
    default_subscribe_topics,
)
from .action_command_components import DeviceResolver
from .action_device_queue import ActionDeviceQueue
from .action_expiry import ActionExpiryPolicy
from .action_mqtt_intake import ActionMqttIntake
from .cctv_metrics import (
    action_bridge_up,
    events_handled,
    events_handled_by_type,
    mqtt_events_received,
)
from .cctv_metrics import (
    pending_events as _metric_pending,
)
from .command_event_handler import CommandEventHandler
from .device_command_transport import (
    DeviceCommandMode,
    normalize_device_command_mode,
)
from .edgex_result_handler import EdgeXResultHandler
from .event_intake_processor import EventIntakeProcessor

logger = logging.getLogger(__name__)

_ACTION_DEFAULTS = ActionBridgeConfig()
_DEVICE_REACHABILITY_CACHE_SECONDS = (
    _device_reachability.DEVICE_REACHABILITY_CACHE_SECONDS
)
_check_tcp_reachable = _device_reachability.check_tcp_reachable
_device_status = _device_reachability.device_status
_REST_ACTION_QUEUE_MAX_SIZE = 1000


class ActionBridge:
    """룰 엔진 출력을 액션으로 변환하는 브리지.

    수신:  MQTT + REST HTTP
    발신:  SpeakerDevice / SignboardDevice / SirenDevice + HttpEventForwarder
    저장:  SQLite

    내부 헬퍼:
        _repo   (_EventRepo)        — DB CRUD
        _sites  (_SiteRegistry)     — 사이트·수동큐 관리
        _alarm  (_AlarmCoordinator) — 알람 타이밍 제어
    """

    _MQTT_MAX_ATTEMPTS: int = 30

    def __init__(
        self,
        mqtt_broker: str = _ACTION_DEFAULTS.mqtt_broker,
        mqtt_port: int = _ACTION_DEFAULTS.mqtt_port,
        subscribe_topics: Optional[Set[str]] = None,
        db_path: str = "/app/data/runtime/action_events.db",
        speaker_config: Optional[SpeakerConfig] = None,
        signboard_config: Optional[SignboardConfig] = None,
        siren_config: Optional[SensorConfig] = None,
        http_targets: Optional[List[HttpEventTarget]] = None,
        external_api_url: Optional[str] = None,  # 하위 호환 단일 URL
        alarm_topics: Optional[Set[str]] = None,
        alarm_cooldown_seconds: int = 10,
        default_mode: ControlMode = ControlMode.AUTO,
        initial_sites: Optional[List[SiteConfig]] = None,
        rest_enabled: bool = False,
        rest_host: str = _ACTION_DEFAULTS.rest_host,
        rest_port: int = _ACTION_DEFAULTS.rest_port,
        edgex_shadow_enabled: bool = False,
        edgex_command_mode: Optional[str] = None,
        edgex_jetson_id: str = "jetson-01",
        edgex_command_topic_prefix: str = "edgex/commands/cctv",
        edgex_device_registry_path: Optional[str] = None,
        edgex_allowed_devices: Optional[Set[str]] = None,
        device_dry_run: bool = False,
        command_response_timeout_seconds: float = 30.0,
        mqtt_intake_db: Optional[str] = None,
        mqtt_client_id: Optional[str] = None,
        mqtt_capture_only: bool = False,
        action_expiry_policy_path: Optional[str] = None,
        physical_queue_enabled: bool = False,
    ) -> None:
        """Action Layer와 장치·EdgeX shadow 설정을 초기화한다."""
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = int(mqtt_port)
        if mqtt_intake_db and not (mqtt_client_id and mqtt_client_id.strip()):
            raise ValueError("영속 MQTT 수신에는 고정 mqtt_client_id가 필요합니다")
        if mqtt_capture_only and not mqtt_intake_db:
            raise ValueError("저장 전용 수신에는 영속 수신 DB가 필요합니다")
        if action_expiry_policy_path and not mqtt_intake_db:
            raise ValueError("만료·안전 재개에는 영속 수신 DB가 필요합니다")
        if physical_queue_enabled and not action_expiry_policy_path:
            raise ValueError("물리 장치 큐에는 명시적인 만료 정책이 필요합니다")
        self.subscribe_topics = subscribe_topics or default_subscribe_topics()
        requested_command_mode = edgex_command_mode
        if requested_command_mode is None:
            requested_command_mode = "shadow" if edgex_shadow_enabled else "direct"
        self._edgex_command_mode = normalize_device_command_mode(requested_command_mode)
        self._edgex_shadow_enabled = self._edgex_command_mode is DeviceCommandMode.SHADOW
        self._edgex_jetson_id = edgex_jetson_id
        self._edgex_command_topic_prefix = edgex_command_topic_prefix
        self._edgex_allowed_devices = {
            AlarmDevice(str(device).strip().lower())
            for device in (edgex_allowed_devices or set())
            if str(device).strip()
        }
        self._edgex_device_registry = (
            DeviceRegistry.from_file(edgex_device_registry_path)
            if edgex_device_registry_path
            else None
        )
        self._device_dry_run = device_dry_run

        self._repo = _EventRepo(Path(db_path), response_timeout_seconds=command_response_timeout_seconds)
        self._sites = _SiteRegistry(
            default_mode=default_mode,
            initial_sites=initial_sites,
        )
        self._alarm = _AlarmCoordinator(
            alarm_topics=alarm_topics or default_alarm_topics(),
            alarm_cooldown_seconds=alarm_cooldown_seconds,
        )
        self._event_intake_processor = EventIntakeProcessor(
            site_registry=self._sites,
            manual_mode=ControlMode.MANUAL,
        )
        self._command_event_handler = CommandEventHandler(
            repository=self._repo,
            publish_status=self._publish_status,
            set_mode=self.set_mode,
            approve_event=self.approve_event,
            reject_event=self.reject_event,
            control_mode_factory=ControlMode,
            mode_topic=CMD_TOPIC_MODE,
            approve_topic=CMD_TOPIC_APPROVE,
            reject_topic=CMD_TOPIC_REJECT,
        )
        self._edgex_result_handler = EdgeXResultHandler(
            repository=self._repo,
            publish_status=self._publish_status,
        )

        self._speaker = SpeakerDevice(speaker_config or SpeakerConfig())
        signboard_settings = signboard_config or SignboardConfig()
        # EdgeX가 출력할 때 로컬 유휴 표시가 원격 경보를 덮지 않도록 한다.
        if self._edgex_command_mode is DeviceCommandMode.EDGEX:
            signboard_settings = replace(signboard_settings, idle_enabled=False)
        self._signboard = SignboardDevice(signboard_settings)
        self._siren = SirenDevice(siren_config or SensorConfig())
        self._device_reachability_cache: Dict[str, Tuple[float, bool]] = {}
        self._device_resolver = DeviceResolver(
            resolve_candidates=self._sites.resolve_alarm_devices,
            is_available=self._device_is_available,
            command_mode=self._edgex_command_mode,
            allowed_devices=self._edgex_allowed_devices,
        )

        targets: List[HttpEventTarget] = list(http_targets or [])
        if external_api_url and not any(t.url == external_api_url for t in targets):
            targets.append(HttpEventTarget(name="default", url=external_api_url))
        self._forwarder = HttpEventForwarder(targets=targets)
        self._executor = _ActionExecutor(
            repo=self._repo,
            alarm=self._alarm,
            forwarder=self._forwarder,
            speaker=self._speaker,
            signboard=self._signboard,
            siren=self._siren,
            resolve_devices=self._resolve_devices,
            publish_status=self._publish_status,
            publish_edgex_command=self._publish_edgex_command
            if self._edgex_command_mode is not DeviceCommandMode.DIRECT
            else None,
            device_command_mode=self._edgex_command_mode,
            device_dry_run=self._device_dry_run,
            resolve_edgex_device_ids=self._resolve_edgex_device_ids,
            edgex_jetson_id=self._edgex_jetson_id,
            edgex_command_topic_prefix=self._edgex_command_topic_prefix,
            build_display_text=build_display_text,
            alarm_device_enum=AlarmDevice,
        )

        self._rest_receiver: Optional[RestEventReceiver] = None
        if rest_enabled:
            self._rest_receiver = RestEventReceiver(
                host=rest_host, port=rest_port, action_layer=self
            )

        if mqtt_intake_db and self._edgex_command_mode is not DeviceCommandMode.EDGEX:
            raise ValueError("영속 수신은 동일 요청 ID를 전달하는 EdgeX 장치 모드에서만 사용합니다")
        self._mqtt_intake = (ActionMqttIntake(
            mqtt_intake_db, lambda topic, raw, **kwargs: self._process_mqtt_payload(topic, raw, **kwargs),
            capture_only=mqtt_capture_only,
            rest_handler=lambda topic, raw, **kwargs: self._process_rest_payload(topic, raw, **kwargs),
        ) if mqtt_intake_db else None)
        if action_expiry_policy_path:
            from ..config.event_type_map import event_type_map
            # 승인되지 않은 TTL은 파일의 빈 규칙으로 보류하고 운영 기본 경로에는 자동 적용하지 않는다.
            expiry = ActionExpiryPolicy(
                self._mqtt_intake.store, action_expiry_policy_path,
                resolve_targets=self._expiry_targets,
                resolve_mode=lambda camera_id: self._sites.resolve_mode(camera_id)[0].value,
                priority_policy=event_type_map.routing_snapshot if physical_queue_enabled else None,
            )
            self._mqtt_intake.expiry_policy = expiry
            self._executor._expiry_policy = expiry
            if physical_queue_enabled:
                queue = ActionDeviceQueue(
                    expiry=expiry, repository=self._repo,
                    publish=lambda topic, payload: self._executor._publish_edgex_command(topic, payload),
                    jetson_id=self._edgex_jetson_id, topic_prefix=self._edgex_command_topic_prefix,
                )
                self._mqtt_intake.device_queue = queue
                self._executor._physical_queue = queue
                self._edgex_result_handler.on_result = queue.observe_result
        self._mqtt_client_id = mqtt_client_id
        self._mqtt_client: Optional[mqtt.Client | MqttTopicSubscriber] = None
        self._running = False
        self._rest_action_queue = _rest_queue.new_rest_action_queue(
            _REST_ACTION_QUEUE_MAX_SIZE
        )
        self._rest_action_worker_stop = _rest_queue.new_rest_action_worker_stop()
        self._rest_action_worker: Optional[object] = None

    def _resolve_edgex_device_ids(self, device_type: str, camera_id: str) -> List[str]:
        """레지스트리에서 이벤트 카메라에 연결된 물리 장치 ID를 조회한다."""
        if self._edgex_device_registry is None:
            return []
        return [
            target.device_id
            for target in self._edgex_device_registry.resolve(
                device_type,
                camera_id=camera_id,
            )
        ]

    @property
    def default_mode(self) -> ControlMode:
        return self._sites.default_mode

    @default_mode.setter
    def default_mode(self, value: ControlMode) -> None:
        self._sites.default_mode = value

    def add_site(self, site: SiteConfig) -> None:
        self._sites.add(site)

    def remove_site(self, site_id: str) -> bool:
        return self._sites.remove(site_id)

    def list_sites(self) -> List[Dict]:
        return self._sites.list_all()

    def list_recent_events(self, limit: int = 20) -> List[Dict]:
        """최근 Action Layer 처리 이력을 반환한다."""
        return self._repo.list_recent(limit=limit)

    def list_commands(self, limit: int = 50) -> List[Dict]:
        """재시작 후에도 남아 있는 출력 제어 명령 상태를 반환한다."""
        self._expire_command_responses()
        return self._repo.list_commands(limit=limit)

    def _expire_command_responses(self) -> None:
        """확정 저장한 미응답 상태만 관제에 알리고 재발행하지 않는다."""
        for result in self._repo.expire_command_responses():
            if self._mqtt_intake is not None and self._mqtt_intake.device_queue is not None:
                self._mqtt_intake.device_queue.observe_result(result)
            self._publish_status("devices/result", result)

    def list_output_devices(self) -> List[Dict]:
        """출력 디바이스 설정 상태를 UI/API용으로 반환한다."""
        return [
            self._output_device_status(
                "speaker", "스피커", self._speaker.config, "HTTP Digest / InterM"
            ),
            self._output_device_status(
                "signboard", "전광판", self._signboard.config, "TCP Socket / Dabit"
            ),
            self._output_device_status(
                "siren", "경광등", self._siren.config, "HTTP Digest / InterM"
            ),
        ]

    def _output_device_status(
        self,
        device: str,
        label: str,
        config,
        protocol: str,
    ) -> Dict:
        configured = config.is_configured
        host = config.host
        port = config.port
        reachable = _check_tcp_reachable(host, port) if configured else None
        return {
            "device": device,
            "label": label,
            "configured": configured,
            "reachable": reachable,
            "status": _device_status(configured, reachable),
            "host": host or None,
            "port": port,
            "protocol": protocol,
        }

    def set_mode(self, mode: ControlMode, site_id: Optional[str] = None) -> None:
        self._sites.set_mode(mode, site_id=site_id)

    def get_default_mode_settings(self) -> Dict:
        """전역 기본 모드와 기본 조치 상세 설정을 반환한다."""
        return self._sites.default_settings()

    def set_default_action_settings(
        self,
        *,
        alarm_devices: Optional[List[AlarmDevice]] = None,
        confidence_threshold: Optional[float] = None,
        display_message: Optional[str] = None,
        tts_message: Optional[str] = None,
    ) -> None:
        self._sites.set_default_action_settings(
            alarm_devices=alarm_devices,
            confidence_threshold=confidence_threshold,
            display_message=display_message,
            tts_message=tts_message,
        )

    def add_site_from_dict(self, data: Dict) -> str:
        """dict에서 사이트를 생성하여 추가하고 site_id를 반환한다."""
        site = SiteConfig.from_dict(data)
        self._sites.add(site)
        return site.site_id

    def set_mode_str(self, mode_str: str, site_id: Optional[str] = None) -> None:
        """문자열 값으로 모드를 설정한다. 잘못된 값이면 ValueError를 전파한다."""
        self._sites.set_mode(ControlMode(mode_str), site_id=site_id)

    def get_pending_events(self) -> List[Dict]:
        return self._sites.list_pending()

    def _publish_status(self, suffix: str, payload: Dict) -> None:
        """Action Layer 상태/명령 결과를 MQTT로 발행한다."""
        if not self._mqtt_client:
            return
        try:
            topic = f"{STATUS_TOPIC_PREFIX}/{suffix.lstrip('/')}"
            body = {
                "timestamp": now_kst_iso(),
                **payload,
            }
            self._mqtt_client.publish(
                topic, json.dumps(body, ensure_ascii=False), qos=0
            )
        except Exception as exc:
            logger.warning("Action status 발행 실패: %s", exc)

    def _publish_edgex_command(self, topic: str, payload: Dict) -> bool:
        """비교용 EdgeX Command를 MQTT로 발행하고 발행 가능 여부를 반환한다."""
        if not self._mqtt_client:
            return False
        try:
            result = self._mqtt_client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                qos=1,
            )
            return getattr(result, "rc", 0) == mqtt.MQTT_ERR_SUCCESS
        except Exception as exc:
            logger.warning("EdgeX shadow Command 발행 실패: topic=%s error=%s", topic, exc)
            return False

    def approve_event(self, event_id: str) -> Tuple[bool, str]:
        """대기 이벤트를 승인하여 즉시 실행한다."""
        # 영속 승인 계약 전에는 REST가 저장 전용/만료 보류를 우회하지 못하게 한다.
        if self._mqtt_intake is not None and (self._mqtt_intake.capture_only or self._mqtt_intake.expiry_policy is not None):
            return False, "영속 수신의 보류 해제·승인 절차가 필요합니다"
        info = self._sites.pop_pending(event_id)
        if info is None:
            return False, f"이벤트 없음: {event_id}"
        logger.info("[수동 승인] event_id=%s", event_id)
        self._execute_action(info["topic"], info["payload"])
        self._publish_status(
            "events/approved",
            {
                "event_id": event_id,
                "camera_id": get_payload_camera_id(info["payload"]),
                "site_id": info.get("site_id"),
                "status": "approved",
            },
        )
        return True, f"승인 완료: {event_id}"

    def reject_event(self, event_id: str) -> Tuple[bool, str]:
        """대기 이벤트를 거부(제거)한다."""
        info = self._sites.pop_pending(event_id)
        if info is None:
            return False, f"이벤트 없음: {event_id}"
        logger.info("[수동 거부] event_id=%s", event_id)
        self._repo.save(
            info["topic"], info["payload"], alarm_played=False, http_sent=False
        )
        self._publish_status(
            "events/rejected",
            {
                "event_id": event_id,
                "camera_id": get_payload_camera_id(info["payload"]),
                "site_id": info.get("site_id"),
                "status": "rejected",
            },
        )
        return True, f"거부 완료: {event_id}"

    def _resolve_devices(
        self,
        camera_id: str,
        *,
        force_refresh: bool = False,
    ) -> List[AlarmDevice]:
        return self._device_resolver.resolve(
            camera_id,
            force_refresh=force_refresh,
        )

    def _device_is_available(
        self,
        device: AlarmDevice,
        *,
        force_refresh: bool = False,
    ) -> bool:
        """설정되지 않았거나 네트워크에 닿지 않는 출력 장치는 실행 대상에서 제외한다."""
        config: Any
        if device == AlarmDevice.SPEAKER:
            config = self._speaker.config
        elif device == AlarmDevice.SIGNBOARD:
            config = self._signboard.config
        elif device == AlarmDevice.SIREN:
            config = self._siren.config
        else:
            return False

        if not config.is_configured:
            return False

        # Dabit 전광판은 짧은 간격의 TCP 연결을 일시 거부할 수 있다. 사전 probe로
        # 이벤트를 버리지 않고 SignboardDevice의 실제 전송 재시도에 맡긴다.
        if device == AlarmDevice.SIGNBOARD:
            return True
        return self._device_reachable_cached(
            device.value,
            str(config.host),
            int(config.port),
            force_refresh=force_refresh,
        )

    def _device_reachable_cached(
        self,
        key: str,
        host: str,
        port: int,
        *,
        force_refresh: bool = False,
    ) -> bool:
        now = time.time()
        cached = self._device_reachability_cache.get(key)
        if (
            not force_refresh
            and cached
            and now - cached[0] < _DEVICE_REACHABILITY_CACHE_SECONDS
        ):
            return cached[1]

        reachable = _check_tcp_reachable(host, port)
        self._device_reachability_cache[key] = (now, reachable)
        if not reachable:
            logger.warning(
                "출력 장치 연결 불가 - 알람 실행에서 제외: device=%s host=%s port=%s",
                key,
                host,
                port,
            )
        return reachable

    def _expiry_targets(self, camera_id: str) -> Dict:
        """수신 ACK 전에 네트워크 접근 없이 물리 대상 설정을 고정한다."""
        return {device.value: sorted(self._resolve_edgex_device_ids(device.value, camera_id))
                for device in self._sites.resolve_alarm_devices(camera_id)
                if not self._edgex_allowed_devices or device in self._edgex_allowed_devices}

    def _handle_event(self, payload: Dict, topic: str = "rest/inbound", *, execution_context=None) -> None:
        """수신된 이벤트를 처리한다 (MQTT·REST 공통 경로).

        - AUTO 모드: 즉시 알람/HTTP 실행
        - MANUAL 모드: 승인 대기 큐에 추가
        """
        prepared_event = self._event_intake_processor.prepare(payload)
        payload = prepared_event.payload
        camera_id = prepared_event.camera_id
        mode = prepared_event.mode
        site_id = prepared_event.site_id
        event_type = prepared_event.event_type
        events_handled.labels(mode=mode.value).inc()
        events_handled_by_type.labels(event_type=event_type, mode=mode.value).inc()

        if prepared_event.is_filtered:
            confidence = prepared_event.confidence
            confidence_threshold = prepared_event.confidence_threshold
            if confidence is not None and confidence_threshold is not None:
                logger.info(
                    "신뢰도 임계값 미달 - 조치 스킵: camera=%s confidence=%.3f threshold=%.3f",
                    camera_id,
                    confidence,
                    confidence_threshold,
                )
                self._repo.save(topic, payload, alarm_played=False, http_sent=False)
                self._publish_status(
                    "events/filtered",
                    {
                        "event_id": get_payload_event_id(payload),
                        **(
                            {"request_id": str(payload["request_id"])}
                            if payload.get("request_id")
                            else {}
                        ),
                        "camera_id": camera_id,
                        "site_id": site_id,
                        "event_type": get_payload_event_type(payload),
                        "confidence": confidence,
                        "threshold": confidence_threshold,
                        "status": "filtered",
                    },
                )
                return

        if mode == ControlMode.MANUAL:
            event_id = str(uuid.uuid4())
            self._sites.push_pending(event_id, topic, payload, site_id)
            _metric_pending.set(len(self._sites.list_pending()))
            self._repo.save(topic, payload, alarm_played=False, http_sent=False)
            self._publish_status(
                "events/pending",
                {
                    "event_id": event_id,
                    **(
                        {"request_id": str(payload["request_id"])}
                        if payload.get("request_id")
                        else {}
                    ),
                    "camera_id": camera_id,
                    "site_id": site_id,
                    "event_type": get_payload_event_type(payload),
                    "status": "pending",
                },
            )
        else:
            if execution_context is None:
                self._execute_action(topic, payload)
            else:
                self._execute_action(topic, payload, execution_context=execution_context)

    def enqueue_rest_event(self, payload: Dict, topic: str = "rest/inbound") -> bool:
        """직접 호출도 영속 수신을 우회하지 않으며 기존 비영속 큐 계약은 유지한다."""
        if self._mqtt_intake is not None:
            self.accept_rest_event(json.dumps({**payload, "topic": topic}, allow_nan=False).encode())
            return True
        return _rest_queue.enqueue_rest_event(self, payload, topic=topic)

    def accept_rest_event(self, raw: bytes) -> Dict:
        """접수 완료만 반환하며 실제 장치 실행·완료를 HTTP 성공으로 오인하지 않는다."""
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("event_id"), str) or not payload["event_id"].strip():
            raise ValueError("event_id가 있는 JSON 객체가 필요합니다")
        topic = payload.pop("topic", "rest/inbound")
        if topic is None or topic == "":
            topic = "rest/inbound"
        # 이벤트 API로 MQTT 제어/완료 응답을 위조해 만료·점유 검사를 우회하지 못하게 한다.
        if (not isinstance(topic, str) or not topic.strip()
                or topic.startswith(("cctv/commands/", "edgex/results/"))):
            raise ValueError("REST 이벤트의 topic이 유효하지 않습니다")
        normalized = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if self._mqtt_intake is None:
            raise RuntimeError("영속 수신이 설정되지 않았습니다")
        ingest_id = self._mqtt_intake.accept_rest(topic, raw, normalized)
        return {"status": "accepted", "queued": True, "durable": True,
                "ingest_id": ingest_id, "event_id": payload["event_id"]}

    def get_rest_receipt(self, ingest_id: str) -> Dict:
        """원본 내용은 노출하지 않고 REST 입력의 처리/보류 상태만 조회한다."""
        if self._mqtt_intake is None:
            raise KeyError(ingest_id)
        receipt = self._mqtt_intake.store.get(ingest_id)
        if receipt["source_namespace"] != "action_rest":
            raise KeyError(ingest_id)
        return {"ingest_id": ingest_id, "received_at": receipt["received_at"],
                "stage": receipt["stage"], "reason": receipt["last_error"]}

    def _process_rest_payload(self, topic: str, raw: bytes, **kwargs) -> None:
        """REST는 이벤트만 처리하고 같은 문자열의 MQTT 제어 경로로 보내지 않는다."""
        payload = json.loads(raw)
        payload.pop("topic", None)
        self._handle_event(payload, topic=topic, **kwargs)

    def _start_rest_action_worker(self) -> None:
        """REST action worker를 필요할 때 시작한다."""
        _rest_queue.start_rest_action_worker(self)

    def _rest_action_worker_loop(self) -> None:
        """REST 이벤트 큐를 소비해 실제 액션을 수행한다."""
        _rest_queue.rest_action_worker_loop(self)

    def _stop_rest_action_worker(self) -> None:
        """REST action worker를 종료한다."""
        _rest_queue.stop_rest_action_worker(self)

    def _execute_action(self, topic: str, payload: Dict, *, execution_context=None) -> None:
        """디바이스 조치 + HTTP 전송을 즉시 실행한다."""
        self._executor._resolve_devices = self._resolve_devices
        self._executor.execute(topic, payload, execution_context=execution_context)

    # ------------------------------------------------------------------
    # MQTT 콜백
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: dict,
        rc: int,
        *args: object,
    ) -> None:
        if rc != 0:
            logger.error("Action Layer MQTT 연결 실패 (rc=%d)", rc)
            return
        logger.info(
            "Action Layer MQTT 연결 성공: %s:%d", self.mqtt_broker, self.mqtt_port
        )
        all_topics = [
            *((t, 0) for t in self.subscribe_topics),
            *((t, 1) for t in (CMD_TOPIC_MODE, CMD_TOPIC_APPROVE, CMD_TOPIC_REJECT)),
            (EDGEX_RESULT_TOPIC, 1),
        ]
        for topic, qos in all_topics:
            client.subscribe(topic, qos=qos)
            logger.info("구독: %s (qos=%d)", topic, qos)

    def _dispatch_command(self, topic: str, payload: Dict) -> None:
        """MQTT 명령 토픽을 처리한다."""
        self._command_event_handler.dispatch(topic, payload)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            self._process_mqtt_payload(msg.topic, msg.payload)
        except json.JSONDecodeError as exc:
            logger.error("JSON 파싱 실패: %s", exc)
        except Exception as exc:
            logger.error("Action 처리 오류: %s", exc, exc_info=True)

    def _process_mqtt_payload(self, topic: str, raw: bytes, *, execution_context=None) -> None:
        """영속 작업자가 오류를 보류할 수 있도록 실제 처리의 예외를 전달한다."""
        payload = json.loads(raw.decode("utf-8"))
        if topic in (CMD_TOPIC_MODE, CMD_TOPIC_APPROVE, CMD_TOPIC_REJECT):
            self._dispatch_command(topic, payload)
        elif topic.startswith("edgex/results/cctv/"):
            self._handle_edgex_result(topic, payload)
        else:
            payloads = payload if isinstance(payload, list) else [payload]
            topic_prefix = "/".join(topic.split("/")[:2])
            mqtt_events_received.labels(topic_prefix=topic_prefix).inc()
            for single in payloads:
                if not isinstance(single, dict):
                    continue
                if topic.startswith("aiot/rules/sensor/"):
                    single = self._normalize_sensor_payload(topic, single)
                if execution_context is None:
                    self._handle_event(single, topic=topic)
                else:
                    self._handle_event(single, topic=topic, execution_context=execution_context)

    def _handle_edgex_result(self, topic: str, payload: Dict) -> None:
        """EdgeX 장치 결과를 Action Layer 명령 이력과 상태 토픽에 반영한다."""
        self._edgex_result_handler.handle(topic, payload)

    @staticmethod
    def _normalize_sensor_payload(topic: str, payload: Dict) -> Dict:
        return normalize_sensor_payload(topic, payload)

    def start(self) -> None:
        """서비스를 시작한다."""
        self._repo.init()
        # 연결/구독 실패도 같은 종료 경로를 거쳐 수신 작업자 소유권을 반납한다.
        try:
            capture_only = self._mqtt_intake is not None and self._mqtt_intake.capture_only
            if not capture_only:
                self._forwarder.start()

            if self._mqtt_intake is not None:
                self._mqtt_intake.start()
                self._mqtt_client = MqttTopicSubscriber(
                    broker=self.mqtt_broker, port=self.mqtt_port,
                    topics=sorted(self.subscribe_topics | {
                        CMD_TOPIC_MODE, CMD_TOPIC_APPROVE, CMD_TOPIC_REJECT, EDGEX_RESULT_TOPIC,
                    }),
                    client_id=self._mqtt_client_id, durable=True,
                    message_handler=self._mqtt_intake.accept,
                )
            else:
                self._mqtt_client = create_mqtt_client("cctv-action-layer")
                self._mqtt_client.on_connect = self._on_connect
                self._mqtt_client.on_message = self._on_message

            for attempt in range(1, self._MQTT_MAX_ATTEMPTS + 1):
                try:
                    if self._mqtt_intake is not None:
                        if not self._mqtt_client.connect():
                            raise ConnectionError("영속 MQTT 구독 확인 대기")
                    else:
                        self._mqtt_client.connect(
                            self.mqtt_broker, self.mqtt_port, keepalive=60
                        )
                    logger.info("MQTT 연결 성공: %s:%d", self.mqtt_broker, self.mqtt_port)
                    break
                except (ConnectionRefusedError, OSError) as exc:
                    if attempt >= self._MQTT_MAX_ATTEMPTS:
                        logger.error("MQTT 연결 실패 - 포기 (%d회 시도): %s", attempt, exc)
                        raise
                    logger.warning(
                        "MQTT 연결 실패 (시도 %d/%d): %s - 5초 후 재시도...",
                        attempt,
                        self._MQTT_MAX_ATTEMPTS,
                        exc,
                    )
                    time.sleep(5)

            if self._mqtt_intake is None:
                self._mqtt_client.loop_start()

            if not capture_only:
                self._start_rest_action_worker()

            if self._rest_receiver:
                self._rest_receiver.start()

            self._running = True
            action_bridge_up.set(1)
            logger.info("Speaker-Bridge Action Layer 실행 중 (Ctrl+C 종료)")

            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            while self._running:
                if self._mqtt_intake is not None:
                    self._mqtt_client.reconnect_if_needed()
                self._expire_command_responses()
                time.sleep(1.0)
        finally:
            self.stop()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("종료 신호 수신 (signum=%d)", signum)
        self._running = False

    def stop(self) -> None:
        """서비스를 안전하게 종료한다."""
        self._running = False
        action_bridge_up.set(0)
        logger.info("Speaker-Bridge 종료 중...")
        if self._rest_receiver:
            self._rest_receiver.stop()
        if self._mqtt_intake is not None:
            # 입력을 먼저 닫고 실행 소유자가 끝난 뒤 기존 출력 자원을 해제한다.
            if self._mqtt_client:
                self._mqtt_client.disconnect()
            self._mqtt_intake.stop()
        self._stop_rest_action_worker()
        self._forwarder.stop()
        if self._mqtt_client and self._mqtt_intake is None:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
        logger.info("Speaker-Bridge 종료 완료")
