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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt

from ..canonical_event import (
    canonicalize_event_payload,
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_event_type,
)
from ..config import ActionBridgeConfig
from ..devices.signboard import SignboardConfig, SignboardDevice, build_display_text
from ..devices.siren import SensorConfig, SirenDevice
from ..devices.speaker import SpeakerConfig, SpeakerDevice
from ..protocols._mqtt_factory import create_mqtt_client
from ..protocols.http import HttpEventForwarder, HttpEventTarget
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
    STATUS_TOPIC_PREFIX,
    default_alarm_topics,
    default_subscribe_topics,
)
from .cctv_metrics import (
    action_bridge_up,
    events_handled,
    mqtt_events_received,
)
from .cctv_metrics import (
    pending_events as _metric_pending,
)

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
    ) -> None:
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = int(mqtt_port)
        self.subscribe_topics = subscribe_topics or default_subscribe_topics()

        self._repo = _EventRepo(Path(db_path))
        self._sites = _SiteRegistry(
            default_mode=default_mode,
            initial_sites=initial_sites,
        )
        self._alarm = _AlarmCoordinator(
            alarm_topics=alarm_topics or default_alarm_topics(),
            alarm_cooldown_seconds=alarm_cooldown_seconds,
        )

        self._speaker = SpeakerDevice(speaker_config or SpeakerConfig())
        self._signboard = SignboardDevice(signboard_config or SignboardConfig())
        self._siren = SirenDevice(siren_config or SensorConfig())
        self._device_reachability_cache: Dict[str, Tuple[float, bool]] = {}

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
            build_display_text=build_display_text,
            alarm_device_enum=AlarmDevice,
        )

        self._rest_receiver: Optional[RestEventReceiver] = None
        if rest_enabled:
            self._rest_receiver = RestEventReceiver(
                host=rest_host, port=rest_port, action_layer=self
            )

        self._mqtt_client: Optional[mqtt.Client] = None
        self._running = False
        self._rest_action_queue = _rest_queue.new_rest_action_queue(
            _REST_ACTION_QUEUE_MAX_SIZE
        )
        self._rest_action_worker_stop = _rest_queue.new_rest_action_worker_stop()
        self._rest_action_worker: Optional[object] = None

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

    def approve_event(self, event_id: str) -> Tuple[bool, str]:
        """대기 이벤트를 승인하여 즉시 실행한다."""
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
        candidates = self._sites.resolve_alarm_devices(camera_id)
        return [
            device
            for device in candidates
            if self._device_is_available(device, force_refresh=force_refresh)
        ]

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

    def _handle_event(self, payload: Dict, topic: str = "rest/inbound") -> None:
        """수신된 이벤트를 처리한다 (MQTT·REST 공통 경로).

        - AUTO 모드: 즉시 알람/HTTP 실행
        - MANUAL 모드: 승인 대기 큐에 추가
        """
        payload = canonicalize_event_payload(
            payload, source="action-layer", source_type="action"
        )
        camera_id = get_payload_camera_id(payload)
        mode, site_id = self._sites.resolve_mode(camera_id)
        action_settings = self._sites.resolve_action_settings(camera_id)

        events_handled.labels(mode=mode.value).inc()

        confidence_threshold = action_settings.get("confidence_threshold")
        if confidence_threshold is not None:
            confidence = get_payload_confidence(payload)
            if confidence is not None and confidence < confidence_threshold:
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
                        "camera_id": camera_id,
                        "site_id": site_id,
                        "event_type": get_payload_event_type(payload),
                        "confidence": confidence,
                        "threshold": confidence_threshold,
                        "status": "filtered",
                    },
                )
                return

        display_message = action_settings.get("display_message") or ""
        tts_message = action_settings.get("tts_message") or ""
        if display_message or tts_message:
            payload = dict(payload)
            event_payload = dict(payload.get("event") or {})
            if display_message:
                event_payload["display_message"] = display_message
                event_payload.setdefault("message", display_message)
            if tts_message:
                event_payload["tts_message"] = tts_message
            payload["event"] = event_payload

        if mode == ControlMode.MANUAL:
            event_id = str(uuid.uuid4())
            self._sites.push_pending(event_id, topic, payload, site_id)
            _metric_pending.set(len(self._sites.list_pending()))
            self._repo.save(topic, payload, alarm_played=False, http_sent=False)
            self._publish_status(
                "events/pending",
                {
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "site_id": site_id,
                    "event_type": get_payload_event_type(payload),
                    "status": "pending",
                },
            )
        else:
            self._execute_action(topic, payload)

    def enqueue_rest_event(self, payload: Dict, topic: str = "rest/inbound") -> bool:
        """REST 수신 이벤트를 백그라운드 큐에 넣는다.

        HTTP 요청 스레드가 실제 장비 제어 timeout에 묶이지 않도록 REST 경로만
        비동기로 분리한다. MQTT 경로는 기존 동기 동작을 유지한다.
        """
        return _rest_queue.enqueue_rest_event(self, payload, topic=topic)

    def _start_rest_action_worker(self) -> None:
        """REST action worker를 필요할 때 시작한다."""
        _rest_queue.start_rest_action_worker(self)

    def _rest_action_worker_loop(self) -> None:
        """REST 이벤트 큐를 소비해 실제 액션을 수행한다."""
        _rest_queue.rest_action_worker_loop(self)

    def _stop_rest_action_worker(self) -> None:
        """REST action worker를 종료한다."""
        _rest_queue.stop_rest_action_worker(self)

    def _execute_action(self, topic: str, payload: Dict) -> None:
        """디바이스 조치 + HTTP 전송을 즉시 실행한다."""
        self._executor._resolve_devices = self._resolve_devices
        self._executor.execute(topic, payload)

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
        ]
        for topic, qos in all_topics:
            client.subscribe(topic, qos=qos)
            logger.info("구독: %s (qos=%d)", topic, qos)

    def _dispatch_command(self, topic: str, payload: Dict) -> None:
        """MQTT 명령 토픽을 처리한다."""
        command_id = str(payload.get("command_id", uuid.uuid4()))
        command_status = "ignored"
        message = ""

        if topic == CMD_TOPIC_MODE:
            mode_val = payload.get("mode")
            if mode_val:
                try:
                    self.set_mode(ControlMode(mode_val), site_id=payload.get("site_id"))
                    command_status = "success"
                    message = f"mode set to {mode_val}"
                except ValueError:
                    logger.warning("MQTT mode 명령 오류: 알 수 없는 값 %r", mode_val)
                    command_status = "error"
                    message = f"invalid mode: {mode_val!r}"

        elif topic == CMD_TOPIC_APPROVE:
            event_id = payload.get("event_id")
            if event_id:
                ok, msg = self.approve_event(event_id)
                logger.info("MQTT approve: %s", msg)
                command_status = "success" if ok else "error"
                message = msg

        elif topic == CMD_TOPIC_REJECT:
            event_id = payload.get("event_id")
            if event_id:
                ok, msg = self.reject_event(event_id)
                logger.info("MQTT reject: %s", msg)
                command_status = "success" if ok else "error"
                message = msg

        self._publish_status(
            "commands/result",
            {
                "command_id": command_id,
                "topic": topic,
                "status": command_status,
                "message": message,
                "site_id": payload.get("site_id"),
                "event_id": payload.get("event_id"),
            },
        )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))
            if topic in (CMD_TOPIC_MODE, CMD_TOPIC_APPROVE, CMD_TOPIC_REJECT):
                self._dispatch_command(topic, payload)
            else:
                payloads = payload if isinstance(payload, list) else [payload]
                topic_prefix = "/".join(topic.split("/")[:2])
                mqtt_events_received.labels(topic_prefix=topic_prefix).inc()
                for single in payloads:
                    if not isinstance(single, dict):
                        continue
                    if topic.startswith("aiot/rules/sensor/"):
                        single = self._normalize_sensor_payload(topic, single)
                    self._handle_event(single, topic=topic)
        except json.JSONDecodeError as exc:
            logger.error("JSON 파싱 실패: %s", exc)
        except Exception as exc:
            logger.error("Action 처리 오류: %s", exc, exc_info=True)

    @staticmethod
    def _normalize_sensor_payload(topic: str, payload: Dict) -> Dict:
        return normalize_sensor_payload(topic, payload)

    def start(self) -> None:
        """서비스를 시작한다."""
        self._repo.init()
        self._forwarder.start()

        self._mqtt_client = create_mqtt_client("cctv-action-layer")
        self._mqtt_client.on_connect = self._on_connect
        self._mqtt_client.on_message = self._on_message

        for attempt in range(1, self._MQTT_MAX_ATTEMPTS + 1):
            try:
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

        self._mqtt_client.loop_start()

        self._start_rest_action_worker()

        if self._rest_receiver:
            self._rest_receiver.start()

        self._running = True
        action_bridge_up.set(1)
        logger.info("Speaker-Bridge Action Layer 실행 중 (Ctrl+C 종료)")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            while self._running:
                time.sleep(1.0)
        finally:
            self.stop()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("종료 신호 수신 (signum=%d)", signum)
        self._running = False

    def stop(self) -> None:
        """서비스를 안전하게 종료한다."""
        action_bridge_up.set(0)
        logger.info("Speaker-Bridge 종료 중...")
        self._stop_rest_action_worker()
        self._forwarder.stop()
        if self._rest_receiver:
            self._rest_receiver.stop()
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
        logger.info("Speaker-Bridge 종료 완료")
