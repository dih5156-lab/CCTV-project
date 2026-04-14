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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt

from ..devices.sensor    import SensorConfig, SirenDevice
from ..devices.signboard import SignboardConfig, SignboardDevice, build_display_text
from ..devices.speaker   import SpeakerConfig, SpeakerDevice
from ..protocols.http    import HttpEventForwarder, HttpEventTarget
from ..protocols.rest    import RestEventReceiver
from ..config            import ActionBridgeConfig
from ._action_bridge_support import (
    _ActionExecutor,
    AlarmDevice,
    ControlMode,
    SiteConfig,
    _AlarmCoordinator,
    _EventRepo,
    _SiteRegistry,
)

logger = logging.getLogger(__name__)

_ACTION_DEFAULTS = ActionBridgeConfig()

# MQTT 명령 토픽
_CMD_TOPIC_MODE    = "cctv/commands/mode"     # {"site_id"?, "mode": "auto|manual"}
_CMD_TOPIC_APPROVE = "cctv/commands/approve"  # {"event_id": "..."}
_CMD_TOPIC_REJECT  = "cctv/commands/reject"   # {"event_id": "..."}
_STATUS_TOPIC_PREFIX = "cctv/status/action"

# ── 공통 토픽 정의 (subscribe_topics / alarm_topics 기본값에 공유) ─────
_ZONE_TOPICS = {
    "cctv/ai/events/+/zone_entered",
    "cctv/ai/events/+/zone_dwelling",
    "cctv/ai/events/+/zone_object_detected",
    "cctv/ai/events/+/crowd_warning",
}
_DETECTION_TOPICS = {
    "cctv/ai/events/+/person",
    "cctv/ai/events/+/fall_detected",
    "cctv/ai/events/+/unsafe_behavior",
    "cctv/ai/events/+/helmet",
    "cctv/ai/events/+/head",
    "cctv/ai/events/+/face_unknown",
    "cctv/ai/events/+/face_recognized",
}
_INTRUSION_TOPICS = {
    "cctv/rules/intrusion/filtered",
    "cctv/rules/intrusion/persisted",
    "cctv/rules/intrusion/critical",
}
_SENSOR_TOPICS = {
    "aiot/rules/sensor/tilt",
    "aiot/rules/sensor/temperature",
    "aiot/rules/sensor/vibration",
}

# subscribe: 모든 이벤트를 수신하여 DB 저장
_DEFAULT_SUBSCRIBE_TOPICS = _INTRUSION_TOPICS | _ZONE_TOPICS | _DETECTION_TOPICS | _SENSOR_TOPICS

# alarm: 알람 장치(스피커/전광판/경광등)를 작동시킬 토픽만
_DEFAULT_ALARM_TOPICS = (
    {"cctv/rules/intrusion/persisted", "cctv/rules/intrusion/critical"}
    | _ZONE_TOPICS
    | {"cctv/ai/events/+/fall_detected", "cctv/ai/events/+/unsafe_behavior"}
    | _SENSOR_TOPICS
)



# ===========================================================================
# ActionBridge  (공개 API)
# ===========================================================================


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
        # MQTT
        mqtt_broker:       str           = _ACTION_DEFAULTS.mqtt_broker,
        mqtt_port:         int           = _ACTION_DEFAULTS.mqtt_port,
        subscribe_topics:  Optional[Set[str]] = None,
        # DB
        db_path:           str           = "action_events.db",
        # 디바이스
        speaker_config:    Optional[SpeakerConfig]   = None,
        signboard_config:  Optional[SignboardConfig]  = None,
        siren_config:      Optional[SensorConfig]     = None,
        # HTTP 플랫폼 전송
        http_targets:      Optional[List[HttpEventTarget]] = None,
        external_api_url:  Optional[str] = None,   # 하위 호환 단일 URL
        # 알람 제어
        alarm_topics:          Optional[Set[str]] = None,
        alarm_cooldown_seconds: int = 10,
        # 모드
        default_mode:   ControlMode              = ControlMode.AUTO,
        initial_sites:  Optional[List[SiteConfig]] = None,
        # REST 서버
        rest_enabled: bool = False,
        rest_host:    str  = _ACTION_DEFAULTS.rest_host,
        rest_port:    int  = _ACTION_DEFAULTS.rest_port,
    ) -> None:
        self.mqtt_broker = mqtt_broker
        self.mqtt_port   = int(mqtt_port)
        self.subscribe_topics = subscribe_topics or set(_DEFAULT_SUBSCRIBE_TOPICS)

        # ── 내부 헬퍼 ──────────────────────────────────────────────
        self._repo  = _EventRepo(Path(db_path))
        self._sites = _SiteRegistry(
            default_mode=default_mode,
            initial_sites=initial_sites,
        )
        self._alarm = _AlarmCoordinator(
            alarm_topics=alarm_topics or set(_DEFAULT_ALARM_TOPICS),
            alarm_cooldown_seconds=alarm_cooldown_seconds,
        )

        # ── 디바이스 ──────────────────────────────────────────────
        self._speaker   = SpeakerDevice(speaker_config   or SpeakerConfig())
        self._signboard = SignboardDevice(signboard_config or SignboardConfig())
        self._siren     = SirenDevice(siren_config        or SensorConfig())

        # ── HTTP 포워더 ───────────────────────────────────────────
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

        # ── REST 서버 ─────────────────────────────────────────────
        self._rest_receiver: Optional[RestEventReceiver] = None
        if rest_enabled:
            self._rest_receiver = RestEventReceiver(
                host=rest_host, port=rest_port, action_layer=self
            )

        self._mqtt_client: Optional[mqtt.Client] = None
        self._running = False

    # ------------------------------------------------------------------
    # default_mode 하위 호환 프로퍼티
    # ------------------------------------------------------------------

    @property
    def default_mode(self) -> ControlMode:
        return self._sites.default_mode

    @default_mode.setter
    def default_mode(self, value: ControlMode) -> None:
        self._sites.default_mode = value

    # ------------------------------------------------------------------
    # 사이트 관리 — _SiteRegistry 위임 (공개 API는 그대로 유지)
    # ------------------------------------------------------------------

    def add_site(self, site: SiteConfig) -> None:
        self._sites.add(site)

    def remove_site(self, site_id: str) -> bool:
        return self._sites.remove(site_id)

    def list_sites(self) -> List[Dict]:
        return self._sites.list_all()


    def set_mode(self, mode: ControlMode, site_id: Optional[str] = None) -> None:
        self._sites.set_mode(mode, site_id=site_id)

    def add_site_from_dict(self, data: Dict) -> str:
        """dict에서 사이트를 생성하여 추가하고 site_id를 반환한다."""
        site = SiteConfig.from_dict(data)
        self._sites.add(site)
        return site.site_id

    def set_mode_str(self, mode_str: str, site_id: Optional[str] = None) -> None:
        """문자열 값으로 모드를 설정한다. 잘못된 값이면 ValueError를 전파한다."""
        self._sites.set_mode(ControlMode(mode_str), site_id=site_id)

    # ------------------------------------------------------------------
    # 수동 승인 큐 (공개 API)
    # ------------------------------------------------------------------

    def get_pending_events(self) -> List[Dict]:
        return self._sites.list_pending()

    def _publish_status(self, suffix: str, payload: Dict) -> None:
        """Action Layer 상태/명령 결과를 MQTT로 발행한다."""
        if not self._mqtt_client:
            return
        try:
            topic = f"{_STATUS_TOPIC_PREFIX}/{suffix.lstrip('/')}"
            body = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            self._mqtt_client.publish(topic, json.dumps(body, ensure_ascii=False), qos=0)
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
                "camera_id": info["payload"].get("camera_id"),
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
        self._repo.save(info["topic"], info["payload"], alarm_played=False, http_sent=False)
        self._publish_status(
            "events/rejected",
            {
                "event_id": event_id,
                "camera_id": info["payload"].get("camera_id"),
                "site_id": info.get("site_id"),
                "status": "rejected",
            },
        )
        return True, f"거부 완료: {event_id}"

    # ------------------------------------------------------------------
    # 이벤트 핸들러 (MQTT / REST 공통)
    # ------------------------------------------------------------------

    def _resolve_devices(self, camera_id: str) -> List[AlarmDevice]:
        site = self._sites.find_by_camera(camera_id)
        return site.alarm_devices if site else list(AlarmDevice)

    def _handle_event(self, payload: Dict, topic: str = "rest/inbound") -> None:
        """수신된 이벤트를 처리한다 (MQTT·REST 공통 경로).

        - AUTO 모드: 즉시 알람/HTTP 실행
        - MANUAL 모드: 승인 대기 큐에 추가
        """
        camera_id = str(payload.get("camera_id", "unknown"))
        mode, site_id = self._sites.resolve_mode(camera_id)

        if mode == ControlMode.MANUAL:
            event_id = str(uuid.uuid4())
            self._sites.push_pending(event_id, topic, payload, site_id)
            self._repo.save(topic, payload, alarm_played=False, http_sent=False)
            self._publish_status(
                "events/pending",
                {
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "site_id": site_id,
                    "event_type": payload.get("type"),
                    "status": "pending",
                },
            )
        else:
            self._execute_action(topic, payload)

    def _execute_action(self, topic: str, payload: Dict) -> None:
        """디바이스 조치 + HTTP 전송을 즉시 실행한다."""
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
    ) -> None:
        if rc != 0:
            logger.error("Action Layer MQTT 연결 실패 (rc=%d)", rc)
            return
        logger.info("Action Layer MQTT 연결 성공: %s:%d", self.mqtt_broker, self.mqtt_port)
        all_topics = [
            *((t, 0) for t in self.subscribe_topics),
            *((t, 1) for t in (_CMD_TOPIC_MODE, _CMD_TOPIC_APPROVE, _CMD_TOPIC_REJECT)),
        ]
        for topic, qos in all_topics:
            client.subscribe(topic, qos=qos)
            logger.info("구독: %s (qos=%d)", topic, qos)

    def _dispatch_command(self, topic: str, payload: Dict) -> None:
        """MQTT 명령 토픽을 처리한다."""
        command_id = str(payload.get("command_id", uuid.uuid4()))
        command_status = "ignored"
        message = ""

        if topic == _CMD_TOPIC_MODE:
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

        elif topic == _CMD_TOPIC_APPROVE:
            event_id = payload.get("event_id")
            if event_id:
                ok, msg = self.approve_event(event_id)
                logger.info("MQTT approve: %s", msg)
                command_status = "success" if ok else "error"
                message = msg

        elif topic == _CMD_TOPIC_REJECT:
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
            topic   = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))
            if topic in (_CMD_TOPIC_MODE, _CMD_TOPIC_APPROVE, _CMD_TOPIC_REJECT):
                self._dispatch_command(topic, payload)
            else:
                # Kuiper sink는 배열 형태로 결과를 발행할 수 있음 → 개별 처리
                payloads = payload if isinstance(payload, list) else [payload]
                for single in payloads:
                    if not isinstance(single, dict):
                        continue
                    # 센서 경보 토픽: device_id → camera_id, type 필드 정규화
                    if topic.startswith("aiot/rules/sensor/"):
                        single = self._normalize_sensor_payload(topic, single)
                    self._handle_event(single, topic=topic)
        except json.JSONDecodeError as exc:
            logger.error("JSON 파싱 실패: %s", exc)
        except Exception as exc:
            logger.error("Action 처리 오류: %s", exc, exc_info=True)

    @staticmethod
    def _normalize_sensor_payload(topic: str, payload: Dict) -> Dict:
        """센서 경보 페이로드를 Action Bridge 공통 형식으로 변환합니다.

        Kuiper 출력: {"dev_eui":..., "device_id":"factory-24", "type":"tilt_alert", ...}
        공통 형식 : {"camera_id":"factory-24", "type":"tilt_alert", "source":"sensor", ...}
        """
        if not isinstance(payload, dict):
            return {"type": f"{topic.split('/')[-1]}_alert", "source": "sensor", "camera_id": "unknown"}
        normalized = dict(payload)
        # device_id → camera_id 매핑
        if "camera_id" not in normalized and "device_id" in normalized:
            normalized["camera_id"] = normalized["device_id"]
        # type이 없는 경우 토픽에서 추출 (aiot/rules/sensor/tilt → tilt_alert)
        if "type" not in normalized:
            sensor_kind = topic.split("/")[-1]
            normalized["type"] = f"{sensor_kind}_alert"
        normalized.setdefault("source", "sensor")
        return normalized

    # ------------------------------------------------------------------
    # 라이프사이클
    # ------------------------------------------------------------------

    def start(self) -> None:
        """서비스를 시작한다."""
        self._repo.init()
        self._forwarder.start()

        self._mqtt_client = mqtt.Client()
        self._mqtt_client.on_connect = self._on_connect
        self._mqtt_client.on_message = self._on_message

        for attempt in range(1, self._MQTT_MAX_ATTEMPTS + 1):
            try:
                self._mqtt_client.connect(
                    self.mqtt_broker, self.mqtt_port, keepalive=60
                )
                logger.info(
                    "MQTT 연결 성공: %s:%d", self.mqtt_broker, self.mqtt_port
                )
                break
            except (ConnectionRefusedError, OSError) as exc:
                if attempt >= self._MQTT_MAX_ATTEMPTS:
                    logger.error(
                        "MQTT 연결 실패 - 포기 (%d회 시도): %s", attempt, exc
                    )
                    raise
                logger.warning(
                    "MQTT 연결 실패 (시도 %d/%d): %s - 5초 후 재시도...",
                    attempt, self._MQTT_MAX_ATTEMPTS, exc,
                )
                time.sleep(5)

        self._mqtt_client.loop_start()

        if self._rest_receiver:
            self._rest_receiver.start()

        self._running = True
        logger.info("Speaker-Bridge Action Layer 실행 중 (Ctrl+C 종료)")

        signal.signal(signal.SIGINT,  self._signal_handler)
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
        logger.info("Speaker-Bridge 종료 중...")
        self._forwarder.stop()
        if self._rest_receiver:
            self._rest_receiver.stop()
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
        logger.info("Speaker-Bridge 종료 완료")
