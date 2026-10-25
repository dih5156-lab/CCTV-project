"""격리 MQTT 원본 전송과 중앙 영속 저장 확인을 잇는 감독자·제어기 작업자."""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import threading

from src.protocols._mqtt_factory import create_mqtt_client
from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError, _fields, _identifier
from src.services.edge_ownership import _encode, validate_result
from src.services.edge_result_delivery import acquire_delivery_lock, deliver_pending
from src.services.edge_service import _private_json

logger = logging.getLogger(__name__)
RAW_PREFIX = "poc/edge/raw"


def raw_topic(payload):
    edge = payload["edge"]
    event_type = (
        "observation"
        if payload["kind"] == "observation"
        else payload["event"].get("type")
    )
    return "/".join(
        (
            RAW_PREFIX,
            _identifier(edge["node_id"]),
            _identifier(edge["assignment_id"]),
            _identifier(edge["camera_id"]),
            _identifier(event_type),
        )
    )


def mqtt_settings(data):
    _fields(data, {"host", "port", "ca_file", "username", "password"})
    host = data["host"]
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host):
        raise ValueError("explicit MQTT host required")
    if type(data["port"]) is not int or not 1 <= data["port"] <= 65535:
        raise ValueError("invalid MQTT port")
    _identifier(data["username"])
    if not isinstance(data["password"], str) or not 1 <= len(data["password"]) <= 256:
        raise ValueError("private MQTT password required")
    context = None
    if data["ca_file"] is None:
        if host != "127.0.0.1":
            raise ValueError("plaintext MQTT is restricted to literal 127.0.0.1")
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_verify_locations(cafile=data["ca_file"])
    return context


class EdgeResultRelay:
    """MQTT PUBACK는 보조 신호이며 중앙 원본 해시가 일치하는 저장 ACK만 완료로 삼는다."""

    def __init__(self, context, node_id, controller, mqtt, *, observation=None):
        self.context = context
        self.node_id = _identifier(node_id)
        self.controller = controller
        self.observation = observation
        self.stop_event = threading.Event()
        self.thread = None
        tls = mqtt_settings(mqtt)
        self.host, self.port = mqtt["host"], mqtt["port"]
        self.client = create_mqtt_client(
            "edge-result", username=mqtt["username"], password=mqtt["password"]
        )
        self.client.enable_logger(logger)
        if tls is not None:
            self.client.tls_set_context(tls)
        self.client.connect_timeout = 3
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.max_queued_messages_set(100)

    @classmethod
    def from_file(cls, path, context, node_id, *, observation=None):
        data = _private_json(path)
        _fields(data, {"controller_url", "controller_ca_file", "node_token", "mqtt"})
        if not isinstance(data["node_token"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{32,128}", data["node_token"]
        ):
            raise ValueError("invalid private node token")
        client = RegistryClient(
            data["controller_url"],
            data["node_token"],
            ca_file=data["controller_ca_file"],
        )
        return cls(context, node_id, client, data["mqtt"], observation=observation)

    def step(self):
        deliver_pending(
            self.context, self.node_id, self.controller, self._publish, self.stop_event
        )

    def _publish(self, payload):
        result = self.client.publish(
            raw_topic(payload), _encode(payload), qos=1, retain=False
        )
        return (
            f"mqtt_rc_{int(result.rc)}" if result.rc else "waiting_for_central_receipt"
        )

    def _run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    if self.observation is not None:
                        self.observation()
                    self.step()
                except Exception as error:
                    logger.warning("엣지 결과 전송 대기: %s", type(error).__name__)
                self.stop_event.wait(1)
        finally:
            os.close(self.delivery_fd)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("result relay already started")
        self.delivery_fd = acquire_delivery_lock(self.context.path)
        try:
            # Paho 2.1은 TLS 협상 제한에도 keepalive를 사용하므로 종료 대기를 제한한다.
            self.client.connect_async(self.host, self.port, keepalive=5)
            self.client.loop_start()
            self.thread = threading.Thread(
                target=self._run, name="edge-results", daemon=True
            )
            self.thread.start()
        except BaseException:
            os.close(self.delivery_fd)
            raise

    def close(self):
        self.stop_event.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=8)
                if self.thread.is_alive():
                    raise RuntimeError("result relay has not stopped")
        finally:
            self.client.disconnect()
            self.client.loop_stop()


class EdgeResultCollector:
    """노드별 broker ACL과 토픽·본문 일치를 확인한 원본만 담당 판정 저장소에 넣는다."""

    def __init__(self, store, node_ids, config):
        _fields(config, {"mqtt", "client_id"})
        self.store = store
        self.stop_event = threading.Event()
        self.thread = None
        self.node_ids = {_identifier(node) for node in node_ids}
        if not self.node_ids:
            raise ValueError("collector requires explicit node identities")
        settings = config["mqtt"]
        tls = mqtt_settings(settings)
        self.subscriber = MqttTopicSubscriber(
            broker=settings["host"],
            port=settings["port"],
            topics=tuple(
                f"{RAW_PREFIX}/{node}/+/+/+" for node in sorted(self.node_ids)
            ),
            message_handler=self.accept,
            durable=True,
            client_id=_identifier(config["client_id"]),
            username=settings["username"],
            password=settings["password"],
            tls_context=tls,
            keepalive=5,
        )

    def accept(self, topic, raw):
        try:
            if len(raw) > 49152:
                raise RegistryError(413, "result_too_large")
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("edge"), dict
            ):
                raise RegistryError(400, "invalid_result")
            node_id = payload["edge"].get("node_id")
            validate_result(node_id, payload, self.store.clock())
            if node_id not in self.node_ids or topic != raw_topic(payload):
                raise RegistryError(403, "mqtt_identity_mismatch")
            self.store.receive(node_id, payload)
        except (
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            RegistryError,
        ) as error:
            reason = (
                error.code
                if isinstance(error, RegistryError)
                else "invalid_mqtt_result"
            )
            self.store.reject_mqtt(topic, raw, reason)

    @classmethod
    def from_file(cls, path, store, node_ids):
        return cls(store, node_ids, _private_json(path))

    def _run(self):
        while not self.stop_event.is_set():
            self.subscriber.connect()
            self.stop_event.wait(1)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("result collector already started")
        self.thread = threading.Thread(
            target=self._run, name="edge-result-collector", daemon=True
        )
        self.thread.start()

    def close(self):
        self.stop_event.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=12)
                if self.thread.is_alive():
                    raise RuntimeError("result collector has not stopped")
        finally:
            self.subscriber.disconnect()
