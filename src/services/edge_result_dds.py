"""격리 ROS 2 보안 토픽과 기존 영속 결과/HTTPS 저장 확인을 연결한다."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading

from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError, _fields, _identifier
from src.services.edge_ownership import _encode, validate_result
from src.services.edge_result_delivery import acquire_delivery_lock, deliver_pending
from src.services.edge_service import _private_json

logger = logging.getLogger(__name__)


def node_component(node_id):
    # 하이픈/점이 있는 기존 ID를 충돌 없이 ROS 이름 규칙에 맞춘다.
    return "n_" + _identifier(node_id).encode("ascii").hex()


def result_topic(node_id):
    return "/soma/edge/raw/" + node_component(node_id)


class _Endpoint:
    def __init__(self, domain_id, enclave):
        if type(domain_id) is not int or not 0 <= domain_id <= 232:
            raise ValueError("explicit ROS domain_id within [0,232] required")
        if any(
            os.environ.get(key) != value
            for key, value in {
                "ROS_SECURITY_ENABLE": "true",
                "ROS_SECURITY_STRATEGY": "Enforce",
                "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            }.items()
        ) or not os.environ.get("ROS_SECURITY_KEYSTORE"):
            raise ValueError("DDS requires SROS2 Enforce, keystore and pinned Fast DDS")
        if os.environ.get("ROS_SECURITY_ENCLAVE_OVERRIDE") or os.environ.get(
            "ROS_SECURITY_LOOKUP_DIRECTORY"
        ):
            raise ValueError("DDS security path overrides are not allowed")
        # 기존 AI/감독자 Python에는 ROS 패키지를 요구하지 않는다.
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import String

        self.context = Context()
        self.node = None
        self.executor = None
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.message_type = String
        # shortcut: 기관 IDL 미제공으로 내부 시험용 JSON을 사용하며, 계약 수령 시 타입을 대조한다.
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        try:
            rclpy.init(
                args=["--ros-args", "--enclave", enclave],
                context=self.context,
                domain_id=domain_id,
                signal_handler_options=SignalHandlerOptions.NO,
            )
            self.node = rclpy.create_node(
                "edge_results",
                namespace="/soma",
                context=self.context,
                enable_rosout=False,
                start_parameter_services=False,
            )
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            while not self.stop.is_set():
                self.executor.spin_once(timeout_sec=0.2)
        except Exception as error:
            self.error = type(error).__name__
            logger.error("DDS 수신 작업자 중지: %s", self.error)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("DDS endpoint already started")
        self.thread = threading.Thread(target=self._spin, name="edge-dds", daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("DDS worker failed: " + self.error)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=8)
            if self.thread.is_alive():
                raise RuntimeError("DDS worker has not stopped")
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1)
        if self.node is not None:
            self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()


class EdgeDdsCollector:
    """SROS2가 노드별 발행 권한을 제한한 토픽의 신원과 본문을 대조한다."""

    def __init__(self, store, node_ids, domain_id):
        self.store = store
        self.node_ids = {_identifier(node) for node in node_ids}
        if not self.node_ids:
            raise ValueError("collector requires explicit node identities")
        self.endpoint = _Endpoint(domain_id, "/soma/controller")
        try:
            self.subscriptions = [
                self.endpoint.node.create_subscription(
                    self.endpoint.message_type,
                    result_topic(node_id),
                    lambda message, node_id=node_id: self._receive(
                        node_id, message.data
                    ),
                    self.endpoint.qos,
                )
                for node_id in sorted(self.node_ids)
            ]
        except BaseException:
            self.endpoint.close()
            raise

    @classmethod
    def from_file(cls, path, store, node_ids):
        data = _private_json(path)
        _fields(data, {"domain_id"})
        return cls(store, node_ids, data["domain_id"])

    def accept(self, node_id, raw):
        encoded = raw.encode("utf-8")
        try:
            if len(encoded) > 49152:
                raise RegistryError(413, "result_too_large")
            if node_id not in self.node_ids:
                raise RegistryError(403, "dds_identity_mismatch")
            payload = json.loads(encoded)
            validate_result(node_id, payload, self.store.clock())
            self.store.receive(node_id, payload)
        except (
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            RegistryError,
        ) as error:
            reason = (
                error.code if isinstance(error, RegistryError) else "invalid_dds_result"
            )
            # 기존 격리 거부 원본 표를 재사용하며 DDS 출처를 토픽 접두어로 남긴다.
            self.store.reject_mqtt("dds:" + result_topic(node_id), encoded, reason)

    def _receive(self, node_id, raw):
        try:
            self.accept(node_id, raw)
        except sqlite3.Error as error:
            # DDS 수신만으로 ACK하지 않으므로 저장 실패 원본은 송신 큐에서 재시도된다.
            logger.warning("DDS 원본 저장 대기: %s", type(error).__name__)

    def start(self):
        self.endpoint.start()

    def check(self):
        self.endpoint.check()

    def close(self):
        self.endpoint.close()


def run_dds_relay(path, context, node_id, stop):
    """호스트 AI의 전용 결과 DB를 공유하는 별도 ROS 프로세스에서 실행한다."""
    data = _private_json(path)
    _fields(data, {"domain_id", "controller_url", "controller_ca_file", "node_token"})
    if not isinstance(data["node_token"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{32,128}", data["node_token"]
    ):
        raise ValueError("invalid private node token")
    node_id = _identifier(node_id)
    controller = RegistryClient(
        data["controller_url"], data["node_token"], ca_file=data["controller_ca_file"]
    )
    fd = acquire_delivery_lock(context.path)
    endpoint = None
    try:
        endpoint = _Endpoint(data["domain_id"], "/soma/" + node_component(node_id))
        publisher = endpoint.node.create_publisher(
            endpoint.message_type, result_topic(node_id), endpoint.qos
        )

        def publish(payload):
            publisher.publish(endpoint.message_type(data=_encode(payload)))
            return "waiting_for_central_receipt"

        endpoint.start()
        print(
            json.dumps({"dds_node_id": node_id, "domain_id": data["domain_id"]}),
            flush=True,
        )
        while not stop.is_set():
            endpoint.check()
            try:
                deliver_pending(context, node_id, controller, publish, stop)
            except sqlite3.Error as error:
                logger.warning("DDS 송신 큐 대기: %s", type(error).__name__)
            stop.wait(1)
    finally:
        try:
            if endpoint is not None:
                endpoint.close()
        finally:
            os.close(fd)
