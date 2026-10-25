"""격리 제어기에 모형 또는 읽기 전용 실측 상태를 주기적으로 보고한다."""

# 구형 JetPack의 Python에서도 최신 타입 표기를 실행 시 평가하지 않는다.
from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import math
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from src.services.jetson_resources import JetsonResourceCollector

logger = logging.getLogger(__name__)


class RegistryClient:
    """로컬 HTTP 또는 명시적으로 신뢰한 인증서의 HTTPS 제어기에 접속한다."""

    def __init__(self, base_url: str, token: str, *, ca_file: str | Path | None = None):
        address = urlsplit(base_url)
        if (
            any(ord(character) <= 32 or ord(character) == 127 for character in base_url)
            or address.scheme not in ("http", "https")
            or not address.hostname
            or not address.port
            or address.path not in ("", "/")
            or address.username
            or address.password
            or address.query
            or address.fragment
        ):
            raise ValueError(
                "an explicit HTTP(S) host and port without a path is required"
            )
        self.context = None
        if address.scheme == "http":
            if address.hostname != "127.0.0.1" or ca_file is not None:
                raise ValueError("plaintext HTTP is restricted to literal 127.0.0.1")
        else:
            if ca_file is None:
                raise ValueError("an explicit CA certificate is required for HTTPS")
            # 서버 신원 검증을 끄거나 운영 환경의 TLS/프록시 설정을 가져오지 않는다.
            self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self.context.minimum_version = ssl.TLSVersion.TLSv1_2
            self.context.load_verify_locations(cafile=str(ca_file))
        self.host = address.hostname
        self.port = address.port
        self.token = token

    def request(
        self, method: str, path: str, payload: dict | None = None
    ) -> tuple[int, dict]:
        for attempt in range(2):
            connection = (
                http.client.HTTPSConnection(
                    self.host, self.port, timeout=3, context=self.context
                )
                if self.context is not None
                else http.client.HTTPConnection(self.host, self.port, timeout=3)
            )
            try:
                connection.request(
                    method,
                    path,
                    body=json.dumps(payload, allow_nan=False)
                    if payload is not None
                    else None,
                    headers={
                        "Authorization": "Bearer " + self.token,
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                code, result = response.status, json.loads(response.read(65536))
            finally:
                connection.close()
            # 일시 저장소 오류의 읽기만 한 번 복구하고 쓰기 재전송 판단은 기존 호출자에 맡긴다.
            if not (
                attempt == 0 and method == "GET" and code == 503
                and result == {"error": "storage_unavailable"}
            ):
                return code, result
            logger.warning("retrying storage read once")
            time.sleep(0.2)


class MockNodeAgent:
    """등록 및 보고 재시도 중 동일 부팅 ID·sequence·본문을 유지한다."""

    simulation = True
    capabilities = ["mock-inference", "status-report"]
    node_ip = "127.0.0.1"

    def __init__(
        self,
        client: RegistryClient,
        node_id: str,
        *,
        interval: float = 1,
        active: bool = True,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.client = client
        self.node_id = node_id
        self.interval = interval
        self.active = active
        self.boot_id = str(uuid.uuid4())
        self.sequence = 0
        self.registered = False
        self.pending = None
        self.started = time.monotonic()
        self.last_error = None

    def _sample(self) -> dict:
        # 수치는 시연용이며 실제 CPU/GPU 계측 또는 추론 성공을 뜻하지 않는다.
        return {
            "boot_id": self.boot_id,
            "sequence": self.sequence + 1,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "simulation": True,
            "resources": {
                "cpu_percent": 40 if self.active else 5,
                "memory_percent": 50 if self.active else 10,
                "gpu_percent": 60 if self.active else 0,
                "network_rx_bytes_per_second": 1000000 if self.active else 1000,
                "network_tx_bytes_per_second": 10000 if self.active else 1000,
            },
            "service": {
                "service_id": "mock-cctv-ai",
                "model_id": "mock-model-v1",
                "status": "running" if self.active else "standby",
                "uptime_seconds": time.monotonic() - self.started,
                "inference_fps": 15 if self.active else 0,
            },
        }

    def report_once(self) -> bool:
        try:
            if not self.registered:
                code, result = self.client.request(
                    "POST",
                    "/v1/nodes/register",
                    {
                        "node_id": self.node_id,
                        "node_type": "AEC-100",
                        "boot_id": self.boot_id,
                        "ip": self.node_ip,
                        "capabilities": self.capabilities,
                        "simulation": self.simulation,
                    },
                )
                if code != 200:
                    self.last_error = result.get("error", "registration_failed")
                    return False
                self.sequence = max(self.sequence, result["last_sequence"])
                self.registered = True
            if self.pending is None:
                self.pending = self._sample()
            code, result = self.client.request(
                "POST", f"/v1/nodes/{self.node_id}/reports", self.pending
            )
            if code != 200 or result.get("sequence") != self.pending["sequence"]:
                self.last_error = result.get("error", "invalid_ack")
                if code == 404:
                    self.registered = False
                return False
            self.sequence = self.pending["sequence"]
            self.pending = None
            self.last_error = None
            return True
        except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            self.last_error = type(exc).__name__
            return False

    def run(self, stop_event, *, report_limit: int | None = None) -> None:
        """중지 요청에 즉시 깨어나며 실패 시 같은 간격으로 재시도한다."""
        if report_limit is not None and (
            type(report_limit) is not int or report_limit <= 0
        ):
            raise ValueError("report_limit must be a positive integer")
        accepted = 0
        while not stop_event.is_set():
            if self.report_once():
                accepted += 1
                if report_limit is not None and accepted >= report_limit:
                    return
            else:
                logger.warning("노드 %s 보고 실패: %s", self.node_id, self.last_error)
            if stop_event.wait(self.interval):
                return


class JetsonNodeAgent(MockNodeAgent):
    """공통 등록·재시도 루프를 재사용하고 모형 표본만 실측으로 교체한다."""

    simulation = False
    capabilities = ["jetson-resources", "status-report"]

    def __init__(
        self, client, node_id, *, interval=1, collector=None, node_ip="127.0.0.1",
        service_client=None,
    ):
        address = ipaddress.IPv4Address(node_ip)
        if address.is_unspecified or address.is_multicast or address.is_reserved:
            raise ValueError("a unicast IPv4 node address is required")
        super().__init__(client, node_id, interval=interval)
        self.node_ip = str(address)
        self.collector = (
            collector if collector is not None else JetsonResourceCollector()
        )
        self.service_client = service_client
        if service_client is not None:
            self.capabilities = [*self.capabilities, "ai-service-status"]

    def _service_status(self):
        if self.service_client is None:
            return None
        # 자원 전용 Agent에는 서버 의존성을 추가하지 않고 명시적으로 연결한 경우만 검사한다.
        from src.services.edge_orchestrator import (
            RegistryError,
            _sample_time,
            validate_service_status,
        )

        try:
            code, result = self.service_client.request("GET", "/v1/node/status")
            if code != 200 or not isinstance(result, dict) or set(result) != {"node_id", "boot_id", "service"} or result.get("node_id") != self.node_id:
                return None
            service = result["service"]
            validate_service_status(service, time.time())
            if result["boot_id"] != service["supervisor_boot_id"] or time.time() - _sample_time(service["observed_at"]) >= 5:
                return None
            return service
        except (OSError, http.client.HTTPException, ValueError, KeyError, RegistryError):
            return None

    def _sample(self) -> dict:
        resources = self.collector.sample()
        service = self._service_status()
        return {
            "boot_id": self.boot_id,
            "sequence": self.sequence + 1,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "simulation": False,
            "resources": resources,
            # 자원 계측만으로 실제 AI 실행 여부를 추정하지 않는다.
            "service": service,
        }
