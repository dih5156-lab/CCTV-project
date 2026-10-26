"""운영 서비스와 분리된 모형·실측 노드 등록 및 상태 보고 제어기."""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import secrets
import sqlite3
import ssl
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from socketserver import TCPServer

from src._http_server import BaseApiHandler, ThreadingApiServer

_APPLICATION_ID = 1162102597
logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RESOURCES = {
    "cpu_percent",
    "memory_percent",
    "gpu_percent",
    "network_rx_bytes_per_second",
    "network_tx_bytes_per_second",
}


class RegistryError(Exception):
    """HTTP 응답으로 전달할 등록/보고 거부 사유."""

    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


def _identifier(value) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RegistryError(400, "invalid_identifier")
    return value


def _fields(value, expected: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise RegistryError(400, "invalid_fields")


def _number(value, maximum: float = 1e15) -> None:
    if type(value) not in (int, float) or not 0 <= value <= maximum:
        raise RegistryError(400, "invalid_number")


def _sample_time(value) -> float:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("UTC required")
        return stamp.timestamp()
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise RegistryError(400, "invalid_sample_time") from None


def _utc(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def validate_service_status(service, now):
    """실측 실행 보고의 신뢰 경계를 검사하며 준비 상태를 자원 수치로 추정하지 않는다."""
    expected = {
        "service_id", "camera_id", "manifest_id", "supervisor_boot_id",
        "assignment_id", "epoch", "session_id", "status", "ready", "backend",
        "uptime_seconds", "inference_fps", "observed_at", "last_error",
        "models_verified", "models",
    }
    # 기존 O2 보고도 계속 수용하며 O3 모드가 제공한 활성화 경계만 추가로 검사한다.
    if isinstance(service, dict) and "activation" in service:
        expected.add("activation")
        activation = service["activation"]
        if activation is not None:
            _fields(activation, {"activation_id", "sequence_floor"})
            _identifier(activation["activation_id"])
            if type(activation["sequence_floor"]) is not int or not 0 <= activation["sequence_floor"] < 2**63:
                raise RegistryError(400, "invalid_activation_boundary")
    _fields(service, expected)
    for name in ("service_id", "camera_id", "manifest_id", "supervisor_boot_id"):
        _identifier(service[name])
    for name in ("assignment_id", "session_id", "last_error"):
        if service[name] is not None:
            _identifier(service[name])
    if type(service["epoch"]) is not int or not 0 <= service["epoch"] < 2**63:
        raise RegistryError(400, "invalid_epoch")
    if (service["assignment_id"] is None) != (service["epoch"] == 0):
        raise RegistryError(400, "invalid_assignment")
    if service["status"] not in ("stopped", "pending", "starting", "running", "unhealthy", "error"):
        raise RegistryError(400, "invalid_service_status")
    if type(service["ready"]) is not bool or type(service["models_verified"]) is not bool:
        raise RegistryError(400, "invalid_readiness")
    if service["backend"] not in (None, "deepstream"):
        raise RegistryError(400, "unsupported_backend")
    _number(service["uptime_seconds"])
    if service["inference_fps"] is not None:
        _number(service["inference_fps"], 10000)
    if _sample_time(service["observed_at"]) > now + 5:
        raise RegistryError(400, "future_service_observation")
    models = service["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= 16:
        raise RegistryError(400, "invalid_models")
    names = set()
    for model in models:
        _fields(model, {"model_id", "sha256"})
        _identifier(model["model_id"])
        if model["model_id"] in names or not isinstance(model["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", model["sha256"]):
            raise RegistryError(400, "invalid_model_inventory")
        names.add(model["model_id"])
    if service["ready"] and (
        service["status"] != "running" or service["backend"] != "deepstream"
        or not service["session_id"] or not service["assignment_id"]
        or not service["models_verified"] or service["inference_fps"] is None
        or service["inference_fps"] <= 0 or service["last_error"] is not None
    ):
        raise RegistryError(400, "inconsistent_readiness")


def observed_camera_service(node, camera_id):
    """물리 노드의 다른 카메라가 보고한 준비 상태를 대신 사용하지 않는다."""
    report = node.get("report") or {}
    if "services" in report:
        service = report["services"].get(camera_id)
        freshness = (node.get("services_freshness") or {}).get(camera_id)
    else:
        service = report.get("service")
        freshness = node.get("service_freshness")
    if (
        node.get("simulation") is not False
        or node.get("freshness") != "online"
        or freshness != "online"
        or not service
        or service.get("camera_id") != camera_id
    ):
        raise RegistryError(409, "node_not_observed")
    return service


class NodeRegistry:
    """별도 SQLite에 최신 보고와 재부팅 세대를 보존한다."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        stale_after: float = 10,
        clock=time.time,
        allow_real_resources: bool = False,
        allow_remote_nodes: bool = False,
        allow_real_services: bool = False,
    ):
        _number(stale_after, 86400)
        if stale_after == 0:
            raise ValueError("stale_after must be positive")
        self.path = Path(db_path).resolve()
        self.stale_after = stale_after
        self.clock = clock
        # 기본 모형 계약은 유지하며 실측 시험 실행기만 명시적으로 허용한다.
        if type(allow_real_resources) is not bool:
            raise ValueError("allow_real_resources must be boolean")
        self.allow_real_resources = allow_real_resources
        if type(allow_remote_nodes) is not bool:
            raise ValueError("allow_remote_nodes must be boolean")
        self.allow_remote_nodes = allow_remote_nodes
        if type(allow_real_services) is not bool:
            raise ValueError("allow_real_services must be boolean")
        self.allow_real_services = allow_real_services
        # 경로 실수로 기존 운영 DB에 테이블을 추가하지 않도록 소유 표식을 확인한다.
        if self.path.exists():
            with closing(
                sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
            ) as conn:
                if (
                    conn.execute("PRAGMA application_id").fetchone()[0]
                    != _APPLICATION_ID
                ):
                    raise ValueError("not an edge node registry database")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(f"""
                PRAGMA application_id = {_APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY, boot_id TEXT NOT NULL,
                    registration TEXT NOT NULL, registered_at REAL NOT NULL,
                    report TEXT, received_at REAL,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    report_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS boots (
                    node_id TEXT NOT NULL, boot_id TEXT NOT NULL,
                    PRIMARY KEY (node_id, boot_id)
                );
            """)

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=2)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def register(self, payload: dict) -> dict:
        _fields(
            payload,
            {"node_id", "node_type", "boot_id", "ip", "capabilities", "simulation"},
        )
        node_id = _identifier(payload["node_id"])
        boot_id = _identifier(payload["boot_id"])
        if payload["node_type"] != "AEC-100":
            raise RegistryError(400, "invalid_node_type")
        if type(payload["simulation"]) is not bool or (
            not payload["simulation"] and not self.allow_real_resources
        ):
            raise RegistryError(400, "simulation_required")
        try:
            if not isinstance(payload["ip"], str):
                raise ValueError("text IP required")
            address = ipaddress.ip_address(payload["ip"])
            if not self.allow_remote_nodes and not address.is_loopback:
                raise RegistryError(400, "loopback_required")
            if self.allow_remote_nodes and (
                address.version != 4
                or address.is_unspecified
                or address.is_multicast
                or address.is_reserved
            ):
                raise ValueError("unicast IPv4 required")
        except (ValueError, TypeError):
            raise RegistryError(400, "invalid_node_ip") from None
        capabilities = payload["capabilities"]
        if not isinstance(capabilities, list) or not 1 <= len(capabilities) <= 16:
            raise RegistryError(400, "invalid_capabilities")
        for capability in capabilities:
            _identifier(capability)
        if len(set(capabilities)) != len(capabilities):
            raise RegistryError(400, "invalid_capabilities")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if row and row["boot_id"] == boot_id:
                if row["registration"] != encoded:
                    raise RegistryError(409, "registration_conflict")
                return {
                    "node_id": node_id,
                    "boot_id": boot_id,
                    "last_sequence": row["last_sequence"],
                }
            # 과거 부팅의 등록 재시도가 새 부팅을 다시 덮어쓰지 못하게 한다.
            if conn.execute(
                "SELECT 1 FROM boots WHERE node_id=? AND boot_id=?", (node_id, boot_id)
            ).fetchone():
                raise RegistryError(409, "retired_boot")
            conn.execute("INSERT INTO boots VALUES (?, ?)", (node_id, boot_id))
            conn.execute(
                """
                INSERT INTO nodes (node_id, boot_id, registration, registered_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    boot_id=excluded.boot_id, registration=excluded.registration,
                    registered_at=excluded.registered_at, report=NULL, received_at=NULL,
                    last_sequence=0, report_count=0
            """,
                (node_id, boot_id, encoded, self.clock()),
            )
        return {"node_id": node_id, "boot_id": boot_id, "last_sequence": 0}

    def report(self, node_id: str, payload: dict) -> dict:
        _identifier(node_id)
        _fields(
            payload,
            {"boot_id", "sequence", "sampled_at", "resources", "service", "simulation"}
            | ({"services"} if isinstance(payload, dict) and "services" in payload else set()),
        )
        _identifier(payload["boot_id"])
        sequence = payload["sequence"]
        if type(sequence) is not int or not 1 <= sequence <= 2**63 - 1:
            raise RegistryError(400, "invalid_sequence")
        if type(payload["simulation"]) is not bool or (
            not payload["simulation"] and not self.allow_real_resources
        ):
            raise RegistryError(400, "simulation_required")
        sampled_at = _sample_time(payload["sampled_at"])
        now = self.clock()
        if sampled_at > now + 5:
            raise RegistryError(400, "future_sample")
        _fields(payload["resources"], _RESOURCES)
        for key, value in payload["resources"].items():
            if value is not None:
                _number(value, 100 if key.endswith("percent") else 1e15)
        service = payload["service"]
        services = payload.get("services")
        if "services" in payload:
            # 단일 보고와 다중 보고를 섞으면 어느 실행이 실제 담당인지 모호해진다.
            if payload["simulation"] or service is not None or not self.allow_real_services:
                raise RegistryError(400, "invalid_services_report")
            if not isinstance(services, dict) or not 1 <= len(services) <= 32:
                raise RegistryError(400, "invalid_services_report")
            for camera_id, observed in services.items():
                _identifier(camera_id)
                if observed is not None:
                    validate_service_status(observed, now)
                    if observed["camera_id"] != camera_id:
                        raise RegistryError(400, "service_camera_mismatch")
        if not payload["simulation"]:
            # 실측 자원과 가상 AI 상태가 한 보고에 섞이지 않게 한다.
            if service is not None and not self.allow_real_services:
                raise RegistryError(400, "unobserved_service_required")
            if service is not None:
                validate_service_status(service, now)
        else:
            _fields(
                service,
                {"service_id", "status", "uptime_seconds", "inference_fps", "model_id"},
            )
            _identifier(service["service_id"])
            _identifier(service["model_id"])
            if service["status"] not in ("running", "standby", "stopped", "error"):
                raise RegistryError(400, "invalid_service_status")
            _number(service["uptime_seconds"])
            _number(service["inference_fps"], 10000)
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not row:
                raise RegistryError(404, "node_not_registered")
            if row["boot_id"] != payload["boot_id"]:
                raise RegistryError(409, "retired_boot")
            if json.loads(row["registration"])["simulation"] != payload["simulation"]:
                raise RegistryError(409, "report_mode_conflict")
            if not payload["simulation"] and (service is not None or services is not None) and "ai-service-status" not in json.loads(row["registration"])["capabilities"]:
                raise RegistryError(409, "service_capability_required")
            if sequence <= row["last_sequence"]:
                if sequence == row["last_sequence"] and encoded == row["report"]:
                    # 응답 유실 재시도는 ACK만 재전송하고 생존 시각을 연장하지 않는다.
                    return {"sequence": sequence, "duplicate": True}
                raise RegistryError(409, "stale_or_conflicting_report")
            if row["report"] and sampled_at < _sample_time(
                json.loads(row["report"])["sampled_at"]
            ):
                raise RegistryError(409, "sample_time_regressed")
            conn.execute(
                """
                UPDATE nodes SET report=?, received_at=?, last_sequence=?,
                    report_count=report_count+1 WHERE node_id=?
            """,
                (encoded, now, sequence, node_id),
            )
        return {"sequence": sequence, "duplicate": False}

    def list_nodes(self) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
        now = self.clock()
        nodes = []
        for row in rows:
            node = json.loads(row["registration"])
            report = json.loads(row["report"]) if row["report"] else None
            age = now - row["received_at"] if report else None
            sample_age = now - _sample_time(report["sampled_at"]) if report else None
            fresh = (
                report is not None
                and 0 <= age < self.stale_after
                and -5 <= sample_age < self.stale_after
            )
            node.update(
                registered_at=_utc(row["registered_at"]),
                received_at=_utc(row["received_at"]) if report else None,
                age_seconds=age,
                freshness="online" if fresh else "stale" if report else "pending",
                last_sequence=row["last_sequence"],
                report_count=row["report_count"],
                report=report,
            )
            if report and not report["simulation"] and report["service"] is not None:
                observed_age = now - _sample_time(report["service"]["observed_at"])
                node["service_freshness"] = "online" if fresh and -5 <= observed_age < self.stale_after else "stale"
            if report and "services" in report:
                node["services_freshness"] = {
                    camera: "online" if fresh and observed is not None
                    and -5 <= now - _sample_time(observed["observed_at"]) < self.stale_after
                    else "stale"
                    for camera, observed in report["services"].items()
                }
            nodes.append(node)
        return nodes


class EdgeRegistryHandler(BaseApiHandler):
    """노드별 토큰을 사용하는 격리 시험용 REST API."""

    _LOG_PREFIX = "[EdgeRegistry]"
    timeout = 3

    def _authorize(self, node_id: str | None = None) -> None:
        expected = (
            self.server.node_tokens.get(node_id)
            if node_id is not None
            else self.server.reader_token
        )
        self._authorize_token(expected)

    def _authorize_token(self, expected):
        provided = self.headers.get("Authorization", "")
        if not expected or not secrets.compare_digest(
            provided.encode(), ("Bearer " + expected).encode()
        ):
            raise RegistryError(401, "unauthorized")

    def _body(self) -> dict:
        # 공유 HTTP 도우미는 그대로 두고 새 경로에만 본문 크기/시간 제한을 둔다.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RegistryError(400, "invalid_content_length") from None
        # 모델 목록을 포함한 카메라별 보고만 기존 명령보다 큰 제한을 사용한다.
        limit = 262144 if re.fullmatch(r"/v1/nodes/[A-Za-z0-9_.-]+/reports", self.path) else 16384
        if self.headers.get("Transfer-Encoding") or not 0 < length <= limit:
            raise RegistryError(400, "invalid_content_length")
        value = self._read_json()
        if not isinstance(value, dict):
            raise RegistryError(400, "invalid_json")
        return value

    def _dispatch(self, method: str) -> None:
        self.close_connection = True
        try:
            if method == "GET" and self.path == "/v1/nodes":
                self._authorize()
                nodes = self.server.registry.list_nodes()
                result = {
                    "simulation": all(node["simulation"] for node in nodes),
                    "nodes": nodes,
                }
            elif method == "POST" and self.path == "/v1/nodes/register":
                body = self._body()
                node_id = _identifier(body.get("node_id"))
                self._authorize(node_id)
                result = self.server.registry.register(body)
            elif method == "POST" and (
                match := re.fullmatch(r"/v1/nodes/([A-Za-z0-9_.-]+)/reports", self.path)
            ):
                node_id = _identifier(match.group(1))
                self._authorize(node_id)
                result = self.server.registry.report(node_id, self._body())
            else:
                raise RegistryError(404, "not_found")
            self._respond(200, result)
        except RegistryError as exc:
            self._respond(exc.status, {"error": exc.code})
        except sqlite3.Error as exc:
            self._storage_unavailable(exc)

    def _storage_unavailable(self, error):
        # 원문 오류·경로·인증값은 남기지 않고 구형 Python에서도 잠금 오류를 식별한다.
        reason = "database_locked" if "locked" in str(error).lower() else type(error).__name__
        logger.warning("edge storage unavailable: %s", reason)
        self._respond(503, {"error": "storage_unavailable"})

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")


class EdgeRegistryServer(ThreadingApiServer):
    """명시한 IP의 역방향 DNS 응답을 기다리지 않고 시험 서버를 시작한다."""

    def server_bind(self):
        # 공유 서버는 유지하고 이 제어기에서만 HTTPServer의 불필요한 getfqdn을 건너뛴다.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def create_registry_server(
    registry: NodeRegistry,
    node_tokens: dict[str, str],
    reader_token: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    tls_context: ssl.SSLContext | None = None,
):
    """기본은 로컬 임의 포트이며 원격은 명시한 IPv4·TLS로만 연다."""
    server = create_edge_server(
        EdgeRegistryHandler, node_tokens, reader_token,
        host=host, port=port, tls_context=tls_context,
        allow_remote=registry.allow_remote_nodes,
    )
    server.registry = registry
    return server


def create_edge_server(
    handler, node_tokens, reader_token, *, host="127.0.0.1", port=0,
    tls_context=None, allow_remote=False, role_tokens=None,
):
    """등록과 실행 제어에 같은 바인딩·TLS·토큰 검증을 적용한다."""
    address = ipaddress.IPv4Address(host)
    if address.is_unspecified or address.is_multicast or address.is_reserved:
        raise ValueError("a specific unicast IPv4 bind address is required")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be within [0, 65535]")
    if not address.is_loopback and (
        tls_context is None or not allow_remote
    ):
        raise ValueError("remote binding requires TLS and explicit remote-node mode")
    if tls_context is not None and tls_context.protocol != ssl.PROTOCOL_TLS_SERVER:
        raise ValueError("a TLS server context is required")
    for node_id in node_tokens:
        _identifier(node_id)
    role_tokens = {} if role_tokens is None else role_tokens
    if not isinstance(role_tokens, dict):
        raise ValueError("role_tokens must be an object")
    for role in role_tokens:
        _identifier(role)
    tokens = [*node_tokens.values(), reader_token, *role_tokens.values()]
    if (
        not node_tokens
        or any(
            not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", t)
            for t in tokens
        )
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("distinct URL-safe tokens of 32-128 characters are required")
    server = EdgeRegistryServer((str(address), port), handler)
    if tls_context is not None:
        try:
            # TLS 협상을 요청 스레드에서 수행해 미완료 연결이 다른 노드를 막지 않게 한다.
            server.socket = tls_context.wrap_socket(
                server.socket,
                server_side=True,
                do_handshake_on_connect=False,
            )
        except Exception:
            server.server_close()
            raise
    server.node_tokens = dict(node_tokens)
    server.reader_token = reader_token
    server.role_tokens = dict(role_tokens)
    return server
