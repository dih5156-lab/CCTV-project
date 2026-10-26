"""허용된 로컬 AI 한 개의 실행 수명·명령 세대·실측 준비 상태를 관리한다."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import logging
import math
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from src.services.edge_orchestrator import (
    EdgeRegistryHandler,
    RegistryError,
    _fields,
    _identifier,
    _number,
    _utc,
    create_edge_server,
)

logger = logging.getLogger(__name__)

_APPLICATION_ID = 1162102611
_COMMAND_FIELDS = {
    "command_id",
    "assignment_id",
    "epoch",
    "desired_state",
    "manifest_id",
    "expected_boot_id",
}


def _private_json(path):
    path = Path(path)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("local JSON must be an owner-only regular file")
    if info.st_size > 65536:
        raise ValueError("local JSON exceeds 64 KiB")
    return json.loads(path.read_text())


class ServiceManifest:
    """제어 요청은 이 로컬 허용 목록을 선택할 수만 있고 명령을 추가할 수 없다."""

    def __init__(self, path):
        data = _private_json(path)
        _fields(
            data,
            {
                "manifest_id",
                "service_id",
                "camera_id",
                "argv",
                "cwd",
                "environment_file",
                "models",
                "metrics_url",
                "ready_timeout_seconds",
                "stop_timeout_seconds",
                "stale_after_seconds",
            },
        )
        for name in ("manifest_id", "service_id", "camera_id"):
            _identifier(data[name])
        argv = data["argv"]
        if (
            not isinstance(argv, list)
            or not 1 <= len(argv) <= 32
            or any(
                not isinstance(arg, str)
                or not arg
                or "\0" in arg
                or "rtsp://" in arg.lower()
                for arg in argv
            )
            or not Path(argv[0]).is_absolute()
        ):
            raise ValueError(
                "local argv requires an absolute executable and no RTSP secrets"
            )
        executable = Path(argv[0]).resolve(strict=True)
        info = executable.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & (
            stat.S_ISUID | stat.S_ISGID | 0o022
        ):
            raise ValueError(
                "executable must not change privileges or be publicly writable"
            )
        if "security.capability" in os.listxattr(executable):
            raise ValueError("executable capabilities clear the parent-death signal")
        for key in ("cwd", "environment_file"):
            if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
                raise ValueError("local paths must be absolute")
        if not Path(data["cwd"]).is_dir():
            raise ValueError("service working directory is missing")
        for name in (
            "ready_timeout_seconds",
            "stop_timeout_seconds",
            "stale_after_seconds",
        ):
            _number(data[name], 300)
            if data[name] < 0.1:
                raise ValueError("timeouts must be at least 0.1 seconds")
        if data["stale_after_seconds"] >= data["ready_timeout_seconds"]:
            raise ValueError("ready timeout must exceed freshness timeout")
        address = urlsplit(data["metrics_url"])
        if (
            address.scheme != "http"
            or address.hostname != "127.0.0.1"
            or not address.port
            or address.path != "/metrics"
            or address.username
            or address.password
            or address.query
            or address.fragment
        ):
            raise ValueError("metrics must use http://127.0.0.1:port/metrics")
        models = data["models"]
        if not isinstance(models, list) or not 1 <= len(models) <= 16:
            raise ValueError("1-16 model artifacts are required")
        names = set()
        model_ids = set()
        for model in models:
            _fields(model, {"model_id", "metric_name", "path", "sha256", "allow_empty"})
            _identifier(model["model_id"])
            _identifier(model["metric_name"])
            if (
                model["metric_name"] in names
                or model["model_id"] in model_ids
                or type(model["allow_empty"]) is not bool
            ):
                raise ValueError("model observations must be unique and explicit")
            names.add(model["metric_name"])
            model_ids.add(model["model_id"])
            if (
                not isinstance(model["path"], str)
                or not Path(model["path"]).is_absolute()
            ):
                raise ValueError("model paths must be absolute")
            if not isinstance(model["sha256"], str) or not re.fullmatch(
                "[0-9a-f]{64}", model["sha256"]
            ):
                raise ValueError("model SHA-256 is required")
        if not any(not model["allow_empty"] for model in models):
            raise ValueError("at least one model must prove tensor output")
        self.data = data
        self.port = address.port

    def environment(self):
        values = _private_json(self.data["environment_file"])
        if not isinstance(values, dict) or any(
            not isinstance(key, str)
            or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str)
            or "\0" in value
            for key, value in values.items()
        ):
            raise ValueError("environment JSON requires string values")
        return values

    def verify_models(self):
        for model in self.data["models"]:
            info = Path(model["path"]).stat()
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022:
                raise ValueError("model_file_unsafe")
            digest = hashlib.sha256()
            with Path(model["path"]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != model["sha256"]:
                raise ValueError("model_hash_mismatch")


def _read_metrics(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        body = response.read(1048577)
        if response.status != 200 or len(body) > 1048576:
            raise ValueError("metrics_unavailable")
    finally:
        connection.close()
    samples = {}
    for line in body.decode().splitlines():
        if line.split("{", 1)[0].split(" ", 1)[0] not in {
            "cctv_edge_service_info",
            "cctv_model_enabled",
            "cctv_model_unique_id_conflict",
            "cctv_model_frames_tensor_output_total",
            "cctv_model_frames_no_objects_total",
            "cctv_model_last_frame_age_seconds",
        }:
            continue
        match = re.fullmatch(r"([a-zA-Z0-9_]+)(?:\{(.*)\})? ([^ ]+)", line)
        if not match:
            raise ValueError("invalid_metrics")
        name, raw_labels, raw_value = match.groups()
        labels = {}
        # 이 계약의 식별자는 ASCII 허용 목록이므로 임의 Prometheus escape를 해석하지 않는다.
        if raw_labels:
            for item in raw_labels.split(","):
                label = re.fullmatch(
                    r'([a-zA-Z_][a-zA-Z0-9_]*)="([A-Za-z0-9_.-]+)"', item
                )
                if not label or label[1] in labels:
                    raise ValueError("invalid_metric_labels")
                labels[label[1]] = label[2]
        key = (name, tuple(sorted(labels.items())))
        value = float(raw_value)
        if key in samples or not math.isfinite(value) or value < 0:
            raise ValueError("invalid_metric_value")
        samples[key] = value
    return samples


class EdgeServiceSupervisor:
    """명령 접수와 실행을 분리하고 자기가 시작한 자식만 종료한다."""

    def __init__(
        self,
        node_id,
        manifest,
        state_dir,
        *,
        clock=time.monotonic,
        enable_results=False,
    ):
        self.node_id = _identifier(node_id)
        self.manifest = manifest
        self.clock = clock
        self.boot_id = str(uuid.uuid4())
        self.mutex = threading.RLock()
        self.process = None
        self.lock_fd = None
        self.db = None
        self.result_context = None
        self.result_observation = None
        self.started = None
        self.not_ready_since = None
        self.last_progress = None
        self.last_sample = None
        self.counters = None
        self.session_id = None
        self.applied = None
        self.status = "stopped"
        self.last_error = None
        self.backend = None
        self.fps = None
        self.models_verified = False
        self.closed = False
        self.creator_thread = threading.get_ident()
        root = Path(state_dir)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("supervisor state directory must be owner-only")
        self.root = root
        try:
            self.lock_fd = os.open(
                root / "service.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            db_path = root / "service.sqlite3"
            if db_path.exists():
                with sqlite3.connect(
                    db_path.resolve().as_uri() + "?mode=ro", uri=True
                ) as conn:
                    if (
                        conn.execute("PRAGMA application_id").fetchone()[0]
                        != _APPLICATION_ID
                    ):
                        raise ValueError("not an edge service database")
            self.db = sqlite3.connect(db_path, check_same_thread=False)
            self.db.executescript(f"""
                PRAGMA application_id={_APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS commands (
                    command_id TEXT PRIMARY KEY, body TEXT NOT NULL, ack TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), node TEXT NOT NULL, manifest TEXT NOT NULL);
            """)
            identity = self.db.execute(
                "SELECT node, manifest FROM identity WHERE id=1"
            ).fetchone()
            expected_identity = (node_id, json.dumps(manifest.data, sort_keys=True))
            if identity and identity != expected_identity:
                raise ValueError("state belongs to another node or manifest")
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO identity VALUES (1, ?, ?)", expected_identity
                )
            row = self.db.execute("SELECT body FROM current WHERE id=1").fetchone()
            self.desired = json.loads(row[0]) if row else None
            # 재시작 뒤 저장된 시작 명령을 재실행하지 않고 새 세대 명령을 기다린다.
            self.minimum_epoch = self.desired["epoch"] + 1 if self.desired else 1
            if self.desired:
                self.last_error = "supervisor_restarted"
                self.applied = self.desired["command_id"]
            if enable_results:
                from src.services.edge_result_context import EdgeResultContext

                self.result_context = EdgeResultContext(
                    root / "results-context.sqlite3", create=True
                )
                self.result_context.end()
        except BaseException:
            if self.result_context is not None:
                self.result_context.close()
            if self.db is not None:
                self.db.close()
            if self.lock_fd is not None:
                os.close(self.lock_fd)
            raise

    def submit(self, camera_id, command):
        _fields(command, _COMMAND_FIELDS)
        for name in ("command_id", "assignment_id", "manifest_id", "expected_boot_id"):
            _identifier(command[name])
        if type(command["epoch"]) is not int or not 1 <= command["epoch"] < 2**63:
            raise RegistryError(400, "invalid_epoch")
        if command["desired_state"] not in ("running", "stopped"):
            raise RegistryError(400, "invalid_desired_state")
        data = self.manifest.data
        if (
            camera_id != data["camera_id"]
            or command["manifest_id"] != data["manifest_id"]
        ):
            raise RegistryError(404, "unknown_manifest")
        encoded = json.dumps(command, sort_keys=True, allow_nan=False)
        with self.mutex:
            if self.closed:
                raise RegistryError(503, "supervisor_closed")
            if command["expected_boot_id"] != self.boot_id:
                raise RegistryError(409, "stale_supervisor_boot")
            row = self.db.execute(
                "SELECT body, ack FROM commands WHERE command_id=?",
                (command["command_id"],),
            ).fetchone()
            if row:
                if row[0] != encoded:
                    raise RegistryError(409, "command_id_conflict")
                return json.loads(row[1])
            previous = self.desired
            if (
                command["epoch"] < self.minimum_epoch
                or previous
                and command["epoch"] < previous["epoch"]
            ):
                raise RegistryError(409, "stale_epoch")
            if (
                previous
                and command["epoch"] == previous["epoch"]
                and (
                    command["assignment_id"] != previous["assignment_id"]
                    or command["desired_state"] == "running"
                )
            ):
                raise RegistryError(409, "assignment_conflict")
            ack = {
                "accepted": True,
                "command_id": command["command_id"],
                "boot_id": self.boot_id,
            }
            with self.db:
                self.db.execute(
                    "INSERT INTO commands VALUES (?, ?, ?)",
                    (command["command_id"], encoded, json.dumps(ack)),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO current VALUES (1, ?)", (encoded,)
                )
            self.desired = dict(command)
            return ack

    def _stop(self):
        child = self.process
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=self.manifest.data["stop_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)
            self.process = None
        if self.result_context is not None and self.session_id is not None:
            self.result_context.end(self.session_id)
        self.status = "stopped"
        self.started = self.last_progress = self.last_sample = self.counters = None
        self.not_ready_since = None
        self.backend = self.fps = self.session_id = None
        self.models_verified = False

    def _start(self):
        self.manifest.verify_models()
        environment = os.environ.copy()
        environment.update(self.manifest.environment())
        self.session_id = str(uuid.uuid4())
        environment["EDGE_SERVICE_SESSION_ID"] = self.session_id
        # 감독자 종료가 다른 카메라까지 멈추지 않도록 프로젝트 런타임을 한 입력으로 제한한다.
        environment["EDGE_SERVICE_CAMERA_ID"] = self.manifest.data["camera_id"]
        environment["METRICS_HOST"] = "127.0.0.1"
        environment["METRICS_PORT"] = str(self.manifest.port)
        environment.pop("EDGE_RESULT_CONTEXT_DB", None)
        if self.result_context is not None:
            # 상위 셸의 운영 저장소를 상속하지 않도록 결과 모드 접수 DB도 감독자 경로로 고정한다.
            environment["AIOT_INGEST_DB"] = str(self.root / "video-ingest.sqlite3")
            self.result_context.begin(
                {
                    "node_id": self.node_id,
                    "camera_id": self.manifest.data["camera_id"],
                    "assignment_id": self.desired["assignment_id"],
                    "epoch": self.desired["epoch"],
                    "supervisor_boot_id": self.boot_id,
                    "session_id": self.session_id,
                }
            )
            environment["EDGE_RESULT_CONTEXT_DB"] = str(
                self.result_context.path.resolve()
            )
        # 원인 진단용 원문 로그는 자격 증명을 포함할 수 있어 전용 0600 파일에만 남긴다.
        log_fd = os.open(
            self.root / f"ai-{self.session_id}.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(log_fd, "ab", buffering=0) as log:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_edge_process.py")),
                    str(os.getpid()),
                    *self.manifest.data["argv"],
                ],
                cwd=self.manifest.data["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                pass_fds=(self.lock_fd,),
            )
        self.started = self.not_ready_since = self.clock()
        self.status = "starting"
        self.last_error = None
        self.models_verified = True

    def _observe(self):
        samples = _read_metrics(self.manifest.port)

        def sample(name, **labels):
            return samples[(name, tuple(sorted(labels.items())))]

        if (
            sample(
                "cctv_edge_service_info",
                session_id=self.session_id,
                backend="deepstream",
            )
            != 1
        ):
            raise ValueError("wrong_runtime_identity")
        self.backend = "deepstream"
        counters = []
        for model in self.manifest.data["models"]:
            name = model["metric_name"]
            if (
                sample("cctv_model_enabled", model=name) != 1
                or sample("cctv_model_unique_id_conflict", model=name) != 0
            ):
                raise ValueError("model_unavailable")
            labels = {"model": name, "camera_id": self.manifest.data["camera_id"]}
            count = sample("cctv_model_frames_tensor_output_total", **labels)
            if model["allow_empty"]:
                count += sample("cctv_model_frames_no_objects_total", **labels)
            age = sample("cctv_model_last_frame_age_seconds", **labels)
            if age >= self.manifest.data["stale_after_seconds"]:
                raise ValueError("frames_stale")
            counters.append(count)
        now = self.clock()
        if self.counters is not None:
            if any(new < old for new, old in zip(counters, self.counters)):
                raise ValueError("counter_regression")
            if all(new > old for new, old in zip(counters, self.counters)):
                self.last_progress = now
                elapsed = now - self.last_sample
                primary = next(
                    i
                    for i, model in enumerate(self.manifest.data["models"])
                    if not model["allow_empty"]
                )
                self.fps = (
                    (counters[primary] - self.counters[primary]) / elapsed
                    if elapsed > 0
                    else None
                )
        self.counters = counters
        self.last_sample = now
        if (
            self.last_progress is None
            or now - self.last_progress >= self.manifest.data["stale_after_seconds"]
        ):
            self.status = "starting" if self.last_progress is None else "unhealthy"
            self.last_error = "waiting_for_inference_progress"
        else:
            self.status = "running"
            self.last_error = None

    def tick(self):
        # PR_SET_PDEATHSIG는 생성 스레드의 죽음에도 발동하므로 HTTP 요청 스레드에서 실행하지 않는다.
        if threading.get_ident() != self.creator_thread:
            raise RuntimeError("tick must run on the persistent supervisor thread")
        with self.mutex:
            if self.closed:
                return
            if self.desired and self.applied != self.desired["command_id"]:
                self._stop()
                self.applied = self.desired["command_id"]
                if self.desired["desired_state"] == "running":
                    try:
                        self._start()
                    except (OSError, ValueError, RegistryError, sqlite3.Error) as exc:
                        self._stop()
                        self.status = "error"
                        self.last_error = (
                            str(exc)
                            if str(exc) in {"model_hash_mismatch", "model_file_unsafe"}
                            else "start_validation_failed"
                        )
            if self.process is None:
                return
            if self.process.poll() is not None:
                self._stop()
                self.status, self.last_error = "error", "process_exited"
                return
            try:
                self._observe()
            except (OSError, http.client.HTTPException, ValueError, KeyError) as error:
                if self.last_error != "metrics_unverified":
                    logger.warning("엣지 지표 검증 실패: %s", type(error).__name__)
                self.status, self.last_error = "unhealthy", "metrics_unverified"
                self.backend = self.fps = None
                self.last_progress = None
                self.counters = self.last_sample = None
            # 정상 실행 뒤에는 시작 시각이 아닌 연속 미준비 시간을 기준으로 종료한다.
            if self.status == "running":
                self.not_ready_since = None
            elif self.not_ready_since is None:
                self.not_ready_since = self.clock()
            if self.not_ready_since is not None and (
                self.clock() - self.not_ready_since
                >= self.manifest.data["ready_timeout_seconds"]
            ):
                self._stop()
                self.status, self.last_error = "error", "readiness_timeout"

    def snapshot(self):
        with self.mutex:
            now = self.clock()
            ready = (
                self.status == "running"
                and self.process is not None
                and self.process.poll() is None
                and self.last_progress is not None
                and now - self.last_progress < self.manifest.data["stale_after_seconds"]
                and self.desired
                and self.applied == self.desired["command_id"]
            )
            response = {
                "node_id": self.node_id,
                "boot_id": self.boot_id,
                "service": {
                    "service_id": self.manifest.data["service_id"],
                    "camera_id": self.manifest.data["camera_id"],
                    "manifest_id": self.manifest.data["manifest_id"],
                    "supervisor_boot_id": self.boot_id,
                    "assignment_id": self.desired["assignment_id"]
                    if self.desired
                    else None,
                    "epoch": self.desired["epoch"] if self.desired else 0,
                    "session_id": self.session_id,
                    "status": "pending"
                    if self.desired and self.applied != self.desired["command_id"]
                    else self.status,
                    "ready": bool(ready),
                    "backend": self.backend,
                    "uptime_seconds": now - self.started
                    if self.started is not None
                    else 0,
                    "inference_fps": self.fps if ready else None,
                    "observed_at": _utc(time.time()),
                    "last_error": self.last_error,
                    "models_verified": self.models_verified,
                    "models": [
                        {"model_id": m["model_id"], "sha256": m["sha256"]}
                        for m in self.manifest.data["models"]
                    ],
                },
            }
            if self.result_context is not None:
                response["service"]["activation"] = (
                    self.result_context.snapshot(self.session_id)
                    if self.session_id is not None
                    else None
                )
            return response

    def capture_observation(self):
        with self.mutex:
            if self.result_context is None or not self.snapshot()["service"]["ready"]:
                return
            primary = next(
                i
                for i, model in enumerate(self.manifest.data["models"])
                if not model["allow_empty"]
            )
            observed = (self.session_id, self.counters[primary])
            if self.result_observation == observed:
                return
            stamp = self.result_context.capture(self.session_id)
            self.result_context.enqueue(
                {
                    "kind": "observation",
                    "edge": stamp["edge"],
                    "event": {
                        "event_id": stamp["event_id"],
                        "camera_id": self.manifest.data["camera_id"],
                        "frames_inferred": observed[1],
                    },
                }
            )
            self.result_observation = observed

    def activate_results(self, camera_id, payload):
        _fields(
            payload,
            {
                "command_id",
                "expected_boot_id",
                "assignment_id",
                "epoch",
                "session_id",
                "activation_id",
                "expected_activation_id",
            },
        )
        if type(payload["epoch"]) is not int or not 1 <= payload["epoch"] < 2**63:
            raise RegistryError(400, "invalid_epoch")
        with self.mutex:
            if self.result_context is None:
                raise RegistryError(409, "result_control_disabled")
            if payload["expected_boot_id"] != self.boot_id:
                raise RegistryError(409, "stale_supervisor_boot")
            if (
                camera_id != self.manifest.data["camera_id"]
                or not self.desired
                or payload["assignment_id"] != self.desired["assignment_id"]
                or payload["epoch"] != self.desired["epoch"]
                or payload["session_id"] != self.session_id
                or not self.snapshot()["service"]["ready"]
            ):
                raise RegistryError(409, "result_session_not_ready")
            return self.result_context.activate(
                self.session_id,
                {
                    "command_id": payload["command_id"],
                    "activation_id": payload["activation_id"],
                    "expected_activation_id": payload["expected_activation_id"],
                },
            )

    def close(self):
        with self.mutex:
            if not self.closed:
                self.closed = True
                self._stop()
                if self.result_context is not None:
                    self.result_context.close()
                self.db.close()
                # 자식이 종료한 뒤 FD를 닫는다. 살아 있는 자식의 상속 락을 해제하지 않는다.
                os.close(self.lock_fd)


class EdgeServiceHandler(EdgeRegistryHandler):
    """읽기 토큰으로 제어할 수 없으며 명령 ACK와 실제 준비 상태를 분리한다."""

    def _dispatch(self, method):
        self.close_connection = True
        try:
            if method == "GET" and self.path == "/v1/node/status":
                self._authorize()
                result = self.server.supervisor.snapshot()
            elif method == "PUT" and (
                match := re.fullmatch(r"/v1/assignments/([A-Za-z0-9_.-]+)", self.path)
            ):
                self._authorize(self.server.supervisor.node_id)
                result = self.server.supervisor.submit(match[1], self._body())
            elif method == "PUT" and (
                match := re.fullmatch(
                    r"/v1/assignments/([A-Za-z0-9_.-]+)/activate", self.path
                )
            ):
                self._authorize(self.server.supervisor.node_id)
                result = self.server.supervisor.activate_results(match[1], self._body())
            else:
                raise RegistryError(404, "not_found")
            self._respond(200, result)
        except RegistryError as exc:
            self._respond(exc.status, {"error": exc.code})
        except sqlite3.Error as exc:
            self._storage_unavailable(exc)

    def do_PUT(self):  # noqa: N802
        self._dispatch("PUT")


def create_service_server(supervisor, control_token, reader_token, **options):
    server = create_edge_server(
        EdgeServiceHandler,
        {supervisor.node_id: control_token},
        reader_token,
        **options,
    )
    server.supervisor = supervisor
    return server
