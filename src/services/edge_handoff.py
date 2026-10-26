"""격리 노드 사이의 인계를 영속 명령 기록과 현재 담당 재대조로 진행한다."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import re
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path

from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError, _fields, _identifier, _number
from src.services.edge_ownership import _encode
from src.services.edge_service import _private_json

_APPLICATION_ID = 1162102608
_TERMINAL = {"complete", "rolled_back", "aborted", "failed", "superseded"}
_WAIT_ERRORS = {
    "node_not_observed",
    "node_observation_stale",
    "observation_predates_controller",
    "activation_not_observed",
    "central_sample_required",
}


class EdgeHandoff:
    """자기 후보만 정리하며 저장 응답을 잃어도 같은 명령 ID와 본문을 재사용한다."""

    @classmethod
    def from_file(cls, path, state_dir):
        controller, nodes, config_id = cls.read_config(path)
        return cls(state_dir, controller, nodes, config_id=config_id)

    @staticmethod
    def read_config(path):
        """수동·자동 인계가 같은 인증/접속 검증을 사용한다."""
        settings = _private_json(path)
        _fields(settings, {"controller", "nodes"})
        if not isinstance(settings["nodes"], dict) or not settings["nodes"]:
            raise ValueError("handoff nodes must be configured")

        def peer(value):
            _fields(value, {"url", "ca_file", "credentials_file"})
            if not isinstance(value["url"], str) or not isinstance(
                value["credentials_file"], str
            ):
                raise ValueError("handoff endpoint and credential path must be text")
            if value["ca_file"] is not None and not isinstance(value["ca_file"], str):
                raise ValueError("handoff CA path must be text or null")
            credentials = _private_json(value["credentials_file"])
            if not isinstance(credentials, dict):
                raise ValueError("handoff credentials must be an object")
            clients = {}
            for role in ("reader", "control"):
                token = credentials.get(role + "_token")
                if not isinstance(token, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]{32,128}", token
                ):
                    raise ValueError("handoff role token is invalid")
                clients[role] = RegistryClient(
                    value["url"], token, ca_file=value["ca_file"]
                )
            return clients

        nodes = {}
        for name, value in settings["nodes"].items():
            _identifier(name)
            if isinstance(value, dict) and "cameras" in value:
                _fields(value, {"cameras"})
                if not isinstance(value["cameras"], dict) or not 1 <= len(value["cameras"]) <= 32:
                    raise ValueError("1-32 camera endpoints are required")
                nodes[name] = {"cameras": {
                    _identifier(camera): peer(item)
                    for camera, item in value["cameras"].items()
                }}
            else:
                nodes[name] = peer(value)
        # 인증값은 기록하지 않되 다른 접속 설정으로 같은 작업 기록을 재사용하지 않는다.
        config_id = hashlib.sha256(_encode(settings).encode()).hexdigest()
        return peer(settings["controller"]), nodes, config_id

    def __init__(self, state_dir, controller, nodes, *, config_id, clock=time.time):
        self.controller, self.nodes = controller, nodes
        self.config_id, self.clock = config_id, clock
        self.lock_fd = None
        self.db = None
        self.storage_failed = False
        root = Path(state_dir)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("handoff state directory must be owner-only")
        try:
            self.lock_fd = os.open(
                root / "handoff.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = root / "handoff.sqlite3"
            if path.is_symlink():
                raise ValueError("handoff database cannot be a symlink")
            if path.exists():
                with closing(
                    sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                ) as check:
                    if (
                        check.execute("PRAGMA application_id").fetchone()[0]
                        != _APPLICATION_ID
                    ):
                        raise ValueError("not an edge handoff database")
            self.db = sqlite3.connect(path, timeout=2)
            self.db.executescript(f"""
                PRAGMA application_id={_APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS operation (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
            """)
            row = self.db.execute("SELECT body FROM operation WHERE id=1").fetchone()
            self.state = json.loads(row[0]) if row else None
        except BaseException:
            self.close()
            raise

    def _save(self):
        if self.storage_failed:
            raise RuntimeError("handoff journal must be reopened after storage failure")
        try:
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO operation VALUES (1,?)",
                    (_encode(self.state),),
                )
        except sqlite3.Error:
            # 저장에 실패한 메모리 상태를 근거로 외부 명령을 보내지 않는다.
            self.storage_failed = True
            raise

    def _status(self, status, error=None):
        if self.state["status"] == status and error is None:
            return
        if self.state["status"] != status:
            self.state["history"].append({"status": status, "at": self.clock()})
        if (self.state["status"], self.state["last_error"]) != (status, error):
            self.state.update(status=status, last_error=error)
            self._save()

    def _read(self, client, path):
        code, response = client.request("GET", path)
        if not isinstance(response, dict):
            raise ValueError("invalid handoff response")
        if code != 200:
            raise RegistryError(
                code, _identifier(response.get("error", "request_failed"))
            )
        return response

    def _command_name(self, name):
        return "rollback:" + name if self.state.get("rollback") else name

    def _send(self, name, peer, method, path, body):
        state = self.state
        name = self._command_name(name)
        identity_key = "request_id" if peer == "controller" else "command_id"
        body = {
            **body,
            identity_key: str(uuid.uuid5(uuid.UUID(state["operation_id"]), name)),
        }
        expected = {"peer": peer, "method": method, "path": path, "body": body}
        saved = state["requests"].get(name)
        if saved is None:
            saved = {**expected, "response": None, "sent_at": self.clock()}
            state["requests"][name] = saved
            # commit 직후 프로세스가 죽어도 작업/명령 ID와 기대 세대를 복원할 수 있다.
            self._save()
        elif any(saved[key] != value for key, value in expected.items()):
            raise ValueError("handoff command identity changed")
        if saved["response"] is not None:
            return saved["response"]
        if self.storage_failed:
            raise RuntimeError("handoff journal storage unavailable")
        client = (
            self.controller["control"]
            if peer == "controller"
            else self._peer(peer, state["intent"]["camera_id"])["control"]
        )
        code, response = client.request(method, path, body)
        if not isinstance(response, dict):
            raise ValueError("invalid handoff command response")
        if code != 200:
            raise RegistryError(
                code, _identifier(response.get("error", "request_failed"))
            )
        saved.update(response=response, acknowledged_at=self.clock())
        self._save()
        return response

    def _peer(self, node, camera_id):
        if node not in self.nodes:
            raise ValueError("handoff node is not configured")
        peer = self.nodes[node]
        if "cameras" in peer:
            # 카메라 주소 누락을 다른 감독자로 대체하면 무관한 AI를 중지할 수 있다.
            peer = peer["cameras"].get(camera_id)
            if peer is None:
                raise ValueError("camera endpoint is not configured")
        return peer

    def _service(self, node):
        entries = self._read(self.controller["reader"], "/v1/nodes")["nodes"]
        for entry in entries:
            if entry["node_id"] == node:
                from src.services.edge_orchestrator import observed_camera_service

                return observed_camera_service(entry, self.state["intent"]["camera_id"])
        raise RegistryError(404, "node_not_registered")

    @staticmethod
    def _matches(service, assignment):
        return all(
            service.get(key) == assignment[key]
            for key in (
                "assignment_id",
                "epoch",
                "supervisor_boot_id",
                "manifest_id",
                "camera_id",
            )
        )

    def _stop_execution(self, name, assignment):
        node = assignment["node_id"]
        snapshot = self._read(self._peer(node, assignment["camera_id"])["reader"], "/v1/node/status")
        if snapshot["node_id"] != node:
            raise ValueError("handoff node identity changed")
        service = snapshot["service"]
        # 새 세대/다른 배정을 중지하지 않는다. 이전 실행의 종료 확인과 구분해 기록한다.
        if not self._matches(service, assignment):
            return "replaced"
        if service["status"] == "stopped":
            return "stopped"
        self._send(
            name,
            node,
            "PUT",
            f"/v1/assignments/{assignment['camera_id']}",
            {
                "assignment_id": assignment["assignment_id"],
                "epoch": assignment["epoch"],
                "desired_state": "stopped",
                "manifest_id": assignment["manifest_id"],
                "expected_boot_id": assignment["supervisor_boot_id"],
            },
        )
        return None

    def _advance(self):
        state = self.state
        expected = state.get("expected_owner")
        if (
            expected is not None
            and "prepare" not in state["requests"]
            and not expected["decided_at"] <= self.clock() <= expected["expires_at"]
        ):
            self._status("failed", "decision_expired")
            return True
        intent = state.get("effective_intent", state["intent"])
        camera_id, node, manifest_id = (
            intent[key] for key in ("camera_id", "target_node", "manifest_id")
        )
        current = self._read(
            self.controller["reader"], f"/v1/cameras/{camera_id}/progress"
        )
        if state["controller_boot_id"] is None:
            active = current["active"]
            # 준비 예약 전에 양쪽 카메라 주소를 확인해 종료 불가능한 이전을 막는다.
            self._peer(node, camera_id)
            if active is not None:
                self._peer(active["node_id"], camera_id)
            # 판단 뒤 담당이 바뀌었다면 준비 명령을 만들기 전에 중단한다.
            if expected is not None and (
                expected["controller_boot_id"] != current["controller_boot_id"]
                or active is None
                or any(
                    active[key] != expected[key] for key in ("assignment_id", "epoch")
                )
            ):
                self._status("failed", "decision_owner_changed")
                return True
            if expected is not None:
                snapshot = self._read(
                    self._peer(active["node_id"], camera_id)["reader"], "/v1/node/status"
                )
                service = snapshot["service"]
                if (
                    snapshot["node_id"] != active["node_id"]
                    or snapshot["boot_id"] != active["supervisor_boot_id"]
                    or not self._matches(service, active)
                    or service.get("session_id") != active["session_id"]
                    or (service.get("activation") or {}).get("activation_id")
                    != active["activation_id"]
                    or service.get("status") != "running"
                ):
                    self._status("failed", "decision_execution_changed")
                    return True
            if current["pending"] is not None:
                self._status("failed", "preparation_in_progress")
                return True
            target = self._service(node)
            if (target["camera_id"], target["manifest_id"]) != (camera_id, manifest_id):
                raise RegistryError(409, "node_manifest_changed")
            previous = current["active"]
            if previous is not None and previous["node_id"] not in self.nodes:
                raise ValueError("previous node is not configured")
            state.update(
                controller_boot_id=current["controller_boot_id"],
                previous=previous,
                target_boot_id=target["supervisor_boot_id"],
            )
            if previous and previous["node_id"] == node:
                state.update(assignment=previous, previous=None)
            self._save()
            return False
        if current["controller_boot_id"] != state["controller_boot_id"]:
            self._status("reconcile_required", "controller_boot_changed")
            return True
        assignment = state["assignment"]
        if assignment is None:
            if (
                expected is not None
                and "prepare" not in state["requests"]
                and not expected["decided_at"] <= self.clock() <= expected["expires_at"]
            ):
                self._status("failed", "decision_expired")
                return True
            self._status("preparing")
            previous = state["previous"]
            state["assignment"] = self._send(
                "prepare",
                "controller",
                "POST",
                f"/v1/cameras/{camera_id}/prepare",
                {
                    "controller_boot_id": state["controller_boot_id"],
                    "node_id": node,
                    "manifest_id": manifest_id,
                    "supervisor_boot_id": state["target_boot_id"],
                    "expected_assignment_id": previous["assignment_id"]
                    if previous
                    else None,
                    "expected_epoch": previous["epoch"] if previous else 0,
                },
            )
            self._save()
            return False
        active = current["active"]
        owns = (
            active is not None
            and active["assignment_id"] == assignment["assignment_id"]
        )
        pending = current["pending"]
        preparing = (
            pending is not None
            and pending["assignment_id"] == assignment["assignment_id"]
        )
        if state["abort_reason"]:
            if owns:
                # 활성화 응답 유실을 준비 실패로 잘못 처리해 새 담당을 중지하지 않는다.
                state["abort_reason"] = None
                self._save()
            else:
                self._status("aborting", state["abort_reason"])
                if preparing:
                    self._send(
                        "abort",
                        "controller",
                        "POST",
                        f"/v1/assignments/{assignment['assignment_id']}/abort",
                        {
                            "controller_boot_id": state["controller_boot_id"],
                        },
                    )
                # 시작 명령 자체를 기록하지 않았다면 후보 실행을 만들지 않았다.
                stopped = (
                    self._stop_execution("stop_candidate", assignment)
                    if self._command_name("start") in state["requests"]
                    else "not_started"
                )
                if stopped:
                    state["candidate_stop"] = stopped
                    self._status(
                        "aborted"
                        if stopped in ("stopped", "not_started")
                        else "reconcile_required",
                        state["abort_reason"]
                        if stopped in ("stopped", "not_started")
                        else "candidate_execution_changed",
                    )
                    return True
                return False
        if owns:
            if not current["accepting_results"]:
                self._status("reconcile_required", "controller_reconciling")
                return True
            target = self._service(node)
            if not self._matches(target, active):
                self._status("reconcile_required", "target_execution_changed")
                return True
            if target["status"] == "error":
                previous = state["previous"]
                if state.get("rollback") or previous is None:
                    self._status("reconcile_required", "active_target_failed")
                    return True
                recovery = self._service(previous["node_id"])
                if not self._matches(recovery, previous):
                    self._status("reconcile_required", "previous_execution_changed")
                    return True
                # 기존 담당도 새 세대로 한 번만 복귀시키며 원래 명령 기록은 보존한다.
                state.update(
                    rollback={
                        "reason": "active_target_failed",
                        "failed_assignment": active,
                        "started_at": self.clock(),
                    },
                    effective_intent={
                        "camera_id": camera_id,
                        "target_node": previous["node_id"],
                        "manifest_id": previous["manifest_id"],
                    },
                    previous=active,
                    target_boot_id=recovery["supervisor_boot_id"],
                    assignment=None,
                    proof=None,
                    activated=None,
                    abort_reason=None,
                    previous_stop=None,
                    candidate_stop=None,
                )
                self._status("rolling_back")
                return False
            if target.get("session_id") != active["session_id"] or target.get(
                "activation"
            ) != {
                "activation_id": active["activation_id"],
                "sequence_floor": active["sequence_floor"],
            }:
                # 이전 활성화의 저장 결과로 현재 실행의 인계 완료를 증명하지 않는다.
                self._status("reconcile_required", "target_activation_changed")
                return True
            if not target["ready"]:
                self._status("waiting_for_result", "target_not_ready")
                return False
            if current["latest_result"] is None:
                self._status("waiting_for_result")
                return False
            if state["proof"] is None:
                state.update(proof=current["latest_result"], activated=active)
                self._save()
            previous = state["previous"]
            if previous is not None:
                self._status("stopping_previous")
                stopped = self._stop_execution("stop_previous", previous)
                if not stopped:
                    return False
                state["previous_stop"] = stopped
                if stopped != "stopped":
                    self._status("reconcile_required", "previous_execution_changed")
                    return True
            state["completed_at"] = self.clock()
            self._status("rolled_back" if state.get("rollback") else "complete")
            return True
        if not preparing:
            # 다른 작업으로 폐기된 후보는 재활성화될 수 없으며 자기 실행만 정리한다.
            self._status("cleaning_superseded", "owner_changed")
            stopped = (
                self._stop_execution("stop_candidate", assignment)
                if self._command_name("start") in state["requests"]
                else "not_started"
            )
            if not stopped:
                return False
            state["candidate_stop"] = stopped
            self._status(
                "superseded"
                if stopped in ("stopped", "not_started")
                else "reconcile_required",
                "owner_changed",
            )
            return True
        if (
            state["requests"].get(self._command_name("start"), {}).get("response")
            is None
        ):
            self._status("starting_target")
        self._send(
            "start",
            node,
            "PUT",
            f"/v1/assignments/{camera_id}",
            {
                "assignment_id": assignment["assignment_id"],
                "epoch": assignment["epoch"],
                "desired_state": "running",
                "manifest_id": manifest_id,
                "expected_boot_id": state["target_boot_id"],
            },
        )
        target = self._service(node)
        if not self._matches(target, assignment):
            self._status("waiting_for_ready")
            return False
        if target.get("status") == "error":
            raise RegistryError(409, "target_start_failed")
        if (
            not target["ready"]
            or pending["sample_id"] is None
            or pending["session_id"] != target["session_id"]
        ):
            self._status("waiting_for_ready")
            return False
        if (
            state["requests"]
            .get(self._command_name("activate_node"), {})
            .get("response")
            is None
        ):
            self._status("activating_target")
        barrier = self._send(
            "activate_node",
            node,
            "PUT",
            f"/v1/assignments/{camera_id}/activate",
            {
                "assignment_id": assignment["assignment_id"],
                "epoch": assignment["epoch"],
                "expected_boot_id": state["target_boot_id"],
                "session_id": target["session_id"],
                "expected_activation_id": None,
                "activation_id": assignment["activation_id"],
            },
        )
        if target.get("activation") != barrier:
            self._status("waiting_for_activation_report")
            return False
        self._status("activating_controller")
        self._send(
            "activate_controller",
            "controller",
            "POST",
            f"/v1/assignments/{assignment['assignment_id']}/activate",
            {
                "controller_boot_id": state["controller_boot_id"],
                "session_id": target["session_id"],
                **barrier,
            },
        )
        return False

    def run(
        self,
        camera_id,
        target_node,
        manifest_id,
        *,
        timeout=60,
        stop=None,
        expected_owner=None,
    ):
        intent = {
            "camera_id": _identifier(camera_id),
            "target_node": _identifier(target_node),
            "manifest_id": _identifier(manifest_id),
        }
        if (
            target_node not in self.nodes
            or type(timeout) not in (int, float)
            or not 0 < timeout <= 600
        ):
            raise ValueError("configured target and timeout within (0,600] required")
        if expected_owner is not None:
            _fields(
                expected_owner,
                {
                    "controller_boot_id",
                    "assignment_id",
                    "epoch",
                    "decided_at",
                    "expires_at",
                },
            )
            _identifier(expected_owner["controller_boot_id"])
            _identifier(expected_owner["assignment_id"])
            if type(expected_owner["epoch"]) is not int or expected_owner["epoch"] < 1:
                raise ValueError("expected owner epoch must be positive")
            _number(expected_owner["expires_at"])
            _number(expected_owner["decided_at"], expected_owner["expires_at"])
            expected_owner = dict(expected_owner)
        if self.state is None:
            self.state = {
                "operation_id": str(uuid.uuid4()),
                "config_id": self.config_id,
                "intent": intent,
                "expected_owner": expected_owner,
                "status": "initializing",
                "last_error": None,
                "history": [],
                "requests": {},
                "controller_boot_id": None,
                "previous": None,
                "target_boot_id": None,
                "assignment": None,
                "abort_reason": None,
                "proof": None,
                "started_at": self.clock(),
            }
            self._save()
        elif (
            self.state["intent"] != intent
            or self.state["config_id"] != self.config_id
            or self.state.get("expected_owner") != expected_owner
        ):
            raise ValueError(
                "handoff journal belongs to another request or configuration"
            )
        if self.storage_failed:
            raise RuntimeError("handoff journal storage unavailable")
        stop = stop or threading.Event()
        deadline = time.monotonic() + timeout
        cleanup_deadline = None
        while not stop.is_set() and self.state["status"] not in _TERMINAL:
            try:
                if self._advance():
                    break
            except (OSError, http.client.HTTPException) as error:
                self._status(self.state["status"], type(error).__name__)
            except RegistryError as error:
                if error.status >= 500 or error.code in _WAIT_ERRORS:
                    self._status(self.state["status"], error.code)
                elif self.state["assignment"] is not None:
                    if self.state["abort_reason"] is None:
                        self.state["abort_reason"] = error.code
                        self._save()
                    self._status("aborting", error.code)
                else:
                    self._status("failed", error.code)
            if time.monotonic() >= deadline:
                # 준비 시간 초과는 중앙 상태를 다시 확인한 다음에만 자기 후보를 정리한다.
                if (
                    self.state["assignment"] is None
                    and self._command_name("prepare") not in self.state["requests"]
                ):
                    self._status("failed", "handoff_timeout")
                    break
                if cleanup_deadline is None and self.state["proof"] is None:
                    self.state["abort_reason"] = (
                        self.state["abort_reason"] or "handoff_timeout"
                    )
                    self._save()
                    cleanup_deadline = time.monotonic() + 10
                elif cleanup_deadline is None or time.monotonic() >= cleanup_deadline:
                    self._status(self.state["status"], "handoff_timeout")
                    break
            stop.wait(0.1)
        if stop.is_set() and self.state["status"] not in _TERMINAL:
            self._status(self.state["status"], "interrupted")
        return {
            key: self.state.get(key)
            for key in (
                "operation_id",
                "intent",
                "effective_intent",
                "rollback",
                "status",
                "last_error",
                "proof",
                "previous_stop",
                "candidate_stop",
                "started_at",
                "completed_at",
                "history",
            )
        }

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
