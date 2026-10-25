"""신선한 자원의 과부하·명시된 회복 조건이 지속될 때 기존 인계 실행기를 호출한다."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import sqlite3
import stat
import time
import uuid
from contextlib import closing
from pathlib import Path

from src.services.edge_handoff import EdgeHandoff
from src.services.edge_orchestrator import (
    RegistryError,
    _fields,
    _identifier,
    _number,
    _sample_time,
)
from src.services.edge_ownership import _encode
from src.services.edge_service import _private_json

_APPLICATION_ID = 1162102610
_RESOURCES = {"cpu_percent", "gpu_percent", "memory_percent"}
_TERMINAL = {"complete", "rolled_back", "failed", "aborted", "superseded"}


def validate_policy(policy):
    """운영 기본값 없이 명시한 두 노드/한 카메라의 시험 정책만 받는다."""
    _fields(
        policy,
        {
            "camera_id",
            "manifests",
            "overload_percent",
            "available_percent",
            "sustain_seconds",
            "min_samples",
            "max_report_age_seconds",
            "cooldown_seconds",
            "poll_seconds",
            "handoff_timeout_seconds",
        } | ({"failback"} if isinstance(policy, dict) and "failback" in policy else set()),
    )
    _identifier(policy["camera_id"])
    manifests = policy["manifests"]
    if not isinstance(manifests, dict) or len(manifests) != 2:
        raise ValueError("exactly two node manifests are required")
    for node, manifest in manifests.items():
        _identifier(node)
        _identifier(manifest)
    if "failback" in policy:
        recovery = policy["failback"]
        _fields(recovery, {"preferred_node", "sustain_seconds"})
        _identifier(recovery["preferred_node"])
        _number(recovery["sustain_seconds"], 600)
        if recovery["preferred_node"] not in manifests or recovery["sustain_seconds"] <= 0:
            raise ValueError("failback requires a configured node and positive recovery duration")
    for field in ("overload_percent", "available_percent"):
        _fields(policy[field], _RESOURCES)
        for value in policy[field].values():
            _number(value, 100)
    if any(
        policy["available_percent"][key] >= policy["overload_percent"][key]
        for key in _RESOURCES
    ):
        raise ValueError(
            "available resource thresholds must be below overload thresholds"
        )
    for key, maximum in {
        "sustain_seconds": 600,
        "max_report_age_seconds": 30,
        "cooldown_seconds": 3600,
        "poll_seconds": 10,
        "handoff_timeout_seconds": 600,
    }.items():
        _number(policy[key], maximum)
        if policy[key] <= 0:
            raise ValueError("policy durations must be positive")
    if type(policy["min_samples"]) is not int or not 2 <= policy["min_samples"] <= 100:
        raise ValueError("min_samples must be within [2,100]")
    if policy["poll_seconds"] > policy["max_report_age_seconds"]:
        raise ValueError("poll interval exceeds report freshness")
    return json.loads(_encode(policy))


class EdgeAutoHandoff:
    """결정을 영속 보존하며 명시된 복귀 정책에서만 회복 후 원래 노드로 돌아간다."""

    @classmethod
    def from_file(cls, config, policy, state_dir):
        controller, nodes, config_id = EdgeHandoff.read_config(config)
        return cls(
            state_dir, controller, nodes, _private_json(policy), config_id=config_id
        )

    def __init__(
        self,
        state_dir,
        controller,
        nodes,
        policy,
        *,
        config_id,
        clock=time.time,
        monotonic=time.monotonic,
    ):
        self.policy = validate_policy(policy)
        if set(nodes) != set(self.policy["manifests"]):
            raise ValueError("policy and handoff nodes must match")
        self.controller, self.nodes = controller, nodes
        self.clock, self.monotonic = clock, monotonic
        self.config_id = hashlib.sha256(
            _encode({"peers": config_id, "policy": self.policy}).encode()
        ).hexdigest()
        self.root = Path(state_dir)
        self.db = self.lock_fd = None
        self.storage_failed = False
        self.window = None
        self.cooldown_until = 0
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("automatic state directory must be owner-only")
        try:
            self.lock_fd = os.open(
                self.root / "auto.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            info = os.fstat(self.lock_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise ValueError("automatic lock must be owner-only")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "decisions.sqlite3"
            if path.is_symlink():
                raise ValueError("automatic database cannot be a symlink")
            if path.exists():
                with closing(
                    sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                ) as check:
                    if (
                        check.execute("PRAGMA application_id").fetchone()[0]
                        != _APPLICATION_ID
                    ):
                        raise ValueError("not an automatic handoff database")
            self.db = sqlite3.connect(path, timeout=2)
            self.db.executescript(f"""
                PRAGMA application_id={_APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), config_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            """)
            row = self.db.execute(
                "SELECT config_id FROM settings WHERE id=1"
            ).fetchone()
            if row and row[0] != self.config_id:
                raise ValueError("automatic journal belongs to another configuration")
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO settings VALUES(1,?)", (self.config_id,)
                )
            row = self.db.execute(
                "SELECT body FROM decisions ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            self.decision = json.loads(row[0]) if row else None
            if self.decision and self.decision["status"] in _TERMINAL:
                # 재시작/벽시계 변경으로 대기 시간을 건너뛰지 않는다.
                self.cooldown_until = self.monotonic() + self.policy["cooldown_seconds"]
        except BaseException:
            self.close()
            raise

    def _save(self):
        if self.storage_failed:
            raise RuntimeError(
                "automatic journal must be reopened after storage failure"
            )
        try:
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO decisions VALUES(?,?)",
                    (self.decision["decision_id"], _encode(self.decision)),
                )
        except sqlite3.Error:
            self.storage_failed = True
            raise

    def _read(self, path):
        code, body = self.controller["reader"].request("GET", path)
        if not isinstance(body, dict):
            raise ValueError("invalid automatic observation")
        if code != 200:
            raise RegistryError(code, _identifier(body.get("error", "request_failed")))
        return body

    def _fresh(self, node, now):
        report = node.get("report") or {}
        service = report.get("service") or {}
        if (
            node.get("simulation") is not False
            or report.get("simulation") is not False
            or node.get("freshness") != "online"
            or node.get("service_freshness") != "online"
            or not isinstance(node.get("boot_id"), str)
            or report.get("boot_id") != node["boot_id"]
            or type(node.get("last_sequence")) is not int
            or node["last_sequence"] < 1
            or report.get("sequence") != node["last_sequence"]
        ):
            return False
        try:
            return all(
                -5 <= now - _sample_time(value) <= self.policy["max_report_age_seconds"]
                for value in (
                    node.get("received_at"),
                    report.get("sampled_at"),
                    service.get("observed_at"),
                )
            )
        except RegistryError:
            return False

    def _wait(self, reason):
        self.window = None
        return None, reason

    def _candidate(self, current, entries):
        owner = current.get("active")
        if (
            not owner
            or not current.get("accepting_results")
            or current.get("pending") is not None
        ):
            return self._wait("owner_not_active")
        names = self.policy["manifests"]
        if (
            owner["node_id"] not in names
            or owner["camera_id"] != self.policy["camera_id"]
        ):
            return self._wait("owner_not_configured")
        by_id = {entry["node_id"]: entry for entry in entries}
        if any(name not in by_id for name in names):
            return self._wait("node_missing")
        source = owner["node_id"]
        target = next(name for name in names if name != source)
        now = self.clock()
        if not all(self._fresh(by_id[name], now) for name in names):
            return self._wait("report_not_fresh")
        reports = {name: by_id[name]["report"] for name in names}
        services = {name: reports[name]["service"] for name in names}
        source_service = services[source]
        if (
            not EdgeHandoff._matches(source_service, owner)
            or source_service.get("session_id") != owner["session_id"]
            or (source_service.get("activation") or {}).get("activation_id")
            != owner["activation_id"]
            or source_service.get("status") != "running"
            or source_service.get("backend") != "deepstream"
            or source_service.get("models_verified") is not True
        ):
            return self._wait("owner_execution_changed")
        if any(
            services[name].get("camera_id") != self.policy["camera_id"]
            or services[name].get("manifest_id") != names[name]
            for name in names
        ) or services[target].get("status") not in {"stopped", "running"}:
            return self._wait("candidate_unavailable")
        resources = {name: reports[name].get("resources") or {} for name in names}
        if any(
            type(resources[name].get(key)) not in (int, float)
            or not 0 <= resources[name][key] <= 100
            for name in names
            for key in _RESOURCES
        ):
            return self._wait("resources_missing")
        if any(
            resources[target][key] > self.policy["available_percent"][key]
            for key in _RESOURCES
        ):
            return self._wait("candidate_busy")
        signals = {
            key
            for key in _RESOURCES
            if resources[source][key] >= self.policy["overload_percent"][key]
        }
        sustain_seconds = self.policy["sustain_seconds"]
        collecting_reason, confirmed_reason = "collecting_overload_samples", "overload_confirmed"
        if not signals:
            recovery = self.policy.get("failback")
            if not recovery or target != recovery["preferred_node"]:
                return self._wait("below_threshold")
            # 복귀도 신선한 전체 자원·후보 여유 검사 뒤에만 허용해 짧은 회복에 왕복하지 않는다.
            signals = {"preferred_node_recovered"}
            sustain_seconds = recovery["sustain_seconds"]
            collecting_reason, confirmed_reason = "collecting_recovery_samples", "recovery_confirmed"
        expected = {
            "controller_boot_id": current["controller_boot_id"],
            "assignment_id": owner["assignment_id"],
            "epoch": owner["epoch"],
        }
        identity = (
            expected,
            tuple(
                (
                    name,
                    by_id[name]["boot_id"],
                    services[name]["supervisor_boot_id"],
                    services[name].get("session_id"),
                )
                for name in sorted(names)
            ),
        )
        sequences = tuple(by_id[name]["last_sequence"] for name in sorted(names))
        tick = self.monotonic()
        if (
            self.window is None
            or self.window["identity"] != identity
            or tick - self.window["last_checked"]
            > self.policy["max_report_age_seconds"]
        ):
            self.window = {"identity": identity, "sequences": None, "metrics": {}}
        self.window["last_checked"] = tick
        for key in set(self.window["metrics"]) - signals:
            self.window["metrics"].pop(key, None)
        previous = self.window["sequences"]
        if previous is not None and any(
            new <= old for new, old in zip(sequences, previous)
        ):
            # 같은 표본을 반복 조회한 시간은 새 과부하 표본 수로 세지 않는다.
            return None, "waiting_for_new_samples"
        self.window["sequences"] = sequences
        for key in signals:
            metric = self.window["metrics"].setdefault(
                key, {"since": tick, "samples": 0}
            )
            metric["samples"] += 1
        ready = {
            key: value
            for key, value in self.window["metrics"].items()
            if value["samples"] >= self.policy["min_samples"]
            and tick - value["since"] >= sustain_seconds
        }
        if not ready:
            return None, collecting_reason
        return {
            "decision_id": str(uuid.uuid4()),
            "status": "executing",
            "decided_at": now,
            "camera_id": self.policy["camera_id"],
            "source_node": source,
            "target_node": target,
            "manifest_id": names[target],
            "expected_owner": dict(
                expected,
                decided_at=now,
                expires_at=now + self.policy["max_report_age_seconds"],
            ),
            "reasons": sorted(ready),
            "resources": resources,
            "observed_sequences": dict(zip(sorted(names), sequences)),
            "sustained_seconds": {
                key: tick - value["since"] for key, value in ready.items()
            },
            "execution_seconds": 0,
            "outcome": None,
        }, confirmed_reason

    def _execute(self, stop):
        decision = self.decision
        task = EdgeHandoff(
            self.root / ("handoff-" + decision["decision_id"]),
            self.controller,
            self.nodes,
            config_id=self.config_id,
            clock=self.clock,
        )
        started = self.monotonic()
        try:
            outcome = task.run(
                decision["camera_id"],
                decision["target_node"],
                decision["manifest_id"],
                expected_owner=decision["expected_owner"],
                timeout=self.policy["handoff_timeout_seconds"],
                stop=stop,
            )
        finally:
            task.close()
        decision["execution_seconds"] += self.monotonic() - started
        decision["outcome"] = outcome
        if outcome["status"] in _TERMINAL | {"reconcile_required"}:
            decision["status"] = outcome["status"]
            decision["finished_at"] = self.clock()
            self.cooldown_until = self.monotonic() + self.policy["cooldown_seconds"]
        self._save()
        return {
            "status": decision["status"],
            "decision_id": decision["decision_id"],
            "outcome": outcome,
        }

    def step(self, stop):
        if self.storage_failed:
            raise RuntimeError("automatic journal storage unavailable")
        if self.decision and self.decision["status"] == "reconcile_required":
            return {
                "status": "reconcile_required",
                "decision_id": self.decision["decision_id"],
            }
        if self.decision and self.decision["status"] not in _TERMINAL:
            return self._execute(stop)
        if self.monotonic() < self.cooldown_until:
            return {"status": "waiting", "reason": "cooldown"}
        try:
            current = self._read(f"/v1/cameras/{self.policy['camera_id']}/progress")
            entries = self._read("/v1/nodes").get("nodes")
        except (OSError, http.client.HTTPException, RegistryError):
            self.window = None
            raise
        if not isinstance(entries, list):
            raise ValueError("node observation must be a list")
        decision, reason = self._candidate(current, entries)
        if decision is None:
            return {"status": "waiting", "reason": reason}
        self.decision = decision
        self._save()
        self.window = None
        return self._execute(stop)

    def run(self, stop):
        previous = None
        while not stop.is_set():
            try:
                outcome = self.step(stop)
            except (OSError, http.client.HTTPException, RegistryError) as error:
                self.window = None
                outcome = {
                    "status": "waiting",
                    "reason": getattr(error, "code", type(error).__name__),
                }
            identity = (
                outcome.get("status"),
                outcome.get("reason"),
                outcome.get("decision_id"),
            )
            if identity != previous:
                print(json.dumps(outcome, ensure_ascii=False), flush=True)
                previous = identity
            stop.wait(self.policy["poll_seconds"])

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
