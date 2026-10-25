"""최초 출력 기한과 실행 선점을 보존하여 오래된 경보의 재출력을 막는다."""

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_event_type,
    get_payload_severity,
)
from ._action_bridge_topics import CMD_TOPIC_APPROVE, CMD_TOPIC_MODE, CMD_TOPIC_REJECT


def _timestamp(value) -> float:
    # 호환 정규화 함수의 현재 시각 보강은 만료 시각을 연장하므로 여기서는 허용하지 않는다.
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone_required")
        value = parsed.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_timestamp")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid_timestamp")
    return float(value)


class ActionExpiryPolicy:
    def __init__(
        self, store, path, *, resolve_targets, resolve_mode, clock=None, monotonic=None,
        priority_policy=None
    ):
        self.store = store
        self.config = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(self.config, dict)
            or self.config.get("schema_version") != 1
            or not isinstance(self.config.get("policy_version"), str)
            or not self.config["policy_version"].strip()
            or not isinstance(self.config.get("rules"), list)
            or not isinstance(self.config.get("clock_verified"), bool)
        ):
            raise ValueError("만료 정책의 버전·시계 확인·rules 설정이 필요합니다")
        for rule in self.config["rules"]:
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("topic"), str)
                or not isinstance(rule.get("event_type"), str)
                or rule.get("time_basis") not in {"occurred_at", "received_at"}
                or not isinstance(rule.get("ttl_seconds"), dict)
            ):
                raise ValueError(
                    "만료 규칙의 토픽·이벤트·시간 기준·장치별 TTL을 확인하세요"
                )
            if rule["time_basis"] == "occurred_at" and rule.get(
                "time_field", "occurred_at"
            ) not in {"occurred_at", "timestamp"}:
                raise ValueError(
                    "발생 시각 필드는 occurred_at 또는 timestamp만 허용합니다"
                )
            for device, ttl in rule["ttl_seconds"].items():
                if (
                    device not in {"speaker", "signboard", "siren"}
                    or isinstance(ttl, bool)
                    or not isinstance(ttl, (int, float))
                    or not math.isfinite(ttl)
                    or ttl <= 0
                ):
                    raise ValueError("장치 TTL은 유한한 양수 초 단위여야 합니다")
        keys = [(rule["topic"], rule["event_type"]) for rule in self.config["rules"]]
        if len(keys) != len(set(keys)):
            raise ValueError("같은 토픽·이벤트 만료 규칙을 중복 지정할 수 없습니다")
        self.policy_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.resolve_targets = resolve_targets
        self.resolve_mode = resolve_mode
        self.priority_policy = priority_policy
        self.clock = clock or time.time
        self.monotonic = monotonic or time.monotonic
        self._wall = self.clock()
        self._monotonic = self.monotonic()
        self._clock_fault = False
        with store._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS action_execution_windows ("
                "event_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, plan_json TEXT NOT NULL, "
                "state TEXT NOT NULL CHECK(state IN ('queued','dispatching','complete','held','expired','uncertain')), "
                "reason TEXT)"
            )

    def _now(self):
        now = self.clock()
        # 실행 중 벽시계가 단조시계에서 1초를 초과해 벗어나면 재동기화 확인 전 출력을 보류한다.
        if (
            not math.isfinite(now)
            or abs(now - self._wall - (self.monotonic() - self._monotonic)) > 1
        ):
            self._clock_fault = True
        return now

    def snapshot(self, topic, raw, received_at):
        if topic.startswith("edgex/results/cctv/"):
            return {"kind": "result"}
        if topic in {CMD_TOPIC_MODE, CMD_TOPIC_APPROVE, CMD_TOPIC_REJECT}:
            return {"kind": "control"}
        plan = {
            "kind": "event",
            "policy_hash": self.policy_hash,
            "policy_version": self.config["policy_version"],
        }
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                if len(payload) != 1:
                    return {**plan, "reason": "batch_requires_split"}
                payload = payload[0]
            if not isinstance(payload, dict):
                return {**plan, "reason": "invalid_payload"}
            event_id = payload.get("event_id")
            if not isinstance(event_id, str) or not event_id.strip():
                return {**plan, "reason": "event_id_required"}
            # 같은 ID의 재수신은 첫 시간·정책을 재사용하고 내용 변경은 별도 보류한다.
            fingerprint = hashlib.sha256(
                json.dumps([topic, payload], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        except (ValueError, UnicodeError, TypeError):
            return {**plan, "reason": "invalid_payload"}
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT fingerprint, plan_json FROM action_execution_windows WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    return {
                        **plan,
                        "event_id": event_id,
                        "reason": "event_content_conflict",
                    }
                return json.loads(row["plan_json"])
            camera_id = get_payload_camera_id(payload)
            event_type = get_payload_event_type(payload).lower()
            targets = self.resolve_targets(camera_id)
            plan.update(
                event_id=event_id,
                camera_id=camera_id,
                event_type=event_type,
                severity=get_payload_severity(payload).lower(),
                targets=targets,
                received_at=_timestamp(received_at),
                deadlines={},
            )
            rule = next(
                (
                    rule
                    for rule in self.config["rules"]
                    if rule["topic"] == topic and rule["event_type"] == event_type
                ),
                None,
            )
            if rule is None:
                plan["reason"] = "policy_unset"
            elif not self.config["clock_verified"]:
                plan["reason"] = "time_unverified"
            elif self.resolve_mode(camera_id) != "auto":
                plan["reason"] = "manual_approval_required"
            elif not any(targets.values()):
                plan["reason"] = "no_target"
            else:
                try:
                    plan["time_basis"] = rule["time_basis"]
                    if rule["time_basis"] == "received_at":
                        occurred = plan["received_at"]
                    else:
                        if rule.get("source_time_verified") is not True:
                            raise ValueError("time_unverified")
                        # 발생 시각 누락을 수신 시각으로 보강해 오래된 입력을 되살리지 않는다.
                        occurred = _timestamp(
                            payload.get(rule.get("time_field", "occurred_at"))
                        )
                    if (
                        occurred > plan["received_at"]
                        or self._now() < plan["received_at"]
                        or self._clock_fault
                    ):
                        raise ValueError("time_unverified")
                    plan["occurred_at"] = occurred
                    plan["deadlines"] = {
                        device: occurred + rule["ttl_seconds"][device]
                        for device in targets
                        if device in rule["ttl_seconds"]
                    }
                    if any(
                        not math.isfinite(value) for value in plan["deadlines"].values()
                    ):
                        plan["deadlines"] = {}
                        raise ValueError("time_unverified")
                    if not plan["deadlines"]:
                        plan["reason"] = "policy_unset"
                except (ValueError, OverflowError):
                    plan["reason"] = "time_unverified"
            if not plan.get("reason") and self.priority_policy is not None:
                try:
                    plan["queue_order"] = self.priority_policy(event_type, plan["severity"])
                except ValueError:
                    plan["reason"] = "priority_policy_unset"
            if not plan.get("reason"):
                reason = self.check(plan)
                if reason:
                    plan["reason"] = reason
            connection.execute(
                "INSERT INTO action_execution_windows VALUES (?, ?, ?, ?, ?)",
                (
                    event_id,
                    fingerprint,
                    json.dumps(plan, sort_keys=True, allow_nan=False),
                    "expired"
                    if plan.get("reason") == "expired"
                    else "held"
                    if plan.get("reason")
                    else "queued",
                    plan.get("reason"),
                ),
            )
        return plan

    def check(self, plan, device=None, device_id=None):
        if not isinstance(plan, dict) or plan.get("kind") != "event":
            return "execution_context_missing"
        if plan.get("reason"):
            return plan["reason"]
        if plan.get("policy_hash") != self.policy_hash:
            return "policy_changed"
        if self.priority_policy is not None:
            try:
                if plan.get("queue_order") != self.priority_policy(plan.get("event_type", ""), plan.get("severity", "")):
                    return "priority_policy_changed"
            except ValueError:
                return "priority_policy_unset"
        if (
            not isinstance(plan.get("targets"), dict)
            or not isinstance(plan.get("deadlines"), dict)
            or not plan["deadlines"]
            or not isinstance(plan.get("camera_id"), str)
            or not isinstance(plan.get("event_id"), str)
            or not plan["event_id"].strip()
        ):
            return "execution_context_missing"
        try:
            # 손상된 저장 값은 반복 예외로 작업자를 막지 않고 원본을 남긴 채 보류한다.
            for value in [plan.get("received_at"), *plan["deadlines"].values()]:
                if not isinstance(value, (int, float)):
                    raise ValueError("invalid_saved_time")
                _timestamp(value)
        except (ValueError, TypeError, OverflowError):
            return "execution_context_missing"
        now = self._now()
        if (
            not self.config["clock_verified"]
            or self._clock_fault
            or now < plan["received_at"]
        ):
            return "time_unverified"
        if self.resolve_mode(plan["camera_id"]) != "auto":
            return "manual_approval_required"
        if self.resolve_targets(plan["camera_id"]) != plan["targets"]:
            return "target_changed"
        if device is not None:
            if device_id is not None and device_id not in plan["targets"].get(
                device, []
            ):
                return "target_changed"
            deadline = plan["deadlines"].get(device)
            return (
                "policy_unset"
                if deadline is None
                else "expired"
                if now >= deadline
                else None
            )
        return (
            "expired"
            if all(now >= value for value in plan["deadlines"].values())
            else None
        )

    def recover(self):
        # 완료 저장 전 종료는 장치 실행 여부를 단정하지 않고 모든 재수신도 함께 차단한다.
        with self.store._connect() as connection:
            connection.execute(
                "UPDATE action_execution_windows SET state='uncertain', reason='execution_unconfirmed' WHERE state='dispatching'"
            )

    def claim(self, plan):
        reason = self.check(plan)
        if reason:
            return reason
        with self.store._connect() as connection:
            row = connection.execute(
                "UPDATE action_execution_windows SET state='dispatching' WHERE event_id=? AND state='queued'",
                (plan["event_id"],),
            )
            if row.rowcount != 1:
                state = connection.execute(
                    "SELECT state FROM action_execution_windows WHERE event_id=?",
                    (plan["event_id"],),
                ).fetchone()
                return (
                    "duplicate_completed"
                    if state and state[0] == "complete"
                    else "execution_unconfirmed"
                )
        return None

    def finish(self, plan, state, reason=None):
        if (
            not isinstance(plan.get("event_id"), str)
            or not plan["event_id"]
            or plan.get("reason") == "event_content_conflict"
        ):
            return
        with self.store._connect() as connection:
            connection.execute(
                "UPDATE action_execution_windows SET state=?, reason=? WHERE event_id=? AND state IN ('queued','dispatching')",
                (state, reason, plan["event_id"]),
            )
