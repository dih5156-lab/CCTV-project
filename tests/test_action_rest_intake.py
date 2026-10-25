"""REST 접수 응답을 실제 저장·복구·출력 경계와 대조한다."""

import base64
import io
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.protocols import rest
from tests.test_action_expiry import TOPIC
from tests.test_action_expiry import make_bridge as expiry_bridge
from tests.test_action_mqtt_intake import make_bridge, wait_until


def request(bridge, raw, *, path="/events", method="POST", headers=None):
    handler = rest._RestHandler.__new__(rest._RestHandler)
    handler.server = SimpleNamespace(action_layer=bridge)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw)), **(headers or {})}
    handler.rfile = io.BytesIO(raw)
    responses = []
    handler._respond = lambda status, body: responses.append((status, body))
    getattr(handler, "do_" + method)()
    return responses[0]


@pytest.fixture(autouse=True)
def internal_auth(monkeypatch):
    monkeypatch.setattr(rest, "_INTERNAL_TOKEN", None)


def event(event_id="event-1", **changes):
    return {"event_id": event_id, "camera_id": "cam1", "type": "fall_detected",
            "severity": "critical", "occurred_at": time.time() - 1, "topic": TOPIC, **changes}


def test_commit_precedes_202_and_receipt_preserves_exact_body(tmp_path):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True, rest_enabled=True)
    intake = bridge._mqtt_intake
    intake.start()
    try:
        raw = b'  { "event_id": "exact", "topic": "rest/inbound", "message": "hello" }\n'
        status, body = request(bridge, raw)
        assert status == 202 and body["durable"] is True
        row = intake.store.get(body["ingest_id"])
        assert row["source_namespace"] == "action_rest"
        assert base64.b64decode(row["raw"]["payload_base64"]) == raw
        assert row["stage"] == "pending"
        code, receipt = request(bridge, b"", path="/receipts/" + body["ingest_id"], method="GET")
        assert code == 200 and receipt["stage"] == "pending"
        assert "raw" not in receipt and "payload" not in receipt
        assert bridge._rest_action_worker is None
    finally:
        intake.stop()


@pytest.mark.parametrize("raw", [b"bad", b"[]", b"null", b"{}", b'{"event_id":7}',
                                    b'{"event_id":" "}', b'{"event_id":"x","v":NaN}',
                                    b'{"event_id":"x","topic":[]}',
                                    b'{"event_id":"x","topic":"cctv/commands/approve"}',
                                    b'{"event_id":"x","topic":"edgex/results/cctv/test"}', b'\xff'])
def test_invalid_event_is_not_acknowledged_or_executed(tmp_path, raw):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True)
    bridge._mqtt_intake.start()
    try:
        assert request(bridge, raw)[0] == 400
        assert bridge._mqtt_intake.store.pending(namespace="action_rest") == []
    finally:
        bridge._mqtt_intake.stop()


@pytest.mark.parametrize("headers,status", [({"Content-Length": "bad"}, 400),
                                          ({"Content-Length": "-1"}, 400),
                                          ({"Content-Length": "1048577"}, 413),
                                          ({"Content-Length": "90"}, 400),
                                          ({"Transfer-Encoding": "chunked"}, 400)])
def test_incomplete_or_oversized_http_body_is_rejected(tmp_path, headers, status):
    bridge = make_bridge(tmp_path)
    assert request(bridge, b'{"event_id":"x"}', headers=headers)[0] == status


def test_storage_failure_returns_503_then_retry_executes_once(tmp_path):
    bridge = expiry_bridge(tmp_path, queue=True)
    intake = bridge._mqtt_intake
    intake.start()
    raw = json.dumps(event()).encode()
    try:
        with intake.store._connect() as conn:
            conn.execute("CREATE TRIGGER fail_rest BEFORE INSERT ON ingest_receipts BEGIN SELECT RAISE(ABORT,'test disk failure'); END")
        assert request(bridge, raw)[0] == 503
        bridge._executor._publish_edgex_command.assert_not_called()
        with intake.store._connect() as conn:
            conn.execute("DROP TRIGGER fail_rest")
        assert request(bridge, raw)[0] == 202
        wait_until(lambda: bridge._executor._publish_edgex_command.call_count == 1)
        status, second = request(bridge, raw)
        assert status == 202
        wait_until(lambda: intake.store.get(second["ingest_id"])["stage"] == "quarantined")
        assert bridge._executor._publish_edgex_command.call_count == 1
    finally:
        intake.stop()


def test_slow_execution_does_not_block_rest_or_mqtt_admission(tmp_path):
    bridge = make_bridge(tmp_path)
    intake = bridge._mqtt_intake
    entered, release = Event(), Event()
    seen = []

    def handle(payload, **kwargs):
        seen.append(payload["event_id"])
        if payload["event_id"] == "slow":
            entered.set()
            assert release.wait(5)

    bridge._handle_event = handle
    intake.start()
    try:
        assert request(bridge, json.dumps(event("slow")).encode())[0] == 202
        assert entered.wait(2)
        second = request(bridge, json.dumps(event("rest-next")).encode())[1]["ingest_id"]
        third = intake.accept(TOPIC, json.dumps(event("mqtt-next")).encode())
        assert intake.store.get(second)["stage"] == intake.store.get(third)["stage"] == "pending"
        assert intake.store.pending(namespace=("action_rest", "action_mqtt"))[-2:] == [second, third]
        release.set()
        wait_until(lambda: intake.store.get(third)["stage"] == "complete")
        assert seen == ["slow", "rest-next", "mqtt-next"]
    finally:
        release.set()
        intake.stop()


def test_legacy_pending_is_held_on_restart_without_policy(tmp_path):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True)
    bridge._mqtt_intake.start()
    key = request(bridge, json.dumps(event()).encode())[1]["ingest_id"]
    bridge._mqtt_intake.stop()
    restored = make_bridge(tmp_path)
    restored._handle_event = Mock()
    restored._mqtt_intake.start()
    try:
        receipt = restored.get_rest_receipt(key)
        assert receipt["stage"] == "quarantined"
        assert receipt["reason"] == "recovery_requires_review"
        restored._handle_event.assert_not_called()
    finally:
        restored._mqtt_intake.stop()


@pytest.mark.parametrize("expired", [False, True])
def test_restart_only_resumes_unexecuted_unexpired_input(tmp_path, expired):
    bridge = expiry_bridge(tmp_path, capture=True, queue=True)
    bridge._mqtt_intake.start()
    raw = json.dumps(event(occurred_at=time.time() - (100 if expired else 1))).encode()
    key = request(bridge, raw)[1]["ingest_id"]
    bridge._mqtt_intake.stop()
    restored = expiry_bridge(tmp_path, queue=True)
    restored._mqtt_intake.start()
    try:
        wait_until(lambda: restored.get_rest_receipt(key)["stage"] in {"complete", "quarantined"})
        if expired:
            assert restored.get_rest_receipt(key)["reason"] == "expired:expired"
            restored._executor._publish_edgex_command.assert_not_called()
        else:
            wait_until(lambda: restored._executor._publish_edgex_command.call_count == 1)
    finally:
        restored._mqtt_intake.stop()


def test_mqtt_and_rest_duplicate_share_first_deadline_and_execution(tmp_path):
    bridge = expiry_bridge(tmp_path, queue=True)
    intake = bridge._mqtt_intake
    intake.start()
    payload = event()
    try:
        rest_key = request(bridge, json.dumps(payload).encode())[1]["ingest_id"]
        payload.pop("topic")
        mqtt_key = intake.accept(TOPIC, json.dumps(payload).encode())
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: request(bridge, json.dumps({**payload, "topic": TOPIC}).encode()), range(4)))
        assert all(status == 202 for status, _ in results)
        wait_until(lambda: bridge._executor._publish_edgex_command.call_count == 1)
        wait_until(lambda: intake.store.get(mqtt_key)["stage"] == "quarantined")
        plans = [json.loads(intake.store.get(k)["projection_json"]) for k in [rest_key, mqtt_key]]
        assert plans[0]["deadlines"] == plans[1]["deadlines"]
        assert bridge._executor._publish_edgex_command.call_count == 1
    finally:
        intake.stop()


def test_capture_only_blocks_direct_enqueue_worker_and_manual_approval(tmp_path):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True, rest_enabled=True)
    bridge._mqtt_intake.start()
    try:
        assert bridge.enqueue_rest_event(event())
        bridge._start_rest_action_worker()
        assert bridge._rest_action_worker is None
        assert bridge._rest_action_queue.empty()
        bridge._sites.push_pending("manual", TOPIC, event(), None)
        bridge._execute_action = Mock()
        assert bridge.approve_event("manual")[0] is False
        bridge._execute_action.assert_not_called()
        assert bridge.get_pending_events()
    finally:
        bridge._mqtt_intake.stop()


def test_auth_storage_unavailability_and_receipt_namespace(tmp_path, monkeypatch):
    bridge = make_bridge(tmp_path, mqtt_capture_only=True)
    raw = json.dumps(event()).encode()
    assert request(bridge, raw)[0] == 503
    bridge._mqtt_intake.start()
    try:
        monkeypatch.setattr(rest, "_INTERNAL_TOKEN", "test-only")
        assert request(bridge, raw)[0] == 401
        assert bridge._mqtt_intake.store.pending(namespace="action_rest") == []
        monkeypatch.setattr(rest, "_INTERNAL_TOKEN", None)
        key = bridge._mqtt_intake.accept(TOPIC, raw)
        assert request(bridge, b"", path="/receipts/" + key, method="GET")[0] == 404
        monkeypatch.setattr(bridge, "get_rest_receipt", Mock(side_effect=sqlite3.OperationalError("test")))
        assert request(bridge, b"", path="/receipts/missing", method="GET")[0] == 503
    finally:
        bridge._mqtt_intake.stop()
    assert request(bridge, raw)[0] == 503


def test_legacy_http_response_stays_compatible():
    bridge = SimpleNamespace(enqueue_rest_event=Mock(return_value=True))
    assert request(bridge, b'{"topic":"test","value":1}') == (200, {"status": "ok", "queued": True})
    bridge.enqueue_rest_event.assert_called_once_with({"value": 1}, topic="test")


def test_snapshot_commit_failure_preserves_raw_without_execution(tmp_path):
    bridge = expiry_bridge(tmp_path, capture=True, queue=True)
    intake = bridge._mqtt_intake
    intake.start()
    try:
        with intake.store._connect() as conn:
            conn.execute("CREATE TRIGGER fail_snapshot BEFORE UPDATE OF projection_json ON sensor_ingest_state BEGIN SELECT RAISE(ABORT,'test snapshot'); END")
        raw = json.dumps(event()).encode()
        assert request(bridge, raw)[0] == 503
        key, = intake.store.pending(namespace="action_rest")
        assert base64.b64decode(intake.store.get(key)["raw"]["payload_base64"]) == raw
        with intake.store._connect() as conn:
            conn.execute("DROP TRIGGER fail_snapshot")
        intake._process(key)
        assert bridge.get_rest_receipt(key)["reason"] == "held:execution_context_missing"
        bridge._executor._publish_edgex_command.assert_not_called()
    finally:
        intake.stop()


def test_process_death_after_http_receipt_never_reexecutes(tmp_path):
    script = '''
import json, os, sys, time
from pathlib import Path
from tests.test_action_expiry import make_bridge
from tests.test_action_rest_intake import event
bridge = make_bridge(Path(sys.argv[1]), capture=True, queue=True)
intake = bridge._mqtt_intake
intake.start()
result = bridge.accept_rest_event(json.dumps(event()).encode())
Path(sys.argv[1], 'receipt.json').write_text(json.dumps(result))
bridge._handle_event = lambda *args, **kwargs: os._exit(23)
intake._process(result['ingest_id'])
'''
    assert subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=15).returncode == 23
    key = json.loads((tmp_path / "receipt.json").read_text())["ingest_id"]
    bridge = expiry_bridge(tmp_path, queue=True)
    bridge._mqtt_intake.start()
    try:
        receipt = bridge.get_rest_receipt(key)
        assert receipt["stage"] == "quarantined" and receipt["reason"] == "execution_unconfirmed"
        row = bridge._mqtt_intake.store.get(key)
        code, duplicate = request(bridge, base64.b64decode(row["raw"]["payload_base64"]))
        assert code == 202
        wait_until(lambda: bridge.get_rest_receipt(duplicate["ingest_id"])["stage"] == "quarantined")
        bridge._executor._publish_edgex_command.assert_not_called()
    finally:
        bridge._mqtt_intake.stop()


@pytest.mark.parametrize("source_id", [None, "original-alert-id"])
def test_public_alert_forwarding_preserves_identity_on_retry(tmp_path, source_id):
    from src.api._event_forwarding import (
        build_alert_action_payload,
        build_alert_action_topic,
    )
    from src.api.schemas.event import AlertIn

    alert = AlertIn(camera_id="cam1", event_type="helmet", confidence=0.9,
                    timestamp=1700000000, event_id=source_id)
    first = build_alert_action_payload(alert)
    second = build_alert_action_payload(alert)
    assert first["event_id"] == second["event_id"]
    if source_id:
        assert first["event_id"] == source_id
    bridge = make_bridge(tmp_path, mqtt_capture_only=True)
    bridge._mqtt_intake.start()
    try:
        status, body = request(bridge, json.dumps({**first, "topic": build_alert_action_topic(alert)}).encode())
        assert status == 202 and body["event_id"] == first["event_id"]
    finally:
        bridge._mqtt_intake.stop()
