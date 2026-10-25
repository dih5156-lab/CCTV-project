"""격리 loopback HTTP에서 느린 실행과 접수 commit 경계를 확인한다."""

import http.client
import json
from contextlib import contextmanager
from threading import Event

from src.protocols import rest
from tests.test_action_mqtt_intake import make_bridge, wait_until
from tests.test_action_rest_intake import event


@contextmanager
def running_rest(tmp_path, monkeypatch, *, capture=False):
    monkeypatch.setattr(rest, "_INTERNAL_TOKEN", "local-test-token")
    bridge = make_bridge(tmp_path, rest_enabled=True, rest_host="127.0.0.1", rest_port=0,
                         mqtt_capture_only=capture)
    bridge._mqtt_intake.start()
    bridge._rest_receiver.start()
    try:
        yield bridge, bridge._rest_receiver._server.server_address[1]
    finally:
        bridge._rest_receiver.stop()
        bridge._rest_receiver._server.server_close()
        bridge._mqtt_intake.stop()


def http_request(port, path, raw=None, *, token="local-test-token"):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("POST" if raw is not None else "GET", path, raw,
                       {"X-Internal-Token": token, "Content-Type": "application/json"})
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def test_http_accept_and_lookup_continue_during_slow_execution(tmp_path, monkeypatch):
    with running_rest(tmp_path, monkeypatch) as (bridge, port):
        entered, release = Event(), Event()

        def slow(payload, **kwargs):
            if payload["event_id"] == "slow":
                entered.set()
                assert release.wait(5)

        bridge._handle_event = slow
        try:
            code, first = http_request(port, "/events", json.dumps(event("slow")))
            assert code == 202 and entered.wait(2)
            code, second = http_request(port, "/events", json.dumps(event("next")))
            assert code == 202
            code, state = http_request(port, "/receipts/" + second["ingest_id"])
            assert code == 200 and state["stage"] == "pending"
            code, state = http_request(port, "/receipts/" + first["ingest_id"])
            assert code == 200 and state["stage"] == "ready"
            release.set()
            wait_until(lambda: bridge.get_rest_receipt(second["ingest_id"])["stage"] == "complete")
        finally:
            release.set()


def test_http_storage_failure_and_auth_do_not_return_false_acceptance(tmp_path, monkeypatch):
    with running_rest(tmp_path, monkeypatch, capture=True) as (bridge, port):
        raw = json.dumps(event())
        assert http_request(port, "/events", raw, token="wrong")[0] == 401
        store = bridge._mqtt_intake.store
        with store._connect() as conn:
            conn.execute("CREATE TRIGGER fail_http BEFORE INSERT ON ingest_receipts BEGIN SELECT RAISE(ABORT,'test'); END")
        assert http_request(port, "/events", raw)[0] == 503
        assert store.pending(namespace="action_rest") == []
        with store._connect() as conn:
            conn.execute("DROP TRIGGER fail_http")
        code, body = http_request(port, "/events", raw)
        assert code == 202
        assert http_request(port, "/receipts/" + body["ingest_id"], token="wrong")[0] == 401
        assert http_request(port, "/receipts/" + body["ingest_id"])[0] == 200
