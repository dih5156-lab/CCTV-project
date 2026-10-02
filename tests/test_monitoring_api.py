import secrets
import sqlite3

from fastapi.testclient import TestClient

from src.monitoring_api import create_monitoring_api
from src.monitoring_contract import parse_record, validate_ack
from tests.test_monitoring_local import setup


def test_broker_unavailable_does_not_hide_committed_ack(tmp_path):
    receiver, store, binding, wire = setup(tmp_path)
    token = secrets.token_urlsafe(32)
    store.accept(binding, wire)
    headers = {"Authorization": "Bearer " + token}
    with TestClient(
        create_monitoring_api(store, binding, token, receiver=receiver)
    ) as client:
        assert client.get("/v1/health", headers=headers).status_code == 503
        reply = client.get(
            "/v1/deliveries/" + parse_record(wire).delivery_id,
            params=binding.as_dict(),
            headers=headers,
        )
        validate_ack(reply.json(), binding, parse_record(wire))


def test_status_requires_auth_and_correct_binding(tmp_path):
    receiver, store, binding, wire = setup(tmp_path)
    token = secrets.token_urlsafe(32)
    with TestClient(create_monitoring_api(store, binding, token)) as client:
        url = "/v1/deliveries/" + parse_record(wire).delivery_id
        assert client.get(url, params=binding.as_dict()).status_code == 401
        headers = {"Authorization": "Bearer " + token}
        wrong = {**binding.as_dict(), "origin_id": "another-origin"}
        assert client.get(url, params=wrong, headers=headers).status_code == 403
        missing = client.get(url, params=binding.as_dict(), headers=headers)
        assert missing.status_code == 404
        assert missing.json()["binding"] == binding.as_dict()
        store.accept(binding, wire)
        response = client.get(url, params=binding.as_dict(), headers=headers)
        assert response.status_code == 200
        validate_ack(response.json(), binding, parse_record(wire))
        assert response.headers["cache-control"] == "no-store"


def test_database_lookup_error_is_not_missing_or_stored(tmp_path):
    receiver, store, binding, wire = setup(tmp_path)
    token = secrets.token_urlsafe(32)
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE monitoring_receipts")
    with TestClient(create_monitoring_api(store, binding, token)) as client:
        response = client.get(
            "/v1/deliveries/job-1",
            params=binding.as_dict(),
            headers={"Authorization": "Bearer " + token},
        )
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "storage_unavailable"}}
