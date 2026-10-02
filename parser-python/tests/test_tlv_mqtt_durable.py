import base64
import json
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mqtt.manager import Manager  # noqa: E402
from test_sensor_projection import envelope, service  # noqa: E402

from src.ingest_store import IngestStore  # noqa: E402


def sensor_service(store):
    instance = service(store)
    instance._receipt_lock = threading.RLock()
    return instance


def wire_payload():
    return json.dumps({"payload": envelope()["payload"]}).encode()


def receipts(store):
    with store._connect() as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT ingest_id FROM ingest_receipts ORDER BY rowid"
            )
        ]


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"{",
        b"[]",
        b"{}",
        b'{"uplink_message": []}',
        b'{"payload": 123}',
        b'{"payload": "abc", "x": NaN}',
        b'{"payload": "abc", "dev_eui": {"wrong": "type"}}',
        b'{"payload": "abc", "end_device_ids": {"application_ids": []}}',
    ],
)
def test_bad_outer_envelope_is_retained_and_quarantined(tmp_path, payload):
    store = IngestStore(tmp_path / "ingest.db")
    instance = sensor_service(store)
    instance.process_mqtt_message("app/dev/up", payload)
    receipt = store.get(receipts(store)[0])
    assert receipt["stage"] == "quarantined"
    assert receipt["raw_format"] == "mqtt_bytes_base64"
    assert base64.b64decode(receipt["raw"]["payload_base64"]) == payload
    assert not instance._edgex_forwarder.mock_calls


def test_raw_recovery_uses_same_receipt_without_live_outputs(tmp_path):
    store = IngestStore(tmp_path / "ingest.db")
    ingest_id = store.accept_mqtt("tlv_mqtt", "app/dev/up", wire_payload())
    instance = sensor_service(store)
    instance._recover_receipt(store.get(ingest_id))
    assert receipts(store) == [ingest_id]
    assert store.get(ingest_id)["stage"] == "ready"
    assert not instance._edgex_forwarder.mock_calls


def test_live_path_prepares_existing_raw_receipt_before_output(tmp_path):
    store = IngestStore(tmp_path / "ingest.db")
    instance = sensor_service(store)
    observed = []
    instance._edgex_forwarder.publish.side_effect = lambda *args, **kw: observed.append(
        store.get(receipts(store)[0])["stage"]
    )
    instance.process_mqtt_message("app/dev/up", wire_payload())
    assert len(receipts(store)) == 1
    assert store.get(receipts(store)[0])["stage"] == "ready"
    # Current live forwarding remains enabled; recovery cannot repeat it.
    assert observed == ["ready"]
    snapshot = list(instance._edgex_forwarder.mock_calls)
    instance._recover_receipt(store.get(receipts(store)[0]))
    assert instance._edgex_forwarder.mock_calls == snapshot


def test_storage_failure_propagates_before_decoding(tmp_path):
    store = IngestStore(tmp_path / "ingest.db")
    instance = sensor_service(store)
    with patch.object(
        store, "accept_mqtt", side_effect=sqlite3.OperationalError("disk full")
    ):
        with pytest.raises(sqlite3.OperationalError):
            instance.process_mqtt_message("app/dev/up", b"invalid")
    assert receipts(store) == []


def test_live_and_recovery_race_keeps_one_projection_and_one_live_output(tmp_path):
    store = IngestStore(tmp_path / "ingest.db")
    instance = sensor_service(store)
    accepted, release = threading.Event(), threading.Event()
    original = instance._prepare_mqtt_receipt
    failures = []

    def prepare(receipt, *, emit_outputs):
        if emit_outputs:
            accepted.set()
            assert release.wait(5)
        return original(receipt, emit_outputs=emit_outputs)

    def run(callback):
        try:
            callback()
        except Exception as error:
            failures.append(error)

    with patch.object(instance, "_prepare_mqtt_receipt", side_effect=prepare):
        live = threading.Thread(
            target=run,
            args=(lambda: instance.process_mqtt_message("app/dev/up", wire_payload()),),
        )
        live.start()
        assert accepted.wait(5)
        key = receipts(store)[0]
        stale_receipt = store.get(key)
        assert stale_receipt["stage"] == "pending"
        recovering = threading.Thread(
            target=run, args=(lambda: instance._recover_receipt(stale_receipt),)
        )
        recovering.start()
        release.set()
        live.join(timeout=5)
        recovering.join(timeout=5)
        assert not live.is_alive() and not recovering.is_alive()
        assert failures == []
        assert instance._prepare_mqtt_receipt.call_count == 1
    assert receipts(store) == [key] and store.get(key)["stage"] == "ready"
    instance._edgex_forwarder.publish.assert_called_once()


def test_durable_manager_requires_stable_id(monkeypatch):
    monkeypatch.delenv("AIOT_MQTT_CLIENT_ID_PREFIX", raising=False)
    with pytest.raises(ValueError, match="AIOT_MQTT_CLIENT_ID_PREFIX"):
        Manager(None, durable=True)


def test_manager_uses_shared_subscriber_and_polls_outside_callback():
    processor = Mock()
    config = SimpleNamespace(host="localhost", port=1883, username="", password="")
    with patch("src.protocols.mqtt_subscriber.MqttTopicSubscriber") as subscriber_type:
        manager = Manager(None, durable=True, client_id_prefix="isolated-parser")
        manager.set_processor(processor)
        manager.connect_client("lab", config)
        options = subscriber_type.call_args.kwargs
        assert options["client_id"] == "isolated-parser-lab"
        assert options["durable"] is True
        options["message_handler"]("app/dev/up", b"original")
        processor.process_mqtt_message.assert_called_once_with(
            "app/dev/up", b"original"
        )
        manager.poll()
        assert subscriber_type.return_value.connect.call_count == 2
        manager.disconnect_all()
        subscriber_type.return_value.disconnect.assert_called_once()


def test_durable_init_does_not_hide_client_configuration_failure():
    config = SimpleNamespace(host="localhost", port=1883, username="", password="")
    configs = SimpleNamespace(ns_park=config, lab=config)
    manager = Manager(configs, durable=True, client_id_prefix="parser")
    with patch.object(
        manager, "connect_client", side_effect=RuntimeError("Paho 2 required")
    ):
        with pytest.raises(RuntimeError, match="Paho 2"):
            manager.init(configs, Mock())


def test_parser_docker_sources_support_durable_imports_in_isolation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text()
    stage = dockerfile.split(" AS parser\n", 1)[1].split("\nFROM ", 1)[0]
    app = tmp_path / "app"
    shutil.copytree(
        root / "parser-python",
        app,
        ignore=shutil.ignore_patterns("__pycache__", "tests"),
    )
    for line in stage.splitlines():
        if line.startswith("COPY src/"):
            *sources, target = line.split()[1:]
            destination = app / target
            destination.mkdir(parents=True, exist_ok=True)
            for source in sources:
                shutil.copy(root / source, destination)
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mqtt.manager import Manager
from service.sensor_service import SensorService
from src.ingest_store import IngestStore
from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from src.sensor_record import SensorRecordStore, SensorRecordWorker
store = SensorRecordStore(sys.argv[1] + "/test-ingest.db")
SensorRecordWorker(store).drain_once()
manager = Manager(None, durable=True, client_id_prefix="test-parser")
assert manager.get_all_clients_status() == {}
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(app)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
