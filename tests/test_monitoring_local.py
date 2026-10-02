"""The MQTT subscriber transport ACK follows commit or durable rejection."""

import sqlite3
from types import SimpleNamespace

import pytest

from src.monitoring_receiver import MonitoringReceiverStore
from src.protocols.monitoring_local import LocalMonitoringReader, LocalMqttReceiver
from tests.test_monitoring_delivery import prepared


def setup(tmp_path):
    source, binding = prepared(tmp_path / "source.db")
    store = MonitoringReceiverStore(
        tmp_path / "receiver.db", consumer_id=binding.consumer_id
    )
    store.register_binding(binding)
    receiver = LocalMqttReceiver(store, binding, "/tmp/not-connected.sock")
    with sqlite3.connect(source.path) as connection:
        wire = connection.execute(
            "SELECT wire_payload FROM monitoring_delivery"
        ).fetchone()[0]
    return receiver, store, binding, wire


class Client:
    def __init__(self, on_ack=lambda: None):
        self.acked, self.disconnected, self.on_ack = [], False, on_ack

    def ack(self, mid, qos):
        self.on_ack()
        self.acked.append((mid, qos))
        return 0

    def disconnect(self):
        self.disconnected = True


def message(receiver, wire):
    return SimpleNamespace(
        topic=receiver.topic, qos=1, retain=False, payload=wire, mid=7
    )


def test_transport_ack_after_commit_and_duplicate_counter(tmp_path):
    receiver, store, binding, wire = setup(tmp_path)

    def check_commit():
        with sqlite3.connect(store.path) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM monitoring_records"
                ).fetchone()[0]
                == 1
            )

    client = Client(check_commit)
    receiver._on_message(client, None, message(receiver, wire))
    receiver._on_message(client, None, message(receiver, wire))
    assert client.acked == [(7, 1), (7, 1)]
    assert receiver.status()["duplicates"] == 1


@pytest.mark.parametrize("reject_failure", [False, True])
def test_invalid_payload_ack_requires_durable_rejection(tmp_path, reject_failure):
    receiver, store, binding, wire = setup(tmp_path)
    if reject_failure:
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "CREATE TRIGGER reject_fail BEFORE INSERT ON monitoring_rejected BEGIN SELECT RAISE(ABORT,'full'); END"
            )
    client = Client()
    receiver._on_message(client, None, message(receiver, b"invalid"))
    assert bool(client.acked) is not reject_failure
    assert client.disconnected is reject_failure


def test_storage_failure_leaves_transport_unacknowledged_and_reader_raises(tmp_path):
    receiver, store, binding, wire = setup(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER insert_fail BEFORE INSERT ON monitoring_receipts BEGIN SELECT RAISE(ABORT,'full'); END"
        )
    client = Client()
    receiver._on_message(client, None, message(receiver, wire))
    assert client.acked == [] and client.disconnected
    reader = LocalMonitoringReader(store)
    assert reader.get_delivery_status(binding, "missing", timeout=1) is None
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE monitoring_receipts")
    with pytest.raises(sqlite3.OperationalError):
        reader.get_delivery_status(binding, "missing", timeout=1)
