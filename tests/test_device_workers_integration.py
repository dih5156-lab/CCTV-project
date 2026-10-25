"""격리 MQTT에서 느린 실호출 모의 중 저장 ACK와 다른 장치 진행을 확인한다."""

import json
import sqlite3
from threading import Event
from uuid import uuid4

from src.protocols.mqtt_subscriber import MqttTopicSubscriber
from tests.test_device_workers import inbox, make_workers
from tests.test_mqtt_durable_integration import broker as broker
from tests.test_mqtt_durable_integration import publish, wait_until
from tests.test_mqtt_durable_integration import publisher as publisher


def test_real_mqtt_slow_device_keeps_ack_status_and_second_device_running(broker, publisher, tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    entered, release = Event(), Event()
    getattr(devices["slow"], method).side_effect = lambda **kwargs: entered.set() or release.wait(8)
    topic = "worker-test/" + uuid4().hex
    subscriber = MqttTopicSubscriber(broker=broker, topics=[topic], durable=True, transport="unix",
                                     client_id="worker-intake-" + uuid4().hex, message_handler=workers.accept)
    acknowledgements = []
    original_build = subscriber._build_client
    def build():
        client = original_build()
        acknowledge = client.ack
        def ack(mid, qos):
            assert inbox(workers)  # ACK보다 먼저 원본 commit이 끝나야 한다.
            acknowledgements.append(mid)
            return acknowledge(mid, qos)
        client.ack = ack
        return client
    subscriber._build_client = build
    workers.start()
    try:
        assert subscriber.connect()
        for data in [request(), request("fast", "two"), request(action="request_status")]:
            publish(publisher, topic, json.dumps(data).encode())
        wait_until(entered.is_set)
        wait_until(lambda: len(acknowledgements) == 3)
        wait_until(lambda: getattr(devices["fast"], method).call_count == 1)
        wait_until(lambda: any(call.args[0]["request_id"] == request()["request_id"]
                              for call in workers.publish.call_args_list))
        assert inbox(workers)[0]["state"] == "running"
        assert not release.is_set()
    finally:
        subscriber.disconnect()
        release.set()
        assert workers.stop()


def test_real_mqtt_store_failure_leaves_unacked_packet_for_redelivery(broker, publisher, tmp_path):
    workers, devices, request, method = make_workers(tmp_path)
    topic = "worker-store-test/" + uuid4().hex
    deliveries, acknowledgements = [], []
    def receive(topic, raw):
        deliveries.append(raw)
        workers.accept(topic, raw)
    subscriber = MqttTopicSubscriber(broker=broker, topics=[topic], durable=True, transport="unix",
                                     client_id="worker-store-" + uuid4().hex, message_handler=receive)
    original_build = subscriber._build_client
    def build():
        client = original_build()
        acknowledge = client.ack
        def ack(mid, qos):
            acknowledgements.append(mid)
            return acknowledge(mid, qos)
        client.ack = ack
        return client
    subscriber._build_client = build
    with sqlite3.connect(workers.cache._database_path) as conn:
        conn.execute("CREATE TRIGGER fail BEFORE INSERT ON device_mqtt_inbox BEGIN SELECT RAISE(ABORT,'full'); END")
    workers.start()
    try:
        assert subscriber.connect()
        publish(publisher, topic, json.dumps(request()).encode())
        wait_until(lambda: subscriber._restart_requested)
        assert not acknowledgements and not inbox(workers)
        getattr(devices["slow"], method).assert_not_called()
        with sqlite3.connect(workers.cache._database_path) as conn:
            conn.execute("DROP TRIGGER fail")
        subscriber._last_attempt_time = 0
        assert subscriber.reconnect_if_needed()
        wait_until(lambda: len(acknowledgements) == 1)
        wait_until(lambda: getattr(devices["slow"], method).call_count == 1)
        assert len(deliveries) == 2 and deliveries[0] == deliveries[1]
        assert len(inbox(workers)) == 1
    finally:
        subscriber.disconnect()
        assert workers.stop()
