"""QoS 1 record transport with broker ACK; no legacy outbox or device commands."""

import json
import math
import time
from threading import Event

import paho.mqtt.client as mqtt

from ..record_delivery import BrokerAck, record_topic


class MqttRecordTransport:
    def __init__(
        self,
        broker: str,
        *,
        port: int = 1883,
        transport: str = "tcp",
        username: str | None = None,
        password: str | None = None,
        tls_context=None,
    ):
        if not broker or transport not in {"tcp", "unix"} or not 1 <= port <= 65535:
            raise ValueError("A broker and supported MQTT transport/port are required")
        self.broker, self.port, self.transport = broker, port, transport
        self.username, self.password, self.tls_context = username, password, tls_context

    @property
    def endpoint(self) -> str:
        return json.dumps(
            {
                "broker": self.broker,
                "port": self.port,
                "transport": self.transport,
                "tls": self.tls_context is not None,
            },
            sort_keys=True,
        )

    def send(self, job: dict, *, timeout: float) -> BrokerAck:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A finite positive timeout is required")
        topic = record_topic(job["destination"])
        if topic != job["topic"]:
            raise ValueError("Frozen destination and topic differ")
        deadline = time.monotonic() + timeout
        connected = Event()
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
            transport=self.transport,
            clean_session=True,
            reconnect_on_failure=False,
        )
        client.connect_timeout = timeout
        if self.username is not None:
            client.username_pw_set(self.username, self.password)
        if self.tls_context is not None:
            client.tls_set_context(self.tls_context)
        client.on_connect = lambda *args: connected.set()
        started = False
        try:
            client.connect(self.broker, self.port)
            client.loop_start()
            started = True
            if (
                not connected.wait(max(0, deadline - time.monotonic()))
                or not client.is_connected()
            ):
                raise TimeoutError("MQTT CONNACK was not successful")
            result = client.publish(topic, job["wire_payload"], qos=1, retain=False)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError("MQTT publish was not accepted")
            result.wait_for_publish(timeout=max(0, deadline - time.monotonic()))
            if not result.is_published():
                raise TimeoutError("MQTT PUBACK was not received")
            return BrokerAck(job["job_id"])
        finally:
            client.disconnect()
            if started:
                client.loop_stop()
